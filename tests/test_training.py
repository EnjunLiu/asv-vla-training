from __future__ import annotations

import json

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from asv_training.dataset import (  # noqa: E402
    EpochSynonymDataset,
    InstructionMetadata,
    mask_task_conditioned_entity_geometry,
)
from asv_training.train import (  # noqa: E402
    _FrameGroupedBatchSampler,
    _acceptance,
    _checkpoint_selection_eligible,
    _checkpoint_selection_eligible_for_modality,
    compute_action_metrics,
    fit_label_mean_action_baseline,
    predict_label_mean_action_baseline,
    _selection_score,
    _improvement_fraction,
    _write_progress,
)


class _FakeFeatureDataset:
    def __init__(self) -> None:
        self.samples = [
            ("RUN", "RUN:1:0:1", "red_a"),
            ("RUN", "RUN:1:0:1", "red_b"),
            ("RUN", "RUN:1:0:1", "stop_a"),
            ("RUN", "RUN:1:1:2", "red_a"),
            ("RUN", "RUN:1:1:2", "red_b"),
            ("RUN", "RUN:1:1:2", "stop_a"),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def sample_metadata(self, index: int):
        run_id, frame_key, instruction_id = self.samples[index]
        return {
            "run_id": run_id,
            "frame_key": frame_key,
            "sample_id": f"sample_{index}",
            "instruction_id": instruction_id,
            "target_stop": instruction_id == "stop_a",
        }

    def __getitem__(self, index: int):
        metadata = self.sample_metadata(index)
        return {
            "instruction_id": metadata["instruction_id"],
            "target_action": torch.zeros(2),
            "target_stop": torch.tensor(
                [float(metadata["target_stop"])], dtype=torch.float32
            ),
        }


def _instruction(
    instruction_id: str,
    intent_group: str,
    action: str,
    target_attribute: str,
    distance_bucket: str,
) -> InstructionMetadata:
    return InstructionMetadata(
        instruction_id=instruction_id,
        intent_group=intent_group,
        action=action,
        target_attribute=target_attribute,
        distance_bucket=distance_bucket,
        split="train",
    )


def test_epoch_synonym_dataset_selects_one_per_frame_label() -> None:
    instructions = {
        "red_a": _instruction(
            "red_a", "follow_red_3m", "follow", "color:red", "3m"
        ),
        "red_b": _instruction(
            "red_b", "follow_red_3m", "follow", "color:red", "3m"
        ),
        "stop_a": _instruction(
            "stop_a", "stop", "stop", "none", "none"
        ),
    }
    first = EpochSynonymDataset(_FakeFeatureDataset(), instructions, seed=17)
    second = EpochSynonymDataset(_FakeFeatureDataset(), instructions, seed=17)

    assert len(first) == 4
    assert [first[index]["instruction_id"] for index in range(len(first))] == [
        second[index]["instruction_id"] for index in range(len(second))
    ]
    first.set_epoch(1)
    groups = [
        (
            first[index]["metadata"]["task_label"],
            first[index]["instruction_id"],
        )
        for index in range(len(first))
    ]
    assert sum(label.startswith("follow|") for label, _ in groups) == 2
    assert sum(label.startswith("stop|") for label, _ in groups) == 2
    assert first.frame_group_indices == ((0, 1), (2, 3))


def test_frame_grouped_sampler_never_splits_an_observation() -> None:
    groups = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    sampler = _FrameGroupedBatchSampler(groups, batch_size=6, seed=17)

    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(9))
    for group in groups:
        assert any(set(group) <= set(batch) for batch in batches)


def test_metrics_expert_and_speed_contract() -> None:
    target = np.zeros((2, 2), dtype=np.float32)
    target[0] = (0.3, 0.0)
    stop = np.asarray([False, True])
    labels = ["follow|color:red|3m", "stop|none|none"]
    logits = np.asarray([-20.0, 20.0], dtype=np.float32)

    metrics = compute_action_metrics(target, target, logits, stop, labels)

    assert metrics["action_error_m"] == pytest.approx(0.0)
    assert metrics["lateral_action_error_m"] == pytest.approx(0.0)
    assert metrics["stop_classification"]["f1"] == pytest.approx(1.0)
    assert metrics["stop_drift"]["within_0_10m_rate"] == pytest.approx(1.0)
    assert metrics["action_bound"]["violation_count"] == 0
    assert metrics["per_label"]["follow|color:red|3m"]["sample_count"] == 1


def test_task_conditioned_geometry_matches_online_single_target_contract() -> None:
    geometry = np.arange(4 * 16, dtype=np.float32).reshape(4, 16)
    mask = np.asarray([True, True, True, False])
    entity_ids = np.asarray(
        ["target_blue", "target_red", "target_left", ""], dtype=object
    )

    red_values, red_mask, red_valid = mask_task_conditioned_entity_geometry(
        geometry, mask, entity_ids, "跟随红色目标船，保持4米距离"
    )
    assert red_valid is True
    assert red_mask.tolist() == [True, False, False, False]
    assert np.allclose(red_values[0], geometry[1])

    stop_values, stop_mask, stop_valid = mask_task_conditioned_entity_geometry(
        geometry, mask, entity_ids, "停止"
    )
    assert stop_valid is True
    assert not np.any(stop_mask)
    assert not np.any(stop_values)

    _, missing_mask, missing_valid = mask_task_conditioned_entity_geometry(
        geometry, mask, entity_ids, "跟随右侧目标船，保持4米距离"
    )
    assert missing_valid is False
    assert not np.any(missing_mask)


def test_label_mean_baseline_and_improvement() -> None:
    actions = np.zeros((4, 2), dtype=np.float32)
    actions[1, 0] = 2.0
    actions[2, 1] = 1.0
    actions[3, 1] = 3.0
    labels = ["a", "a", "b", "b"]

    means = fit_label_mean_action_baseline(actions, labels)
    prediction, logits = predict_label_mean_action_baseline(means, ["a", "b"])

    assert np.allclose(prediction[0, 0], 1.0)
    assert np.allclose(prediction[1, 1], 2.0)
    assert np.all(logits == -20.0)
    assert _improvement_fraction(0.7, 1.0) == pytest.approx(0.3)


def test_checkpoint_selection_requires_both_stop_gates() -> None:
    config = {
        "selection_constraints": {
            "minimum_stop_f1": 0.95,
            "minimum_stop_within_0_10m_rate": 0.95,
        }
    }
    metrics = {
        "stop_classification": {"f1": 0.97},
        "stop_drift": {"within_0_10m_rate": 0.96},
    }

    assert _checkpoint_selection_eligible(metrics, config)
    metrics["stop_classification"]["f1"] = 0.94
    assert not _checkpoint_selection_eligible(metrics, config)
    metrics["stop_classification"]["f1"] = 0.97
    metrics["stop_drift"]["within_0_10m_rate"] = 0.94
    assert not _checkpoint_selection_eligible(metrics, config)
    assert _checkpoint_selection_eligible_for_modality(
        metrics, config, "entity_only"
    )


def test_single_point_selection_rejects_trajectory_metric() -> None:
    metrics = {"action_error_m": 0.1}
    assert _selection_score(
        metrics, {"selection_metric": "action_error_m"}
    ) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="single-step training supports only"):
        _selection_score(metrics, {"selection_metric": "ade_plus_half_fde"})


def test_acceptance_requires_every_frozen_gate() -> None:
    metrics = {
        "action_error_m": 0.6,
        "stop_drift": {"within_0_10m_rate": 0.95},
        "stop_classification": {"f1": 0.95},
        "action_bound": {"violation_rate": 0.0},
        "invalid_count": 0,
    }
    baseline = {"action_error_m": 1.0}
    config = {
        "minimum_ade_improvement_over_label_mean": 0.30,
        "minimum_stop_within_0_10m_rate": 0.95,
        "minimum_stop_f1": 0.95,
        "maximum_speed_violation_rate": 0.0,
        "maximum_invalid_count": 0,
    }

    result = _acceptance(metrics, baseline, config)

    assert result["passed"] is True
    metrics["stop_classification"]["f1"] = 0.94
    assert _acceptance(metrics, baseline, config)["passed"] is False


def test_single_point_acceptance_rejects_fde_gate() -> None:
    with pytest.raises(ValueError, match="cannot define an FDE gate"):
        _acceptance(
            {
                "action_error_m": 0.6,
                "stop_drift": {"within_0_10m_rate": 0.95},
                "stop_classification": {"f1": 0.95},
                "action_bound": {"violation_rate": 0.0},
                "invalid_count": 0,
            },
            {"action_error_m": 1.0},
            {"minimum_fde_improvement_over_label_mean": 0.3},
        )


def test_progress_snapshot_is_atomic_and_records_training_identity(tmp_path) -> None:
    path = tmp_path / "progress.json"

    _write_progress(
        path,
        output_root=tmp_path,
        stage="epoch_running",
        seed=29,
        modality="full",
        epoch_started=1,
        epoch_completed=0,
        epochs_total=80,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "training_progress_v1"
    assert payload["stage"] == "epoch_running"
    assert payload["pid"] > 0
    assert payload["seed"] == 29
    assert payload["epoch_started"] == 1
    assert not list(tmp_path.glob(".progress.json.*.tmp"))
