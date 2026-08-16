from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

import asv_training.dataset as dataset_module  # noqa: E402
from asv_training.dataset import (  # noqa: E402
    FrozenFeatureDataset,
    POLICY_INPUT_KEYS,
    _SampleRef,
    mask_task_conditioned_entity_geometry,
    policy_inputs_from_batch,
    task_target_id_from_instruction,
)


ENTITY_IDS = (
    "target_blue",
    "target_red",
    "target_left",
    "target_right",
)


def _geometry() -> np.ndarray:
    return np.arange(4 * 16, dtype=np.float32).reshape(4, 16)


@pytest.mark.parametrize(
    ("instruction", "target_id"),
    (
        ("follow target_red", "target_red"),
        ("follow target_blue", "target_blue"),
        ("follow target_left", "target_left"),
        ("follow target_right", "target_right"),
    ),
)
def test_task_conditioned_mask_keeps_only_instruction_target(
    instruction: str, target_id: str
) -> None:
    geometry = _geometry()
    mask = np.ones(4, dtype=np.bool_)

    masked, masked_mask, selected = mask_task_conditioned_entity_geometry(
        geometry, mask, ENTITY_IDS, instruction
    )

    assert selected
    assert task_target_id_from_instruction(instruction) == target_id
    assert masked_mask.tolist() == [True, False, False, False]
    np.testing.assert_array_equal(masked[0], geometry[ENTITY_IDS.index(target_id)])
    np.testing.assert_array_equal(masked[1:], 0.0)


def test_stop_mask_is_all_false_and_zero() -> None:
    masked, masked_mask, selected = mask_task_conditioned_entity_geometry(
        _geometry(),
        np.ones(4, dtype=np.bool_),
        ENTITY_IDS,
        "STOP",
    )

    assert selected
    assert not bool(np.any(masked_mask))
    np.testing.assert_array_equal(masked, 0.0)


def test_missing_target_is_fail_closed() -> None:
    masked, masked_mask, selected = mask_task_conditioned_entity_geometry(
        _geometry(),
        np.ones(4, dtype=np.bool_),
        ("target_blue", "", "", ""),
        "follow target_red",
    )

    assert not selected
    assert not bool(np.any(masked_mask))
    np.testing.assert_array_equal(masked, 0.0)


def test_require_valid_filters_missing_task_target(
    tmp_path, monkeypatch
) -> None:
    cache = SimpleNamespace(
        run_id="RUN",
        frame_indices=np.asarray([0], dtype=np.int64),
        sample_frame_rows=np.asarray([0, 0], dtype=np.int64),
        sample_instruction_rows=np.asarray([0, 1], dtype=np.int64),
        policy_input_valid=np.asarray([True], dtype=np.bool_),
        target_safe_stop=np.asarray([False, False], dtype=np.bool_),
        target_actions=np.zeros((2, 2), dtype=np.float32),
        previous_expert_actions=np.zeros((2, 2), dtype=np.float32),
        previous_action_valid=np.asarray([False, False], dtype=np.bool_),
        entity_geometry=_geometry()[None, ...],
        entity_geometry_mask=np.ones((1, 4), dtype=np.bool_),
        entity_ids=np.asarray([("target_blue", "", "", "")]),
        instruction_texts=np.asarray(["follow target_red", "follow target_blue"]),
        language=np.zeros((2, 8), dtype=np.float32),
        language_splits=np.asarray(["train", "train"]),
        sample_ids=np.asarray(["red", "blue"]),
        frame_keys=np.asarray(["RUN:1:0:1"]),
        instruction_ids=np.asarray(["red", "blue"]),
    )
    monkeypatch.setattr(dataset_module, "_load_cache", lambda path: cache)

    dataset = FrozenFeatureDataset([tmp_path / "cache"], require_valid=True)

    assert dataset.raw_sample_count == 1
    assert len(dataset) == 1
    assert dataset[0]["instruction_id"] == "blue"


def _dataset_for_sample(
    instruction: str, *, safe_stop: bool = False, entity_ids=ENTITY_IDS
) -> FrozenFeatureDataset:
    cache = SimpleNamespace(
        sample_frame_rows=np.asarray([0], dtype=np.int64),
        sample_instruction_rows=np.asarray([0], dtype=np.int64),
        entity_geometry=_geometry()[None, ...],
        entity_geometry_mask=np.ones((1, 4), dtype=np.bool_),
        target_actions=np.zeros((1, 2), dtype=np.float32),
        target_safe_stop=np.asarray([safe_stop], dtype=np.bool_),
        previous_expert_actions=np.zeros((1, 2), dtype=np.float32),
        previous_action_valid=np.asarray([False], dtype=np.bool_),
        language=np.zeros((1, 8), dtype=np.float32),
        instruction_texts=np.asarray([instruction]),
        entity_ids=np.asarray([entity_ids]),
        policy_input_valid=np.asarray([True], dtype=np.bool_),
        run_id="RUN",
        frame_keys=np.asarray(["RUN:1:0:1"]),
        sample_ids=np.asarray(["sample"]),
        instruction_ids=np.asarray(["instruction"]),
    )
    dataset = FrozenFeatureDataset.__new__(FrozenFeatureDataset)
    dataset._samples = [_SampleRef(0, 0)]
    dataset._caches = [cache]
    dataset._augment = False
    dataset._instruction_swap_prob = 0.0
    dataset._mirror_prob = 0.0
    dataset._runtime_first_step_limit_m = None
    dataset._swap_rows = {}
    return dataset


def test_frozen_dataset_getitem_applies_final_instruction_mask() -> None:
    dataset = _dataset_for_sample("follow target_red")

    item = dataset[0]

    assert item["policy_input_valid"]
    assert item["entity_geometry_mask"].tolist() == [True, False, False, False]
    np.testing.assert_array_equal(
        item["entity_geometry"][0].numpy(), _geometry()[1]
    )
    np.testing.assert_array_equal(item["entity_geometry"][1:].numpy(), 0.0)


def test_frozen_dataset_stop_getitem_keeps_valid_all_false_geometry() -> None:
    dataset = _dataset_for_sample("STOP", safe_stop=True)

    item = dataset[0]

    assert item["policy_input_valid"]
    assert not bool(torch.any(item["entity_geometry_mask"]))
    assert not bool(torch.any(item["entity_geometry"]))


def test_policy_inputs_still_reject_entity_ids() -> None:
    batch = {key: torch.zeros(1) for key in POLICY_INPUT_KEYS}
    batch["entity_ids"] = ["target_red"]

    with pytest.raises(ValueError, match="privileged fields"):
        policy_inputs_from_batch(batch)
