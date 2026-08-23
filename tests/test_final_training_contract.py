import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_final import (  # noqa: E402
    FINAL_SCHEMA,
    control_window_actions,
    build_policy_dataset,
    build_policy_checkpoint,
    load_language_embeddings,
    load_episode_records,
    split_moving_target_slots,
    split_slots,
    stamp_language_standoff,
    task_key,
    teacher_action,
    validate_policy_checkpoint,
)
from decision import (  # noqa: E402
    ENTITY_EMBEDDING_DIM,
    ENTITY_FEATURE_DIM,
    ENTITY_GEOMETRY_DIM,
    SmallActionPolicy,
    SmallPolicyConfig,
)


def _control_window_records(record, action=(0.12, -0.04)):
    """Return five contiguous 0.2 s frames with a complete first 0.5 s window."""
    start_stamp = int(record.ego["stamp_us"])
    return [
        replace(
            record,
            frame_index=record.frame_index + index,
            ego={**record.ego, "stamp_us": start_stamp + index * 200_000},
            action=action,
        )
        for index in range(5)
    ]

def _with_target_xy(record, xy: tuple[float, float]):
    entities = []
    for item in record.entities:
        value = dict(item)
        if item["entity_id"] == "target_blue":
            value["relative_position_m"] = [float(xy[0]), float(xy[1]), 0.0]
        entities.append(value)
    return replace(record, entities=tuple(entities))


def test_final_episode_loader_reads_real_complete_frames() -> None:
    episodes = Path("D:/asv-vla-training/data/episodes/moving_target_valid")
    records = load_episode_records(episodes)
    assert len(records) == 8874
    assert {record.slot_id for record in records} == {
        "BLUE_3M_TEST", "BLUE_3M_VALIDATION", "RED_3M_TEST", "RED_3M_VALIDATION",
        "RED_4M_TEST", "RED_4M_VALIDATION",
        *[
            f"{color}_{distance}_TRAIN_{index:02d}"
            for color, distance in (("BLUE", "3M"), ("RED", "3M"), ("RED", "4M"))
            for index in range(1, 5)
        ],
        "BLUE_3M_TRAIN_START_01", "BLUE_3M_TRAIN_START_02",
        "RED_3M_TRAIN_START_01", "RED_3M_TRAIN_START_02",
        "RED_4M_TRAIN_START_01", "RED_4M_TRAIN_START_02",
    }
    assert all(record.image_path.is_file() for record in records)
    assert all(record.entities for record in records)


def test_policy_checkpoint_uses_final_contract_without_legacy_ego_key() -> None:
    checkpoint = build_policy_checkpoint()
    validate_policy_checkpoint(checkpoint)
    assert checkpoint["schema_version"] == FINAL_SCHEMA
    assert "use_ego_state" not in checkpoint["model_config"]
    assert checkpoint["model_config"]["ego_state_dim"] == 2


def test_policy_training_fixes_torch_initialization_seed() -> None:
    source = (ROOT / "src" / "train_final.py").read_text(encoding="utf-8")
    function = source[source.index("def save_policy_checkpoint("):source.index("def build_policy_dataset(")]
    assert "torch.manual_seed(20260819)" in function


def test_policy_training_uses_stable_full_batch_mse_optimization() -> None:
    source = (ROOT / "src" / "train_final.py").read_text(encoding="utf-8")
    function = source[source.index("def save_policy_checkpoint("):source.index("def build_policy_dataset(")]
    assert "mse_loss(output.action, batch_action)" in function
    assert "batch_size = min(4096, len(language))" in function


def test_policy_checkpoint_rejects_legacy_configuration() -> None:
    checkpoint = build_policy_checkpoint()
    checkpoint["model_config"]["use_ego_state"] = True
    with pytest.raises(ValueError, match="unknown model configuration"):
        validate_policy_checkpoint(checkpoint)


def test_real_qwen_embeddings_cover_final_tasks() -> None:
    table = load_language_embeddings(
        Path("D:/asv-vla-training/data/qwen_final_embeddings.npz")
    )
    assert set(table) == {"red_3m", "blue_3m", "red_4m"}
    assert all(value.shape == (256,) for value in table.values())
    assert all(np.isclose(np.linalg.norm(value), 1.0, atol=1.0e-5) for value in table.values())


def test_run_split_has_no_seed_leakage() -> None:
    slots = [f"L7_S2_R{i}" for i in range(1, 7)] + [f"L7B_S2_R{i}" for i in range(1, 7)]
    split = split_slots(slots)
    assert set(split) == {"train", "validation", "test"}
    assert not (set(split["train"]) & set(split["validation"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["validation"]) & set(split["test"]))
    assert set().union(*map(set, split.values())) == set(slots)
    assert set(split["validation"]) == {"L7_S2_R5", "L7B_S2_R5"}
    assert set(split["test"]) == {"L7_S2_R6", "L7B_S2_R6"}


def test_moving_target_split_uses_explicit_train_validation_test_slots() -> None:
    slots = [
        "RED_3M_TRAIN_01",
        "RED_3M_VALIDATION",
        "RED_3M_TEST",
        "BLUE_3M_TRAIN_01",
        "BLUE_3M_VALIDATION",
        "BLUE_3M_TEST",
    ]

    split = split_moving_target_slots(slots)

    assert split == {
        "train": ["BLUE_3M_TRAIN_01", "RED_3M_TRAIN_01"],
        "validation": ["BLUE_3M_VALIDATION", "RED_3M_VALIDATION"],
        "test": ["BLUE_3M_TEST", "RED_3M_TEST"],
    }


def test_teacher_action_tracks_current_speed_at_standoff() -> None:
    action = teacher_action(
        relative_xy=np.asarray([3.0, 0.0]),
        standoff_m=3.0,
        surge_velocity_mps=1.0,
        yaw_rate_radps=0.0,
    )
    np.testing.assert_allclose(action, [0.5, 0.0], atol=1.0e-6)


def test_teacher_action_closes_radial_error_with_bounded_small_step() -> None:
    far = teacher_action(np.asarray([5.0, 0.0]), 3.0, 0.0, 0.0)
    near = teacher_action(np.asarray([1.0, 0.0]), 3.0, 0.0, 0.0)
    assert far[0] > 0.0 and near[0] < 0.0
    assert np.linalg.norm(far) <= 0.500001
    assert np.linalg.norm(near) <= 0.500001


def test_teacher_action_saturates_when_far_from_standoff() -> None:
    action = teacher_action(np.asarray([4.5, 0.0]), 3.0, 0.0, 0.0)
    np.testing.assert_allclose(action, [0.5, 0.0], atol=1.0e-6)


def test_teacher_action_is_radial_without_current_motion() -> None:
    relative = np.asarray([4.0, 3.0])
    action = teacher_action(relative, 3.0, 0.0, 0.0)
    assert abs(action[0] * relative[1] - action[1] * relative[0]) < 1.0e-6
    assert float(np.dot(action, relative)) > 0.0


def test_policy_dataset_matches_final_tensor_contract() -> None:
    record = _with_target_xy(
        load_episode_records(
            Path("D:/asv-vla-training/data/episodes/moving_target_valid")
        )[0],
        (3.0, 0.0),
    )
    records = _control_window_records(record, action=(0.05, 0.0))
    embeddings = load_language_embeddings(
        Path("D:/asv-vla-training/data/qwen_final_embeddings.npz")
    )
    dataset = build_policy_dataset(records, embeddings, distance_scales=(1.0,))
    assert dataset["language"].shape == (4, 256)
    assert dataset["entity_geometry"].shape == (4, 16, ENTITY_FEATURE_DIM)
    assert dataset["ego_state"].shape == (4, 2)
    assert dataset["action"].shape == (4, 2)
    assert dataset["entity_geometry_mask"].shape == (4, 16)
    assert np.sum(dataset["entity_geometry_mask"], axis=1).tolist() == [4, 1, 4, 1]
    np.testing.assert_allclose(dataset["action"][0], dataset["action"][1])
    assert np.max(np.linalg.norm(dataset["action"], axis=1)) <= 0.500001


def test_policy_dataset_uses_recorded_action_and_real_ego() -> None:
    record = _with_target_xy(
        load_episode_records(
            Path("D:/asv-vla-training/data/episodes/moving_target_valid")
        )[0],
        (3.0, 0.0),
    )
    records = _control_window_records(record)
    records[0] = replace(
        records[0], ego={**records[0].ego, "surge_velocity_mps": 1.25, "yaw_rate_radps": -0.2}
    )
    embeddings = load_language_embeddings(
        Path("D:/asv-vla-training/data/qwen_final_embeddings.npz")
    )

    dataset = build_policy_dataset(records, embeddings, distance_scales=(1.0,))

    assert dataset["language"].shape == (4, 256)
    assert dataset["action"].shape == (4, 2)
    np.testing.assert_allclose(dataset["action"][0], [0.30, -0.10])
    np.testing.assert_allclose(dataset["action"][1], [0.30, -0.10])
    np.testing.assert_allclose(dataset["ego_state"][0], [0.25, -0.2])
    np.testing.assert_allclose(dataset["language"][0, -1], -1.0)


def test_control_window_actions_accumulate_frame_actions_to_control_period() -> None:
    """A 0.5 s policy command must cover the three 0.2 s expert steps."""
    record = load_episode_records(
        Path("D:/asv-vla-training/data/episodes/moving_target_valid")
    )[0]
    records = [
        replace(record, frame_index=index, ego={"stamp_us": index * 200_000}, action=(0.12, 0.0))
        for index in range(5)
    ]

    labels = control_window_actions(records)

    np.testing.assert_allclose(labels[0], [0.30, 0.0])
    np.testing.assert_allclose(labels[1], [0.30, 0.0])
    assert labels[2] is None


def test_control_window_actions_weights_partial_final_interval() -> None:
    record = load_episode_records(
        Path("D:/asv-vla-training/data/episodes/moving_target_valid")
    )[0]
    interval_actions = (0.1, 0.2, 0.3, 0.4)
    records = [
        replace(
            record,
            frame_index=index,
            ego={"stamp_us": index * 200_000},
            action=(value, 0.0),
        )
        for index, value in enumerate(interval_actions)
    ] + [
        replace(record, frame_index=4, ego={"stamp_us": 800_000}, action=None)
    ]

    labels = control_window_actions(records)

    np.testing.assert_allclose(labels[0], [0.45, 0.0])


def test_control_window_actions_clips_accumulated_displacement() -> None:
    record = load_episode_records(
        Path("D:/asv-vla-training/data/episodes/moving_target_valid")
    )[0]
    records = [
        replace(
            record,
            frame_index=index,
            ego={"stamp_us": index * 200_000},
            action=(0.4, 0.0),
        )
        for index in range(5)
    ]

    labels = control_window_actions(records)

    np.testing.assert_allclose(labels[0], [0.50, 0.0])
    assert np.linalg.norm(labels[0]) <= 0.500001


def test_policy_dataset_adds_teacher_labeled_recovery_distances() -> None:
    record = _with_target_xy(
        load_episode_records(
            Path("D:/asv-vla-training/data/episodes/moving_target_valid")
        )[0],
        (3.0, 0.0),
    )
    embeddings = load_language_embeddings(
        Path("D:/asv-vla-training/data/qwen_final_embeddings.npz")
    )
    recovery = teacher_action(np.asarray([4.5, 0.0], dtype=np.float32), 3.0, 0.0, 0.0)

    dataset = build_policy_dataset(
        _control_window_records(record), embeddings, distance_scales=(1.0, 1.5)
    )

    assert dataset["action"].shape == (8, 2)
    np.testing.assert_allclose(dataset["action"][0], [0.30, -0.10])
    np.testing.assert_allclose(dataset["action"][1], [0.30, -0.10])
    np.testing.assert_allclose(dataset["action"][2], recovery)
    np.testing.assert_allclose(dataset["action"][3], recovery)
    np.testing.assert_allclose(dataset["ego_state"][2], [0.0, 0.0])
    np.testing.assert_allclose(dataset["ego_state"][3], [0.0, 0.0])
    assert np.linalg.norm(recovery) <= 0.500001
    assert dataset["task_id"].tolist() == ["blue_3m"] * 8
    np.testing.assert_allclose(dataset["language"][0, -1], -1.0)


def test_policy_dataset_relabels_far_unscaled_samples_for_cold_start_chase() -> None:
    record = load_episode_records(
        Path("D:/asv-vla-training/data/episodes/moving_target_valid")
    )[0]
    far_entities = []
    for item in record.entities:
        value = dict(item)
        if item["entity_id"] == "target_blue":
            value["relative_position_m"] = [5.5, -0.34, 0.0]
        far_entities.append(value)
    far_record = replace(record, entities=tuple(far_entities), task_text="follow the blue boat, keep 3 meters distance")
    embeddings = load_language_embeddings(
        Path("D:/asv-vla-training/data/qwen_final_embeddings.npz")
    )
    expected = teacher_action(np.asarray([5.5, -0.34], dtype=np.float32), 3.0, 0.0, 0.0)

    dataset = build_policy_dataset(_control_window_records(far_record), embeddings, distance_scales=(1.0,))

    np.testing.assert_allclose(dataset["action"][0], expected)
    np.testing.assert_allclose(dataset["ego_state"][0], [0.0, 0.0])
    assert float(np.linalg.norm(expected)) >= 0.49


def test_task_key_reads_chinese_color_and_distance() -> None:
    assert task_key("跟随蓝色目标船，保持3米距离") == "blue_3m"
    assert task_key("跟随红色目标船，保持4米距离") == "red_4m"
    assert task_key("跟随红色目标船，保持3米距离") == "red_3m"


def test_stamped_language_separates_3m_and_4m_standoff() -> None:
    embeddings = load_language_embeddings(
        Path("D:/asv-vla-training/data/qwen_final_embeddings.npz")
    )
    three = stamp_language_standoff(embeddings["red_3m"], 3.0)
    four = stamp_language_standoff(embeddings["red_4m"], 4.0)
    assert three[-1] == -1.0
    assert four[-1] == 1.0
    assert float(np.linalg.norm(three - four)) > 1.5


def test_policy_dataset_keeps_expert_labels_inside_near_standoff_band() -> None:
    record = load_episode_records(
        Path("D:/asv-vla-training/data/episodes/moving_target_valid")
    )[0]
    mid_entities = []
    for item in record.entities:
        value = dict(item)
        if item["entity_id"] == "target_blue":
            value["relative_position_m"] = [3.8, 0.0, 0.0]
        mid_entities.append(value)
    mid_record = replace(
        record,
        entities=tuple(mid_entities),
        task_text="follow the blue boat, keep 3 meters distance",
        ego={**record.ego, "surge_velocity_mps": 0.6, "yaw_rate_radps": 0.0},
    )
    embeddings = load_language_embeddings(
        Path("D:/asv-vla-training/data/qwen_final_embeddings.npz")
    )

    dataset = build_policy_dataset(
        _control_window_records(mid_record, action=(0.12, -0.04)),
        embeddings,
        distance_scales=(1.0,),
    )

    # Two complete 0.5 s windows x (expert pair + lag-teacher pair) = 8 rows.
    assert dataset["action"].shape[0] == 8
    np.testing.assert_allclose(dataset["action"][0], [0.30, -0.10])
    np.testing.assert_allclose(dataset["ego_state"][0], [0.12, 0.0])
    expected = teacher_action(np.asarray([3.8, 0.0], dtype=np.float32), 3.0, 0.6, 0.0)
    np.testing.assert_allclose(dataset["action"][2], expected)
    np.testing.assert_allclose(dataset["ego_state"][2], [0.12, 0.0])
    assert float(expected[0]) > 0.35
    np.testing.assert_allclose(dataset["language"][0, -1], -1.0)


def test_policy_checkpoint_uses_entity_feature_contract() -> None:
    checkpoint = build_policy_checkpoint()
    validate_policy_checkpoint(checkpoint)
    config = checkpoint["model_config"]
    assert config["entity_geometry_dim"] == ENTITY_GEOMETRY_DIM
    assert config["entity_embedding_dim"] == ENTITY_EMBEDDING_DIM
    assert config["language_conditioned_entity_attention"] is True
    assert config["entity_attention_mode"] == "language_only"


def test_entity_features_encode_velocity() -> None:
    from decision import build_entity_features  # noqa: E402

    class _Entity:
        entity_id = "boat"
        visible = True
        valid = True
        relative_x = 2.0
        relative_y = 0.0
        relative_velocity_x = 1.0
        relative_velocity_y = -0.5
        velocity_valid = True
        entity_embedding = [0.0] * ENTITY_EMBEDDING_DIM

    row = build_entity_features([_Entity()]).features[0]
    assert row[2] > 0.0
    assert row[3] < 0.0
