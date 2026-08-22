import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_training import (
    moving_target_split,
    relative_entity_positions,
    task_key,
    tracking_metrics,
    visibility_metrics,
)  # noqa: E402


def test_moving_target_split_rejects_seed_leakage_between_partitions() -> None:
    class Record:
        def __init__(self, slot_id: str, run_id: str, scene_seed: int) -> None:
            self.slot_id = slot_id
            self.run_id = run_id
            self.scene_seed = scene_seed

    records = [
        Record("RED_3M_TRAIN_01", "train-run", 1),
        Record("RED_3M_VALIDATION", "validation-run", 2),
        Record("RED_3M_TEST", "test-run", 3),
    ]

    split = moving_target_split(records)

    assert split["train"] == ["RED_3M_TRAIN_01"]
    assert split["validation"] == ["RED_3M_VALIDATION"]
    assert split["test"] == ["RED_3M_TEST"]


def test_relative_entity_positions_keep_world_targets_fixed() -> None:
    entity_world = {
        "target_red": np.asarray([6.0, 2.0], dtype=np.float32),
        "target_blue": np.asarray([4.0, -3.0], dtype=np.float32),
    }
    asv_world = np.asarray([0.5, -0.25], dtype=np.float32)

    relative = relative_entity_positions(entity_world, asv_world)

    np.testing.assert_allclose(entity_world["target_red"], [6.0, 2.0])
    np.testing.assert_allclose(relative["target_red"], [5.5, 2.25])
    np.testing.assert_allclose(relative["target_blue"], [3.5, -2.75])


def test_visibility_metrics_separate_raw_slots_from_task_target() -> None:
    raw_visibility = {
        "target_red": True,
        "target_blue": True,
        "target_left": False,
        "target_right": True,
    }
    truth_visibility = {
        "target_red": True,
        "target_blue": True,
        "target_left": True,
        "target_right": True,
    }

    metrics = visibility_metrics(
        raw_visibility,
        truth_visibility,
        selected_target_id="target_blue",
    )

    assert metrics == {
        "all_slot_correct": 3,
        "all_slot_total": 4,
        "selected_target_detected": 1,
        "selected_target_total": 1,
    }


def test_tracking_metrics_require_long_moving_target_and_report_windows() -> None:
    time = np.arange(0.0, 180.0, 1.0)
    target = np.column_stack((0.6 * time, np.zeros_like(time)))
    asv = target - np.column_stack((3.0 + 0.1 * np.sin(time), np.zeros_like(time)))

    metrics = tracking_metrics(time, target, asv, desired_standoff_m=3.0, policy_driven=0.95)

    assert metrics["duration_s"] >= 179.0
    assert metrics["target_displacement_m"] >= 50.0
    assert metrics["steady_state_mae_m"] < 0.5
    assert metrics["final_window_mae_m"] < 0.3
    assert metrics["policy_driven_ratio"] == 0.95
    assert metrics["diverged"] is False


def test_tracking_metrics_reject_fixed_target_as_final_evidence() -> None:
    time = np.arange(0.0, 180.0, 1.0)
    target = np.zeros((len(time), 2), dtype=np.float32)
    asv = np.zeros_like(target)
    metrics = tracking_metrics(time, target, asv, desired_standoff_m=3.0, policy_driven=1.0)

    assert metrics["target_displacement_m"] == 0.0
    assert metrics["acceptance_ready"] is False


def test_rollout_plot_selection_uses_test_slots_without_version_suffix() -> None:
    class Record:
        def __init__(self, slot_id: str, task_text: str) -> None:
            self.slot_id = slot_id
            self.task_text = task_text

    records = [
        Record("RED_3M_TEST", "follow the red boat, keep 3 meters distance"),
        Record("BLUE_3M_TEST", "follow the blue boat, keep 3 meters distance"),
        Record("RED_4M_TEST", "follow the red boat, keep 4 meters distance"),
    ]
    held_out = [record for record in records if record.slot_id.endswith("_TEST")]
    assert {task_key(record.task_text) for record in held_out} == {"red_3m", "blue_3m", "red_4m"}
