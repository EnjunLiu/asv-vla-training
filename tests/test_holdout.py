from __future__ import annotations

import pytest

from asv_training.make_holdout_split import make_holdout_split


def _entry(index: int) -> dict[str, object]:
    return {
        "run_id": f"run_{index}",
        "collection_slot": f"L{index}_S0_R1",
        "scene_seed": 160000 + index,
        "layout_id": f"L{index}",
        "motion_state": "S0",
        "training_eligible": True,
    }


def _plan(count: int = 2) -> dict[str, object]:
    return {
        "minimum_complete_runs": count,
        "slots": [
            {
                "slot_id": f"L{index}_S0_R1",
                "scene_seed": 160000 + index,
                "layout_id": f"L{index}",
                "motion_state": "S0",
            }
            for index in range(1, count + 1)
        ],
    }


def test_holdout_split_assigns_every_planned_run_to_test() -> None:
    result = make_holdout_split([_entry(1), _entry(2)], _plan())

    assert result["training_ready"] is False
    assert result["split_run_counts"] == {
        "train": 0,
        "validation": 0,
        "test": 2,
    }
    assert result["assignments"] == {"run_1": "test", "run_2": "test"}


def test_holdout_split_rejects_plan_drift() -> None:
    entry = _entry(1)
    entry["scene_seed"] = 999

    with pytest.raises(ValueError, match="scene_seed"):
        make_holdout_split([entry, _entry(2)], _plan())


def test_holdout_split_rejects_ineligible_run() -> None:
    entry = _entry(1)
    entry["training_eligible"] = False

    with pytest.raises(ValueError, match="not evaluation-eligible"):
        make_holdout_split([entry, _entry(2)], _plan())
