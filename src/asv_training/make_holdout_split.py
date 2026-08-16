"""Create a frozen all-test split from a collection plan and Run registry.

This is intentionally separate from ``make_group_splits``: a holdout set is
never training-ready and every planned Run is assigned to ``test``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from asv_training.make_group_splits import SPLIT_SCHEMA_VERSION, _load_jsonl


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def make_holdout_split(
    entries: list[dict[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one eligible Run per planned slot and assign all Runs to test."""

    slots = plan.get("slots")
    if not isinstance(slots, list) or not slots:
        raise ValueError("holdout plan must contain a non-empty slots list")
    expected_count = int(plan.get("minimum_complete_runs", len(slots)))
    if expected_count != len(slots):
        raise ValueError("holdout plan count must equal its slot count")
    if len(entries) != expected_count:
        raise ValueError(
            f"holdout requires exactly {expected_count} registry entries, "
            f"found {len(entries)}"
        )

    planned = {
        str(slot["slot_id"]): {
            "scene_seed": int(slot["scene_seed"]),
            "layout_id": str(slot["layout_id"]),
            "motion_state": str(slot["motion_state"]),
        }
        for slot in slots
    }
    if len(planned) != len(slots):
        raise ValueError("holdout plan contains duplicate slot IDs")

    by_slot: dict[str, dict[str, Any]] = {}
    run_ids: list[str] = []
    scene_seeds: list[int] = []
    for entry in entries:
        if not bool(entry.get("training_eligible")):
            raise ValueError(
                f"Run {entry.get('run_id', '?')} is not evaluation-eligible"
            )
        slot_id = str(entry.get("collection_slot", ""))
        if slot_id not in planned:
            raise ValueError(f"unplanned holdout slot {slot_id!r}")
        if slot_id in by_slot:
            raise ValueError(f"duplicate holdout slot {slot_id!r}")
        expected = planned[slot_id]
        for key in ("scene_seed", "layout_id", "motion_state"):
            actual = entry.get(key)
            if key == "scene_seed":
                actual = int(actual)
            else:
                actual = str(actual)
            if actual != expected[key]:
                raise ValueError(
                    f"slot {slot_id}: {key}={actual!r}, "
                    f"expected {expected[key]!r}"
                )
        run_id = str(entry.get("run_id", "")).strip()
        if not run_id:
            raise ValueError(f"slot {slot_id}: empty Run ID")
        run_ids.append(run_id)
        scene_seeds.append(int(entry["scene_seed"]))
        by_slot[slot_id] = entry

    missing = sorted(set(planned) - set(by_slot))
    if missing:
        raise ValueError(f"missing planned holdout slots: {missing}")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("holdout registry contains duplicate Run IDs")
    if len(set(scene_seeds)) != len(scene_seeds):
        raise ValueError("holdout Scene Seeds must be unique")

    sorted_run_ids = sorted(run_ids)
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "frozen_all_test_holdout",
        "training_ready": False,
        "reason": "all Runs are reserved for one-time holdout evaluation",
        "registry_run_count": len(entries),
        "run_count": len(run_ids),
        "scene_seed_count": len(scene_seeds),
        "scene_seeds": sorted(scene_seeds),
        "split_run_counts": {
            "train": 0,
            "validation": 0,
            "test": len(run_ids),
        },
        "assignments": {run_id: "test" for run_id in sorted_run_ids},
        "train_run_ids": [],
        "validation_run_ids": [],
        "test_run_ids": sorted_run_ids,
        "rejected_run_ids": [],
        "holdout_slots": {
            slot_id: str(by_slot[slot_id]["run_id"])
            for slot_id in sorted(by_slot)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a plan-checked, all-test holdout split."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = make_holdout_split(
        _load_jsonl(args.registry),
        _read_json(args.plan),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "HOLDOUT_SPLIT_PASS "
        f"runs={result['run_count']} "
        f"test={result['split_run_counts']['test']} "
        "training_ready=False"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
