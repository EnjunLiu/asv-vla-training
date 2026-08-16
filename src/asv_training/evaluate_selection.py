"""Evaluate direct-action colour-selection correctness on held-out sine runs.

The demo task is "follow the red boat" / "follow the blue boat" when the
red/blue pair moves side by side.  The policy must steer toward the
commanded colour.  This script checks, for every follow-red / follow-blue
sample in the TEST split, whether the model's first executed step points
toward the task-selected entity's bearing.

Selection correctness = fraction of samples whose first-step direction is
within 45 deg of the commanded entity's bearing.

Usage:
    python evaluate_selection.py --checkpoint <best.pt> \
        --features <features_sine> --split <sine_group_split_v1.json> \
        --model-config <model_small_v2.yaml> [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from asv_training.dataset import (
    load_split_assignments,
    mask_task_conditioned_entity_geometry,
    task_target_id_from_instruction,
)


def _bearing_deg(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def _angle_between_deg(a_deg: float, b_deg: float) -> float:
    delta = abs(a_deg - b_deg) % 360.0
    return min(delta, 360.0 - delta)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from asv_training.model import SmallActionPolicy, SmallPolicyConfig
    from asv_training.dataset import _load_cache

    import yaml

    model_cfg = SmallPolicyConfig.from_mapping(
        yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    )
    model = SmallActionPolicy(model_cfg).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    assignments = load_split_assignments(args.split)
    test_run_ids = {
        run_id for run_id, split in assignments.items() if split == "test"
    }
    if not test_run_ids:
        print("SELECTION_EVALUATION_FAIL: no test runs in split")
        return 1

    cache_dirs = sorted(
        path
        for path in args.features.iterdir()
        if path.is_dir() and path.name in test_run_ids
    )
    # Layout (L6 vs mirrored L6B) comes from the registry collection_slot.
    slot_by_run: dict[str, str] = {}
    registry_path = args.split.parent / "sine_registry_v1.jsonl"
    if registry_path.is_file():
        for line in registry_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            slot_by_run[str(entry["run_id"])] = str(
                entry.get("collection_slot", "")
            )
    correct = 0
    total = 0
    by_layout: dict[str, dict[str, int]] = {}
    with torch.no_grad():
        for cache_dir in cache_dirs:
            cache = _load_cache(cache_dir)
            slot = slot_by_run.get(cache.run_id, "")
            is_mirrored = "L6B" in slot
            language_source = np.load(
                cache_dir / "language.npz", allow_pickle=False
            )
            instruction_texts = language_source["instruction_texts"]
            frames_source = np.load(
                cache_dir / "frames_000.npz", allow_pickle=False
            )
            entity_ids = frames_source["entity_ids"]
            for sample_row in range(len(cache.sample_ids)):
                frame_row = int(cache.sample_frame_rows[sample_row])
                instruction_row = int(cache.sample_instruction_rows[sample_row])
                instruction = str(instruction_texts[instruction_row])
                commanded_id = task_target_id_from_instruction(instruction)
                if commanded_id not in {"target_red", "target_blue"}:
                    continue
                if not bool(cache.policy_input_valid[frame_row]):
                    continue
                entities = entity_ids[frame_row]
                geometry, geometry_mask, target_valid = (
                    mask_task_conditioned_entity_geometry(
                        cache.entity_geometry[frame_row],
                        cache.entity_geometry_mask[frame_row],
                        entities,
                        instruction,
                    )
                )
                if not target_valid or not bool(geometry_mask[0]):
                    continue
                expected = _bearing_deg(
                    float(geometry[0][0]), float(geometry[0][1])
                )

                item = {
                    "language": torch.from_numpy(
                        cache.language[instruction_row].copy()
                    ).unsqueeze(0).to(args.device),
                    "entity_geometry": torch.from_numpy(
                        geometry.copy()
                    ).unsqueeze(0).to(args.device),
                    "previous_action": torch.from_numpy(
                        cache.previous_expert_actions[sample_row].copy()
                    ).unsqueeze(0).to(args.device),
                    "language_valid": torch.tensor([True], dtype=torch.bool),
                    "entity_geometry_mask": torch.from_numpy(
                        geometry_mask.copy()
                    ).unsqueeze(0).to(args.device),
                    "previous_action_valid": torch.tensor(
                        [bool(cache.previous_action_valid[sample_row])],
                        dtype=torch.bool,
                    ),
                    "policy_input_valid": torch.tensor(
                        [bool(cache.policy_input_valid[frame_row])],
                        dtype=torch.bool,
                    ),
                }
                output = model(**item)
                action = output.action[0].cpu().numpy()
                stop_logit = float(output.stop_logit[0][0])
                stop = stop_logit > 0.0
                first_dx = float(action[0])
                first_dy = float(action[1])
                if math.hypot(first_dx, first_dy) < 1e-4:
                    # A zero step cannot be classified; count as failure
                    # only for non-stop commands (a stop is not a selection).
                    if stop:
                        continue
                step_bearing = _bearing_deg(first_dx, first_dy)
                toward_expected = _angle_between_deg(step_bearing, expected)
                is_correct = toward_expected <= 45.0
                correct += int(is_correct)
                total += 1
                layout = "L6B" if is_mirrored else "L6"
                bucket = by_layout.setdefault(layout, {"correct": 0, "total": 0})
                bucket["correct"] += int(is_correct)
                bucket["total"] += 1

    if total == 0:
        print("SELECTION_EVALUATION_FAIL: no evaluable follow-colour samples")
        return 1
    rate = correct / total
    print(f"SELECTION_PASS rate={rate:.3f} correct={correct}/{total}")
    for layout, bucket in sorted(by_layout.items()):
        r = bucket["correct"] / bucket["total"]
        print(f"  {layout}: {r:.3f} ({bucket['correct']}/{bucket['total']})")
    return 0 if rate >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
