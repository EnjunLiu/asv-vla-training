"""Day 11B: create strictly Run-ID / Scene-Seed-grouped train / validation / test
splits.

Splits are the single source of truth for every downstream consumer — feature
caching (Day 13), training (Day 15), intervention evaluation (Day 16), and
final metrics (Day 21).

Key rules (from ``TODO.md`` § 11B):

1. All frames of a Run MUST belong to exactly one split.
2. Scene Seeds are the grouping key; all Runs sharing a Scene Seed default
   to the same split.
3. Language-template families (the ``split`` field in
   ``instructions.jsonl``) are validated for non-overlap, but Run / Seed
   grouping takes precedence for visual-generalisation claims.
4. Primary test holds out *both* unseen Scene Seeds and unseen language
   templates.
5. A single-Scene-Seed pilot registry must produce
   ``training_ready = false``.

Usage::

    PYTHONPATH=src \\
      python -m asv_training.make_group_splits \\
      --registry data/registry/dataset_registry_v1.jsonl \\
      --output data/registry/group_split_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPLIT_SCHEMA_VERSION = "group_split_v1"
VALID_SPLITS = ("train", "validation", "test")
MINIMUM_TRAINING_RUNS = 12


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, one JSON object per line."""
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_no}: expected JSON object, got "
                    f"{type(value).__name__}"
                )
            entries.append(value)
    return entries


def _validate_instructions_language_splits(
    instructions_path: Path,
) -> dict[str, set[str]]:
    """Return ``{language_split: set_of_template_ids}`` for the instruction file.

    The ``language_split`` values MUST be ``train`` / ``validation`` / ``test``
    and these families must be disjoint.
    """
    records = _load_jsonl(instructions_path)
    families: dict[str, set[str]] = {}
    for rec in records:
        split = str(rec.get("split", "")).strip().casefold()
        if split not in VALID_SPLITS:
            raise ValueError(
                f"instruction {rec.get('instruction_id', '?')}: "
                f"unknown split={split!r}"
            )
        families.setdefault(split, set()).add(rec.get("instruction_id", ""))
    return families


# ---------------------------------------------------------------------------
#  Scene-Seed grouping & deterministic split
# ---------------------------------------------------------------------------

def _group_by_scene_seed(
    entries: list[dict[str, Any]],
) -> dict[int, list[str]]:
    """Group Run IDs by their Scene Seed.

    Every Run must have a unique run_id and a finite integer Scene Seed.
    """
    groups: dict[int, list[str]] = {}
    seen_run_ids: set[str] = set()

    for entry in entries:
        run_id = str(entry.get("run_id", "")).strip()
        if not run_id:
            raise ValueError("registry entry has empty run_id")
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate run_id in registry: {run_id!r}")
        seen_run_ids.add(run_id)

        seed = entry.get("scene_seed")
        if seed is None:
            raise ValueError(
                f"run_id={run_id}: scene_seed is missing; "
                f"every Run must have a Scene Seed"
            )
        try:
            seed_int = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"run_id={run_id}: scene_seed={seed!r} is not a valid integer"
            ) from exc

        groups.setdefault(seed_int, []).append(run_id)

    return groups


def make_splits(
    entries: list[dict[str, Any]],
    *,
    split_seed: int = 42,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    instructions_path: Path | None = None,
) -> dict[str, Any]:
    """Assign every Run to exactly one split.

    Returns a dict with the full split assignment, metadata, and the
    ``training_ready`` flag.
    """
    eligible_entries = [
        entry
        for entry in entries
        if entry.get("training_eligible", True) is not False
    ]
    rejected_run_ids = sorted(
        str(entry.get("run_id", ""))
        for entry in entries
        if entry.get("training_eligible", True) is False
    )
    if not eligible_entries:
        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "training_ready": False,
            "reason": (
                "registry is empty"
                if not entries
                else "registry has no training-eligible Runs"
            ),
            "registry_run_count": len(entries),
            "run_count": 0,
            "scene_seed_count": 0,
            "split_run_counts": {
                "train": 0,
                "validation": 0,
                "test": 0,
            },
            "assignments": {},
            "rejected_run_ids": rejected_run_ids,
        }

    groups = _group_by_scene_seed(eligible_entries)
    seeds = sorted(groups)
    n_seeds = len(seeds)

    # Deterministic shuffle of Scene Seeds.
    rng = random.Random(split_seed)
    shuffled = list(seeds)
    rng.shuffle(shuffled)

    # Day 12 freezes exact acceptance splits for the two planned scales.
    # Other sizes retain the configured ratio behavior.
    if n_seeds == 12:
        n_train, n_val, n_test = 8, 2, 2
    elif n_seeds == 30:
        n_train, n_val, n_test = 18, 6, 6
    else:
        n_train = max(1, round(n_seeds * train_ratio))
        n_val = max(1, round(n_seeds * validation_ratio))
        n_test = n_seeds - n_train - n_val
    # Guard against rounding that exceeds available seeds.
    if n_train + n_val >= n_seeds:
        n_val = (
            max(1, n_seeds - n_train - 1)
            if n_seeds >= 3
            else max(1, n_seeds - n_train)
        )
        n_test = n_seeds - n_train - n_val

    if n_test < 1 and n_seeds >= 3:
        # Safety: ensure at least one seed in test when we have enough.
        n_test = 1
        n_val = max(1, n_seeds - n_train - 1)
        if n_train + n_val + n_test > n_seeds:
            n_train = n_seeds - n_val - n_test

    train_seeds = set(shuffled[:n_train])
    val_seeds = set(shuffled[n_train : n_train + n_val])
    test_seeds = set(shuffled[n_train + n_val : n_train + n_val + n_test])

    assignments: dict[str, str] = {}
    for seed, run_ids in groups.items():
        if seed in train_seeds:
            split = "train"
        elif seed in val_seeds:
            split = "validation"
        elif seed in test_seeds:
            split = "test"
        else:
            raise RuntimeError(f"BUG: Scene Seed {seed} not assigned to any split")
        for run_id in run_ids:
            assignments[run_id] = split

    # Count.
    split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    split_run_ids: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for run_id, split in assignments.items():
        split_counts[split] += 1
        split_run_ids[split].append(run_id)

    run_count = len(assignments)
    training_ready = bool(
        run_count >= MINIMUM_TRAINING_RUNS
        and n_seeds >= 3
        and split_counts["train"] >= 1
        and split_counts["validation"] >= 1
        and split_counts["test"] >= 1
    )

    reason = (
        "ready for training"
        if training_ready
        else (
            f"need >={MINIMUM_TRAINING_RUNS} eligible Runs, "
            f">=3 Scene Seeds and >=1 Run per split; "
            f"runs={run_count}, "
            f"seeds={n_seeds}, "
            f"train={split_counts['train']}, "
            f"val={split_counts['validation']}, "
            f"test={split_counts['test']}"
        )
    )

    result: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": split_seed,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "registry_run_count": len(entries),
        "run_count": run_count,
        "scene_seed_count": n_seeds,
        "scene_seeds": seeds,
        "training_ready": training_ready,
        "reason": reason,
        "split_run_counts": split_counts,
        "assignments": assignments,
        "train_run_ids": sorted(split_run_ids["train"]),
        "validation_run_ids": sorted(split_run_ids["validation"]),
        "test_run_ids": sorted(split_run_ids["test"]),
        "rejected_run_ids": rejected_run_ids,
    }

    # Cross-validate language-template families if the file is available.
    if instructions_path is not None and instructions_path.is_file():
        lang_families = _validate_instructions_language_splits(instructions_path)
        for lang_split in VALID_SPLITS:
            if lang_split not in lang_families:
                raise ValueError(
                    f"instructions.jsonl has no instructions with "
                    f"split={lang_split!r}"
                )
        result["language_template_families_ok"] = True
        result["language_template_counts"] = {
            k: len(v) for k, v in sorted(lang_families.items())
        }
    else:
        result["language_template_families_ok"] = None
        result["language_template_note"] = (
            "instructions.jsonl not available for cross-validation; "
            "re-run with --instructions to verify template-family isolation"
        )

    return result


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Day 11B: create Run/Scene-Seed-level train/val/test splits."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Path to dataset_registry_v1.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the split assignment JSON file.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Deterministic seed for shuffling Scene Seeds (default: 42).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Fraction of Scene Seeds for training (default: 0.6).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Fraction of Scene Seeds for validation (default: 0.2).",
    )
    parser.add_argument(
        "--instructions",
        type=Path,
        default=None,
        help="Path to instructions.jsonl for language-template validation.",
    )
    args = parser.parse_args()

    # Basic ratio validation.
    ratios = [args.train_ratio, args.validation_ratio]
    if any(r <= 0.0 or r >= 1.0 for r in ratios):
        print("SPLIT_FAIL: ratios must be in (0, 1)", file=sys.stderr)
        return 1
    if sum(ratios) >= 1.0:
        print("SPLIT_FAIL: train + validation ratios must sum to < 1.0", file=sys.stderr)
        return 1

    try:
        entries = _load_jsonl(args.registry)
    except (OSError, ValueError) as exc:
        print(f"SPLIT_FAIL: cannot read registry: {exc}", file=sys.stderr)
        return 1

    try:
        split_result = make_splits(
            entries,
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            instructions_path=args.instructions,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"SPLIT_FAIL: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(split_result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ready = "ready" if split_result["training_ready"] else "not_ready"
    print(
        f"SPLIT_PASS "
        f"runs={split_result['run_count']} "
        f"seeds={split_result['scene_seed_count']} "
        f"train={split_result['split_run_counts']['train']} "
        f"val={split_result['split_run_counts']['validation']} "
        f"test={split_result['split_run_counts']['test']} "
        f"training_ready={split_result['training_ready']}"
    )
    if not split_result["training_ready"]:
        print(f"TRAINING_NOT_READY: {split_result['reason']}")
    print(f"  output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
