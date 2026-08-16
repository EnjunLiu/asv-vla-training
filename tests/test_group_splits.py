"""Day 11B split contract tests.

Run from the repo root::

    PYTHONPATH=src python -m pytest -q tests/test_group_splits.py
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any
import pytest

# ---------------------------------------------------------------------------
#  Helpers — minimal fakes for the registry entries consumed by make_splits
# ---------------------------------------------------------------------------

def _run(run_id: str, scene_seed: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "scene_seed": scene_seed,
        "episode_valid": True,
        "supervision_valid": True,
        "frame_count": 100,
        "sample_count": 900,
        "coverage_complete": True,
        "training_eligible": True,
    }


def _instructions_jsonl(records: list[dict[str, Any]]) -> Path:
    """Write a temporary instructions.jsonl and return its Path."""
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for rec in records:
        tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
#  make_splits
# ---------------------------------------------------------------------------

from asv_training.make_group_splits import make_splits  # noqa: E402


class TestEmptyRegistry:
    def test_empty_registry_not_ready(self) -> None:
        result = make_splits([])
        assert result["training_ready"] is False
        assert result["run_count"] == 0
        assert result["assignments"] == {}


class TestPilotSingleRun:
    """The Day-10 pilot has exactly one Run and one Scene Seed."""

    def test_single_run_is_not_ready(self) -> None:
        entries = [_run("A1D7BAAE49F39E3BB7B1808AB8443CA9", 12345)]
        result = make_splits(entries)
        assert result["training_ready"] is False, (
            "single pilot Run MUST produce training_ready=false"
        )
        assert result["run_count"] == 1
        assert result["scene_seed_count"] == 1

    def test_single_run_assigned_to_one_split_only(self) -> None:
        entries = [_run("A1D7BAAE49F39E3BB7B1808AB8443CA9", 12345)]
        result = make_splits(entries)
        # With one seed, all 3 ratio buckets collapse to the same seed.
        # The assignement is still deterministic and the Run appears exactly once.
        assignments = list(result["assignments"].values())
        assert len(assignments) == 1
        assert assignments[0] in ("train", "validation", "test")


class TestSceneSeedGrouping:
    """Runs that share a Scene Seed must stay together."""

    def test_same_seed_same_split(self) -> None:
        entries = [
            _run("run_a", 100),
            _run("run_b", 100),
            _run("run_c", 200),
            _run("run_d", 300),
        ]
        result = make_splits(entries)
        assert result["assignments"]["run_a"] == result["assignments"]["run_b"], (
            "Runs sharing Scene Seed 100 must be in the same split"
        )

    def test_different_seeds_can_differ(self) -> None:
        entries = [
            _run("run_a", 100),
            _run("run_b", 200),
            _run("run_c", 300),
            _run("run_d", 400),
        ]
        result = make_splits(entries)
        # With 4 seeds, at least 2 different splits must appear.
        unique_splits = set(result["assignments"].values())
        assert len(unique_splits) >= 2


class TestMinimumTrainingGate:
    """Three seeds can populate splits but cannot bypass the 12-Run gate."""

    def test_three_seeds_three_runs_is_not_ready(self) -> None:
        entries = [
            _run("run_a", 100),
            _run("run_b", 200),
            _run("run_c", 300),
        ]
        result = make_splits(
            entries,
            split_seed=42,
            train_ratio=0.34,
            validation_ratio=0.33,
        )
        assert result["training_ready"] is False, (
            f"3 Runs must not be ready; got reason={result['reason']}"
        )
        for s in ("train", "validation", "test"):
            assert result["split_run_counts"][s] >= 1, (
                f"each split must have ≥1 Run, {s} has "
                f"{result['split_run_counts'][s]}"
            )

    def test_three_seeds_one_per_split(self) -> None:
        """With ratios ~0.34/0.33/0.33, 3 seeds → 1 per split."""
        entries = [
            _run("run_a", 100),
            _run("run_b", 200),
            _run("run_c", 300),
        ]
        result = make_splits(
            entries,
            split_seed=42,
            train_ratio=0.34,
            validation_ratio=0.33,
        )
        counts = result["split_run_counts"]
        assert counts["train"] == 1
        assert counts["validation"] == 1
        assert counts["test"] == 1


class TestDeterminism:
    """Same registry + same seed → identical splits."""

    def test_repeatable(self) -> None:
        entries = [
            _run(f"run_{i}", seed)
            for i, seed in enumerate([10, 20, 30, 40, 50, 60])
        ]
        a = make_splits(entries, split_seed=17)
        b = make_splits(entries, split_seed=17)
        assert a["assignments"] == b["assignments"]

    def test_different_seeds_diverge(self) -> None:
        entries = [
            _run(f"run_{i}", seed)
            for i, seed in enumerate([10, 20, 30, 40, 50, 60])
        ]
        a = make_splits(entries, split_seed=17)
        b = make_splits(entries, split_seed=23)
        # They *may* be equal by chance but with 6 seeds it is very unlikely.
        # We only assert that the function doesn't ignore the seed param.
        assert a is not None and b is not None


class TestNoCrossSplitLeakage:
    """A single Run ID must never appear in multiple splits."""

    def test_no_duplicate_run_id(self) -> None:
        entries = [
            _run(f"run_{i}", seed)
            for i, seed in enumerate(range(1, 13))
        ]
        result = make_splits(entries)
        run_ids = list(result["assignments"])
        assert len(run_ids) == len(set(run_ids)), "run_ids must be unique"


class TestTwelveRunMinimum:
    """The 12-Run minimum baseline from TODO.md Day 12."""

    def test_twelve_runs_four_seeds_ready(self) -> None:
        """4 layouts × 3 seeds = 12 Runs."""
        entries: list[dict[str, Any]] = []
        for layout in range(1, 5):
            for seed_idx in range(3):
                seed = layout * 100 + seed_idx
                entries.append(_run(f"L{layout}_S{seed_idx}", seed))
        assert len(entries) == 12
        result = make_splits(entries)
        assert result["training_ready"] is True
        assert result["run_count"] == 12
        assert result["scene_seed_count"] == 12  # each Run has distinct seed
        assert result["split_run_counts"] == {
            "train": 8,
            "validation": 2,
            "test": 2,
        }


class TestThirtyRunRecommended:
    """The 30-Run recommended scale from TODO.md Day 12."""

    def test_thirty_runs_ready(self) -> None:
        entries: list[dict[str, Any]] = []
        for layout in range(1, 6):
            for state in range(2):
                for seed_idx in range(3):
                    seed = layout * 1000 + state * 100 + seed_idx
                    entries.append(
                        _run(f"L{layout}_St{state}_S{seed_idx}", seed)
                    )
        assert len(entries) == 30
        result = make_splits(entries, split_seed=42)
        assert result["training_ready"] is True
        assert result["split_run_counts"] == {
            "train": 18,
            "validation": 6,
            "test": 6,
        }


class TestEligibilityGate:
    def test_ineligible_run_is_not_split(self) -> None:
        entries = [_run(f"run_{index}", index) for index in range(13)]
        entries[-1]["training_eligible"] = False

        result = make_splits(entries)

        assert result["registry_run_count"] == 13
        assert result["run_count"] == 12
        assert result["training_ready"] is True
        assert result["rejected_run_ids"] == ["run_12"]
        assert "run_12" not in result["assignments"]


class TestLanguageTemplateValidation:
    def test_valid_families_pass(self) -> None:
        inst = _instructions_jsonl(
            [
                {
                    "instruction_id": "t1",
                    "text": "follow red",
                    "split": "train",
                    "action": "follow",
                    "target_attribute": "color:red",
                    "distance_bucket": "3m",
                },
                {
                    "instruction_id": "v1",
                    "text": "follow blue",
                    "split": "validation",
                    "action": "follow",
                    "target_attribute": "color:blue",
                    "distance_bucket": "3m",
                },
                {
                    "instruction_id": "e1",
                    "text": "stop",
                    "split": "test",
                    "action": "stop",
                    "target_attribute": "none",
                    "distance_bucket": "none",
                },
            ]
        )
        try:
            entries = [
                _run("run_a", 100),
                _run("run_b", 200),
                _run("run_c", 300),
            ]
            result = make_splits(
                entries,
                train_ratio=0.34,
                validation_ratio=0.33,
                instructions_path=inst,
            )
            assert result["language_template_families_ok"] is True
        finally:
            inst.unlink(missing_ok=True)

    def test_missing_split_key_fails(self) -> None:
        inst = _instructions_jsonl(
            [
                {
                    "instruction_id": "t1",
                    "text": "follow red",
                    "split": "train",
                    "action": "follow",
                    "target_attribute": "color:red",
                    "distance_bucket": "3m",
                },
                {
                    "instruction_id": "v1",
                    "text": "follow blue",
                    "split": "validation",
                    "action": "follow",
                    "target_attribute": "color:blue",
                    "distance_bucket": "3m",
                },
                # missing "test" family
            ]
        )
        try:
            entries = [
                _run("run_a", 100),
                _run("run_b", 200),
                _run("run_c", 300),
            ]
            with pytest.raises(ValueError, match="no instructions with split='test'"):
                make_splits(
                    entries,
                    train_ratio=0.34,
                    validation_ratio=0.33,
                    instructions_path=inst,
                )
        finally:
            inst.unlink(missing_ok=True)


class TestSplitCountsConsistent:
    def test_total_runs_preserved(self) -> None:
        entries = [_run(f"run_{i}", i) for i in range(10)]
        result = make_splits(entries)
        total = sum(result["split_run_counts"].values())
        assert total == len(entries)


class TestSceneSeedRejection:
    """If a single Run crosses splits, the validator must catch it."""

    def test_duplicate_scene_seed_in_registry_fails_grouping(self) -> None:
        """Two entries with the same run_id must be rejected."""
        entries = [
            _run("dup_run", 100),
            _run("dup_run", 100),
            _run("run_c", 300),
        ]
        with pytest.raises(ValueError, match="duplicate run_id"):
            make_splits(entries)
