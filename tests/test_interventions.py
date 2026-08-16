from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from asv_training.interventions import (  # noqa: E402
    apply_intervention,
    compute_pair_response,
    pair_response_passed,
)


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "language": torch.ones(2, 3),
        "entity_geometry": torch.arange(16, dtype=torch.float32).reshape(1, 4, 4).repeat(2, 1, 1),
        "previous_action": torch.zeros(2, 2),
        "language_valid": torch.ones(2, dtype=torch.bool),
        "entity_geometry_mask": torch.ones(2, 4, dtype=torch.bool),
        "previous_action_valid": torch.zeros(2, dtype=torch.bool),
        "policy_input_valid": torch.ones(2, dtype=torch.bool),
    }


def test_decision_interventions_do_not_mutate_source() -> None:
    source = _inputs()
    original_language = source["language"].clone()
    original_geometry = source["entity_geometry"].clone()

    no_language = apply_intervention(source, "no_language")
    no_geometry = apply_intervention(source, "no_entity_geometry")
    misaligned = apply_intervention(source, "entity_alignment_error")

    assert torch.count_nonzero(no_language["language"]) == 0
    assert torch.count_nonzero(no_geometry["entity_geometry"]) == 0
    assert not torch.any(misaligned["policy_input_valid"])
    assert torch.equal(source["language"], original_language)
    assert torch.equal(source["entity_geometry"], original_geometry)


def test_fail_closed_faults_clear_validity() -> None:
    source = _inputs()
    assert not torch.any(apply_intervention(source, "missing_language")["language_valid"])
    assert not torch.any(
        apply_intervention(source, "missing_entity_geometry")["policy_input_valid"]
    )


def test_pair_response_detects_correct_and_swapped_assignment() -> None:
    left = np.zeros((4, 2), dtype=np.float32)
    right = np.zeros_like(left)
    right[:, 0] = 2.0
    correct = compute_pair_response(left, right, left, right)
    wrong = compute_pair_response(right, left, left, right)
    thresholds = {
        "minimum_directional_accuracy": 0.6,
        "minimum_assignment_accuracy": 0.6,
        "minimum_response_ratio": 0.02,
    }
    assert correct["directional_accuracy"] == pytest.approx(1.0)
    assert pair_response_passed(correct, thresholds)
    assert wrong["directional_accuracy"] == pytest.approx(0.0)
    assert not pair_response_passed(wrong, thresholds)


def test_pair_response_rejects_identical_expert_actions() -> None:
    values = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="never differ"):
        compute_pair_response(values, values, values, values)
