from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from asv_training.losses import (
    PolicyLossWeights,
    action_policy_loss,
    paired_action_contrastive_loss,
)
from asv_training.model import PolicyOutput, SmallActionPolicy, SmallPolicyConfig


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "model_small_v3.yaml"


def _inputs(batch_size: int, *, requires_grad: bool = False):
    entity_mask = torch.ones(batch_size, 16, dtype=torch.bool)
    return {
        "language": torch.randn(batch_size, 256, requires_grad=requires_grad),
        "entity_geometry": torch.randn(
            batch_size, 16, 16, requires_grad=requires_grad
        ),
        "previous_action": torch.randn(batch_size, 2, requires_grad=requires_grad),
        "language_valid": torch.ones(batch_size, dtype=torch.bool),
        "entity_geometry_mask": entity_mask,
        "previous_action_valid": torch.ones(batch_size, dtype=torch.bool),
        "policy_input_valid": torch.ones(batch_size, dtype=torch.bool),
    }


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_policy_emits_one_bounded_body_frame_action(batch_size: int) -> None:
    torch.manual_seed(42)
    output = SmallActionPolicy()(**_inputs(batch_size))
    assert output.action.shape == (batch_size, 2)
    assert output.stop_logit.shape == (batch_size, 1)
    assert output.valid_mask.shape == (batch_size,)
    assert torch.isfinite(output.action).all()
    assert torch.isfinite(output.stop_logit).all()
    assert torch.max(torch.linalg.vector_norm(output.action, dim=-1)) <= 0.3 + 1e-6


def test_decision_head_contract_excludes_visual_and_ego_inputs() -> None:
    source = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    contract = source["contract"]
    assert contract["decision_inputs"] == [
        "language",
        "entity_geometry",
        "previous_action",
        "language_valid",
        "entity_geometry_mask",
        "previous_action_valid",
        "policy_input_valid",
    ]
    assert set(contract["decision_input_exclusions"]) == {
        "global_visual",
        "entity_visual",
        "ego",
    }


def test_policy_rejects_legacy_visual_or_ego_inputs() -> None:
    inputs = _inputs(1)
    inputs["ego"] = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="legacy inputs"):
        SmallActionPolicy()(**inputs)


def test_invalid_previous_action_is_zero_sentinel_without_invalidating_sample() -> None:
    inputs = _inputs(1)
    inputs["previous_action"].fill_(float("nan"))
    inputs["previous_action_valid"].zero_()
    output = SmallActionPolicy()(**inputs)
    assert bool(output.valid_mask[0])
    assert torch.isfinite(output.action).all()


def test_policy_is_deterministic_and_all_false_entity_mask_is_finite() -> None:
    model = SmallActionPolicy().eval()
    inputs = _inputs(2)
    inputs["entity_geometry_mask"].zero_()
    inputs["entity_geometry"].fill_(float("nan"))
    first = model(**inputs)
    second = model(**inputs)
    assert torch.equal(first.action, second.action)
    assert torch.isfinite(first.action).all()
    assert first.valid_mask.all()


def test_missing_language_fails_closed_without_nan() -> None:
    inputs = _inputs(2)
    inputs["language_valid"][0] = False
    inputs["language"][0].fill_(float("nan"))
    output = SmallActionPolicy()(**inputs)
    assert not bool(output.valid_mask[0])
    assert torch.count_nonzero(output.action[0]) == 0
    assert output.stop_logit[0, 0] == 20.0
    assert torch.isfinite(output.action).all()


def test_invalid_policy_input_masks_nonfinite_modalities_before_encoding() -> None:
    inputs = _inputs(1)
    inputs["policy_input_valid"][0] = False
    inputs["language"][0].fill_(float("nan"))
    inputs["previous_action"][0].fill_(float("nan"))
    output = SmallActionPolicy()(**inputs)
    assert not bool(output.valid_mask[0])
    assert torch.count_nonzero(output.action[0]) == 0
    assert torch.isfinite(output.action).all()
    assert torch.isfinite(output.stop_logit).all()


def test_stop_logit_hard_gates_action_motion() -> None:
    model = SmallActionPolicy().eval()
    inputs = _inputs(1)
    with torch.no_grad():
        model.stop_head.weight.zero_()
        model.stop_head.bias.fill_(-20.0)
    moving = model(**inputs)
    with torch.no_grad():
        model.stop_head.bias.fill_(20.0)
    stopped = model(**inputs)
    assert torch.max(torch.abs(moving.action)) > 1e-3
    assert torch.count_nonzero(stopped.action) == 0


def test_action_loss_is_smooth_l1_and_filters_invalid_samples() -> None:
    model = SmallActionPolicy()
    inputs = _inputs(2)
    inputs["policy_input_valid"][0] = False
    output = model(**inputs)
    target = torch.zeros(2, 2)
    losses = action_policy_loss(output, target, torch.zeros(2, 1))
    assert int(losses["valid_samples"]) == 1
    assert torch.isfinite(losses["total"])
    target[1, 0] = float("nan")
    with pytest.raises(ValueError, match="target action"):
        action_policy_loss(output, target, torch.zeros(2, 1))


def test_action_loss_rejects_legacy_trajectory_target() -> None:
    output = PolicyOutput(
        action=torch.zeros(1, 2),
        stop_logit=torch.zeros(1, 1),
        valid_mask=torch.ones(1, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="target action shape"):
        action_policy_loss(output, torch.zeros(1, 20, 2), torch.zeros(1, 1))


def test_paired_action_auxiliary_loss_uses_two_dimensional_actions() -> None:
    prediction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = prediction.clone()
    exact = paired_action_contrastive_loss(prediction, target, ["pair", "pair"])
    swapped = paired_action_contrastive_loss(
        prediction, target.flip(0), ["pair", "pair"]
    )
    assert exact < swapped


def test_cached_inputs_are_detached_but_policy_gradients_flow() -> None:
    model = SmallActionPolicy()
    inputs = _inputs(8, requires_grad=True)
    output = model(**inputs)
    losses = action_policy_loss(output, torch.zeros(8, 2), torch.zeros(8, 1))
    losses["total"].backward()
    assert inputs["language"].grad is None
    assert inputs["entity_geometry"].grad is not None
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_checkpoint_config_has_no_legacy_horizon() -> None:
    source = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert "horizon" not in source["model"]
    config = SmallPolicyConfig.from_mapping(source)
    assert asdict(config)["action_dim"] == 2
