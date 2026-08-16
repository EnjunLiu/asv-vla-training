"""Losses for the single-step body-frame action policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as functional

from asv_training.model import PolicyOutput


@dataclass(frozen=True)
class PolicyLossWeights:
    action: float = 1.0
    stop: float = 0.2
    smoothness: float = 0.05
    pairwise: float = 0.0

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, float] | None
    ) -> "PolicyLossWeights":
        if value is None:
            return cls()
        known = {
            "action",
            "stop",
            "smoothness",
            "pairwise",
        }
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown loss weight keys: {sorted(unknown)}")
        action = float(value.get("action", cls.action))
        return cls(
            action=action,
            stop=float(value.get("stop", cls.stop)),
            smoothness=float(value.get("smoothness", cls.smoothness)),
            pairwise=float(value.get("pairwise", cls.pairwise)),
        )

    def __post_init__(self) -> None:
        weights = (self.action, self.stop, self.smoothness, self.pairwise)
        if any(not torch.isfinite(torch.tensor(value)) for value in weights):
            raise ValueError("loss weights must be finite")
        if any(value < 0.0 for value in weights):
            raise ValueError("loss weights must be non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one loss weight must be positive")


def _validate_finite(tensor: Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")


def paired_action_contrastive_loss(
    prediction: Tensor,
    target: Tensor,
    group_ids: Sequence[str],
    *,
    minimum_target_difference: float = 0.10,
    assignment_margin_m: float = 0.05,
) -> Tensor:
    """Keep paired language variants directionally consistent with labels."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("paired actions must have equal [B, 2] shapes")
    if prediction.shape[1] != 2:
        raise ValueError("paired actions must have action_dim=2")
    if len(group_ids) != int(prediction.shape[0]):
        raise ValueError("group_ids must match the batch size")
    if minimum_target_difference <= 0.0 or assignment_margin_m < 0.0:
        raise ValueError("pairwise thresholds are invalid")

    grouped: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(str(group_id), []).append(index)
    directional_losses: list[Tensor] = []
    assignment_losses: list[Tensor] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        pairs = torch.combinations(
            torch.tensor(indices, device=prediction.device), r=2
        )
        predicted_delta = prediction[pairs[:, 1]] - prediction[pairs[:, 0]]
        target_delta = target[pairs[:, 1]] - target[pairs[:, 0]]
        meaningful = (
            torch.linalg.vector_norm(target_delta, dim=1)
            >= minimum_target_difference
        )
        if not torch.any(meaningful):
            continue
        predicted_delta = predicted_delta[meaningful]
        target_delta = target_delta[meaningful]
        directional_losses.append(
            1.0
            - functional.cosine_similarity(
                predicted_delta, target_delta, dim=1, eps=1.0e-6
            )
        )

        selected_pairs = pairs[meaningful]
        left = selected_pairs[:, 0]
        right = selected_pairs[:, 1]

        def _error(first: Tensor, second: Tensor) -> Tensor:
            return torch.linalg.vector_norm(first - second, dim=-1)

        correct = _error(prediction[left], target[left]) + _error(
            prediction[right], target[right]
        )
        swapped = _error(prediction[left], target[right]) + _error(
            prediction[right], target[left]
        )
        assignment_losses.append(
            functional.relu(assignment_margin_m + correct - swapped)
        )

    if not directional_losses:
        return prediction.sum() * 0.0
    loss = torch.cat(directional_losses).mean() + torch.cat(
        assignment_losses
    ).mean()
    _validate_finite(loss, "pairwise action loss")
    return loss


def action_policy_loss(
    output: PolicyOutput,
    target_action: Tensor,
    target_stop: Tensor,
    *,
    previous_action: Tensor | None = None,
    previous_action_valid: Tensor | None = None,
    smoothness_margin_m: float = 0.02,
    sample_valid: Tensor | None = None,
    weights: PolicyLossWeights | None = None,
    group_ids: Sequence[str] | None = None,
) -> dict[str, Tensor]:
    """Compute SmoothL1 action loss and stop BCE over valid samples."""

    if output.action.ndim != 2 or output.action.shape[1] != 2:
        raise ValueError("policy action must have shape [B, 2]")
    batch_size = int(output.action.shape[0])
    if tuple(target_action.shape) != (batch_size, 2):
        raise ValueError(
            f"target action shape {tuple(target_action.shape)} does not match "
            f"{(batch_size, 2)}"
        )
    if tuple(output.stop_logit.shape) != (batch_size, 1):
        raise ValueError("stop_logit must have shape [B, 1]")
    if tuple(target_stop.shape) not in {(batch_size,), (batch_size, 1)}:
        raise ValueError("target_stop must have shape [B] or [B, 1]")
    if tuple(output.valid_mask.shape) != (batch_size,):
        raise ValueError("output.valid_mask must have shape [B]")
    if previous_action is not None and tuple(previous_action.shape) != (
        batch_size,
        2,
    ):
        raise ValueError("previous_action must have shape [B, 2]")
    if previous_action_valid is not None and tuple(previous_action_valid.shape) != (
        batch_size,
    ):
        raise ValueError("previous_action_valid must have shape [B]")
    if not torch.isfinite(torch.tensor(float(smoothness_margin_m))) or float(
        smoothness_margin_m
    ) < 0.0:
        raise ValueError("smoothness_margin_m must be finite and non-negative")

    valid = output.valid_mask
    if sample_valid is not None:
        if tuple(sample_valid.shape) != (batch_size,):
            raise ValueError("sample_valid must have shape [B]")
        valid = valid & sample_valid.to(device=valid.device, dtype=torch.bool)
    if not torch.any(valid):
        raise ValueError("loss batch has no valid samples")

    prediction = output.action[valid]
    action_target = target_action.to(
        device=prediction.device, dtype=prediction.dtype
    )[valid]
    stop_prediction = output.stop_logit[valid]
    stop_target = target_stop.reshape(batch_size, 1).to(
        device=stop_prediction.device, dtype=stop_prediction.dtype
    )[valid]
    _validate_finite(prediction, "policy action")
    _validate_finite(action_target, "target action")
    _validate_finite(stop_prediction, "stop_logit")
    _validate_finite(stop_target, "target_stop")
    if torch.any((stop_target < 0.0) | (stop_target > 1.0)):
        raise ValueError("target_stop values must be in [0, 1]")

    loss_weights = weights or PolicyLossWeights()
    action = functional.smooth_l1_loss(prediction, action_target)
    stop = functional.binary_cross_entropy_with_logits(
        stop_prediction, stop_target
    )
    # Penalize only action jumps larger than the expert's own frame-to-frame
    # jump. This keeps the expert-following objective dominant while reducing
    # closed-loop oscillation when perception jitters.
    smoothness = prediction.sum() * 0.0
    if previous_action is not None and previous_action_valid is not None:
        previous = previous_action.to(
            device=prediction.device, dtype=prediction.dtype
        )
        previous_valid = previous_action_valid.to(
            device=prediction.device, dtype=torch.bool
        )
        history_valid = valid & previous_valid
        if torch.any(history_valid):
            previous_subset = previous[history_valid]
            _validate_finite(previous_subset, "previous action")
            predicted_jump = torch.linalg.vector_norm(
                prediction[previous_valid[valid]], dim=-1
            )
            expert_jump = torch.linalg.vector_norm(
                action_target[previous_valid[valid]] - previous_subset,
                dim=-1,
            )
            smoothness = functional.relu(
                predicted_jump - expert_jump - float(smoothness_margin_m)
            ).square().mean()
    if loss_weights.pairwise > 0.0:
        if group_ids is None:
            raise ValueError("positive pairwise weight requires group_ids")
        pairwise = paired_action_contrastive_loss(
            prediction,
            action_target,
            [group_ids[index] for index, keep in enumerate(valid) if bool(keep)],
        )
    else:
        pairwise = prediction.sum() * 0.0
    total = (
        loss_weights.action * action
        + loss_weights.stop * stop
        + loss_weights.smoothness * smoothness
        + loss_weights.pairwise * pairwise
    )
    _validate_finite(total, "total loss")
    return {
        "total": total,
        "action": action,
        "stop": stop,
        "smoothness": smoothness,
        "pairwise": pairwise,
        "valid_samples": torch.tensor(
            int(torch.count_nonzero(valid)),
            device=total.device,
            dtype=torch.int64,
        ),
    }


__all__ = [
    "PolicyLossWeights",
    "action_policy_loss",
    "paired_action_contrastive_loss",
]
