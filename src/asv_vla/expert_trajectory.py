"""Deterministic Day 9 FOLLOW/STOP expert trajectory labels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from .trajectory_contract import DT_SEC


MODEL_VERSION = "deterministic_follow_stop_expert_action_v2"
DEFAULT_MAX_SPEED_MPS = 1.5
BEARING_DEADBAND_M = 0.25
SUPPORTED_TARGET_ATTRIBUTES = {
    "color:red",
    "color:blue",
    "bearing:left",
    "bearing:right",
}
COLOR_ALIASES = {
    "red": "red",
    "红": "red",
    "红色": "red",
    "blue": "blue",
    "蓝": "blue",
    "蓝色": "blue",
}


class ExpertTrajectoryError(ValueError):
    """Raised when a deterministic expert label cannot be generated."""


@dataclass(frozen=True)
class ExpertTask:
    action: str
    target_attribute: str
    desired_distance_m: float


@dataclass(frozen=True)
class ExpertActionResult:
    expert_action: tuple[float, float]
    safe_stop: bool
    selected_entity_id: str
    detail: str


@dataclass(frozen=True)
class _Target:
    entity: Any
    entity_id: str
    distance_m: float


def task_from_labels(
    action: str,
    target_attribute: str,
    distance_bucket: str,
) -> ExpertTask:
    normalized_action = str(action).strip().casefold()
    normalized_attribute = str(target_attribute).strip().casefold()
    normalized_distance = str(distance_bucket).strip().casefold()

    if normalized_action == "stop":
        if (
            normalized_attribute not in {"", "none"}
            or normalized_distance not in {"", "none"}
        ):
            raise ExpertTrajectoryError(
                "STOP labels must use target_attribute=none and "
                "distance_bucket=none"
            )
        return ExpertTask(
            action="stop",
            target_attribute="none",
            desired_distance_m=0.0,
        )

    if normalized_action != "follow":
        raise ExpertTrajectoryError(
            f"unsupported action: {action!r}; expected follow or stop"
        )
    if normalized_attribute not in SUPPORTED_TARGET_ATTRIBUTES:
        raise ExpertTrajectoryError(
            f"unsupported target_attribute: {target_attribute!r}"
        )
    distances = {"2.5m": 2.5, "3m": 3.0, "4m": 4.0, "10m": 10.0}
    if normalized_distance not in distances:
        raise ExpertTrajectoryError(
            f"unsupported distance_bucket: {distance_bucket!r}"
        )
    return ExpertTask(
        action="follow",
        target_attribute=normalized_attribute,
        desired_distance_m=distances[normalized_distance],
    )


def _finite_entity(entity: Any) -> tuple[str, tuple[float, float, float, float]]:
    entity_id = str(entity.entity_id).strip()
    if not entity_id:
        raise ExpertTrajectoryError("target entity_id must not be empty")
    values = (
        float(entity.relative_x),
        float(entity.relative_y),
        float(entity.relative_velocity_x),
        float(entity.relative_velocity_y),
    )
    if not all(math.isfinite(value) for value in values):
        raise ExpertTrajectoryError(
            f"target {entity_id!r} contains NaN or Inf"
        )
    return entity_id, values


def _matches(entity: Any, target_attribute: str) -> bool:
    selector, expected = target_attribute.split(":", maxsplit=1)
    if selector == "color":
        color = COLOR_ALIASES.get(
            str(entity.color).strip().casefold(),
            str(entity.color).strip().casefold(),
        )
        return color == expected
    y = float(entity.relative_y)
    if selector == "bearing" and expected == "left":
        return y > BEARING_DEADBAND_M
    if selector == "bearing" and expected == "right":
        return y < -BEARING_DEADBAND_M
    return False


def select_target(
    entities: Iterable[Any],
    target_attribute: str,
) -> Any:
    if target_attribute not in SUPPORTED_TARGET_ATTRIBUTES:
        raise ExpertTrajectoryError(
            f"unsupported target_attribute: {target_attribute!r}"
        )

    candidates: list[_Target] = []
    seen_ids: set[str] = set()
    for entity in entities:
        if not (
            bool(entity.valid)
            and bool(entity.visible)
            and bool(entity.is_target)
        ):
            continue
        entity_id, (x, y, _, _) = _finite_entity(entity)
        if entity_id in seen_ids:
            raise ExpertTrajectoryError(
                f"duplicate visible target entity_id: {entity_id!r}"
            )
        seen_ids.add(entity_id)
        if _matches(entity, target_attribute):
            candidates.append(
                _Target(
                    entity=entity,
                    entity_id=entity_id,
                    distance_m=math.hypot(x, y),
                )
            )

    if not candidates:
        raise ExpertTrajectoryError(
            f"no valid visible target matches {target_attribute}"
        )
    return min(
        candidates,
        key=lambda candidate: (
            candidate.distance_m,
            candidate.entity_id,
        ),
    ).entity


def _clip_step(
    dx: float,
    dy: float,
    max_step_m: float,
) -> tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm <= 1.0e-12:
        return 0.0, 0.0
    # A bounded proportional map retains distance information in one action
    # even when the requested standoff correction exceeds one control step.
    scale = max_step_m * math.tanh(norm / max_step_m) / norm
    return dx * scale, dy * scale


def generate_expert_trajectory(
    task: ExpertTask,
    entities: Iterable[Any],
    *,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> ExpertActionResult:
    if not math.isfinite(max_speed_mps) or max_speed_mps <= 0.0:
        raise ExpertTrajectoryError("max_speed_mps must be positive and finite")

    if task.action == "stop":
        return ExpertActionResult(
            expert_action=(0.0, 0.0),
            safe_stop=True,
            selected_entity_id="",
            detail="STOP: deterministic zero-displacement safety label",
        )
    if task.action != "follow":
        raise ExpertTrajectoryError(f"unsupported action: {task.action!r}")
    if task.target_attribute not in SUPPORTED_TARGET_ATTRIBUTES:
        raise ExpertTrajectoryError(
            f"unsupported target_attribute: {task.target_attribute!r}"
        )
    if (
        not math.isfinite(task.desired_distance_m)
        or task.desired_distance_m <= 0.0
    ):
        raise ExpertTrajectoryError(
            "FOLLOW desired_distance_m must be positive and finite"
        )

    target = select_target(entities, task.target_attribute)
    entity_id, (x, y, vx, vy) = _finite_entity(target)
    if x <= 0.0:
        # Fail-closed label: a target at/behind the camera is not visually
        # observable, so a plain FOLLOW label would teach the policy to
        # drive away from what it sees (the baseline UE5 scene flips the ASV
        # yaw mid-run, leaving targets behind the camera in base_link).
        # Label those frames STOP instead of following the inverted
        # coordinate.
        return ExpertActionResult(
            expert_action=(0.0, 0.0),
            safe_stop=True,
            selected_entity_id=entity_id,
            detail=(
                f"FOLLOW fail-closed: target {entity_id!r} is not in front "
                f"of the camera (x={x:.3f} m); deterministic STOP label"
            ),
        )
    predicted_x = x + vx * DT_SEC
    predicted_y = y + vy * DT_SEC
    predicted_distance = math.hypot(predicted_x, predicted_y)
    if predicted_distance <= 1.0e-9:
        raise ExpertTrajectoryError(
            f"target {entity_id!r} reaches the ASV origin"
        )

    # Return one body-frame displacement for this source frame. The desired
    # standoff displacement is clipped by the maximum distance allowed in dt.
    scale = (
        predicted_distance - task.desired_distance_m
    ) / predicted_distance
    desired_x = predicted_x * scale
    desired_y = predicted_y * scale
    action_x, action_y = _clip_step(
        desired_x,
        desired_y,
        max_speed_mps * DT_SEC,
    )
    expert_action = (action_x, action_y)
    if len(expert_action) != 2 or not all(
        math.isfinite(value) for value in expert_action
    ):
        raise ExpertTrajectoryError("expert action contains NaN or Inf")

    return ExpertActionResult(
        expert_action=expert_action,
        safe_stop=False,
        selected_entity_id=entity_id,
        detail=(
            f"FOLLOW:target={entity_id};"
            f"selector={task.target_attribute};"
            f"distance_m={task.desired_distance_m:.3f};"
            f"max_speed_mps={max_speed_mps:.3f}"
        ),
    )


