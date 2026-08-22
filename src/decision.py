"""Decision algorithm and ROS boundary for ASV displacement commands.

The perception algorithm owns image understanding and temporal velocity:

    camera -> perception -> structured entity tensor -> decision

This node consumes ``TaskEmbedding``, tracked entities and the current
``ASVState``, constructs its structured feature tensor, then publishes one
bounded body-frame displacement for the next control interval. There is no
trajectory horizon, global visual token, entity crop token or bbox input.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from interfaces.msg import ASVState, DesiredDisplacement, EntityArray, TaskEmbedding
except ModuleNotFoundError:  # Allow the algorithm half to run in offline tests.
    rclpy = None

    class Node:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("ROS 2 dependencies are required to instantiate DecisionNode")

    class _QoS:
        RELIABLE = 1
        TRANSIENT_LOCAL = 1

    DurabilityPolicy = ReliabilityPolicy = _QoS

    class QoSProfile:
        def __init__(self, **kwargs) -> None:
            self.settings = kwargs

    ASVState = DesiredDisplacement = EntityArray = TaskEmbedding = Any

import math
from typing import Protocol


# ``HORIZON`` remains an offline/model constant.  It is intentionally not
# part of the online ROS control contract: the policy adapter consumes the
# frozen model output and publishes one displacement command per frame.
HORIZON = 20
ACTION_DIM = 2
DT_SEC = 0.5
FRAME_ID = "base_link"
# The trained decision head and online desired_x/desired_y contract share one
# bounded single-step displacement.  The 0.2 s control interval is unchanged.
MAX_DISPLACEMENT_M = 0.50
SAFE_STOP_SOURCE = "safe_stop"
FLOAT_TOLERANCE = 1.0e-6


class DesiredDisplacementLike(Protocol):
    stamp_us: int
    run_id: str
    scene_seed: int
    frame_index: int
    frame_id: str
    source: str
    step_dt: float
    desired_x: float
    desired_y: float
    safe_stop: bool
    valid: bool


def finite_zero(value: float, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return math.isfinite(value) and abs(value) <= tolerance


def is_safe_stop(message: DesiredDisplacementLike) -> bool:
    """Validate a non-executable, single-point safe-stop marker.

    A safe stop is deliberately invalid.  Downstream adapters must interpret
    it as a hold, rather than as a valid zero displacement that could trigger
    position-hold compensation in a physical controller.
    """

    return (
        int(message.stamp_us) > 0
        and bool(str(message.run_id).strip())
        and message.frame_id == FRAME_ID
        and message.source == SAFE_STOP_SOURCE
        and math.isfinite(float(message.step_dt))
        and abs(float(message.step_dt) - DT_SEC) <= FLOAT_TOLERANCE
        and finite_zero(float(message.desired_x))
        and finite_zero(float(message.desired_y))
        and bool(message.safe_stop)
        and not bool(message.valid)
    )



"""Image-only standoff guidance from normalized ``EntityFeatures``.

This module is intentionally ROS-free.  It consumes only the entity IDs,
mask, normalized geometry, and optional tracker velocity already present in a
``EntityFeatures``-shaped object. UE truth entities are neither imported nor
accepted as an input.
"""

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

import numpy as np



FEATURE_DIM = 16
POSITION_SCALE_M = 20.0
VELOCITY_SCALE_MPS = 5.0
DEFAULT_STANDOFF_M = 3.0
DEFAULT_GUARD_MAX_STEP_M = MAX_DISPLACEMENT_M
# Keep the hold region narrow enough that the trained raw action remains the
# default outside a small standoff tolerance.
DEFAULT_DEADBAND_M = 0.20
# A deadband may suppress numerical jitter, but it must not erase a meaningful
# policy command while the target is moving through the standoff band.
HOLD_ACTION_NORM_M = 0.02
DEFAULT_PREDICTION_HORIZON_SEC = DT_SEC
MAX_TARGET_DISTANCE_M = 20.0
MAX_TARGET_SPEED_MPS = 5.0
TARGET_IDS = ("target_red", "target_blue", "target_left", "target_right")

# Backstop semantics: the learned policy drives the executed point; the
# deterministic radial step only replaces a clearly reversed policy action.
BACKSTOP_DOT_THRESHOLD = -0.25
# Retained as a keyword-compatible parameter; action magnitude alone no
# longer activates the backstop.
BACKSTOP_ZERO_STEP_M = 1.0e-3

# Guard outcomes returned by ``apply_standoff_guard``.
GUARD_PASS_THROUGH = "pass_through"  # non-FOLLOW task, action unchanged
GUARD_FAIL_CLOSED = "fail_closed"  # FOLLOW, target missing/OOD -> safe stop
GUARD_BACKSTOP = "backstop"  # FOLLOW, policy step unsafe -> radial replacement
GUARD_POLICY_DRIVEN = "policy_driven"  # FOLLOW, policy step kept
GUARD_HOLD = "deadband_hold"  # FOLLOW, standoff deadband -> zero hold

_DISTANCE_RE = re.compile(
    r"(?<![\d.])([0-9]+(?:\.[0-9]+)?)\s*(?:m|米|meters?|metres?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TargetObservation:
    """Finite image/tracker geometry in the vehicle ``base_link`` frame."""

    entity_id: str
    relative_x: float
    relative_y: float
    relative_velocity_x: float = 0.0
    relative_velocity_y: float = 0.0
    velocity_valid: bool = True

    @property
    def distance_m(self) -> float:
        return math.hypot(self.relative_x, self.relative_y)


def _instruction_text(entity_features: Any, instruction: str | None) -> str:
    if instruction is not None:
        return str(instruction).strip()
    return str(getattr(entity_features, "instruction", "")).strip()


def is_follow_instruction(instruction: str) -> bool:
    """Return whether a task asks for FOLLOW-like standoff behavior."""

    text = str(instruction).strip().casefold()
    if not text or any(token in text for token in ("stop", "停止", "停船")):
        return False
    return any(
        token in text
        for token in ("follow", "track", "跟随", "跟踪", "追踪", "保持", "靠近")
    )


def target_id_from_instruction(instruction: str) -> str | None:
    """Map color/bearing words to the canonical task-tensor entity ID."""

    text = str(instruction).strip().casefold()
    if not text:
        return None
    for target_id in TARGET_IDS:
        if target_id in text:
            return target_id
    if any(token in text for token in ("red", "红", "赤")):
        return "target_red"
    if any(token in text for token in ("blue", "蓝")):
        return "target_blue"
    if any(token in text for token in ("left", "左")):
        return "target_left"
    if any(token in text for token in ("right", "右")):
        return "target_right"
    return None


def desired_standoff_from_instruction(
    instruction: str,
    *,
    default_m: float = DEFAULT_STANDOFF_M,
) -> float:
    """Parse ``3m``/``3米``/``10 meters`` with a safe 3 m default."""

    default = float(default_m)
    if not math.isfinite(default) or default <= 0.0:
        raise ValueError("default standoff must be finite and positive")
    match = _DISTANCE_RE.search(str(instruction))
    if match is None:
        return default
    value = float(match.group(1))
    if not math.isfinite(value) or value <= 0.0 or value > MAX_TARGET_DISTANCE_M:
        return default
    return value


def _feature_row(entity_features: Any, index: int) -> np.ndarray | None:
    try:
        feature_dim = int(getattr(entity_features, "feature_dim", FEATURE_DIM))
        max_entities = int(
            getattr(
                entity_features,
                "max_entities",
                len(getattr(entity_features, "entity_ids", [])),
            )
        )
        values = np.asarray(entity_features.features, dtype=np.float64).reshape(
            max_entities, feature_dim
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if feature_dim < 5 or index < 0 or index >= max_entities:
        return None
    return values[index]


def extract_target_observation(
    entity_features: Any,
    instruction: str | None = None,
) -> TargetObservation | None:
    """Extract the instruction-selected, visible target from image features."""

    text = _instruction_text(entity_features, instruction)
    target_id = target_id_from_instruction(text)
    if target_id is None or not bool(getattr(entity_features, "valid", False)):
        return None
    try:
        entity_ids = [str(value).strip() for value in entity_features.entity_ids]
        masks = [bool(value) for value in entity_features.mask]
        index = entity_ids.index(target_id)
    except (AttributeError, ValueError, TypeError):
        return None
    if index >= len(masks) or not masks[index]:
        return None
    row = _feature_row(entity_features, index)
    if row is None or not np.all(np.isfinite(row[:5])):
        return None
    relative_x = float(row[0] * POSITION_SCALE_M)
    relative_y = float(row[1] * POSITION_SCALE_M)
    relative_velocity_x = float(row[3] * VELOCITY_SCALE_MPS)
    relative_velocity_y = float(row[4] * VELOCITY_SCALE_MPS)
    distance = math.hypot(relative_x, relative_y)
    if not (
        math.isfinite(distance)
        and 0.05 <= relative_x <= MAX_TARGET_DISTANCE_M
        and abs(relative_y) <= MAX_TARGET_DISTANCE_M
        and distance <= MAX_TARGET_DISTANCE_M
        and abs(relative_velocity_x) <= MAX_TARGET_SPEED_MPS
        and abs(relative_velocity_y) <= MAX_TARGET_SPEED_MPS
    ):
        return None
    return TargetObservation(
        entity_id=target_id,
        relative_x=relative_x,
        relative_y=relative_y,
        relative_velocity_x=relative_velocity_x,
        relative_velocity_y=relative_velocity_y,
        velocity_valid=True,
    )


def compute_standoff_step(
    observation: TargetObservation,
    desired_standoff_m: float = DEFAULT_STANDOFF_M,
    *,
    guard_max_step_m: float = DEFAULT_GUARD_MAX_STEP_M,
    deadband_m: float = DEFAULT_DEADBAND_M,
    prediction_horizon_sec: float = DEFAULT_PREDICTION_HORIZON_SEC,
) -> tuple[float, float] | None:
    """Return one bounded radial step toward/away from the predicted target."""

    desired = float(desired_standoff_m)
    max_step = float(guard_max_step_m)
    deadband = float(deadband_m)
    horizon = float(prediction_horizon_sec)
    if not (
        math.isfinite(desired)
        and desired > 0.0
        and math.isfinite(max_step)
        and max_step >= 0.0
        and math.isfinite(deadband)
        and deadband >= 0.0
        and math.isfinite(horizon)
        and horizon >= 0.0
    ):
        return None
    values = (
        float(observation.relative_x),
        float(observation.relative_y),
        float(observation.relative_velocity_x),
        float(observation.relative_velocity_y),
    )
    if not all(math.isfinite(value) for value in values):
        return None
    predicted_x = values[0]
    predicted_y = values[1]
    if observation.velocity_valid:
        predicted_x += values[2] * horizon
        predicted_y += values[3] * horizon
    distance = math.hypot(predicted_x, predicted_y)
    if not (
        0.05 <= predicted_x <= MAX_TARGET_DISTANCE_M
        and abs(predicted_y) <= MAX_TARGET_DISTANCE_M
        and 0.05 <= distance <= MAX_TARGET_DISTANCE_M
    ):
        return None
    error = distance - desired
    if abs(error) <= deadband:
        return 0.0, 0.0
    step_norm = min(abs(error), max_step)
    sign = 1.0 if error > 0.0 else -1.0
    return (
        float(sign * step_norm * predicted_x / distance),
        float(sign * step_norm * predicted_y / distance),
    )


def apply_standoff_guard(
    displacement: Sequence[float] | np.ndarray,
    entity_features: Any,
    *,
    dot_threshold: float = BACKSTOP_DOT_THRESHOLD,
    zero_step_m: float = BACKSTOP_ZERO_STEP_M,
    deadband_m: float = DEFAULT_DEADBAND_M,
) -> tuple[tuple[float, ...] | None, str]:
    """Apply the learned-policy-first standoff backstop.

    Returns ``(desired_displacement, reason)`` where ``reason`` is one of
    ``GUARD_PASS_THROUGH`` / ``GUARD_FAIL_CLOSED`` / ``GUARD_BACKSTOP`` /
    ``GUARD_POLICY_DRIVEN`` / ``GUARD_HOLD``.

    - Non-FOLLOW tasks retain the finite policy displacement unchanged.
    - A FOLLOW task with a missing/OOD target returns ``(None, GUARD_FAIL_CLOSED)``
      so the caller can publish a safe stop.
    - A FOLLOW task with a visible target keeps the policy's direct action
      (the learned policy drives the executed command) unless the action
      has a substantial reverse projection (``dot(action, target_dir) <
      dot_threshold``, -0.25 m by default). Only then is it replaced by the
      deterministic radial standoff action (``GUARD_BACKSTOP``). Lateral
      geometry remains a decision-head responsibility; it does not silently
      replace a valid learned action.
    """

    # ``zero_step_m`` is retained for callers of the previous interface.  A
    # small but directionally correct learned action must now pass through.
    _ = zero_step_m
    try:
        values = np.asarray(displacement, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None, GUARD_FAIL_CLOSED
    if values.size != 2 or not np.all(np.isfinite(values)):
        return None, GUARD_FAIL_CLOSED
    instruction = _instruction_text(entity_features, None)
    if not is_follow_instruction(instruction):
        return tuple(float(value) for value in values), GUARD_PASS_THROUGH
    observation = extract_target_observation(entity_features, instruction)
    if observation is None:
        return None, GUARD_FAIL_CLOSED
    desired = desired_standoff_from_instruction(instruction)
    try:
        base_deadband = float(deadband_m)
    except (TypeError, ValueError):
        return None, GUARD_FAIL_CLOSED
    if not math.isfinite(base_deadband) or base_deadband < 0.0:
        return None, GUARD_FAIL_CLOSED
    error = observation.distance_m - desired
    effective_deadband = base_deadband
    step = compute_standoff_step(
        observation,
        desired,
        deadband_m=effective_deadband,
    )
    if step is None:
        return None, GUARD_FAIL_CLOSED

    # Only suppress a near-zero command inside the deadband. A meaningful raw
    # action remains policy-driven so the controller can follow a moving target.
    if abs(error) <= effective_deadband and np.linalg.norm(values) <= HOLD_ACTION_NORM_M:
        values = values.copy()
        values[:] = (0.0, 0.0)
        return tuple(float(value) for value in values), GUARD_HOLD

    first_step = values
    target_dir = (observation.relative_x, observation.relative_y)
    target_norm = math.hypot(*target_dir)
    if target_norm <= 0.0:
        return None, GUARD_FAIL_CLOSED
    target_dir = (target_dir[0] / target_norm, target_dir[1] / target_norm)
    dot_value = first_step[0] * target_dir[0] + first_step[1] * target_dir[1]
    if dot_value < dot_threshold:
        values = values.copy()
        values[:] = step
        return tuple(float(value) for value in values), GUARD_BACKSTOP
    return tuple(float(value) for value in values), GUARD_POLICY_DRIVEN



"""Deterministic safety gate for one online body-frame displacement.

The gate is the only component between the CUDA policy and the kinematic
controller.  It validates the current ``(desired_x, desired_y)`` command,
checks its one-step collision envelope, and fails closed on every rejection.
The model's offline [20, 2] output is not accepted at this boundary.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Sequence


PASS = "PASS"
POLICY_STOP = "POLICY_STOP"
STALE_INPUT = "STALE_INPUT"
INVALID_MODALITY = "INVALID_MODALITY"
INVALID_SHAPE = "INVALID_SHAPE"
NONFINITE = "NONFINITE"
SPEED_LIMIT = "SPEED_LIMIT"
COLLISION_RISK = "COLLISION_RISK"
CONTROL_UNREACHABLE = "CONTROL_UNREACHABLE"
ESTOP = "ESTOP"

REJECTION_CODES = frozenset(
    {
        STALE_INPUT,
        INVALID_MODALITY,
        INVALID_SHAPE,
        NONFINITE,
        SPEED_LIMIT,
        COLLISION_RISK,
        CONTROL_UNREACHABLE,
        ESTOP,
    }
)

DEFAULT_MAX_STEP_M = MAX_DISPLACEMENT_M
DEFAULT_MAX_SPEED_MPS = DEFAULT_MAX_STEP_M / DT_SEC
DEFAULT_STALE_TIMEOUT_SEC = 1.0
DEFAULT_ESTOP_TIMEOUT_SEC = 2.0
DEFAULT_COLLISION_MARGIN_M = 0.5


@dataclass(frozen=True)
class SafetyGateConfig:
    """Limits for one body-frame command evaluated over ``dt``."""

    max_step_m: float = DEFAULT_MAX_STEP_M
    stale_timeout_sec: float = DEFAULT_STALE_TIMEOUT_SEC
    estop_timeout_sec: float = DEFAULT_ESTOP_TIMEOUT_SEC
    collision_margin_m: float = DEFAULT_COLLISION_MARGIN_M

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_step_m) or self.max_step_m <= 0.0:
            raise ValueError("max_step_m must be positive and finite")
        if not math.isfinite(self.stale_timeout_sec) or self.stale_timeout_sec <= 0.0:
            raise ValueError("stale_timeout_sec must be positive and finite")
        if not math.isfinite(self.estop_timeout_sec) or self.estop_timeout_sec <= 0.0:
            raise ValueError("estop_timeout_sec must be positive and finite")
        if self.estop_timeout_sec < self.stale_timeout_sec:
            raise ValueError("estop_timeout_sec must be >= stale_timeout_sec")
        if not math.isfinite(self.collision_margin_m) or self.collision_margin_m <= 0.0:
            raise ValueError("collision_margin_m must be positive and finite")


@dataclass(frozen=True)
class SafetyGateResult:
    desired_x: float
    desired_y: float
    safe_stop: bool
    valid: bool
    reason: str
    detail: str = ""


def limit_displacement_rate(
    result: SafetyGateResult,
    previous: tuple[float, float] | None,
    *,
    max_delta_m: float = DEFAULT_MAX_STEP_M,
) -> SafetyGateResult:
    """Limit the change between consecutive executable displacements."""

    if not result.valid or previous is None:
        return result
    if not math.isfinite(max_delta_m) or max_delta_m <= 0.0:
        return _reject(CONTROL_UNREACHABLE, "invalid rate limit")
    previous_x, previous_y = map(float, previous)
    if not (math.isfinite(previous_x) and math.isfinite(previous_y)):
        return _reject(CONTROL_UNREACHABLE, "invalid previous displacement")
    delta_x = float(result.desired_x) - previous_x
    delta_y = float(result.desired_y) - previous_y
    delta_norm = math.hypot(delta_x, delta_y)
    if delta_norm <= max_delta_m + FLOAT_TOLERANCE:
        return result
    scale = max_delta_m / delta_norm
    return SafetyGateResult(
        desired_x=previous_x + delta_x * scale,
        desired_y=previous_y + delta_y * scale,
        safe_stop=False,
        valid=True,
        reason=result.reason,
        detail="RATE_LIMITED",
    )


@dataclass(frozen=True)
class _Entity:
    entity_id: str
    relative_x: float
    relative_y: float
    relative_vx: float
    relative_vy: float


def _reject(code: str, detail: str = "") -> SafetyGateResult:
    """Return a non-executable zero command for every rejected input."""

    return SafetyGateResult(
        desired_x=0.0,
        desired_y=0.0,
        safe_stop=True,
        valid=False,
        reason=code,
        detail=detail,
    )


def _check_modality_and_shape(
    *,
    stamp_us: int,
    run_id: str,
    frame_id: str,
    model_version: str,
    dt: float,
    desired_x: float,
    desired_y: float,
    policy_valid: bool,
    language_valid: bool,
    entity_valid: bool,
    last_valid_stamp_us: int,
) -> str | None:
    """Return a rejection code before any command is executed."""

    if int(stamp_us) <= 0 or int(stamp_us) <= int(last_valid_stamp_us):
        return STALE_INPUT
    if not all((language_valid, entity_valid)):
        return INVALID_MODALITY
    if not policy_valid:
        return INVALID_MODALITY
    if not str(run_id).strip() or not str(model_version).strip():
        return INVALID_MODALITY
    if frame_id != FRAME_ID:
        return INVALID_MODALITY
    if not math.isfinite(float(dt)) or abs(float(dt) - DT_SEC) > FLOAT_TOLERANCE:
        return INVALID_SHAPE
    try:
        x = float(desired_x)
        y = float(desired_y)
    except (TypeError, ValueError):
        return INVALID_SHAPE
    if not (math.isfinite(x) and math.isfinite(y)):
        return NONFINITE
    return None


def _check_kinematics(
    desired_x: float,
    desired_y: float,
    config: SafetyGateConfig,
) -> str | None:
    """Check the norm of the one-step displacement."""

    norm = math.hypot(float(desired_x), float(desired_y))
    if not math.isfinite(norm):
        return NONFINITE
    if norm > config.max_step_m + FLOAT_TOLERANCE:
        return SPEED_LIMIT
    return None


def _check_entity_finiteness(entities: Sequence[_Entity]) -> str | None:
    for entity in entities:
        values = (
            entity.relative_x,
            entity.relative_y,
            entity.relative_vx,
            entity.relative_vy,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return NONFINITE
    return None


def _check_collision(
    desired_x: float,
    desired_y: float,
    entities: Sequence[_Entity],
    config: SafetyGateConfig,
    *,
    dt: float = DT_SEC,
) -> str | None:
    """Check the one executed setpoint against constant-velocity entities."""

    for entity in entities:
        predicted_x = float(entity.relative_x) + float(entity.relative_vx) * float(dt)
        predicted_y = float(entity.relative_y) + float(entity.relative_vy) * float(dt)
        distance = math.hypot(
            float(desired_x) - predicted_x,
            float(desired_y) - predicted_y,
        )
        if not math.isfinite(distance):
            return NONFINITE
        if distance < config.collision_margin_m:
            return COLLISION_RISK
    return None


def evaluate_safety_gate(
    *,
    stamp_us: int,
    run_id: str,
    frame_id: str,
    model_version: str,
    dt: float,
    desired_x: float,
    desired_y: float,
    safe_stop: bool,
    valid: bool,
    reason: str,
    language_valid: bool = True,
    entity_valid: bool = True,
    entities: Sequence[_Entity] | None = None,
    last_valid_stamp_us: int = 0,
    time_since_last_valid_sec: float = 0.0,
    config: SafetyGateConfig | None = None,
) -> SafetyGateResult:
    """Evaluate one body-frame displacement through the safety gate."""

    cfg = config or SafetyGateConfig()
    rejection = _check_modality_and_shape(
        stamp_us=stamp_us,
        run_id=run_id,
        frame_id=frame_id,
        model_version=model_version,
        dt=dt,
        desired_x=desired_x,
        desired_y=desired_y,
        policy_valid=valid,
        language_valid=language_valid,
        entity_valid=entity_valid,
        last_valid_stamp_us=last_valid_stamp_us,
    )
    if rejection is not None:
        return _reject(rejection, reason)

    # A stop is an invalid hold marker, never a valid zero action.
    if safe_stop:
        return _reject(POLICY_STOP, reason or "policy requested stop")

    try:
        age = float(time_since_last_valid_sec)
    except (TypeError, ValueError):
        return _reject(NONFINITE, "invalid policy age")
    if not math.isfinite(age) or age < 0.0:
        return _reject(NONFINITE, "invalid policy age")
    if age > cfg.estop_timeout_sec:
        return _reject(ESTOP, f"no valid policy for {age:.2f}s")
    if age > cfg.stale_timeout_sec:
        return _reject(STALE_INPUT, f"last valid policy is {age:.2f}s old")

    rejection = _check_kinematics(desired_x, desired_y, cfg)
    if rejection is not None:
        return _reject(rejection)

    checked_entities = tuple(entities or ())
    rejection = _check_entity_finiteness(checked_entities)
    if rejection is not None:
        return _reject(rejection)
    if checked_entities:
        rejection = _check_collision(
            desired_x,
            desired_y,
            checked_entities,
            cfg,
            dt=float(dt),
        )
        if rejection is not None:
            return _reject(rejection)

    return SafetyGateResult(
        desired_x=float(desired_x),
        desired_y=float(desired_y),
        safe_stop=False,
        valid=True,
        reason=PASS,
    )




"""Small multimodal policy with one bounded body-frame action output."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SmallPolicyConfig:
    language_dim: int = 256
    entity_count: int = 16
    entity_geometry_dim: int = 16
    action_dim: int = 2
    ego_state_dim: int = 2
    language_hidden: int = 128
    entity_geometry_hidden: int = 64
    entity_hidden: int = 192
    ego_state_hidden: int = 64
    fusion_hidden: int = 256
    maximum_action_m: float = 0.5
    invalid_stop_logit: float = 20.0
    maximum_trainable_parameters: int = 2_000_000
    language_conditioned_entity_attention: bool = False
    entity_attention_mode: str = "legacy"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SmallPolicyConfig":
        model = value.get("model", value)
        if not isinstance(model, Mapping):
            raise ValueError("model configuration must be a mapping")
        model = dict(model)
        if "horizon" in model:
            raise ValueError(
                "legacy trajectory policy config is unsupported; remove "
                "horizon and retrain with a single [B, 2] action"
            )
        if "maximum_step_m" in model and "maximum_action_m" not in model:
            model["maximum_action_m"] = model.pop("maximum_step_m")
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(model) - known
        if unknown:
            raise ValueError(f"unknown model configuration keys: {sorted(unknown)}")
        return cls(**dict(model))

    def __post_init__(self) -> None:
        positive_ints = (
            self.language_dim,
            self.entity_count,
            self.entity_geometry_dim,
            self.action_dim,
            self.ego_state_dim,
            self.language_hidden,
            self.entity_geometry_hidden,
            self.entity_hidden,
            self.ego_state_hidden,
            self.fusion_hidden,
            self.maximum_trainable_parameters,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("all dimensions and parameter limits must be positive")
        if self.action_dim != 2:
            raise ValueError("single-step body-frame action_dim must be 2")
        if self.ego_state_dim != 2:
            raise ValueError("ego_state_dim must be 2: surge_velocity and yaw_rate")
        if self.maximum_action_m <= 0.0:
            raise ValueError("maximum_action_m must be positive")
        if not torch.isfinite(torch.tensor(self.invalid_stop_logit)):
            raise ValueError("invalid_stop_logit must be finite")
        if self.entity_attention_mode not in {
            "legacy",
            "language_additive",
            "language_only",
        }:
            raise ValueError(
                f"unsupported entity_attention_mode={self.entity_attention_mode!r}"
            )
        if (
            self.entity_attention_mode != "legacy"
            and not self.language_conditioned_entity_attention
        ):
            raise ValueError(
                "language attention mode requires "
                "language_conditioned_entity_attention=true"
            )

    @property
    def maximum_step_m(self) -> float:
        """Compatibility spelling for callers that used the old bound."""
        return self.maximum_action_m


@dataclass(frozen=True)
class PolicyOutput:
    """Policy tensors plus an explicit fail-closed sample-validity mask."""

    action: Tensor
    stop_logit: Tensor
    valid_mask: Tensor


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU(),
    )


class SmallActionPolicy(nn.Module):
    """Fuse language and structured entity geometry into one bounded action.

    The decision head receives task language, structured entity geometry and
    the current vessel dynamics. It does not receive the previous action or
    image/bbox tensors; temporal smoothness is learned from current dynamics.
    """

    def __init__(self, config: SmallPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or SmallPolicyConfig()
        cfg = self.config

        self.language_encoder = _mlp(
            cfg.language_dim, cfg.language_hidden, cfg.language_hidden
        )
        self.entity_geometry_encoder = _mlp(
            cfg.entity_geometry_dim,
            cfg.entity_geometry_hidden,
            cfg.entity_geometry_hidden,
        )
        self.entity_fusion = nn.Sequential(
            nn.Linear(
                cfg.entity_geometry_hidden,
                cfg.entity_hidden,
            ),
            nn.GELU(),
        )
        self.ego_state_encoder = _mlp(
            cfg.ego_state_dim,
            cfg.ego_state_hidden,
            cfg.ego_state_hidden,
        )
        self.entity_attention = nn.Linear(cfg.entity_hidden, 1)
        if cfg.language_conditioned_entity_attention:
            self.entity_language_query: nn.Linear | None = nn.Linear(
                cfg.language_hidden,
                cfg.entity_hidden,
                bias=False,
            )
        else:
            # Keep the earlier state_dict format loadable.
            self.entity_language_query = None
        # The validity bit is part of the decision input so a real zero action
        # is distinguishable from the zero sentinel used at episode start.
        fusion_input_dim = (
            cfg.language_hidden + cfg.entity_hidden + cfg.ego_state_hidden + 1
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.fusion_hidden),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden),
            nn.GELU(),
        )
        self.action_head = nn.Linear(cfg.fusion_hidden, cfg.action_dim)
        self.stop_head = nn.Linear(cfg.fusion_hidden, 1)

        parameter_count = self.trainable_parameter_count()
        if parameter_count > cfg.maximum_trainable_parameters:
            raise ValueError(
                f"policy has {parameter_count} trainable parameters; "
                f"limit is {cfg.maximum_trainable_parameters}"
            )

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def _expect_shape(tensor: Tensor, shape: tuple[int, ...], name: str) -> None:
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match {shape}"
            )

    @staticmethod
    def _as_mask(
        mask: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        name: str,
    ) -> Tensor:
        if mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if tuple(mask.shape) != (batch_size,):
            raise ValueError(
                f"{name} shape {tuple(mask.shape)} does not match "
                f"{(batch_size,)}"
            )
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _sanitize_masked(values: Tensor, mask: Tensor, name: str) -> Tensor:
        expanded = mask
        while expanded.ndim < values.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(values)
        active_values = values[expanded]
        if active_values.numel() and not torch.isfinite(active_values).all():
            raise ValueError(f"{name} contains NaN or Inf in an active position")
        return torch.where(expanded, values, torch.zeros_like(values))

    def forward(
        self,
        *,
        language: Tensor,
        entity_geometry: Tensor,
        ego_state: Tensor,
        language_valid: Tensor | None = None,
        entity_geometry_mask: Tensor | None = None,
        ego_state_valid: Tensor | None = None,
        policy_input_valid: Tensor | None = None,
        **legacy_inputs: Tensor,
    ) -> PolicyOutput:
        if legacy_inputs:
            raise ValueError(
                "decision policy accepts language, entity_geometry, "
                "ego_state, and their validity masks; "
                f"legacy inputs are unsupported: {sorted(legacy_inputs)}"
            )
        cfg = self.config
        if language.ndim != 2:
            raise ValueError("language must have shape [B, language_dim]")
        batch_size = int(language.shape[0])
        device = language.device
        dtype = language.dtype
        if (
            entity_geometry.device != device
            or ego_state.device != device
        ):
            raise ValueError("all policy inputs must be on the same device")
        if entity_geometry.dtype != dtype or ego_state.dtype != dtype:
            raise ValueError("all floating policy inputs must share one dtype")

        self._expect_shape(
            language, (batch_size, cfg.language_dim), "language"
        )
        self._expect_shape(
            entity_geometry,
            (
                batch_size,
                cfg.entity_count,
                cfg.entity_geometry_dim,
            ),
            "entity_geometry",
        )
        self._expect_shape(ego_state, (batch_size, cfg.ego_state_dim), "ego_state")
        language_mask = self._as_mask(
            language_valid,
            batch_size=batch_size,
            device=device,
            name="language_valid",
        )
        input_mask = self._as_mask(
            policy_input_valid,
            batch_size=batch_size,
            device=device,
            name="policy_input_valid",
        )

        if entity_geometry_mask is None:
            entity_geometry_mask = torch.ones(
                batch_size,
                cfg.entity_count,
                dtype=torch.bool,
                device=device,
            )
        self._expect_shape(
            entity_geometry_mask,
            (batch_size, cfg.entity_count),
            "entity_geometry_mask",
        )
        geometry_entity_mask = entity_geometry_mask.to(
            device=device, dtype=torch.bool
        )
        ego_mask = self._as_mask(
            ego_state_valid,
            batch_size=batch_size,
            device=device,
            name="ego_state_valid",
        )

        valid_mask = language_mask & input_mask
        geometry_entity_mask = geometry_entity_mask & valid_mask.unsqueeze(1)

        language_clean = self._sanitize_masked(
            language.detach(), language_mask & valid_mask, "language"
        )
        entity_geometry_clean = self._sanitize_masked(
            entity_geometry, geometry_entity_mask, "entity_geometry"
        )
        ego_state_clean = self._sanitize_masked(
            ego_state.detach(), ego_mask & valid_mask, "ego_state"
        )
        language_token = self.language_encoder(language_clean)
        entity_geometry_token = self.entity_geometry_encoder(
            entity_geometry_clean
        )
        entity_token = self.entity_fusion(
            entity_geometry_token
        )
        ego_state_token = self.ego_state_encoder(ego_state_clean)

        attention_score = self.entity_attention(entity_token).squeeze(-1)
        if self.entity_language_query is not None:
            language_query = self.entity_language_query(language_token)
            language_score = torch.sum(
                entity_token * language_query.unsqueeze(1), dim=-1
            ) / (cfg.entity_hidden**0.5)
            mode = cfg.entity_attention_mode
            # Earlier checkpoints predate the explicit mode field and
            # used the additive form.  Preserve that behavior when its boolean
            # flag is true and the new field retains the legacy default.
            if mode in {"legacy", "language_additive"}:
                attention_score = attention_score + language_score
            else:
                attention_score = language_score
        attention_weight = torch.exp(attention_score.clamp(-20.0, 20.0))
        attention_weight = attention_weight * geometry_entity_mask.to(
            dtype=dtype
        )
        attention_weight = attention_weight / attention_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-12)
        pooled_entity = torch.sum(
            entity_token * attention_weight.unsqueeze(-1), dim=1
        )
        fused = self.fusion(
            torch.cat(
                (
                    language_token,
                    pooled_entity,
                    ego_state_token,
                    ego_mask.to(dtype=dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        stop_logit = self.stop_head(fused)
        raw_action = self.action_head(fused)
        raw_norm = torch.linalg.vector_norm(raw_action, dim=-1, keepdim=True)
        radial_scale = torch.where(
            raw_norm > 1.0e-6,
            torch.tanh(raw_norm) / raw_norm.clamp_min(1.0e-6),
            torch.ones_like(raw_norm),
        )
        movement_gate = torch.sigmoid(-stop_logit)
        action = raw_action * radial_scale * cfg.maximum_action_m * movement_gate
        # A positive STOP decision is an execution contract, not merely a
        # request to move less. Keep the continuous gate above for useful
        # training gradients, then make the published policy output exactly
        # stationary whenever the classifier selects STOP.
        movement_selected = (stop_logit < 0.0).view(batch_size, 1)
        action = torch.where(movement_selected, action, torch.zeros_like(action))

        action = torch.where(
            valid_mask.view(batch_size, 1), action, torch.zeros_like(action)
        )
        stop_logit = torch.where(
            valid_mask.view(batch_size, 1),
            stop_logit,
            torch.full_like(stop_logit, cfg.invalid_stop_logit),
        )
        return PolicyOutput(
            action=action,
            stop_logit=stop_logit,
            valid_mask=valid_mask,
        )


DEFAULT_POLICY_MODEL_PATH = (
    "models/"
    "policy.pt"
)

_FLOAT_INPUT_NAMES = (
    "language",
    "entity_geometry",
    "ego_state",
)
_MASK_INPUT_NAMES = (
    "language_valid",
    "entity_geometry_mask",
    "ego_state_valid",
    "policy_input_valid",
)


class PolicyRuntimeError(RuntimeError):
    """Raised when a policy checkpoint cannot satisfy the CUDA contract."""


@dataclass
class TorchPolicyRunner:
    """A checked, inference-only instance of the action policy."""

    model: SmallActionPolicy
    config: SmallPolicyConfig
    device: torch.device
    model_path: str

    @classmethod
    def load(
        cls, model_path: str | Path, *, device: str = "cuda"
    ) -> "TorchPolicyRunner":
        """Load a training checkpoint onto CUDA without a CPU fallback."""

        requested = str(device).strip().lower()
        if requested != "cuda":
            raise PolicyRuntimeError(
                f"torch policy requires device='cuda', got {device!r}"
            )
        if not torch.cuda.is_available():
            raise PolicyRuntimeError("CUDA is unavailable for the torch policy")

        path = Path(model_path).expanduser()
        if not path.is_file():
            raise PolicyRuntimeError(f"policy checkpoint not found: {path}")

        try:
            try:
                checkpoint = torch.load(
                    path, map_location="cpu", weights_only=True
                )
            except TypeError:
                # Torch versions before ``weights_only`` accepted only the
                # map-location argument.  Validate before copying to CUDA.
                checkpoint = torch.load(path, map_location="cpu")
            if not isinstance(checkpoint, Mapping):
                raise PolicyRuntimeError("policy checkpoint must be a mapping")
            raw_config = checkpoint.get("model_config")
            config = (
                SmallPolicyConfig()
                if raw_config is None
                else SmallPolicyConfig.from_mapping(raw_config)
            )
            state_dict = checkpoint.get("model_state_dict")
            if not isinstance(state_dict, Mapping):
                raise PolicyRuntimeError(
                    "policy checkpoint is missing model_state_dict"
                )
            model = SmallActionPolicy(config)
            model.load_state_dict(state_dict, strict=True)
            target = torch.device("cuda")
            model.to(device=target)
            model.eval()
        except PolicyRuntimeError:
            raise
        except Exception as exc:
            raise PolicyRuntimeError(
                f"failed to load torch policy checkpoint {path}: {exc}"
            ) from exc

        try:
            parameter_device = next(model.parameters()).device
        except StopIteration as exc:
            raise PolicyRuntimeError("torch policy has no parameters") from exc
        if parameter_device.type != "cuda":
            raise PolicyRuntimeError(
                f"torch policy loaded on {parameter_device}, expected CUDA"
            )
        return cls(
            model=model,
            config=config,
            device=target,
            model_path=str(path),
        )

    def run(
        self, inputs: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one or more policy samples and return NumPy outputs."""

        missing = [
            name for name in (*_FLOAT_INPUT_NAMES, *_MASK_INPUT_NAMES)
            if name not in inputs
        ]
        if missing:
            raise PolicyRuntimeError(
                f"policy inputs missing required fields: {', '.join(missing)}"
            )

        try:
            tensors: dict[str, torch.Tensor] = {}
            for name in _FLOAT_INPUT_NAMES:
                tensors[name] = torch.as_tensor(
                    np.asarray(inputs[name]),
                    dtype=torch.float32,
                    device=self.device,
                )
            for name in _MASK_INPUT_NAMES:
                tensors[name] = torch.as_tensor(
                    np.asarray(inputs[name]),
                    dtype=torch.bool,
                    device=self.device,
                )
            with torch.inference_mode():
                output: PolicyOutput = self.model(**tensors)
            action = output.action.detach().to("cpu").numpy()
            stop_logit = output.stop_logit.detach().to("cpu").numpy()
            valid_mask = output.valid_mask.detach().to("cpu").numpy()
        except Exception as exc:
            raise PolicyRuntimeError(f"torch policy inference failed: {exc}") from exc

        if (
            action.ndim != 2
            or action.shape[1:] != (self.config.action_dim,)
            or stop_logit.ndim != 2
            or stop_logit.shape[1:] != (1,)
            or valid_mask.ndim != 1
            or valid_mask.shape[0] != action.shape[0]
        ):
            raise PolicyRuntimeError(
                "torch policy returned an invalid output shape: "
                f"action={action.shape}, stop_logit={stop_logit.shape}, "
                f"valid_mask={valid_mask.shape}"
            )
        if not (
            np.all(np.isfinite(action))
            and np.all(np.isfinite(stop_logit))
        ):
            raise PolicyRuntimeError("torch policy returned NaN or Inf")
        return action, stop_logit, valid_mask.astype(bool, copy=False)


__all__ = [
    "DEFAULT_POLICY_MODEL_PATH",
    "PolicyOutput",
    "PolicyRuntimeError",
    "SmallActionPolicy",
    "SmallPolicyConfig",
    "TorchPolicyRunner",
]



DEFAULT_POLICY_BACKEND = "torch_cuda"
POLICY_MODEL_VERSION = "vla_torch_cuda_ego_dynamics"
ENTITY_COUNT = 16
LANGUAGE_DIM = 256
ENTITY_GEOMETRY_DIM = 16
EGO_STATE_DIM = 2
ENTITY_FEATURE_BACKEND = "deterministic_entity_tensor"
POSITION_SCALE_M = 20.0
HEIGHT_SCALE_M = 5.0
VELOCITY_SCALE_MPS = 5.0
RISK_HORIZON_SEC = 4.0
RISK_RADIUS_M = 3.0
STALE_SEC = 1.0
MIN_INFERENCE_INTERVAL_SEC = DT_SEC
SYNC_CACHE_SIZE = 256
SYNC_CACHE_TTL_SEC = 5.0
SYNC_FAIL_PUBLISH_PERIOD_SEC = 1.0
POLICY_MAX_STEP_M = MAX_DISPLACEMENT_M
POLICY_MAX_ACTION_DELTA_M = 0.05
POLICY_TRACE_LIMIT = 5
POLICY_AUDIT_PERIOD = 100
_POLICY_LANGUAGE_TABLE: dict[str, np.ndarray] | None = None
_POLICY_LANGUAGE_TABLE_PATHS = (
    Path("/home/jetson/jetson_asv_ws/models/qwen_final_embeddings.npz"),
    Path("D:/asv-vla-training/data/qwen_final_embeddings.npz"),
)


def policy_task_key(text: str) -> str:
    folded = str(text).casefold()
    color = "blue" if "blue" in folded or "蓝" in text else "red"
    distance = "4m" if "4m" in folded or "4 m" in folded or "4米" in text else "3m"
    return f"{color}_{distance}"


def stamp_language_standoff(embedding: np.ndarray, standoff_m: float) -> np.ndarray:
    stamped = np.asarray(embedding, dtype=np.float32).copy()
    if stamped.ndim != 1 or stamped.size < 1:
        raise ValueError("language embedding must be a 1-D vector")
    stamped[-1] = (float(standoff_m) - 3.5) / 0.5
    return stamped


def load_policy_language_table(
    path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    global _POLICY_LANGUAGE_TABLE
    if path is None and _POLICY_LANGUAGE_TABLE is not None:
        return _POLICY_LANGUAGE_TABLE
    candidates = (Path(path),) if path is not None else _POLICY_LANGUAGE_TABLE_PATHS
    for candidate in candidates:
        if not candidate.is_file():
            continue
        data = np.load(candidate, allow_pickle=True)
        ids = [str(value) for value in data["instruction_ids"]]
        values = np.asarray(data["embeddings"], dtype=np.float32)
        table = {key: values[index] for index, key in enumerate(ids)}
        if path is None:
            _POLICY_LANGUAGE_TABLE = table
        return table
    raise FileNotFoundError("policy language embedding table is missing")


def condition_policy_language(embedding: np.ndarray, instruction: str) -> np.ndarray:
    """Map live instructions onto the training-table embedding and stamp standoff."""

    table = load_policy_language_table()
    key = policy_task_key(instruction)
    if key not in table:
        raise KeyError(f"policy language table missing task {key}")
    standoff_m = desired_standoff_from_instruction(instruction)
    return stamp_language_standoff(table[key], standoff_m)

LANG_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

FrameKey = tuple[str, int, int]
IDENTITY_FIELDS = ("run_id", "scene_seed", "frame_index")


@dataclass(frozen=True)
class _PendingAction:
    stamp_us: int
    action: tuple[float, float]


@dataclass(frozen=True)
class _SyncEntry:
    message: Any
    received_at: float


class EntityFeaturesError(RuntimeError):
    pass


@dataclass(frozen=True)
class EntityMetrics:
    distance_m: float
    bearing_sin: float
    bearing_cos: float
    closing_speed_mps: float
    time_to_cpa_sec: float
    cpa_distance_m: float
    is_risk: bool


@dataclass(frozen=True)
class EntityFeaturesResult:
    features: np.ndarray
    mask: np.ndarray
    entity_ids: tuple[str, ...]
    entity_count: int
    target_count: int
    risk_count: int
    dropped_count: int


@dataclass
class DecisionEntityFrame:
    """Internal structured entity tensor; never exposed as a ROS message."""

    stamp_us: int
    run_id: str
    scene_seed: int
    frame_index: int
    frame_id: str
    max_entities: int
    feature_dim: int
    entity_count: int
    entity_ids: list[str]
    features: list[float]
    mask: list[bool]
    valid: bool
    instruction_id: str
    instruction: str
    detail: str
    safety_entities: tuple[_Entity, ...] = ()


def _clip_feature(value: float, scale: float, low: float = -1.0) -> float:
    return float(np.clip(value / scale, low, 1.0))


def compute_entity_metrics(entity: Any) -> EntityMetrics:
    x, y = float(entity.relative_x), float(entity.relative_y)
    vx, vy = float(entity.relative_velocity_x), float(entity.relative_velocity_y)
    distance = math.hypot(x, y)
    if distance > 1.0e-9:
        bearing_sin, bearing_cos = y / distance, x / distance
        closing_speed = -(x * vx + y * vy) / distance
    else:
        bearing_sin, bearing_cos, closing_speed = 0.0, 1.0, 0.0
    speed_squared = vx * vx + vy * vy
    raw_time_to_cpa = -(x * vx + y * vy) / speed_squared if speed_squared > 1.0e-12 else math.inf
    time_to_cpa = min(max(raw_time_to_cpa, 0.0), RISK_HORIZON_SEC)
    cpa_distance = math.hypot(x + vx * time_to_cpa, y + vy * time_to_cpa)
    return EntityMetrics(
        distance, bearing_sin, bearing_cos, closing_speed, time_to_cpa,
        cpa_distance,
        closing_speed > 0.0 and 0.0 < raw_time_to_cpa <= RISK_HORIZON_SEC
        and cpa_distance <= RISK_RADIUS_M,
    )


def _entity_candidate(entity: Any) -> tuple[Any, str, EntityMetrics]:
    entity_id = str(entity.entity_id).strip()
    values = (
        entity.relative_x, entity.relative_y, entity.relative_z,
        entity.relative_velocity_x, entity.relative_velocity_y,
        entity.relative_velocity_z,
    )
    if not entity_id:
        raise EntityFeaturesError("a valid visible entity has an empty entity_id")
    if not all(math.isfinite(float(value)) for value in values):
        raise EntityFeaturesError(f"entity {entity_id!r} contains NaN or Inf")
    return entity, entity_id, compute_entity_metrics(entity)


def _entity_sort_key(candidate: tuple[Any, str, EntityMetrics]) -> tuple[Any, ...]:
    entity, entity_id, metrics = candidate
    if bool(entity.is_target):
        return 0, metrics.distance_m, entity_id
    if metrics.is_risk:
        return 1, metrics.cpa_distance_m, metrics.time_to_cpa_sec, metrics.distance_m, entity_id
    return 2, metrics.distance_m, entity_id


def _entity_row(candidate: tuple[Any, str, EntityMetrics]) -> np.ndarray:
    entity, _, metrics = candidate
    color = str(entity.color).strip().casefold()
    return np.asarray((
        _clip_feature(float(entity.relative_x), POSITION_SCALE_M),
        _clip_feature(float(entity.relative_y), POSITION_SCALE_M),
        _clip_feature(float(entity.relative_z), HEIGHT_SCALE_M),
        0.0,
        0.0,
        0.0,
        _clip_feature(metrics.distance_m, POSITION_SCALE_M, 0.0),
        metrics.bearing_sin, metrics.bearing_cos,
        0.0,
        0.0,
        0.0,
        float(bool(entity.is_target)), float(metrics.is_risk),
        float(color in {"red", "红", "红色"}),
        float(color in {"blue", "蓝", "蓝色"}),
    ), dtype=np.float32)


def build_entity_features(entities: Iterable[Any]) -> EntityFeaturesResult:
    candidates, seen_ids = [], set()
    for entity in entities:
        if not bool(entity.valid) or not bool(entity.visible):
            continue
        candidate = _entity_candidate(entity)
        if candidate[1] in seen_ids:
            raise EntityFeaturesError(f"duplicate valid visible entity_id {candidate[1]!r}")
        seen_ids.add(candidate[1])
        candidates.append(candidate)
    selected = sorted(candidates, key=_entity_sort_key)[:ENTITY_COUNT]
    features = np.zeros((ENTITY_COUNT, ENTITY_GEOMETRY_DIM), dtype=np.float32)
    mask = np.zeros(ENTITY_COUNT, dtype=np.bool_)
    entity_ids = [""] * ENTITY_COUNT
    for index, candidate in enumerate(selected):
        features[index], mask[index], entity_ids[index] = _entity_row(candidate), True, candidate[1]
    if not np.all(np.isfinite(features)):
        raise EntityFeaturesError("entity features contain NaN or Inf")
    return EntityFeaturesResult(
        features, mask, tuple(entity_ids), len(selected),
        sum(bool(entity.is_target) for entity, _, _ in selected),
        sum(metrics.is_risk for _, _, metrics in selected),
        max(0, len(candidates) - len(selected)),
    )


def _identity_tuple(message: Any) -> tuple[str, int, int] | None:
    """Return complete frame identity, or None when it cannot be trusted."""

    try:
        run_id = str(getattr(message, "run_id")).strip()
        scene_seed = int(getattr(message, "scene_seed"))
        frame_index = int(getattr(message, "frame_index"))
    except (AttributeError, TypeError, ValueError):
        return None
    if not run_id or scene_seed <= 0 or frame_index < 0:
        return None
    return run_id, scene_seed, frame_index


def identity_mismatch_reason(
    language: Any,
    entities: Any,
) -> str | None:
    """Validate task identity without treating language as a camera frame.

    ``TaskEmbedding`` is task-level: its ``run_id`` identifies the encoder
    provenance and its ``stamp_us`` is publication time.  Neither is compared
    with the camera-frame identity.  The task text, and ``instruction_id``
    when both message interfaces provide it, are the synchronization keys.
    """

    language_instruction = str(getattr(language, "instruction", "")).strip()
    entity_instruction = str(getattr(entities, "instruction", "")).strip()
    if not language_instruction or not entity_instruction:
        return "IDENTITY_MISMATCH"
    if language_instruction != entity_instruction:
        return "IDENTITY_MISMATCH"
    language_instruction_id = str(
        getattr(language, "instruction_id", "")
    ).strip()
    entity_instruction_id = str(getattr(entities, "instruction_id", "")).strip()
    if (
        language_instruction_id
        and entity_instruction_id
        and language_instruction_id != entity_instruction_id
    ):
        return "IDENTITY_MISMATCH"
    return None


def entity_features_identity_reason(
    message: Any,
    previous_identity: FrameKey | None = None,
) -> str | None:
    """Validate EntityFeatures identity and monotonic same-run frame order."""

    identity = _identity_tuple(message)
    if identity is None:
        return "IDENTITY_MISMATCH"
    try:
        stamp_us = int(getattr(message, "stamp_us"))
    except (AttributeError, TypeError, ValueError):
        return "IDENTITY_MISMATCH"
    if stamp_us <= 0:
        return "IDENTITY_MISMATCH"
    if previous_identity is not None:
        if (
            identity[:2] == previous_identity[:2]
            and identity[2] <= previous_identity[2]
        ):
            return "IDENTITY_MISMATCH"
    return None


def bound_policy_displacement(
    displacement: Sequence[float] | np.ndarray,
    *,
    safe_stop: bool = False,
    valid: bool = True,
    max_step_m: float = POLICY_MAX_STEP_M,
) -> tuple[float, float] | None:
    """Validate and norm-bound one direct ``[desired_x, desired_y]`` action."""

    if safe_stop or not valid:
        return None
    try:
        maximum = float(max_step_m)
        values = np.asarray(displacement, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(maximum) or maximum < 0.0:
        return None
    if values.size != ACTION_DIM or not np.all(np.isfinite(values)):
        return None
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm):
        return None
    if norm > maximum and norm > 0.0:
        values = values * (maximum / norm)
    return float(values[0]), float(values[1])


def smooth_policy_displacement(
    displacement: Sequence[float] | np.ndarray,
    *,
    previous_action: Sequence[float] | np.ndarray | None = None,
    max_step_m: float = POLICY_MAX_STEP_M,
    max_delta_m: float = POLICY_MAX_ACTION_DELTA_M,
) -> tuple[float, float] | None:
    """Apply a bounded per-frame action change around the previous command.

    Delta smoothing is only meaningful when a previous command is available.
    Without one, return the bounded policy action directly.
    """

    current = bound_policy_displacement(displacement, max_step_m=max_step_m)
    if current is None:
        return None
    try:
        maximum_delta = float(max_delta_m)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(maximum_delta) or maximum_delta < 0.0:
        return None
    if previous_action is None:
        return current
    else:
        try:
            previous = np.asarray(previous_action, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if previous.size != ACTION_DIM or not np.all(np.isfinite(previous)):
            return None
    current_array = np.asarray(current, dtype=np.float64)
    delta = current_array - previous
    delta_norm = float(np.linalg.norm(delta))
    if not math.isfinite(delta_norm):
        return None
    if delta_norm > maximum_delta and delta_norm > 0.0:
        current_array = previous + delta * (maximum_delta / delta_norm)
    return bound_policy_displacement(current_array, max_step_m=max_step_m)


class FrameSyncCache:
    """Bounded, scene-isolated cache for structured entity frames."""

    def __init__(
        self,
        *,
        cache_size: int = SYNC_CACHE_SIZE,
        ttl_sec: float = SYNC_CACHE_TTL_SEC,
    ) -> None:
        if int(cache_size) <= 0:
            raise ValueError("cache_size must be positive")
        if not math.isfinite(float(ttl_sec)) or float(ttl_sec) <= 0.0:
            raise ValueError("ttl_sec must be finite and positive")
        self.cache_size = int(cache_size)
        self.ttl_sec = float(ttl_sec)
        self._entities: OrderedDict[FrameKey, _SyncEntry] = OrderedDict()
        self._active_run: tuple[str, int] | None = None

    @staticmethod
    def key_for(message: Any) -> FrameKey:
        return (
            str(message.run_id),
            int(message.scene_seed),
            int(message.frame_index),
        )

    @property
    def active_run(self) -> tuple[str, int] | None:
        return self._active_run

    @property
    def entity_size(self) -> int:
        return len(self._entities)

    def keys(self) -> tuple[FrameKey, ...]:
        return tuple(self._entities.keys())

    def clear(self) -> None:
        self._entities.clear()

    def put_entities(
        self, message: Any, received_at: float | None = None
    ) -> tuple[FrameKey, bool]:
        key = self.key_for(message)
        run = (key[0], key[1])
        switched = self._active_run is not None and run != self._active_run
        if switched:
            self.clear()
        self._active_run = run
        self._entities[key] = _SyncEntry(
            message=message,
            received_at=time.monotonic() if received_at is None else float(received_at),
        )
        self._entities.move_to_end(key)
        while len(self._entities) > self.cache_size:
            self._entities.popitem(last=False)
        return key, switched

    def entity_for(self, key: FrameKey) -> Any | None:
        entry = self._entities.get(key)
        return entry.message if entry is not None else None

    def expire(self, now: float | None = None) -> int:
        current = time.monotonic() if now is None else float(now)
        removed = 0
        for key, entry in tuple(self._entities.items()):
            if current - entry.received_at > self.ttl_sec:
                del self._entities[key]
                removed += 1
        return removed

    def consume(self, key: FrameKey) -> None:
        self._entities.pop(key, None)


class DecisionNode(Node):
    """Run the language-conditioned decision head at the camera cadence."""

    def __init__(self, model_path: str = "") -> None:
        super().__init__("decision")
        self.declare_parameter("model_path", model_path)
        self.declare_parameter("device", "cuda")
        self._language_released = bool(
            self.declare_parameter("language_release_after_encode", False).value
        )
        self._backend = DEFAULT_POLICY_BACKEND
        policy_device = str(self.get_parameter("device").value).strip() or "cuda"
        self._frame_sync = FrameSyncCache(
            cache_size=int(self.declare_parameter("sync_cache_size", SYNC_CACHE_SIZE).value),
            ttl_sec=float(self.declare_parameter("sync_cache_ttl_sec", SYNC_CACHE_TTL_SEC).value),
        )
        self._language: TaskEmbedding | None = None
        self._language_stamp = 0.0
        self._language_task_key: str | None = None
        self._entities: DecisionEntityFrame | None = None
        self._last_out_stamp_us = 0
        self._last_inference_time = 0.0
        self._last_sync_fail_time = 0.0
        self._frame_seq = 0
        self._policy_trace_count = 0
        self._policy_audit_events = 0
        self._policy_driven_count = 0
        self._backstop_count = 0
        self._hold_count = 0
        self._fail_closed_count = 0
        self._policy_stop_count = 0
        self._policy_audit_shutdown_logged = False
        self._last_audit_guard_reason = "none"
        self._last_audit_raw_dx, self._last_audit_raw_dy = "nan", "nan"
        self._last_audit_guarded_dx, self._last_audit_guarded_dy = "nan", "nan"
        self._last_audit_final_dx, self._last_audit_final_dy = "nan", "nan"
        self._inference_count = 0
        self._active_run: tuple[str, int] | None = None
        self._retired_runs: set[tuple[str, int]] = set()
        self._last_entity_identity: FrameKey | None = None
        self._last_entity_frame_index = -1
        self._last_inferred_frame_index = -1
        self._last_gate_frame_index = -1
        self._ego_state: ASVState | None = None
        self._safety_config = SafetyGateConfig(
            max_step_m=float(
                self.declare_parameter(
                    "safety_max_step_m", MAX_DISPLACEMENT_M
                ).value
            ),
            stale_timeout_sec=float(
                self.declare_parameter("safety_stale_timeout_sec", 1.0).value
            ),
            estop_timeout_sec=float(
                self.declare_parameter("safety_estop_timeout_sec", 2.0).value
            ),
            collision_margin_m=float(
                self.declare_parameter("safety_collision_margin_m", 0.5).value
            ),
        )

        self._lang_sub = self.create_subscription(
            TaskEmbedding, "/vla/language_embedding", self._on_language, LANG_QOS
        )
        self._ent_sub = self.create_subscription(
            EntityArray, "/vla/entities", self._on_entities, 10
        )
        self._ego_sub = self.create_subscription(
            ASVState, "/ue/asv_state", self._on_ego_state, 10
        )
        self._pub = self.create_publisher(
            DesiredDisplacement, "/control/desired_displacement", 10
        )
        self.create_timer(1.0, self._expire_cache)

        self._torch_runner = None
        self._policy_load_error = ""
        self._model_version = POLICY_MODEL_VERSION
        try:
            requested_path = str(self.get_parameter("model_path").value) or model_path
            self._torch_runner = TorchPolicyRunner.load(
                requested_path or DEFAULT_POLICY_MODEL_PATH,
                device=policy_device,
            )
            self.get_logger().info(
                f"POLICY_READY backend={self._backend} device={policy_device} "
                f"model={requested_path or DEFAULT_POLICY_MODEL_PATH} "
                "inputs=task_embedding+structured_entities+ego_dynamics "
                "output=[desired_x,desired_y]"
            )
        except Exception as exc:
            self._policy_load_error = f"MODEL_LOAD_ERROR:{exc}"
            self.get_logger().error(
                f"POLICY_LOAD_ERROR backend={self._backend} error={self._policy_load_error}"
            )

        self._smooth_max_step_m = float(
            self.declare_parameter(
                "smoothing_max_step_m", POLICY_MAX_STEP_M
            ).value
        )
        self._smooth_max_delta_m = float(
            self.declare_parameter(
                "smoothing_max_delta_m", POLICY_MAX_ACTION_DELTA_M
            ).value
        )

    def _clear_control_history(self) -> None:
        self._last_gate_frame_index = -1

    def _on_ego_state(self, message: ASVState) -> None:
        """Subscribe to current vessel dynamics for the decision input."""

        self._ego_state = message

    def _expire_cache(self) -> None:
        self._frame_sync.expire()

    @staticmethod
    def _audit_action(action: Sequence[float] | np.ndarray | None) -> tuple[str, str]:
        if action is None:
            return "nan", "nan"
        try:
            values = np.asarray(action, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return "nan", "nan"
        if values.size != ACTION_DIM or not np.all(np.isfinite(values)):
            return "nan", "nan"
        return f"{float(values[0]):.6f}", f"{float(values[1]):.6f}"

    def _maybe_policy_audit(self, *, force: bool = False, trigger: str = "periodic") -> None:
        events = int(self._policy_audit_events)
        if not force and (events == 0 or events % POLICY_AUDIT_PERIOD != 0):
            return
        self.get_logger().info(
            "POLICY_AUDIT "
            f"trigger={trigger} events={events} "
            f"policy_driven={int(self._policy_driven_count)} "
            f"backstop={int(self._backstop_count)} "
            f"hold={int(self._hold_count)} "
            f"fail_closed={int(self._fail_closed_count)} "
            f"policy_stop={int(self._policy_stop_count)} "
            f"guard_reason={self._last_audit_guard_reason} "
            f"raw_dx={self._last_audit_raw_dx} raw_dy={self._last_audit_raw_dy} "
            f"guarded_dx={self._last_audit_guarded_dx} "
            f"guarded_dy={self._last_audit_guarded_dy} "
            f"final_dx={self._last_audit_final_dx} "
            f"final_dy={self._last_audit_final_dy}"
        )

    def _record_guard_outcome(
        self,
        guard_reason: str,
        *,
        raw_action: Sequence[float] | np.ndarray | None = None,
        guarded_action: Sequence[float] | np.ndarray | None = None,
        final_action: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if guard_reason == GUARD_POLICY_DRIVEN:
            self._policy_driven_count += 1
        elif guard_reason == GUARD_BACKSTOP:
            self._backstop_count += 1
        elif guard_reason == GUARD_HOLD:
            self._hold_count += 1
        self._last_audit_guard_reason = str(guard_reason)
        self._last_audit_raw_dx, self._last_audit_raw_dy = self._audit_action(
            raw_action
        )
        self._last_audit_guarded_dx, self._last_audit_guarded_dy = self._audit_action(
            guarded_action
        )
        self._last_audit_final_dx, self._last_audit_final_dy = self._audit_action(
            final_action
        )
        self._policy_audit_events += 1
        self._maybe_policy_audit()

    def _record_fail_closed(
        self,
        *,
        raw_action: Sequence[float] | np.ndarray | None = None,
        reason: str = "",
    ) -> None:
        self._fail_closed_count += 1
        if str(reason) == "POLICY_STOP":
            self._policy_stop_count += 1
        self._last_audit_guard_reason = GUARD_FAIL_CLOSED
        self._last_audit_raw_dx, self._last_audit_raw_dy = self._audit_action(
            raw_action
        )
        self._last_audit_guarded_dx, self._last_audit_guarded_dy = "nan", "nan"
        self._last_audit_final_dx, self._last_audit_final_dy = "0.000000", "0.000000"
        self._policy_audit_events += 1
        self._maybe_policy_audit()

    def _trace_policy_decision(
        self,
        ent: DecisionEntityFrame,
        *,
        policy_valid: bool,
        stop: bool,
        lang_valid: bool,
        ent_valid: bool,
        guard_result: str,
        guard_reason: str,
        raw_action: Sequence[float] | np.ndarray | None = None,
        guarded_action: Sequence[float] | np.ndarray | None = None,
        final_action: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if self._policy_trace_count >= POLICY_TRACE_LIMIT:
            return
        self._policy_trace_count += 1
        raw_dx, raw_dy = self._audit_action(raw_action)
        guarded_dx, guarded_dy = self._audit_action(guarded_action)
        final_dx, final_dy = self._audit_action(final_action)
        self.get_logger().info(
            "POLICY_TRACE "
            f"sample={self._policy_trace_count}/{POLICY_TRACE_LIMIT} "
            f"run_id={ent.run_id} scene_seed={int(ent.scene_seed)} "
            f"frame_index={int(ent.frame_index)} policy_valid={bool(policy_valid)} "
            f"stop={bool(stop)} lang_valid={bool(lang_valid)} "
            f"entity_valid={bool(ent_valid)} guard_result={guard_result} "
            f"guard_reason={guard_reason} "
            f"raw_dx={raw_dx} raw_dy={raw_dy} "
            f"guarded_dx={guarded_dx} guarded_dy={guarded_dy} "
            f"final_dx={final_dx} final_dy={final_dy} "
            f"policy_driven={int(self._policy_driven_count)} "
            f"backstop={int(self._backstop_count)} "
            f"hold={int(self._hold_count)} "
            f"fail_closed={int(self._fail_closed_count)} "
            f"policy_stop={int(self._policy_stop_count)}"
        )

    def _on_language(self, message: TaskEmbedding) -> None:
        task_key = str(getattr(message, "instruction", "")).strip()
        if self._language_task_key != task_key:
            # A new instruction invalidates both queued frames and gate
            # history. The next matching EntityFeatures frame starts cold.
            self._frame_sync.clear()
            self._last_entity_identity = None
            self._last_entity_frame_index = -1
            self._last_inferred_frame_index = -1
            self._clear_control_history()
        self._language_task_key = task_key
        self._language = message
        self._language_stamp = time.monotonic()
        for key in self._frame_sync.keys():
            self._maybe_infer(key, trigger="language")

    def _new_entity_features(self, source: EntityArray) -> DecisionEntityFrame:
        safety_entities = tuple(
            _Entity(
                entity_id=str(entity.entity_id),
                relative_x=float(entity.relative_x),
                relative_y=float(entity.relative_y),
                relative_vx=float(entity.relative_velocity_x),
                relative_vy=float(entity.relative_velocity_y),
            )
            for entity in source.entities
            if bool(entity.valid) and bool(entity.visible)
        )
        return DecisionEntityFrame(
            stamp_us=int(source.stamp_us),
            run_id=str(source.run_id),
            scene_seed=int(source.scene_seed),
            frame_index=int(source.frame_index),
            frame_id=str(source.frame_id),
            max_entities=ENTITY_COUNT,
            feature_dim=ENTITY_GEOMETRY_DIM,
            entity_count=0,
            entity_ids=[""] * ENTITY_COUNT,
            features=[0.0] * (ENTITY_COUNT * ENTITY_GEOMETRY_DIM),
            mask=[False] * ENTITY_COUNT,
            valid=False,
            instruction_id=str(source.instruction_id),
            instruction=str(source.instruction),
            detail="UNINITIALIZED",
            safety_entities=safety_entities,
        )

    def _on_entities(self, source: EntityArray) -> None:
        message = self._new_entity_features(source)
        if not source.valid:
            message.detail = f"INVALID_SOURCE:{source.detail}"
        elif not message.run_id:
            message.detail = "INVALID_RUN_ID: run_id is empty"
        elif message.frame_id != FRAME_ID:
            message.detail = f"INVALID_FRAME: expected {FRAME_ID}, got {message.frame_id!r}"
        else:
            try:
                result = build_entity_features(source.entities)
                message.entity_count = result.entity_count
                message.entity_ids = list(result.entity_ids)
                message.features = result.features.reshape(-1).tolist()
                message.mask = result.mask.tolist()
                message.valid = True
                message.detail = (
                    f"OK:selected={result.entity_count};targets={result.target_count};"
                    f"risks={result.risk_count};dropped={result.dropped_count}"
                )
            except (EntityFeaturesError, ValueError) as exc:
                message.detail = f"{type(exc).__name__.upper()}:{exc}"
            except Exception as exc:
                message.detail = f"UNEXPECTED_ENTITY_TENSOR_ERROR:{type(exc).__name__}:{exc}"
        self._on_feature_message(message)

    def _on_feature_message(self, message: DecisionEntityFrame) -> None:
        self._entities = message
        now = time.monotonic()
        identity = _identity_tuple(message)
        continuity_reason = entity_features_identity_reason(
            message, self._last_entity_identity
        )
        if continuity_reason is not None:
            self._frame_sync.clear()
            self._clear_control_history()
            self._last_entity_identity = identity
            self._last_entity_frame_index = (
                identity[2] if identity is not None else -1
            )
            self._last_inferred_frame_index = -1
            self._publish_fail_closed(message, continuity_reason)
            return
        if identity is None:
            self._publish_fail_closed(message, "IDENTITY_MISMATCH")
            return

        run = identity[:2]
        if self._active_run is not None and run != self._active_run:
            if run in self._retired_runs:
                self._frame_sync.clear()
                self._clear_control_history()
                self._publish_fail_closed(message, "IDENTITY_MISMATCH")
                return
            self._retired_runs.add(self._active_run)

        if (
            self._language is not None
            and identity_mismatch_reason(self._language, message) is not None
        ):
            self._frame_sync.clear()
            self._clear_control_history()
            self._last_entity_identity = identity
            self._last_entity_frame_index = identity[2]
            self._last_inferred_frame_index = -1
            self._publish_fail_closed(message, "IDENTITY_MISMATCH")
            return

        key, switched = self._frame_sync.put_entities(message, received_at=now)
        if switched or self._active_run != run:
            self._active_run = run
            self._last_entity_frame_index = -1
            self._last_inferred_frame_index = -1
            self._last_entity_identity = None
            self._clear_control_history()
        elif (
            self._last_entity_frame_index >= 0
            and identity[2] <= self._last_entity_frame_index
        ):
            self._clear_control_history()
        self._last_entity_identity = identity
        self._last_entity_frame_index = identity[2]
        self._inference_count += 1
        if self._inference_count <= 5 or self._inference_count % 100 == 0:
            self.get_logger().info(
                f"ENT_IN_TRACE count={self._inference_count} "
                f"frame_index={int(message.frame_index)} valid={bool(message.valid)} "
                f"detail={str(message.detail)[:80]}"
            )
        self._maybe_infer(key, trigger="entities", now=now)

    def _new_output(self, ent: DecisionEntityFrame) -> DesiredDisplacement:
        message = DesiredDisplacement()
        stamp = int(ent.stamp_us)
        self._last_out_stamp_us = max(stamp, self._last_out_stamp_us + 1)
        message.stamp_us = self._last_out_stamp_us
        message.run_id = str(ent.run_id)
        message.scene_seed = int(ent.scene_seed)
        message.frame_index = int(ent.frame_index)
        message.frame_id = FRAME_ID
        message.source = self._model_version
        message.step_dt = DT_SEC
        return message

    def _publish_fail_closed(
        self,
        ent: DecisionEntityFrame,
        reason: str,
        *,
        raw_action: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self._last_sync_fail_time = time.monotonic()
        self._record_fail_closed(raw_action=raw_action, reason=reason)
        self._clear_control_history()
        message = self._new_output(ent)
        message.desired_x = 0.0
        message.desired_y = 0.0
        message.safe_stop = True
        message.valid = False
        message.reason = str(reason)
        self._pub.publish(message)
        self._frame_seq += 1

    def _maybe_infer(
        self,
        key: FrameKey,
        *,
        trigger: str,
        now: float | None = None,
    ) -> None:
        current = time.monotonic() if now is None else float(now)
        if current - self._last_inference_time < MIN_INFERENCE_INTERVAL_SEC:
            return
        ent = self._frame_sync.entity_for(key)
        if ent is None:
            return
        if self._language is None:
            return
        if (
            current - self._language_stamp > STALE_SEC
            and not self._language_released
        ):
            return
        identity_reason = identity_mismatch_reason(self._language, ent)
        if identity_reason is not None:
            self._frame_sync.consume(key)
            self._clear_control_history()
            self._last_inference_time = current
            self._last_inferred_frame_index = int(ent.frame_index)
            self._publish_fail_closed(ent, identity_reason)
            return
        self._frame_sync.consume(key)
        self._last_inference_time = current
        message = self._new_output(ent)
        self._last_inferred_frame_index = int(ent.frame_index)

        ego = self._ego_state
        if ego is None or _identity_tuple(ego) != key or not bool(ego.valid):
            self._publish_fail_closed(ent, "MISSING_OR_MISMATCHED_EGO_STATE")
            return

        if self._torch_runner is None:
            self._publish_fail_closed(ent, self._policy_load_error or "NO_MODEL_LOADED")
            return

        try:
            inputs = self._build_inputs(
                self._language,
                ent,
                ego_state=ego,
            )
            action, stop_logit, valid_mask = self._torch_runner.run(inputs)
            action = np.asarray(action, dtype=np.float32)
            stop_logit = np.asarray(stop_logit, dtype=np.float32)
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if (
                action.shape != (1, ACTION_DIM)
                or stop_logit.shape != (1, 1)
                or valid_mask.shape != (1,)
                or not np.all(np.isfinite(action))
                or not np.all(np.isfinite(stop_logit))
            ):
                raise ValueError(
                    f"invalid direct policy shapes action={action.shape} "
                    f"stop_logit={stop_logit.shape} valid_mask={valid_mask.shape}"
                )
        except Exception as exc:
            self._publish_fail_closed(ent, f"INFERENCE_ERROR:{exc}")
            return

        policy_valid = bool(valid_mask[0]) and bool(self._language.valid) and bool(ent.valid)
        stop = float(stop_logit[0, 0]) >= 0.0
        if stop or not policy_valid:
            self._publish_fail_closed(ent, "POLICY_STOP" if stop else "POLICY_INVALID")
            self._trace_policy_decision(
                ent,
                policy_valid=policy_valid,
                stop=stop,
                lang_valid=bool(self._language.valid),
                ent_valid=bool(ent.valid),
                guard_result="skipped",
                guard_reason="POLICY_STOP" if stop else "POLICY_INVALID",
                raw_action=action[0],
                final_action=(0.0, 0.0),
            )
            return

        displacement = bound_policy_displacement(
            action[0], valid=True, max_step_m=self._smooth_max_step_m
        )
        if displacement is None:
            self._publish_fail_closed(ent, "POLICY_ACTION_INVALID")
            return
        guarded, guard_reason = apply_standoff_guard(displacement, ent)
        if guarded is None:
            self._publish_fail_closed(
                ent, "VISUAL_TARGET_MISSING", raw_action=action[0]
            )
            self._trace_policy_decision(
                ent,
                policy_valid=policy_valid,
                stop=False,
                lang_valid=bool(self._language.valid),
                ent_valid=bool(ent.valid),
                guard_result=GUARD_FAIL_CLOSED,
                guard_reason=guard_reason,
                raw_action=action[0],
                final_action=(0.0, 0.0),
            )
            return

        current_action = np.asarray(guarded, dtype=np.float32)
        shaped = smooth_policy_displacement(
            current_action,
            previous_action=None,
            max_step_m=self._smooth_max_step_m,
            max_delta_m=self._smooth_max_delta_m,
        )
        if shaped is None:
            self._publish_fail_closed(ent, "POLICY_SMOOTHING_INVALID")
            return

        safety = evaluate_safety_gate(
            stamp_us=int(message.stamp_us),
            run_id=str(ent.run_id),
            frame_id=FRAME_ID,
            model_version=self._model_version,
            dt=DT_SEC,
            desired_x=float(shaped[0]),
            desired_y=float(shaped[1]),
            safe_stop=False,
            valid=True,
            reason="POLICY_INFERRED_SMOOTHED",
            language_valid=bool(self._language.valid),
            entity_valid=bool(ent.valid),
            entities=ent.safety_entities,
            last_valid_stamp_us=0,
            time_since_last_valid_sec=0.0,
            config=self._safety_config,
        )
        if not safety.valid or safety.safe_stop:
            self._publish_fail_closed(ent, safety.reason, raw_action=shaped)
            return

        self._record_guard_outcome(
            guard_reason,
            raw_action=action[0],
            guarded_action=guarded,
            final_action=shaped,
        )
        message.desired_x = float(shaped[0])
        message.desired_y = float(shaped[1])
        message.safe_stop = False
        message.valid = True
        message.reason = "POLICY_INFERRED_SMOOTHED"
        self._trace_policy_decision(
            ent,
            policy_valid=True,
            stop=False,
            lang_valid=bool(self._language.valid),
            ent_valid=bool(ent.valid),
            guard_result=guard_reason,
            guard_reason=guard_reason,
            raw_action=action[0],
            guarded_action=guarded,
            final_action=shaped,
        )
        self._pub.publish(message)
        self._frame_seq += 1

    @staticmethod
    def _build_inputs(
        language: TaskEmbedding,
        entities: DecisionEntityFrame,
        *,
        ego_state: ASVState,
    ) -> dict[str, np.ndarray]:
        """Build language, entity and current-dynamics policy inputs."""

        identity_reason = identity_mismatch_reason(language, entities)
        if identity_reason is not None:
            raise ValueError(identity_reason)
        embedding = np.asarray(language.embedding, dtype=np.float32).reshape(-1)
        if embedding.size != LANGUAGE_DIM or not np.all(np.isfinite(embedding)):
            raise ValueError("task embedding must be finite float32[256]")
        embedding = condition_policy_language(embedding, str(language.instruction))
        if embedding.size != LANGUAGE_DIM or not np.all(np.isfinite(embedding)):
            raise ValueError("conditioned task embedding must be finite float32[256]")
        if int(entities.max_entities) != ENTITY_COUNT:
            raise ValueError("EntityFeatures max_entities does not match policy contract")
        if int(entities.feature_dim) != ENTITY_GEOMETRY_DIM:
            raise ValueError("EntityFeatures feature_dim does not match policy contract")
        geometry = np.asarray(entities.features, dtype=np.float32).reshape(
            ENTITY_COUNT, ENTITY_GEOMETRY_DIM
        )
        mask = np.asarray(entities.mask, dtype=bool).reshape(ENTITY_COUNT)
        if not np.all(np.isfinite(geometry[mask])):
            raise ValueError("active structured entity features are non-finite")
        if not str(language.instruction).strip() == str(entities.instruction).strip() and str(entities.instruction).strip():
            raise ValueError("language/entity instruction mismatch")
        ego_values = np.asarray(
            [
                np.clip(float(ego_state.surge_velocity) / 5.0, -1.0, 1.0),
                np.clip(float(ego_state.yaw_rate) / 1.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        ego_valid = bool(ego_state.valid) and bool(np.all(np.isfinite(ego_values)))
        policy_valid = bool(language.valid) and bool(entities.valid) and bool(np.any(mask)) and ego_valid
        return {
            "language": embedding.reshape(1, LANGUAGE_DIM),
            "entity_geometry": geometry.reshape(1, ENTITY_COUNT, ENTITY_GEOMETRY_DIM),
            "ego_state": ego_values.reshape(1, EGO_STATE_DIM),
            "language_valid": np.asarray([bool(language.valid)], dtype=bool),
            "entity_geometry_mask": mask.reshape(1, ENTITY_COUNT),
            "ego_state_valid": np.asarray([ego_valid], dtype=bool),
            "policy_input_valid": np.asarray([policy_valid], dtype=bool),
        }

    def destroy_node(self) -> bool:
        if not self._policy_audit_shutdown_logged:
            self._policy_audit_shutdown_logged = True
            self._maybe_policy_audit(force=True, trigger="shutdown")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
