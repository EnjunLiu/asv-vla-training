from __future__ import annotations

import math
from typing import Protocol


# ``HORIZON`` remains an offline/model constant.  It is intentionally not
# part of the online ROS control contract: the policy adapter consumes the
# frozen model output and publishes one DecisionPoint per frame.
HORIZON = 20
ACTION_DIM = 2
DT_SEC = 0.2
FRAME_ID = "base_link"
# The trained decision head and online desired_x/desired_y contract share one
# bounded single-step displacement.  The 0.2 s control interval is unchanged.
MAX_DISPLACEMENT_M = 0.30
SAFE_STOP_MODEL_VERSION = "safe_stop:none"
FLOAT_TOLERANCE = 1.0e-6


class DecisionPointLike(Protocol):
    stamp_us: int
    run_id: str
    scene_seed: int
    frame_index: int
    frame_id: str
    model_version: str
    dt: float
    desired_x: float
    desired_y: float
    safe_stop: bool
    valid: bool


def finite_zero(value: float, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return math.isfinite(value) and abs(value) <= tolerance


def is_safe_stop(message: DecisionPointLike) -> bool:
    """Validate a non-executable, single-point safe-stop marker.

    A safe stop is deliberately invalid.  Downstream adapters must interpret
    it as a hold, rather than as a valid zero displacement that could trigger
    position-hold compensation in a physical controller.
    """

    return (
        int(message.stamp_us) > 0
        and bool(str(message.run_id).strip())
        and message.frame_id == FRAME_ID
        and message.model_version == SAFE_STOP_MODEL_VERSION
        and math.isfinite(float(message.dt))
        and abs(float(message.dt) - DT_SEC) <= FLOAT_TOLERANCE
        and finite_zero(float(message.desired_x))
        and finite_zero(float(message.desired_y))
        and bool(message.safe_stop)
        and not bool(message.valid)
    )


