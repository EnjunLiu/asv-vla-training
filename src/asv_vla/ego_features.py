"""Canonical ego-motion features shared by collection, training and runtime.

The policy does not receive UE truth entities, but it may use the vessel's own
odometry/INS state.  The two values are normalized once at the interface so
the PC cache and Jetson runtime cannot silently learn different units.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# Conservative ranges for the present UE5/USV contract.  Values outside these
# ranges are clipped, while non-finite values invalidate the modality.
SURGE_SCALE_MPS = 5.0
YAW_RATE_SCALE_RADPS = 1.0


def normalize_ego(surge_velocity_mps: float, yaw_rate_radps: float) -> tuple[np.ndarray, bool]:
    """Return ``[surge/scale, yaw_rate/scale]`` and a finite validity flag."""

    values = (float(surge_velocity_mps), float(yaw_rate_radps))
    if not all(math.isfinite(value) for value in values):
        return np.zeros(2, dtype=np.float32), False
    result = np.asarray(
        [
            np.clip(values[0] / SURGE_SCALE_MPS, -1.0, 1.0),
            np.clip(values[1] / YAW_RATE_SCALE_RADPS, -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    return result, True


def normalize_ego_message(message: Any) -> tuple[np.ndarray, bool]:
    """Normalize a ``UEASVState``-compatible message."""

    vector, finite = normalize_ego(message.surge_velocity, message.yaw_rate)
    return vector, bool(message.valid) and finite


