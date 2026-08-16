from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


MAX_ENTITIES = 16
FEATURE_DIM = 16
BACKEND_ID = "deterministic_entity_tensor_v1"

# Each row is:
# [x, y, z, vx, vy, vz, planar_distance, bearing_sin, bearing_cos,
#  closing_speed, time_to_cpa, cpa_distance, is_target, is_risk,
#  color_red, color_blue]
# Continuous values are normalized and clipped to the ranges below.
POSITION_SCALE_M = 20.0
HEIGHT_SCALE_M = 5.0
VELOCITY_SCALE_MPS = 5.0
DEFAULT_RISK_HORIZON_SEC = 4.0
DEFAULT_RISK_RADIUS_M = 3.0


class TaskEntityTensorError(RuntimeError):
    """Raised when an entity array cannot satisfy the Day 7 contract."""


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
class EntityTensorResult:
    features: np.ndarray
    mask: np.ndarray
    entity_ids: tuple[str, ...]
    entity_count: int
    target_count: int
    risk_count: int
    dropped_count: int


@dataclass(frozen=True)
class _Candidate:
    entity: Any
    entity_id: str
    metrics: EntityMetrics


def _clip(value: float, scale: float, low: float = -1.0) -> float:
    return float(np.clip(value / scale, low, 1.0))


def compute_entity_metrics(
    entity: Any,
    *,
    risk_horizon_sec: float = DEFAULT_RISK_HORIZON_SEC,
    risk_radius_m: float = DEFAULT_RISK_RADIUS_M,
) -> EntityMetrics:
    if risk_horizon_sec <= 0.0:
        raise ValueError("risk_horizon_sec must be positive")
    if risk_radius_m <= 0.0:
        raise ValueError("risk_radius_m must be positive")

    x = float(entity.relative_x)
    y = float(entity.relative_y)
    vx = float(entity.relative_velocity_x)
    vy = float(entity.relative_velocity_y)
    distance = math.hypot(x, y)
    if distance > 1.0e-9:
        bearing_sin = y / distance
        bearing_cos = x / distance
        closing_speed = -(x * vx + y * vy) / distance
    else:
        bearing_sin = 0.0
        bearing_cos = 1.0
        closing_speed = 0.0

    velocity_squared = vx * vx + vy * vy
    if velocity_squared > 1.0e-12:
        raw_time_to_cpa = -(x * vx + y * vy) / velocity_squared
        time_to_cpa = min(max(raw_time_to_cpa, 0.0), risk_horizon_sec)
    else:
        raw_time_to_cpa = math.inf
        time_to_cpa = risk_horizon_sec

    cpa_x = x + vx * time_to_cpa
    cpa_y = y + vy * time_to_cpa
    cpa_distance = math.hypot(cpa_x, cpa_y)
    is_risk = (
        closing_speed > 0.0
        and 0.0 < raw_time_to_cpa <= risk_horizon_sec
        and cpa_distance <= risk_radius_m
    )
    return EntityMetrics(
        distance_m=distance,
        bearing_sin=bearing_sin,
        bearing_cos=bearing_cos,
        closing_speed_mps=closing_speed,
        time_to_cpa_sec=time_to_cpa,
        cpa_distance_m=cpa_distance,
        is_risk=is_risk,
    )


def _validate_visible_entity(entity: Any) -> tuple[str, tuple[float, ...]]:
    entity_id = str(entity.entity_id).strip()
    if not entity_id:
        raise TaskEntityTensorError(
            "a valid visible entity has an empty entity_id"
        )
    values = (
        float(entity.relative_x),
        float(entity.relative_y),
        float(entity.relative_z),
        float(entity.relative_velocity_x),
        float(entity.relative_velocity_y),
        float(entity.relative_velocity_z),
    )
    if not all(math.isfinite(value) for value in values):
        raise TaskEntityTensorError(
            f"entity {entity_id!r} contains NaN or Inf"
        )
    return entity_id, values


def _entity_row(
    candidate: _Candidate,
    *,
    risk_horizon_sec: float,
) -> np.ndarray:
    entity = candidate.entity
    metrics = candidate.metrics
    color = str(entity.color).strip().casefold()
    is_red = color in {"red", "红", "红色"}
    is_blue = color in {"blue", "蓝", "蓝色"}
    return np.asarray(
        (
            _clip(float(entity.relative_x), POSITION_SCALE_M),
            _clip(float(entity.relative_y), POSITION_SCALE_M),
            _clip(float(entity.relative_z), HEIGHT_SCALE_M),
            _clip(float(entity.relative_velocity_x), VELOCITY_SCALE_MPS),
            _clip(float(entity.relative_velocity_y), VELOCITY_SCALE_MPS),
            _clip(float(entity.relative_velocity_z), VELOCITY_SCALE_MPS),
            _clip(metrics.distance_m, POSITION_SCALE_M, low=0.0),
            metrics.bearing_sin,
            metrics.bearing_cos,
            _clip(metrics.closing_speed_mps, VELOCITY_SCALE_MPS),
            _clip(metrics.time_to_cpa_sec, risk_horizon_sec, low=0.0),
            _clip(metrics.cpa_distance_m, POSITION_SCALE_M, low=0.0),
            1.0 if bool(entity.is_target) else 0.0,
            1.0 if metrics.is_risk else 0.0,
            1.0 if is_red else 0.0,
            1.0 if is_blue else 0.0,
        ),
        dtype=np.float32,
    )


def build_entity_tensor(
    entities: Iterable[Any],
    *,
    max_entities: int = MAX_ENTITIES,
    risk_horizon_sec: float = DEFAULT_RISK_HORIZON_SEC,
    risk_radius_m: float = DEFAULT_RISK_RADIUS_M,
) -> EntityTensorResult:
    if max_entities <= 0:
        raise ValueError("max_entities must be positive")
    if risk_horizon_sec <= 0.0:
        raise ValueError("risk_horizon_sec must be positive")
    if risk_radius_m <= 0.0:
        raise ValueError("risk_radius_m must be positive")

    candidates = []
    seen_ids = set()
    for entity in entities:
        if not bool(entity.valid) or not bool(entity.visible):
            continue
        entity_id, _ = _validate_visible_entity(entity)
        if entity_id in seen_ids:
            raise TaskEntityTensorError(
                f"duplicate valid visible entity_id {entity_id!r}"
            )
        seen_ids.add(entity_id)
        metrics = compute_entity_metrics(
            entity,
            risk_horizon_sec=risk_horizon_sec,
            risk_radius_m=risk_radius_m,
        )
        candidates.append(_Candidate(entity, entity_id, metrics))

    targets = sorted(
        (item for item in candidates if bool(item.entity.is_target)),
        key=lambda item: (item.metrics.distance_m, item.entity_id),
    )
    risks = sorted(
        (
            item
            for item in candidates
            if not bool(item.entity.is_target) and item.metrics.is_risk
        ),
        key=lambda item: (
            item.metrics.cpa_distance_m,
            item.metrics.time_to_cpa_sec,
            item.metrics.distance_m,
            item.entity_id,
        ),
    )
    normal = sorted(
        (
            item
            for item in candidates
            if not bool(item.entity.is_target) and not item.metrics.is_risk
        ),
        key=lambda item: (item.metrics.distance_m, item.entity_id),
    )
    selected = (targets + risks + normal)[:max_entities]

    features = np.zeros((max_entities, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(max_entities, dtype=np.bool_)
    entity_ids = [""] * max_entities
    for index, candidate in enumerate(selected):
        features[index] = _entity_row(
            candidate,
            risk_horizon_sec=risk_horizon_sec,
        )
        mask[index] = True
        entity_ids[index] = candidate.entity_id

    if features.shape != (max_entities, FEATURE_DIM):
        raise TaskEntityTensorError(
            f"internal feature shape is invalid: {features.shape}"
        )
    if not np.all(np.isfinite(features)):
        raise TaskEntityTensorError("entity tensor contains NaN or Inf")

    selected_targets = sum(
        bool(item.entity.is_target) for item in selected
    )
    selected_risks = sum(item.metrics.is_risk for item in selected)
    return EntityTensorResult(
        features=features,
        mask=mask,
        entity_ids=tuple(entity_ids),
        entity_count=len(selected),
        target_count=selected_targets,
        risk_count=selected_risks,
        dropped_count=max(0, len(candidates) - len(selected)),
    )


