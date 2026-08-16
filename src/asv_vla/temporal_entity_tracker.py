"""Pure temporal tracking for geometry-only entity observations.

The UE bridge can provide positions and semantics without a velocity field.
This module keeps that transport boundary honest: velocity is zero and marked
invalid until two time-ordered observations of the same entity are available.
No ROS imports are required, so the tracker can be tested independently of a
running graph and its records can be adapted to the existing ``UEEntity``
message shape by callers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Sequence


Position3 = tuple[float, float, float]
BBox4 = tuple[float, float, float, float]
VelocityFilter = Literal["none", "ema", "alpha_beta"]


class TemporalEntityTrackerError(ValueError):
    """Raised when a geometry frame cannot satisfy the tracker contract."""


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Identity and timing attached to one observation frame."""

    run_id: str
    scene_seed: int
    frame_index: int
    stamp_us: int

    def __post_init__(self) -> None:
        run_id = str(self.run_id).strip()
        if not run_id:
            raise TemporalEntityTrackerError("run_id must not be empty")
        scene_seed = int(self.scene_seed)
        frame_index = int(self.frame_index)
        stamp_us = int(self.stamp_us)
        if frame_index < 0:
            raise TemporalEntityTrackerError("frame_index must be non-negative")
        if stamp_us < 0:
            raise TemporalEntityTrackerError("stamp_us must be non-negative")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "scene_seed", scene_seed)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "stamp_us", stamp_us)

    @property
    def identity(self) -> tuple[str, int]:
        return self.run_id, self.scene_seed


@dataclass(frozen=True, slots=True)
class GeometryObservation:
    """One entity observation containing geometry and semantic metadata."""

    entity_id: str
    relative_x: float
    relative_y: float
    relative_z: float
    class_name: str = ""
    color: str = ""
    is_target: bool = False
    visible: bool = True
    bbox: BBox4 | None = None
    confidence: float = 1.0
    run_id: str = ""
    scene_seed: int = 0
    frame_index: int = 0
    stamp_us: int = 0

    def __post_init__(self) -> None:
        entity_id = str(self.entity_id).strip()
        if not entity_id:
            raise TemporalEntityTrackerError("entity_id must not be empty")
        values = (
            float(self.relative_x),
            float(self.relative_y),
            float(self.relative_z),
        )
        if not all(math.isfinite(value) for value in values):
            raise TemporalEntityTrackerError(
                f"entity {entity_id!r} position must be finite"
            )
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise TemporalEntityTrackerError(
                f"entity {entity_id!r} confidence must be in [0, 1]"
            )
        bbox = None if self.bbox is None else tuple(float(v) for v in self.bbox)
        if bbox is not None and (
            len(bbox) != 4 or not all(math.isfinite(value) for value in bbox)
        ):
            raise TemporalEntityTrackerError(
                f"entity {entity_id!r} bbox must contain four finite values"
            )
        metadata = FrameMetadata(
            run_id=self.run_id,
            scene_seed=self.scene_seed,
            frame_index=self.frame_index,
            stamp_us=self.stamp_us,
        )
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "relative_x", values[0])
        object.__setattr__(self, "relative_y", values[1])
        object.__setattr__(self, "relative_z", values[2])
        object.__setattr__(self, "class_name", str(self.class_name))
        object.__setattr__(self, "color", str(self.color))
        object.__setattr__(self, "is_target", bool(self.is_target))
        object.__setattr__(self, "visible", bool(self.visible))
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "run_id", metadata.run_id)
        object.__setattr__(self, "scene_seed", metadata.scene_seed)
        object.__setattr__(self, "frame_index", metadata.frame_index)
        object.__setattr__(self, "stamp_us", metadata.stamp_us)

    @property
    def position(self) -> Position3:
        return self.relative_x, self.relative_y, self.relative_z

    @property
    def metadata(self) -> FrameMetadata:
        return FrameMetadata(
            self.run_id, self.scene_seed, self.frame_index, self.stamp_us
        )


@dataclass(frozen=True, slots=True)
class TrackedEntity:
    """Current geometry plus an explicitly validity-gated velocity estimate."""

    entity_id: str
    relative_x: float
    relative_y: float
    relative_z: float
    relative_velocity_x: float
    relative_velocity_y: float
    relative_velocity_z: float
    velocity_valid: bool
    class_name: str
    color: str
    is_target: bool
    visible: bool
    bbox: BBox4 | None
    confidence: float
    run_id: str
    scene_seed: int
    frame_index: int
    stamp_us: int
    frame_gap: int = 0
    valid: bool = True
    source: str = "temporal_tracker"

    @property
    def position(self) -> Position3:
        return self.relative_x, self.relative_y, self.relative_z

    @property
    def velocity(self) -> Position3:
        return (
            self.relative_velocity_x,
            self.relative_velocity_y,
            self.relative_velocity_z,
        )

    def as_ue_entity_kwargs(self) -> dict[str, object]:
        """Return fields accepted by the existing ``UEEntity`` message."""

        bbox = self.bbox or (0.0, 0.0, 0.0, 0.0)
        return {
            "entity_id": self.entity_id,
            "class_name": self.class_name,
            "color": self.color,
            "is_target": self.is_target,
            "visible": self.visible,
            "relative_x": self.relative_x,
            "relative_y": self.relative_y,
            "relative_z": self.relative_z,
            "relative_velocity_x": self.relative_velocity_x,
            "relative_velocity_y": self.relative_velocity_y,
            "relative_velocity_z": self.relative_velocity_z,
            "valid": self.valid,
            "source": self.source,
            "bbox_x_min": bbox[0],
            "bbox_y_min": bbox[1],
            "bbox_x_max": bbox[2],
            "bbox_y_max": bbox[3],
            "bbox_valid": self.bbox is not None,
            "confidence": self.confidence,
            "velocity_valid": self.velocity_valid,
        }

    # Short alias for adapters that use the noun rather than the message name.
    to_ue_entity_kwargs = as_ue_entity_kwargs


@dataclass(slots=True)
class _TrackState:
    position: Position3
    filter_position: Position3
    frame_index: int
    stamp_us: int
    velocity: Position3 = (0.0, 0.0, 0.0)
    velocity_valid: bool = False


class TemporalEntityTracker:
    """Track geometry observations and estimate velocity by finite difference."""

    def __init__(
        self,
        *,
        ttl_frames: int = 2,
        ttl_sec: float | None = None,
        velocity_filter: VelocityFilter = "none",
        alpha: float = 1.0,
        beta: float = 0.85,
    ) -> None:
        if ttl_frames < 0:
            raise ValueError("ttl_frames must be non-negative")
        if ttl_sec is not None and ttl_sec <= 0.0:
            raise ValueError("ttl_sec must be positive when provided")
        if velocity_filter not in {"none", "ema", "alpha_beta"}:
            raise ValueError("velocity_filter must be none, ema, or alpha_beta")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 < beta <= 2.0:
            raise ValueError("beta must be in (0, 2]")
        self.ttl_frames = int(ttl_frames)
        self.ttl_sec = None if ttl_sec is None else float(ttl_sec)
        self.velocity_filter = velocity_filter
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._tracks: dict[str, _TrackState] = {}
        self._identity: tuple[str, int] | None = None
        self._last_frame_index: int | None = None
        self._last_stamp_us: int | None = None

    @property
    def identity(self) -> tuple[str, int] | None:
        return self._identity

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tracks))

    def reset(self) -> None:
        self._tracks.clear()
        self._identity = None
        self._last_frame_index = None
        self._last_stamp_us = None

    def update(
        self,
        observations: Iterable[GeometryObservation],
        *,
        frame: FrameMetadata | None = None,
    ) -> tuple[TrackedEntity, ...]:
        """Consume one frame and return records for entities seen in it.

        A frame with no observations can be represented by passing ``frame``;
        this advances TTL bookkeeping without fabricating entity positions.
        Frames with a non-increasing index are ignored.  A timestamp regression
        is accepted as a geometry update but invalidates velocity for that
        frame, rather than guessing a time interval.
        """

        items = tuple(observations)
        metadata = self._metadata_for(items, frame)
        if any(item.metadata != metadata for item in items):
            raise TemporalEntityTrackerError(
                "all observations in a frame must share run/scene/frame/stamp"
            )
        ids = [item.entity_id for item in items]
        if len(ids) != len(set(ids)):
            raise TemporalEntityTrackerError("duplicate entity_id in frame")

        if self._identity != metadata.identity:
            self._tracks.clear()
            self._identity = metadata.identity
            self._last_frame_index = None
            self._last_stamp_us = None

        if (
            self._last_frame_index is not None
            and metadata.frame_index <= self._last_frame_index
        ):
            return ()

        monotonic_stamp = (
            self._last_stamp_us is None
            or metadata.stamp_us > self._last_stamp_us
        )
        self._expire_tracks(metadata)
        records = tuple(
            self._record_for(item, metadata, monotonic_stamp) for item in items
        )
        self._last_frame_index = metadata.frame_index
        self._last_stamp_us = metadata.stamp_us
        return records

    # Explicit alias makes the intended frame-processing operation discoverable.
    process_frame = update

    def _metadata_for(
        self,
        items: Sequence[GeometryObservation],
        frame: FrameMetadata | None,
    ) -> FrameMetadata:
        if frame is not None and not isinstance(frame, FrameMetadata):
            raise TypeError("frame must be FrameMetadata")
        if items:
            metadata = items[0].metadata
            if frame is not None and frame != metadata:
                raise TemporalEntityTrackerError(
                    "explicit frame metadata does not match observation"
                )
            return metadata
        if frame is None:
            raise TemporalEntityTrackerError(
                "an empty frame requires explicit FrameMetadata"
            )
        return frame

    def _expire_tracks(self, metadata: FrameMetadata) -> None:
        expired = []
        for entity_id, state in self._tracks.items():
            frame_gap = metadata.frame_index - state.frame_index
            time_gap = (metadata.stamp_us - state.stamp_us) / 1.0e6
            too_many_frames = frame_gap > self.ttl_frames
            too_long = self.ttl_sec is not None and time_gap > self.ttl_sec
            if too_many_frames or (time_gap >= 0.0 and too_long):
                expired.append(entity_id)
        for entity_id in expired:
            del self._tracks[entity_id]

    def _record_for(
        self,
        item: GeometryObservation,
        metadata: FrameMetadata,
        monotonic_stamp: bool,
    ) -> TrackedEntity:
        state = self._tracks.get(item.entity_id)
        velocity = (0.0, 0.0, 0.0)
        velocity_valid = False
        frame_gap = 0
        if state is not None:
            frame_gap = metadata.frame_index - state.frame_index
            dt_sec = (metadata.stamp_us - state.stamp_us) / 1.0e6
            if monotonic_stamp and frame_gap > 0 and dt_sec > 0.0:
                raw_velocity = tuple(
                    (current - previous) / dt_sec
                    for current, previous in zip(item.position, state.position)
                )
                velocity, filter_position = self._filter_velocity(
                    state, raw_velocity, dt_sec, item
                )
                velocity_valid = all(math.isfinite(value) for value in velocity)
                if not velocity_valid:
                    velocity = (0.0, 0.0, 0.0)
                    filter_position = item.position
            else:
                filter_position = item.position
        else:
            filter_position = item.position

        self._tracks[item.entity_id] = _TrackState(
            position=item.position,
            filter_position=filter_position,
            frame_index=metadata.frame_index,
            stamp_us=metadata.stamp_us,
            velocity=velocity,
            velocity_valid=velocity_valid,
        )
        return TrackedEntity(
            entity_id=item.entity_id,
            relative_x=item.relative_x,
            relative_y=item.relative_y,
            relative_z=item.relative_z,
            relative_velocity_x=velocity[0],
            relative_velocity_y=velocity[1],
            relative_velocity_z=velocity[2],
            velocity_valid=velocity_valid,
            class_name=item.class_name,
            color=item.color,
            is_target=item.is_target,
            visible=item.visible,
            bbox=item.bbox,
            confidence=item.confidence,
            run_id=metadata.run_id,
            scene_seed=metadata.scene_seed,
            frame_index=metadata.frame_index,
            stamp_us=metadata.stamp_us,
            frame_gap=frame_gap,
        )

    def _filter_velocity(
        self,
        state: _TrackState,
        raw_velocity: Position3,
        dt_sec: float,
        item: GeometryObservation,
    ) -> tuple[Position3, Position3]:
        if not state.velocity_valid or self.velocity_filter == "none":
            return raw_velocity, item.position
        if self.velocity_filter == "ema":
            return (
                tuple(
                    self.alpha * raw + (1.0 - self.alpha) * previous
                    for raw, previous in zip(raw_velocity, state.velocity)
                ),
                item.position,
            )

        # Alpha-beta: the position residual corrects the prior velocity.  The
        # reported position remains the actual geometry observation.
        residual = tuple(
            current - (previous + velocity * dt_sec)
            for current, previous, velocity in zip(
                item.position, state.filter_position, state.velocity
            )
        )
        predicted_position = tuple(
            previous + velocity * dt_sec
            for previous, velocity in zip(state.filter_position, state.velocity)
        )
        corrected_position = tuple(
            predicted + self.alpha * error
            for predicted, error in zip(predicted_position, residual)
        )
        return (
            tuple(
                velocity + self.beta * error / dt_sec
                for velocity, error in zip(state.velocity, residual)
            ),
            corrected_position,
        )


__all__ = [
    "FrameMetadata",
    "GeometryObservation",
    "TemporalEntityTracker",
    "TemporalEntityTrackerError",
    "TrackedEntity",
]


