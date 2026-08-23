"""Match Jetson perception_node entity construction for offline training."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from perception import (
    CameraProfile,
    FrameMetadata,
    GeometryObservation,
    ImageEntityModel,
    ImageEntityPrediction,
    TemporalEntityTracker,
    TrackedEntity,
)


def task_key(text: str) -> str:
    folded = text.casefold()
    color = "blue" if "blue" in folded or "蓝" in text else "red"
    distance = "4m" if "4m" in folded or "4 m" in folded or "4米" in text else "3m"
    return f"{color}_{distance}"


def prediction_to_entity_item(
    prediction: ImageEntityPrediction,
    *,
    template: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not prediction.visible:
        return None
    if not all(
        math.isfinite(float(value))
        for value in (prediction.relative_x, prediction.relative_y, prediction.relative_z)
    ):
        return None
    source = dict(template or {})
    return {
        "entity_id": prediction.entity_id,
        "class_name": str(source.get("class_name", "boat")),
        "color": str(source.get("color", "")),
        "is_target": bool(source.get("is_target", False)),
        "visible": True,
        "valid": True,
        "relative_position_m": [
            float(prediction.relative_x),
            float(prediction.relative_y),
            float(prediction.relative_z),
        ],
        "relative_velocity_mps": [0.0, 0.0, 0.0],
        "velocity_valid": False,
    }


def tracked_to_entity_item(
    tracked: TrackedEntity,
    *,
    template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = dict(template or {})
    return {
        "entity_id": tracked.entity_id,
        "class_name": str(source.get("class_name", tracked.class_name or "boat")),
        "color": str(source.get("color", tracked.color)),
        "is_target": bool(source.get("is_target", tracked.is_target)),
        "visible": bool(tracked.visible),
        "valid": bool(tracked.valid),
        "relative_position_m": [
            float(tracked.relative_x),
            float(tracked.relative_y),
            float(tracked.relative_z),
        ],
        "relative_velocity_mps": [
            float(tracked.relative_velocity_x),
            float(tracked.relative_velocity_y),
            float(tracked.relative_velocity_z),
        ],
        "velocity_valid": bool(tracked.velocity_valid),
    }


def entities_from_predictions(
    predictions: Sequence[ImageEntityPrediction],
    *,
    templates: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    templates = templates or {}
    items: list[dict[str, Any]] = []
    for prediction in predictions:
        item = prediction_to_entity_item(
            prediction,
            template=templates.get(prediction.entity_id),
        )
        if item is not None:
            items.append(item)
    return items


def _record_key(record: Any) -> tuple[str, str, int, int]:
    return (
        str(record.slot_id),
        str(record.run_id),
        int(record.scene_seed),
        int(record.frame_index),
    )


def _run_key(record: Any) -> tuple[str, int]:
    return (str(record.run_id), int(record.scene_seed))


def _predictions_for_record(
    record: Any,
    *,
    model: ImageEntityModel,
    task_embedding: np.ndarray,
) -> tuple[ImageEntityPrediction, ...]:
    image = Image.open(record.image_path).convert("RGB")
    return model.predict(image, device="cuda", task_embedding=task_embedding)


def _tracked_items_for_record(
    record: Any,
    *,
    model: ImageEntityModel,
    task_embedding: np.ndarray,
    tracker: TemporalEntityTracker,
) -> list[dict[str, Any]]:
    templates = {str(item["entity_id"]): item for item in record.entities}
    predictions = _predictions_for_record(
        record,
        model=model,
        task_embedding=task_embedding,
    )
    observations = [
        GeometryObservation(
            entity_id=prediction.entity_id,
            relative_x=prediction.relative_x,
            relative_y=prediction.relative_y,
            relative_z=prediction.relative_z,
            class_name=str(templates.get(prediction.entity_id, {}).get("class_name", "boat")),
            color=str(templates.get(prediction.entity_id, {}).get("color", "")),
            visible=prediction.visible,
            confidence=float(prediction.confidence),
            run_id=str(record.run_id),
            scene_seed=int(record.scene_seed),
            frame_index=int(record.frame_index),
            stamp_us=int(record.ego["stamp_us"]),
        )
        for prediction in predictions
        if prediction.visible
    ]
    frame = FrameMetadata(
        run_id=str(record.run_id),
        scene_seed=int(record.scene_seed),
        frame_index=int(record.frame_index),
        stamp_us=int(record.ego["stamp_us"]),
    )
    tracked = tracker.update(observations, frame=frame)
    return [
        tracked_to_entity_item(item, template=templates.get(item.entity_id))
        for item in tracked
        if item.valid and item.visible
    ]


def build_online_entity_cache(
    records: Sequence[Any],
    *,
    model: ImageEntityModel,
    embeddings: Mapping[str, np.ndarray],
) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for record in records:
        grouped[_run_key(record)].append(record)
    cache: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for group in grouped.values():
        group.sort(key=lambda item: (int(item.ego["stamp_us"]), int(item.frame_index)))
        tracker = TemporalEntityTracker(velocity_filter="ema", alpha=0.35)
        for record in group:
            task_embedding = task_embedding_for_record(record, embeddings)
            cache[_record_key(record)] = _tracked_items_for_record(
                record,
                model=model,
                task_embedding=task_embedding,
                tracker=tracker,
            )
    return cache


def entities_for_record(
    record: Any,
    *,
    model: ImageEntityModel,
    task_embedding: np.ndarray,
    profile: CameraProfile | None = None,
    entity_cache: Mapping[tuple[str, str, int, int], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    del profile  # kept for call-site compatibility
    if entity_cache is not None:
        return list(entity_cache.get(_record_key(record), []))
    tracker = TemporalEntityTracker(velocity_filter="ema", alpha=0.35)
    return _tracked_items_for_record(
        record,
        model=model,
        task_embedding=task_embedding,
        tracker=tracker,
    )


def task_embedding_for_record(
    record: Any,
    embeddings: Mapping[str, np.ndarray],
) -> np.ndarray:
    return np.asarray(embeddings[task_key(record.task_text)], dtype=np.float32)
