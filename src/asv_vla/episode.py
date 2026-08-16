"""Day 8 episode recording helpers and deterministic quality checks."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

from PIL import Image

from .frame_record import (
    CAMERA_FOV_ANGLE_DEG,
    CAMERA_HEIGHT_PX,
    CAMERA_MOUNT_POSITION_M,
    CAMERA_MOUNT_RPY_UE_DEG,
    CAMERA_WIDTH_PX,
    SCHEMA_VERSION,
    FrameRecordError,
    read_frame_record,
    write_frame_record,
)


EPISODE_MANIFEST_VERSION = "episode_manifest_v1"
QUALITY_REPORT_VERSION = "episode_quality_v1"
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
EXECUTION_MODES = frozenset({
    "observation_only",
    "ue5_kinematic_expert_v1",
    "legacy_thruster",
})


class EpisodeError(ValueError):
    """Raised when an episode cannot be written or replayed safely."""


def frame_key(message: Any) -> tuple[str, int, int, int]:
    return (
        str(message.run_id),
        int(message.scene_seed),
        int(message.frame_index),
        int(message.stamp_us),
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, payload)


def validate_run_id_path(run_id: str) -> None:
    if not SAFE_RUN_ID.fullmatch(run_id):
        raise EpisodeError(
            "run_id is not safe for an episode directory: "
            f"{run_id!r}"
        )


def _entity_record(entity: Any) -> dict[str, Any]:
    bbox_valid = bool(getattr(entity, "bbox_valid", False))
    return {
        "entity_id": str(entity.entity_id),
        "class_name": str(entity.class_name),
        "color": str(entity.color),
        "is_target": bool(entity.is_target),
        "visible": bool(entity.visible),
        "relative_position_m": [
            float(entity.relative_x),
            float(entity.relative_y),
            float(entity.relative_z),
        ],
        "relative_velocity_mps": [
            float(entity.relative_velocity_x),
            float(entity.relative_velocity_y),
            float(entity.relative_velocity_z),
        ],
        "source": str(getattr(entity, "source", "ue_truth")),
        "bbox_px": [
            float(getattr(entity, "bbox_x_min", 0.0)),
            float(getattr(entity, "bbox_y_min", 0.0)),
            float(getattr(entity, "bbox_x_max", 0.0)),
            float(getattr(entity, "bbox_y_max", 0.0)),
        ] if bbox_valid else None,
        "confidence": float(getattr(entity, "confidence", 1.0)),
        "velocity_valid": bool(
            getattr(entity, "velocity_valid", True)
        ),
        "valid": bool(entity.valid),
    }


def make_frame_record(
    *,
    task_text: str,
    task_stamp_us: int,
    state: Any,
    camera: Any,
    entities: Any,
    image_path: str,
) -> dict[str, Any]:
    keys = {frame_key(state), frame_key(camera), frame_key(entities)}
    if len(keys) != 1:
        raise EpisodeError(
            "ego, camera, and entities do not share the exact frame key"
        )
    run_id, scene_seed, frame_index, stamp_us = next(iter(keys))
    validate_run_id_path(run_id)

    task_valid = bool(task_text.strip())
    ego_valid = bool(state.valid)
    camera_valid = bool(camera.valid) and bool(camera.data)
    entities_valid = bool(entities.valid)
    modality_mask = {
        "task": task_valid,
        "ego": ego_valid,
        "camera": camera_valid,
        "entities": entities_valid,
    }
    valid = all(modality_mask.values())

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scene_seed": scene_seed,
        "frame_index": frame_index,
        "stamp_us": stamp_us,
        "frame_id": "base_link",
        "task": {
            "stamp_us": int(task_stamp_us),
            "text": task_text if task_valid else "",
            "valid": task_valid,
        },
        "ego": {
            "stamp_us": int(state.stamp_us),
            "world_frame_id": "ue_world",
            "simulation_time_s": float(state.simulation_time),
            "position_m": [
                float(state.position_x),
                float(state.position_y),
                float(state.position_z),
            ],
            "rpy_ue_rad": [
                float(state.roll),
                float(state.pitch),
                float(state.yaw),
            ],
            "surge_velocity_mps": float(state.surge_velocity),
            "yaw_rate_radps": float(state.yaw_rate),
            "valid": ego_valid,
        },
        "camera": {
            "stamp_us": int(camera.stamp_us),
            "image_path": image_path if camera_valid else None,
            "encoding": "jpeg",
            "width_px": CAMERA_WIDTH_PX,
            "height_px": CAMERA_HEIGHT_PX,
            "fov_angle_deg": CAMERA_FOV_ANGLE_DEG,
            "mount_frame_id": "base_link",
            "mount_position_m": list(CAMERA_MOUNT_POSITION_M),
            "mount_rpy_ue_deg": list(CAMERA_MOUNT_RPY_UE_DEG),
            "valid": camera_valid,
        },
        "entities": {
            "stamp_us": int(entities.stamp_us),
            "frame_id": str(entities.frame_id),
            "items": [_entity_record(item) for item in entities.entities],
            "valid": entities_valid,
        },
        "modality_mask": modality_mask,
        "valid": valid,
        "detail": "ok" if valid else "one or more source modalities are invalid",
    }


def write_episode_frame(
    episode_dir: Path,
    *,
    task_text: str,
    task_stamp_us: int,
    state: Any,
    camera: Any,
    entities: Any,
) -> Path:
    if str(camera.encoding).lower() not in {"jpeg", "jpg"}:
        raise EpisodeError(
            f"camera encoding must be jpeg, got {camera.encoding!r}"
        )
    image_relative = f"camera/{int(camera.frame_index):012d}.jpg"
    record_relative = f"frames/{int(camera.frame_index):012d}.json"
    image_path = episode_dir / image_relative
    record_path = episode_dir / record_relative
    if image_path.exists() or record_path.exists():
        raise EpisodeError(
            f"episode frame already exists: {int(camera.frame_index)}"
        )

    record = make_frame_record(
        task_text=task_text,
        task_stamp_us=task_stamp_us,
        state=state,
        camera=camera,
        entities=entities,
        image_path=image_relative,
    )
    if not record["valid"]:
        raise EpisodeError("only complete valid frames are recorded in Day 8")

    _atomic_bytes(image_path, bytes(camera.data))
    try:
        write_frame_record(record_path, record, image_root=episode_dir)
    except Exception:
        image_path.unlink(missing_ok=True)
        raise
    return record_path


def make_manifest(
    *,
    run_id: str,
    scene_seed: int,
    task_text: str,
    frame_indices: Iterable[int],
    stamp_values: Iterable[int],
    status: str,
    execution_mode: str = "observation_only",
    collection_slot: str = "",
    layout_id: str = "",
    motion_state: str = "",
) -> dict[str, Any]:
    if execution_mode not in EXECUTION_MODES:
        raise EpisodeError(f"invalid execution_mode: {execution_mode!r}")
    indices = list(frame_indices)
    stamps = list(stamp_values)
    manifest = {
        "schema_version": EPISODE_MANIFEST_VERSION,
        "frame_record_schema": SCHEMA_VERSION,
        "run_id": run_id,
        "scene_seed": int(scene_seed),
        "task_text": task_text,
        "status": status,
        "execution_mode": execution_mode,
        "frame_count": len(indices),
        "first_frame_index": indices[0] if indices else None,
        "last_frame_index": indices[-1] if indices else None,
        "first_stamp_us": stamps[0] if stamps else None,
        "last_stamp_us": stamps[-1] if stamps else None,
        "camera_profile": {
            "width_px": CAMERA_WIDTH_PX,
            "height_px": CAMERA_HEIGHT_PX,
            "fov_angle_deg": CAMERA_FOV_ANGLE_DEG,
            "mount_position_m": list(CAMERA_MOUNT_POSITION_M),
            "mount_rpy_ue_deg": list(CAMERA_MOUNT_RPY_UE_DEG),
        },
    }
    collection_values = (collection_slot, layout_id, motion_state)
    if any(collection_values):
        if not all(str(value).strip() for value in collection_values):
            raise EpisodeError(
                "collection_slot, layout_id and motion_state must either "
                "all be set or all be empty"
            )
        manifest["collection"] = {
            "schema_version": "collection_slot_v1",
            "slot_id": str(collection_slot).strip(),
            "layout_id": str(layout_id).strip(),
            "motion_state": str(motion_state).strip(),
        }
    return manifest


def load_episode_records(episode_dir: Path) -> list[dict[str, Any]]:
    frame_paths = sorted((episode_dir / "frames").glob("*.json"))
    if not frame_paths:
        raise EpisodeError(f"episode has no frame records: {episode_dir}")
    return [
        read_frame_record(path, image_root=episode_dir)
        for path in frame_paths
    ]


def _image_shape(path: Path) -> tuple[int, int]:
    try:
        payload = path.read_bytes()
        with Image.open(BytesIO(payload)) as image:
            image.load()
            return int(image.width), int(image.height)
    except Exception as exc:
        raise EpisodeError(f"invalid JPEG {path}: {exc}") from exc


def evaluate_episode(
    episode_dir: Path,
    *,
    min_frames: int = 1,
    write_report: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []

    for path in sorted((episode_dir / "frames").glob("*.json")):
        try:
            records.append(read_frame_record(path, image_root=episode_dir))
        except (FrameRecordError, OSError) as exc:
            errors.append(f"{path.name}: {exc}")

    if len(records) < min_frames:
        errors.append(
            f"frame_count={len(records)} is below required minimum={min_frames}"
        )

    run_ids = {record["run_id"] for record in records}
    scene_seeds = {record["scene_seed"] for record in records}
    frame_indices = [record["frame_index"] for record in records]
    stamp_values = [record["stamp_us"] for record in records]

    if len(run_ids) > 1:
        errors.append("multiple run_id values in one episode")
    if len(scene_seeds) > 1:
        errors.append("Scene_Seed changed within one episode")
    if any(
        current <= previous
        for previous, current in zip(frame_indices, frame_indices[1:])
    ):
        errors.append("Frame_Index values are not strictly increasing")
    if any(
        current < previous
        for previous, current in zip(stamp_values, stamp_values[1:])
    ):
        errors.append("stamp_us values move backwards")
    duplicate_timestamps = sum(
        current == previous
        for previous, current in zip(stamp_values, stamp_values[1:])
    )
    if duplicate_timestamps:
        warnings.append(
            f"{duplicate_timestamps} adjacent frame pair(s) share stamp_us"
        )

    frame_gaps = sum(
        max(0, current - previous - 1)
        for previous, current in zip(frame_indices, frame_indices[1:])
    )
    if frame_gaps:
        warnings.append(f"UE transport dropped {frame_gaps} frame(s)")

    entity_counts: list[int] = []
    for record in records:
        if not record["valid"]:
            errors.append(
                f"frame {record['frame_index']} is not fully multimodal-valid"
            )
        image_path = episode_dir / record["camera"]["image_path"]
        try:
            shape = _image_shape(image_path)
        except EpisodeError as exc:
            errors.append(str(exc))
        else:
            expected = (CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX)
            if shape != expected:
                errors.append(
                    f"frame {record['frame_index']} image shape "
                    f"{shape} != {expected}"
                )
        entity_counts.append(len(record["entities"]["items"]))

    manifest_path = episode_dir / "manifest.json"
    manifest: dict[str, Any] | None = None
    execution_mode = "observation_only"
    if not manifest_path.is_file():
        errors.append("manifest.json is missing")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"manifest.json is invalid: {exc}")
        else:
            if manifest.get("schema_version") != EPISODE_MANIFEST_VERSION:
                errors.append("manifest schema_version is invalid")
            execution_mode = manifest.get(
                "execution_mode",
                "observation_only",
            )
            if execution_mode not in EXECUTION_MODES:
                errors.append("manifest execution_mode is invalid")
            if manifest.get("frame_count") != len(records):
                errors.append("manifest frame_count does not match records")
            if run_ids and manifest.get("run_id") != next(iter(run_ids)):
                errors.append("manifest run_id does not match records")
            if (
                scene_seeds
                and manifest.get("scene_seed") != next(iter(scene_seeds))
            ):
                errors.append("manifest scene_seed does not match records")

    finite = True
    for record in records:
        stack = [record]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, float) and not math.isfinite(value):
                finite = False
    if not finite:
        errors.append("episode contains NaN or Inf")

    report = {
        "schema_version": QUALITY_REPORT_VERSION,
        "passed": not errors,
        "episode_dir": str(episode_dir.resolve()),
        "frame_count": len(records),
        "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None,
        "scene_seed": (
            next(iter(scene_seeds)) if len(scene_seeds) == 1 else None
        ),
        "execution_mode": execution_mode,
        "first_frame_index": frame_indices[0] if frame_indices else None,
        "last_frame_index": frame_indices[-1] if frame_indices else None,
        "frame_gaps": frame_gaps,
        "duplicate_timestamps": duplicate_timestamps,
        "all_modalities_valid": bool(records)
        and all(record["valid"] for record in records),
        "all_numbers_finite": finite,
        "camera_shape_px": [CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX],
        "entity_count_min": min(entity_counts) if entity_counts else None,
        "entity_count_max": max(entity_counts) if entity_counts else None,
        "errors": errors,
        "warnings": warnings,
    }
    if write_report:
        write_json_atomic(episode_dir / "quality_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a complete Day 8 FrameRecord episode."
    )
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--min-frames", type=int, default=1)
    args = parser.parse_args()

    report = evaluate_episode(
        args.episode_dir,
        min_frames=args.min_frames,
        write_report=True,
    )
    if not report["passed"]:
        print(
            "EPISODE_EPISODE_QUALITY_FAIL "
            + "; ".join(report["errors"])
        )
        return 1
    print(
        "EPISODE_EPISODE_QUALITY_PASS "
        f"frames={report['frame_count']} "
        f"gaps={report['frame_gaps']} "
        f"run_id={report['run_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


