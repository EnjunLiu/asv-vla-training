"""FrameRecord v1 JSON schema, semantic validation, and atomic I/O."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


SCHEMA_VERSION = "frame_record_v1"
CAMERA_WIDTH_PX = 1280
CAMERA_HEIGHT_PX = 720
CAMERA_FOV_ANGLE_DEG = 90.0
CAMERA_MOUNT_POSITION_M = (0.42, 0.0, 0.20)
CAMERA_MOUNT_RPY_UE_DEG = (0.0, -5.0, 0.0)
MODALITIES = ("task", "ego", "camera", "entities")


class FrameRecordError(ValueError):
    """Raised when a record violates the frozen v1 contract."""


def _schema_path() -> Path:
    package_candidate = Path(__file__).resolve().parent / "schema" / "frame_record_v1.schema.json"
    if package_candidate.is_file():
        return package_candidate

    source_path = Path(__file__).resolve().parents[1] / "schema"
    source_candidate = source_path / "frame_record_v1.schema.json"
    if source_candidate.is_file():
        return source_candidate

    try:
        from ament_index_python.packages import get_package_share_directory

        share_path = Path(get_package_share_directory("asv_vla"))
    except Exception as exc:
        raise FrameRecordError("FrameRecord v1 schema is unavailable") from exc

    installed_candidate = share_path / "schema" / "frame_record_v1.schema.json"
    if not installed_candidate.is_file():
        raise FrameRecordError(
            f"FrameRecord v1 schema is missing: {installed_candidate}"
        )
    return installed_candidate


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    return json.loads(_schema_path().read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _reject_constant(value: str) -> None:
    raise FrameRecordError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise FrameRecordError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _find_nonfinite(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path} must be finite")
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_find_nonfinite(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_find_nonfinite(child, f"{path}[{index}]"))
    return errors


def _format_schema_error(error: Any) -> str:
    path = "$"
    for part in error.absolute_path:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return f"{path}: {error.message}"


def _close_vector(
    actual: list[float],
    expected: tuple[float, float, float],
    tolerance: float = 1e-9,
) -> bool:
    try:
        return len(actual) == 3 and all(
            math.isclose(float(value), target, abs_tol=tolerance)
            for value, target in zip(actual, expected)
        )
    except (TypeError, ValueError):
        return False


def validate_frame_record(
    record: Any,
    *,
    image_root: Path | None = None,
) -> None:
    """Validate schema plus cross-field and data-quality invariants."""

    schema_errors = sorted(
        _schema_validator().iter_errors(record),
        key=lambda error: list(error.absolute_path),
    )
    errors = [_format_schema_error(error) for error in schema_errors]
    errors.extend(_find_nonfinite(record))

    if isinstance(record, dict):
        mask = record.get("modality_mask")
        if isinstance(mask, dict):
            for modality in MODALITIES:
                block = record.get(modality)
                if (
                    modality in mask
                    and isinstance(block, dict)
                    and isinstance(block.get("valid"), bool)
                    and mask[modality] != block["valid"]
                ):
                    errors.append(
                        f"$.modality_mask.{modality} must equal "
                        f"$.{modality}.valid"
                    )

            if all(isinstance(mask.get(name), bool) for name in MODALITIES):
                expected_valid = all(mask[name] for name in MODALITIES)
                if record.get("valid") != expected_valid:
                    errors.append("$.valid must equal all modality_mask values")

        stamp_us = record.get("stamp_us")
        if isinstance(stamp_us, int) and not isinstance(stamp_us, bool):
            for modality in ("ego", "camera", "entities"):
                block = record.get(modality)
                if (
                    isinstance(block, dict)
                    and block.get("valid") is True
                    and block.get("stamp_us") != stamp_us
                ):
                    errors.append(
                        f"$.{modality}.stamp_us must equal $.stamp_us"
                    )

            task = record.get("task")
            if (
                isinstance(task, dict)
                and isinstance(task.get("stamp_us"), int)
                and task["stamp_us"] > stamp_us
            ):
                errors.append("$.task.stamp_us must not be in the future")

        camera = record.get("camera")
        if isinstance(camera, dict):
            image_path = camera.get("image_path")
            camera_valid = camera.get("valid")
            if camera_valid is True and not image_path:
                errors.append("$.camera.image_path is required when camera is valid")
            if camera_valid is False and image_path is not None:
                errors.append("$.camera.image_path must be null when camera is invalid")

            if isinstance(image_path, str):
                relative_path = Path(image_path)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    errors.append("$.camera.image_path must be a safe relative path")
                elif image_root is not None and not (
                    Path(image_root) / relative_path
                ).is_file():
                    errors.append("$.camera.image_path does not exist")

            mount_position = camera.get("mount_position_m")
            if isinstance(mount_position, list) and not _close_vector(
                mount_position, CAMERA_MOUNT_POSITION_M
            ):
                errors.append("$.camera.mount_position_m violates Day 4 profile")

            mount_rotation = camera.get("mount_rpy_ue_deg")
            if isinstance(mount_rotation, list) and not _close_vector(
                mount_rotation, CAMERA_MOUNT_RPY_UE_DEG
            ):
                errors.append("$.camera.mount_rpy_ue_deg violates Day 4 profile")

        entities = record.get("entities")
        if isinstance(entities, dict) and isinstance(entities.get("items"), list):
            entity_ids = [
                item.get("entity_id")
                for item in entities["items"]
                if isinstance(item, dict)
                and isinstance(item.get("entity_id"), str)
            ]
            if len(entity_ids) != len(set(entity_ids)):
                errors.append("$.entities.items contains duplicate entity_id")

        task = record.get("task")
        if isinstance(task, dict):
            text = task.get("text")
            if task.get("valid") is True and (
                not isinstance(text, str) or not text.strip()
            ):
                errors.append("$.task.text must be non-empty when task is valid")
            if task.get("valid") is False and text != "":
                errors.append("$.task.text must be empty when task is invalid")

        if record.get("valid") is True and record.get("detail") != "ok":
            errors.append("$.detail must be 'ok' when record is valid")
        if record.get("valid") is False and record.get("detail") == "ok":
            errors.append("$.detail must explain why the record is invalid")

    if errors:
        raise FrameRecordError("; ".join(dict.fromkeys(errors)))


def read_frame_record(
    path: str | Path,
    *,
    image_root: Path | None = None,
) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            record = json.load(
                stream,
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise FrameRecordError(f"failed to read {source}: {exc}") from exc

    validate_frame_record(record, image_root=image_root)
    return record


def write_frame_record(
    path: str | Path,
    record: dict[str, Any],
    *,
    image_root: Path | None = None,
) -> None:
    destination = Path(path)
    validate_frame_record(record, image_root=image_root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a FrameRecord v1 JSON file.")
    parser.add_argument("record", type=Path)
    parser.add_argument("--image-root", type=Path)
    args = parser.parse_args()

    try:
        record = read_frame_record(args.record, image_root=args.image_root)
    except FrameRecordError as exc:
        print(f"FRAME_RECORD_INVALID: {exc}")
        return 1

    print(
        "FRAME_RECORD_VALID "
        f"run_id={record['run_id']} frame_index={record['frame_index']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

