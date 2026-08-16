"""PC-only training entry for the language-conditioned entity model.

The input labels come from frame ``Entities`` only during asv_training.  The
resulting model consumes a camera image and a real task embedding at runtime;
velocity is intentionally not a model output and is supplied later by the
temporal tracker. Do not run this module on Jetson: it is for Windows/PC data
processing, feature construction, training and evaluation only.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import json
import math
import platform
import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from asv_vla.image_entity_perception import (
    ENTITY_COUNT,
    ENTITY_IDS,
    FEATURE_DIM,
    FUSED_FEATURE_DIM,
    LANGUAGE_EMBEDDING_DIM,
    MODEL_INPUT_CONTRACT,
    MODEL_SCHEMA_VERSION,
    MODEL_VERSION,
    OUTPUT_DIM,
    POSITION_SCALE_M,
    extract_image_features,
    save_model,
)
from asv_vla.visual_encoder import (
    CameraProfile,
    enhance_low_light_image,
    project_target_to_pixel,
)


CAMERA_PROFILE = CameraProfile()

ACCEPTANCE_MIN_VISIBILITY_ACCURACY = 0.95
ACCEPTANCE_MAX_GEOMETRY_RMSE_M = 0.5
ACCEPTANCE_MIN_RUN_VISIBILITY_ACCURACY = 0.95
ACCEPTANCE_MAX_RUN_GEOMETRY_RMSE_M = 0.5
GEOMETRY_METRIC_MASK = "camera_projected_visibility_only"
LANGUAGE_MANIFEST_SCHEMA = "task_embedding_manifest_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_language_embeddings(
    embeddings_path: Path | None,
    manifest_path: Path | None,
    *,
    expected_model_id: str = "",
    expected_weights_sha256: str = "",
) -> dict[str, Any]:
    """Load an explicit instruction_id -> frozen embedding table."""

    if embeddings_path is None or manifest_path is None:
        raise RuntimeError(
            "LANGUAGE_EMBEDDINGS_REQUIRED: provide both --language-embeddings "
            "(.npy/.npz) and --language-manifest, or use "
            "--legacy-image-only explicitly"
        )
    manifest_path = Path(manifest_path).expanduser().resolve()
    embeddings_path = Path(embeddings_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"language manifest not found: {manifest_path}")
    if not embeddings_path.is_file():
        raise RuntimeError(f"language embeddings not found: {embeddings_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read language manifest: {exc}") from exc
    if manifest.get("schema_version") != LANGUAGE_MANIFEST_SCHEMA:
        raise RuntimeError(
            "LANGUAGE_MANIFEST_SCHEMA_MISMATCH: expected "
            f"{LANGUAGE_MANIFEST_SCHEMA}"
        )
    model_id = str(manifest.get("model_id", "")).strip()
    weights_sha256 = str(manifest.get("weights_sha256", "")).strip().lower()
    if not model_id:
        raise RuntimeError("LANGUAGE_MODEL_ID_MISSING in language manifest")
    if len(weights_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in weights_sha256
    ):
        raise RuntimeError(
            "LANGUAGE_MODEL_HASH_INVALID in language manifest; expected SHA-256"
        )
    if expected_model_id and expected_model_id != model_id:
        raise RuntimeError(
            f"LANGUAGE_MODEL_ID_MISMATCH: expected {expected_model_id!r}, "
            f"manifest has {model_id!r}"
        )
    if expected_weights_sha256 and expected_weights_sha256.lower() != weights_sha256:
        raise RuntimeError("LANGUAGE_MODEL_HASH_MISMATCH: CLI and manifest differ")
    if "embeddings_sha256" in manifest:
        expected_embeddings_sha256 = str(manifest["embeddings_sha256"]).lower()
        if _sha256_file(embeddings_path) != expected_embeddings_sha256:
            raise RuntimeError("LANGUAGE_EMBEDDINGS_FILE_HASH_MISMATCH")

    ids = [str(value).strip() for value in manifest.get("instruction_ids", [])]
    texts = [str(value) for value in manifest.get("instruction_texts", [])]
    if not ids and isinstance(manifest.get("instructions"), list):
        rows = manifest["instructions"]
        ids = [str(row.get("instruction_id", "")).strip() for row in rows]
        texts = [str(row.get("text", "")) for row in rows]
    try:
        loaded = np.load(embeddings_path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "embeddings" not in loaded:
                loaded.close()
                raise RuntimeError("language npz is missing 'embeddings'")
            array = np.asarray(loaded["embeddings"], dtype=np.float32)
            if not ids and "instruction_ids" in loaded:
                ids = [str(value).strip() for value in loaded["instruction_ids"]]
            loaded.close()
        else:
            array = np.asarray(loaded, dtype=np.float32)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"cannot load language embeddings: {exc}") from exc
    if not ids or any(not value for value in ids):
        raise RuntimeError("LANGUAGE_INSTRUCTION_IDS_MISSING")
    if len(ids) != len(set(ids)):
        raise RuntimeError("LANGUAGE_INSTRUCTION_IDS_DUPLICATED")
    if texts and len(texts) != len(ids):
        raise RuntimeError("LANGUAGE_INSTRUCTION_TEXTS_SHAPE_MISMATCH")
    if array.shape != (len(ids), LANGUAGE_EMBEDDING_DIM):
        raise RuntimeError(
            "LANGUAGE_EMBEDDING_SHAPE_MISMATCH: got "
            f"{array.shape}; expected ({len(ids)}, {LANGUAGE_EMBEDDING_DIM})"
        )
    if not np.all(np.isfinite(array)):
        raise RuntimeError("LANGUAGE_EMBEDDINGS_NONFINITE")
    norms = np.linalg.norm(array, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        raise RuntimeError("LANGUAGE_EMBEDDINGS_ZERO_NORM")
    text_to_id = {}
    if texts:
        for instruction_id, text in zip(ids, texts):
            if text:
                if text in text_to_id and text_to_id[text] != instruction_id:
                    raise RuntimeError("LANGUAGE_INSTRUCTION_TEXT_AMBIGUOUS")
                text_to_id[text] = instruction_id
    return {
        "by_id": {
            instruction_id: np.ascontiguousarray(array[index], dtype=np.float32)
            for index, instruction_id in enumerate(ids)
        },
        "text_to_id": text_to_id,
        "model_id": model_id,
        "weights_sha256": weights_sha256,
        "manifest_path": str(manifest_path),
        "embeddings_path": str(embeddings_path),
    }


def _resolve_instruction_id(
    record: dict[str, Any], episode_manifest: dict[str, Any], table: dict[str, Any]
) -> str | None:
    candidates = (
        record.get("instruction_id"),
        record.get("task", {}).get("instruction_id"),
        episode_manifest.get("instruction_id"),
        episode_manifest.get("task", {}).get("instruction_id"),
    )
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value in table["by_id"]:
            return value
    texts = (
        record.get("task", {}).get("text"),
        episode_manifest.get("task_text"),
        episode_manifest.get("task", {}).get("text"),
    )
    for text in texts:
        value = table["text_to_id"].get(str(text or "").strip())
        if value:
            return value
    return None


def _load_supervision_instruction_ids(
    supervision_root: Path,
    run_id: str,
) -> dict[int, tuple[str, ...]]:
    """Index the offline instruction labels associated with each image frame."""

    samples_path = Path(supervision_root) / run_id / "samples.jsonl"
    if not samples_path.is_file():
        raise RuntimeError(
            f"supervision samples not found for run {run_id}: {samples_path}"
        )
    by_frame: dict[int, list[str]] = {}
    seen: set[tuple[int, str]] = set()
    try:
        lines = samples_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read supervision samples: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{samples_path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        source = sample.get("source", {})
        instruction = sample.get("instruction", {})
        try:
            frame_index = int(source["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{samples_path}:{line_number}: missing frame_index"
            ) from exc
        instruction_id = str(instruction.get("instruction_id", "")).strip()
        if not instruction_id:
            raise RuntimeError(
                f"{samples_path}:{line_number}: missing instruction_id"
            )
        key = (frame_index, instruction_id)
        if key in seen:
            raise RuntimeError(
                f"{samples_path}:{line_number}: duplicate frame/instruction {key}"
            )
        seen.add(key)
        by_frame.setdefault(frame_index, []).append(instruction_id)
    if not by_frame:
        raise RuntimeError(f"supervision samples are empty: {samples_path}")
    return {frame: tuple(ids) for frame, ids in by_frame.items()}


def _augment_image(image: Image.Image) -> Image.Image:
    """Random photometric perturbation for the training split.

    The online windowed render puts the blue target at geometries the
    expert-driven collections undersample (e.g. the stationary 4.5 m / +0.4 m
    startup position, where the un-augmented detector drops to ~3% frame
    visibility while the red target at image centre stays near 100%).
    The augmentation must preserve image geometry because the target labels
    are not transformed alongside the image.
    """

    factor = random.uniform(0.88, 1.12)
    image = ImageEnhance.Brightness(image).enhance(factor)
    return image


def _read_samples(
    root: Path,
    *,
    max_primary_distance_m: float,
    max_abs_yaw_rad: float,
    max_abs_surge_velocity_mps: float,
    augment: bool = False,
    low_light: bool = False,
    low_light_gamma: float = 0.92,
    low_light_brightness: float = 1.04,
    low_light_contrast: float = 1.03,
    language_table: dict[str, Any] | None = None,
    supervision_root: Path | None = None,
    legacy_image_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], int, int, int]:
    if language_table is None and not legacy_image_only:
        raise RuntimeError(
            "LANGUAGE_EMBEDDINGS_REQUIRED: image-conditioned training needs "
            "a precomputed embedding table; use legacy_image_only=True only "
            "for an explicit old image-only artifact"
        )
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    run_ids: list[str] = []
    skipped_far = 0
    skipped_yaw = 0
    skipped_speed = 0
    for episode in sorted((p for p in root.iterdir() if p.is_dir())):
        manifest_path = episode / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest.get("run_id", episode.name))
        supervision_by_frame = (
            _load_supervision_instruction_ids(supervision_root, run_id)
            if language_table is not None and supervision_root is not None
            else None
        )
        print(f"read_samples episode={episode.name} augment={augment}", flush=True)
        frame_paths = sorted((episode / "frames").glob("*.json"))
        for frame_path in frame_paths:
            try:
                record = json.loads(frame_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # A corrupted/empty frame record from collection cannot
                # contribute a sample; skip it instead of failing the run.
                continue
            yaw = abs(float(record.get("ego", {}).get("rpy_ue_rad", [0.0, 0.0, 0.0])[2]))
            if yaw > max_abs_yaw_rad:
                skipped_yaw += 1
                continue
            surge_velocity = abs(
                float(record.get("ego", {}).get("surge_velocity_mps", 0.0))
            )
            if (
                not math.isfinite(surge_velocity)
                or surge_velocity > max_abs_surge_velocity_mps
            ):
                skipped_speed += 1
                continue
            distances = [
                math.hypot(
                    float(entity["relative_position_m"][0]),
                    float(entity["relative_position_m"][1]),
                )
                for entity in record["entities"]["items"]
            ]
            if not distances or min(distances) > max_primary_distance_m:
                skipped_far += 1
                continue
            instruction_ids: tuple[str | None, ...]
            if language_table is not None:
                if supervision_by_frame is not None:
                    indexed_ids = supervision_by_frame.get(int(record["frame_index"]))
                    if not indexed_ids:
                        raise RuntimeError(
                            f"{frame_path}: supervision has no instruction rows"
                        )
                    unknown = sorted(
                        set(indexed_ids) - set(language_table["by_id"])
                    )
                    if unknown:
                        raise RuntimeError(
                            f"{frame_path}: supervision references unknown instruction IDs: "
                            f"{unknown[:5]}"
                        )
                    instruction_ids = tuple(indexed_ids)
                else:
                    instruction_id = _resolve_instruction_id(
                        record, manifest, language_table
                    )
                    if instruction_id is None:
                        raise RuntimeError(
                            f"{frame_path}: no language embedding for instruction_id "
                            "or task text; refusing to generate a fallback"
                        )
                    instruction_ids = (instruction_id,)
            else:
                instruction_ids = (None,)
            image_path = episode / str(record["camera"]["image_path"])
            try:
                with Image.open(image_path) as image:
                    if low_light:
                        image = enhance_low_light_image(
                            image,
                            gamma=low_light_gamma,
                            brightness=low_light_brightness,
                            contrast=low_light_contrast,
                        )
                    if augment:
                        image = _augment_image(image)
                    image_features = extract_image_features(image)
            except (OSError, KeyError, ValueError) as exc:
                raise RuntimeError(f"cannot read {frame_path}: {exc}") from exc
            by_id = {
                str(entity["entity_id"]): entity
                for entity in record["entities"]["items"]
            }
            output = np.zeros(OUTPUT_DIM, dtype=np.float32)
            for slot, entity_id in enumerate(ENTITY_IDS):
                entity = by_id.get(entity_id)
                if entity is None:
                    raise RuntimeError(f"{frame_path}: missing {entity_id}")
                x, y, z = entity["relative_position_m"]
                visible = bool(entity.get("visible", False))
                try:
                    project_target_to_pixel(x, y, z, CAMERA_PROFILE)
                    in_view = True
                except Exception:
                    in_view = False
                # UE visibility and camera visibility are distinct.  The
                # image model is trained on the latter so it cannot emit a
                # high-confidence entity that is outside the camera image.
                visible = visible and in_view
                offset = slot * 4
                output[offset] = 1.0 if visible else -1.0
                output[offset + 1 : offset + 4] = np.asarray(
                    (x, y, z), dtype=np.float32
                ) / POSITION_SCALE_M
            for instruction_id in instruction_ids:
                if instruction_id is None:
                    features.append(image_features)
                else:
                    features.append(
                        np.concatenate(
                            (image_features, language_table["by_id"][instruction_id])
                        )
                    )
                targets.append(output.copy())
                run_ids.append(run_id)
    if not features:
        raise RuntimeError(f"no episode frames found under {root}")
    return (
        np.stack(features).astype(np.float32),
        np.stack(targets).astype(np.float32),
        run_ids,
        skipped_far,
        skipped_yaw,
        skipped_speed,
    )


def _ridge_fit(
    x: np.ndarray, y: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale = np.where(feature_scale < 1.0e-4, 1.0, feature_scale)
    normalized = (x - feature_mean) / feature_scale
    design = np.concatenate(
        [normalized, np.ones((len(normalized), 1), dtype=np.float32)], axis=1
    )
    # Ridge solution, choosing the cheaper normal equation side: with the
    # near-standoff collection the frame count (~24k) far exceeds the feature
    # dimension, so the dual N x N form would be O(N^3); the primal
    # (D+1) x (D+1) form is mathematically equivalent and seconds-fast.
    if len(design) > design.shape[1]:
        gram = design.T @ design
        gram.flat[:: gram.shape[0] + 1] += float(ridge)
        solution = np.linalg.solve(gram, design.T @ y)
    else:
        kernel = design @ design.T
        kernel.flat[:: kernel.shape[0] + 1] += float(ridge)
        alpha = np.linalg.solve(kernel, y)
        solution = design.T @ alpha
    return feature_mean, feature_scale, solution[:-1], solution[-1]


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    visibility = prediction[:, 0::4] >= 0.0
    target_visibility = target[:, 0::4] >= 0.0
    geometry_pred = prediction.reshape(-1, ENTITY_COUNT, 4)[:, :, 1:]
    geometry_target = target.reshape(-1, ENTITY_COUNT, 4)[:, :, 1:]
    geometry_mask = np.broadcast_to(target_visibility[:, :, None], geometry_pred.shape)
    geometry_error = (geometry_pred - geometry_target)[geometry_mask]
    rmse_normalized = float(
        np.sqrt(np.mean(geometry_error**2)) if geometry_error.size else 0.0
    )
    return {
        "visibility_accuracy": float(np.mean(visibility == target_visibility)),
        "geometry_rmse_normalized": rmse_normalized,
        "geometry_rmse_m": float(
            rmse_normalized * np.linalg.norm(POSITION_SCALE_M) / math.sqrt(3.0)
        ),
        "visible_geometry_rmse_normalized": rmse_normalized,
        "visible_geometry_rmse_m": float(
            rmse_normalized * np.linalg.norm(POSITION_SCALE_M) / math.sqrt(3.0)
        ),
        "visible_geometry_slots": int(np.sum(target_visibility)),
        "frames": float(len(target)),
    }


def _acceptance_gate(
    validation: dict[str, float | int],
    validation_by_run: dict[str, dict[str, float | int]],
    *,
    velocity_output: bool,
    geometry_metric_mask: str,
    min_visibility_accuracy: float = ACCEPTANCE_MIN_VISIBILITY_ACCURACY,
    max_geometry_rmse_m: float = ACCEPTANCE_MAX_GEOMETRY_RMSE_M,
    min_run_visibility_accuracy: float = ACCEPTANCE_MIN_RUN_VISIBILITY_ACCURACY,
    max_run_geometry_rmse_m: float = ACCEPTANCE_MAX_RUN_GEOMETRY_RMSE_M,
) -> dict[str, object]:
    """Evaluate the explicit, validation-only perception acceptance gate."""

    thresholds = {
        "min_visibility_accuracy": float(min_visibility_accuracy),
        "max_geometry_rmse_m": float(max_geometry_rmse_m),
        "min_run_visibility_accuracy": float(min_run_visibility_accuracy),
        "max_run_geometry_rmse_m": float(max_run_geometry_rmse_m),
    }
    checks = {
        "velocity_output_false": not velocity_output,
        "geometry_metric_mask": geometry_metric_mask == GEOMETRY_METRIC_MASK,
        "validation_visibility": float(validation["visibility_accuracy"])
        >= min_visibility_accuracy,
        "validation_geometry": float(validation["visible_geometry_rmse_m"])
        <= max_geometry_rmse_m,
    }
    run_results: dict[str, dict[str, object]] = {}
    for run_id, metrics in sorted(validation_by_run.items()):
        run_visibility = float(metrics["visibility_accuracy"])
        run_geometry = float(metrics["visible_geometry_rmse_m"])
        visible_slots = int(metrics["visible_geometry_slots"])
        run_passed = (
            visible_slots > 0
            and run_visibility >= min_run_visibility_accuracy
            and run_geometry <= max_run_geometry_rmse_m
        )
        run_results[run_id] = {
            "visibility_accuracy": run_visibility,
            "visible_geometry_rmse_m": run_geometry,
            "visible_geometry_slots": visible_slots,
            "passed": run_passed,
        }
    checks["validation_runs"] = bool(run_results) and all(
        bool(result["passed"]) for result in run_results.values()
    )
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed_checks,
        "thresholds": thresholds,
        "checks": checks,
        "validation_by_run": run_results,
        "failed_checks": failed_checks,
    }


def train(
    root: Path,
    output: Path,
    *,
    ridge: float = 300.0,
    max_primary_distance_m: float = 5.0,
    max_abs_yaw_rad: float = 0.1,
    max_abs_surge_velocity_mps: float = 1.0,
    acceptance_min_visibility_accuracy: float = ACCEPTANCE_MIN_VISIBILITY_ACCURACY,
    acceptance_max_geometry_rmse_m: float = ACCEPTANCE_MAX_GEOMETRY_RMSE_M,
    acceptance_min_run_visibility_accuracy: float = ACCEPTANCE_MIN_RUN_VISIBILITY_ACCURACY,
    acceptance_max_run_geometry_rmse_m: float = ACCEPTANCE_MAX_RUN_GEOMETRY_RMSE_M,
    low_light: bool = False,
    low_light_gamma: float = 0.92,
    low_light_brightness: float = 1.04,
    low_light_contrast: float = 1.03,
    brightness_augmentation: bool = False,
    language_embeddings: Path | None = None,
    language_manifest: Path | None = None,
    supervision_root: Path | None = None,
    language_model_id: str = "",
    language_weights_sha256: str = "",
    legacy_image_only: bool = False,
) -> dict[str, object]:
    if platform.machine().lower() in {"aarch64", "arm64"}:
        raise RuntimeError(
            "PC_ONLY_TRAINING: image perception data processing, feature cache, "
            "training and evaluation must run on Windows/PC; Jetson only loads "
            "the synchronized artifact and performs inference"
        )
    if language_embeddings is None and language_manifest is not None:
        raise RuntimeError("LANGUAGE_MANIFEST_WITHOUT_EMBEDDINGS")
    if language_embeddings is not None and language_manifest is None:
        raise RuntimeError("LANGUAGE_EMBEDDINGS_REQUIRE_MANIFEST")
    if legacy_image_only and language_embeddings is not None:
        raise RuntimeError("LEGACY_MODE_CANNOT_USE_LANGUAGE_EMBEDDINGS")
    language_table = (
        _load_language_embeddings(
            language_embeddings,
            language_manifest,
            expected_model_id=language_model_id,
            expected_weights_sha256=language_weights_sha256,
        )
        if language_embeddings is not None
        else None
    )
    if language_table is None and not legacy_image_only:
        raise RuntimeError(
            "LANGUAGE_EMBEDDINGS_REQUIRED: pass --language-embeddings and "
            "--language-manifest, or explicitly pass --legacy-image-only"
        )
    supervision_source = (
        None
        if supervision_root is None
        else Path(supervision_root).expanduser().resolve()
    )
    if language_table is not None and supervision_source is None:
        candidate = root.parent / "day10_supervised"
        if candidate.is_dir():
            supervision_source = candidate
        else:
            raise RuntimeError(
                "SUPERVISION_ROOT_REQUIRED: language-conditioned perception "
                "training needs per-frame instruction labels"
            )
    if max_primary_distance_m <= 0.0:
        raise ValueError("max_primary_distance_m must be positive")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    if max_abs_yaw_rad < 0.0:
        raise ValueError("max_abs_yaw_rad must be non-negative")
    if max_abs_surge_velocity_mps < 0.0:
        raise ValueError("max_abs_surge_velocity_mps must be non-negative")
    if not 0.0 <= acceptance_min_visibility_accuracy <= 1.0:
        raise ValueError("acceptance_min_visibility_accuracy must be in [0, 1]")
    if acceptance_max_geometry_rmse_m <= 0.0:
        raise ValueError("acceptance_max_geometry_rmse_m must be positive")
    if not 0.0 <= acceptance_min_run_visibility_accuracy <= 1.0:
        raise ValueError("acceptance_min_run_visibility_accuracy must be in [0, 1]")
    if acceptance_max_run_geometry_rmse_m <= 0.0:
        raise ValueError("acceptance_max_run_geometry_rmse_m must be positive")
    (
        x,
        y,
        run_ids,
        skipped_far,
        skipped_yaw,
        skipped_speed,
    ) = _read_samples(
        root,
        max_primary_distance_m=max_primary_distance_m,
        max_abs_yaw_rad=max_abs_yaw_rad,
        max_abs_surge_velocity_mps=max_abs_surge_velocity_mps,
        augment=False,
        low_light=low_light,
        low_light_gamma=low_light_gamma,
        low_light_brightness=low_light_brightness,
        low_light_contrast=low_light_contrast,
        language_table=language_table,
        supervision_root=supervision_source,
        legacy_image_only=legacy_image_only,
    )
    (
        x_aug,
        _,
        _,
        _,
        _,
        _,
    ) = _read_samples(
        root,
        max_primary_distance_m=max_primary_distance_m,
        max_abs_yaw_rad=max_abs_yaw_rad,
        max_abs_surge_velocity_mps=max_abs_surge_velocity_mps,
        augment=brightness_augmentation,
        low_light=low_light,
        low_light_gamma=low_light_gamma,
        low_light_brightness=low_light_brightness,
        low_light_contrast=low_light_contrast,
        language_table=language_table,
        supervision_root=supervision_source,
        legacy_image_only=legacy_image_only,
    )
    unique_runs = sorted(set(run_ids))
    if len(unique_runs) < 2:
        raise RuntimeError("at least two runs are required for a group split")
    validation_runs = set(unique_runs[::5] or unique_runs[-1:])
    train_mask = np.asarray([run not in validation_runs for run in run_ids])
    val_mask = ~train_mask
    # The fixed low-light transform is the input contract. Optional random
    # brightness jitter is separate and disabled for the calibrated run.
    mean, scale, weights, bias = _ridge_fit(x_aug[train_mask], y[train_mask], ridge)
    prediction = ((x - mean) / scale) @ weights + bias
    run_array = np.asarray(run_ids)
    validation_by_run = {
        run: _metrics(
            prediction[val_mask & (run_array == run)],
            y[val_mask & (run_array == run)],
        )
        for run in sorted(validation_runs)
    }
    acceptance_gate = _acceptance_gate(
        _metrics(prediction[val_mask], y[val_mask]),
        validation_by_run,
        velocity_output=False,
        geometry_metric_mask=GEOMETRY_METRIC_MASK,
        min_visibility_accuracy=acceptance_min_visibility_accuracy,
        max_geometry_rmse_m=acceptance_max_geometry_rmse_m,
        min_run_visibility_accuracy=acceptance_min_run_visibility_accuracy,
        max_run_geometry_rmse_m=acceptance_max_run_geometry_rmse_m,
    )
    report = {
        "model_version": MODEL_VERSION if not legacy_image_only else "image_entity_ridge_v2",
        "model_schema_version": (
            MODEL_SCHEMA_VERSION if not legacy_image_only else "image_only_legacy_schema"
        ),
        "entity_ids": list(ENTITY_IDS),
        "input_shape": [18, 32, 7],
        "feature_dim": FUSED_FEATURE_DIM if not legacy_image_only else FEATURE_DIM,
        "feature_contract": MODEL_INPUT_CONTRACT if not legacy_image_only else "(camera_image_rgb)->structured_entities",
        "language_embedding_dim": LANGUAGE_EMBEDDING_DIM if not legacy_image_only else 0,
        "language_model_id": language_table["model_id"] if language_table else "",
        "language_weights_sha256": language_table["weights_sha256"] if language_table else "",
        "language_manifest": language_table["manifest_path"] if language_table else "",
        "language_embeddings": language_table["embeddings_path"] if language_table else "",
        "supervision_root": str(supervision_source) if supervision_source else "",
        "language_sample_source": (
            "offline_supervision_samples_per_frame"
            if language_table is not None
            else "single_image_sample"
        ),
        "legacy_image_only": legacy_image_only,
        "label_source": "frame_record_v1.entities",
        "velocity_output": False,
        "geometry_metric_mask": GEOMETRY_METRIC_MASK,
        "train_runs": sorted(set(np.asarray(run_ids)[train_mask].tolist())),
        "validation_runs": sorted(validation_runs),
        "train": _metrics(prediction[train_mask], y[train_mask]),
        "validation": _metrics(prediction[val_mask], y[val_mask]),
        "validation_by_run": validation_by_run,
        "ridge": ridge,
        "max_primary_distance_m": max_primary_distance_m,
        "skipped_far_frames": skipped_far,
        "max_abs_yaw_rad": max_abs_yaw_rad,
        "skipped_yaw_frames": skipped_yaw,
        "max_abs_surge_velocity_mps": max_abs_surge_velocity_mps,
        "skipped_speed_frames": skipped_speed,
        "low_light_preprocess": {
            "enabled": low_light,
            "gamma": low_light_gamma,
            "brightness": low_light_brightness,
            "contrast": low_light_contrast,
        },
        "brightness_augmentation": brightness_augmentation,
        "acceptance_gate": acceptance_gate,
        "acceptance_ready": bool(acceptance_gate["passed"]),
        "acceptance_note": (
            "Validation metrics satisfy the configured image-perception gate."
            if acceptance_gate["passed"]
            else "Acceptance gate failed: "
            + ", ".join(acceptance_gate["failed_checks"])
        ),
    }
    save_model(
        output,
        feature_mean=mean,
        feature_scale=scale,
        weights=weights,
        bias=bias,
        model_version=MODEL_VERSION if not legacy_image_only else "image_entity_ridge_v2",
        schema_version=(
            MODEL_SCHEMA_VERSION if not legacy_image_only else "image_only_legacy_schema"
        ),
        input_contract=(
            MODEL_INPUT_CONTRACT
            if not legacy_image_only
            else "(camera_image_rgb)->structured_entities"
        ),
        task_embedding_dim=LANGUAGE_EMBEDDING_DIM if not legacy_image_only else 0,
        language_model_id=language_table["model_id"] if language_table else "",
        language_weights_sha256=language_table["weights_sha256"] if language_table else "",
        velocity_output=False,
        metadata=report,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=300.0)
    parser.add_argument("--max-primary-distance-m", type=float, default=5.0)
    parser.add_argument("--max-abs-yaw-rad", type=float, default=0.1)
    parser.add_argument(
        "--acceptance-min-visibility-accuracy",
        type=float,
        default=ACCEPTANCE_MIN_VISIBILITY_ACCURACY,
    )
    parser.add_argument(
        "--acceptance-max-geometry-rmse-m",
        type=float,
        default=ACCEPTANCE_MAX_GEOMETRY_RMSE_M,
    )
    parser.add_argument(
        "--acceptance-min-run-visibility-accuracy",
        type=float,
        default=ACCEPTANCE_MIN_RUN_VISIBILITY_ACCURACY,
    )
    parser.add_argument(
        "--acceptance-max-run-geometry-rmse-m",
        type=float,
        default=ACCEPTANCE_MAX_RUN_GEOMETRY_RMSE_M,
    )
    parser.add_argument(
        "--max-abs-surge-velocity-mps", type=float, default=1.0
    )
    parser.add_argument(
        "--low-light", action="store_true",
        help="apply the fixed low-light transform before feature extraction",
    )
    parser.add_argument("--low-light-gamma", type=float, default=0.92)
    parser.add_argument("--low-light-brightness", type=float, default=1.04)
    parser.add_argument("--low-light-contrast", type=float, default=1.03)
    parser.add_argument(
        "--brightness-augmentation",
        action="store_true",
        help="add random brightness jitter after the fixed low-light transform",
    )
    parser.add_argument(
        "--language-embeddings",
        type=Path,
        help="PC-generated .npy/.npz table of float32 task embeddings",
    )
    parser.add_argument(
        "--language-manifest",
        type=Path,
        help="manifest mapping instruction_id to the precomputed embeddings",
    )
    parser.add_argument(
        "--supervision-root",
        type=Path,
        help="root containing one samples.jsonl directory per episode Run",
    )
    parser.add_argument("--language-model-id", default="")
    parser.add_argument("--language-weights-sha256", default="")
    parser.add_argument(
        "--legacy-image-only",
        action="store_true",
        help="explicitly train/save the old image-only schema for migration only",
    )
    args = parser.parse_args()
    report = train(
        args.episodes,
        args.output,
        ridge=args.ridge,
        max_primary_distance_m=args.max_primary_distance_m,
        max_abs_yaw_rad=args.max_abs_yaw_rad,
        max_abs_surge_velocity_mps=args.max_abs_surge_velocity_mps,
        acceptance_min_visibility_accuracy=args.acceptance_min_visibility_accuracy,
        acceptance_max_geometry_rmse_m=args.acceptance_max_geometry_rmse_m,
        acceptance_min_run_visibility_accuracy=args.acceptance_min_run_visibility_accuracy,
        acceptance_max_run_geometry_rmse_m=args.acceptance_max_run_geometry_rmse_m,
        low_light=args.low_light,
        low_light_gamma=args.low_light_gamma,
        low_light_brightness=args.low_light_brightness,
        low_light_contrast=args.low_light_contrast,
        brightness_augmentation=args.brightness_augmentation,
        language_embeddings=args.language_embeddings,
        language_manifest=args.language_manifest,
        supervision_root=args.supervision_root,
        language_model_id=args.language_model_id,
        language_weights_sha256=args.language_weights_sha256,
        legacy_image_only=args.legacy_image_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
