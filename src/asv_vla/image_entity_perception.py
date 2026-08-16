"""Image-only entity geometry model used by the online VLA path.

This is deliberately small and auditable: the PC trainer learns a multi-output
ridge regressor from RGB image tiles to the calibrated relative geometry of
four canonical boat slots.  At runtime JPEG/PIL decoding stays on the host,
while feature construction, normalization and the learned projection run on
the requested CUDA device.  UE ``Entities`` never enter this module; they are
used by the trainer as supervision labels only.

It is a first perception model, not a claim of general-purpose object
detection.  The manifest records its data split and metrics, and a later
detector can replace this file without changing the ROS topic contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
from PIL import Image


LEGACY_MODEL_VERSION = "image_entity_ridge_v1"
IMAGE_ONLY_MODEL_VERSION = "image_entity_ridge_v2"
MODEL_VERSION = "image_entity_ridge_language_v3"
MODEL_SCHEMA_VERSION = "image_entity_perception_schema_v3"
MODEL_INPUT_CONTRACT = (
    "(camera_image_rgb,task_embedding_float32[256])->structured_entities"
)
LANGUAGE_EMBEDDING_DIM = 256
LANGUAGE_EMBEDDING_CONTRACT = (
    "task_embedding:float32[256];finite;nonzero_l2;normalized_by_language_encoder"
)
STRUCTURED_ENTITY_OUTPUT_CONTRACT = (
    "entity_id,class_name,color,relative_position_m,visible_mask,bbox_px,"
    "confidence,valid,relative_velocity_mps,velocity_valid"
)
VELOCITY_SOURCE = "temporal_entity_tracker"
GRID_WIDTH = 32
GRID_HEIGHT = 18
CHANNELS = 7  # RGB plus red/blue/white/bright spatial evidence maps
BASE_FEATURE_DIM = GRID_WIDTH * GRID_HEIGHT * CHANNELS
MOMENT_MAPS = 4
MOMENT_FEATURES_PER_MAP = 8
FEATURE_DIM = BASE_FEATURE_DIM + MOMENT_MAPS * MOMENT_FEATURES_PER_MAP
FUSED_FEATURE_DIM = FEATURE_DIM + LANGUAGE_EMBEDDING_DIM
ENTITY_IDS = ("target_red", "target_blue", "target_left", "target_right")
ENTITY_COUNT = len(ENTITY_IDS)
OUTPUT_DIM = ENTITY_COUNT * 4  # visible logit + relative x/y/z per slot
POSITION_SCALE_M = np.asarray((40.0, 40.0, 5.0), dtype=np.float32)
COLOR_CALIBRATED_MODEL_VERSION = "image_entity_color_calibrated_v1"
COLOR_CALIBRATED_MODEL_VERSION_V2 = IMAGE_ONLY_MODEL_VERSION
IMAGE_ONLY_MODEL_VERSIONS = frozenset(
    {
        LEGACY_MODEL_VERSION,
        IMAGE_ONLY_MODEL_VERSION,
        COLOR_CALIBRATED_MODEL_VERSION,
        COLOR_CALIBRATED_MODEL_VERSION_V2,
    }
)
COLOR_CALIBRATION_WIDTH = 320
COLOR_CALIBRATION_HEIGHT = 180
# The image model is retrained against this exact transform. Keep these
# values in the perception contract rather than exposing per-node tuning.
LOW_LIGHT_PREPROCESS_CONTRACT = "ue5_capture_gamma065_brightness100_contrast100_v2"
# The UE5 bridge applies this transform before publishing JPEG bytes. Applying
# it again in the Jetson node would double-lift the image and change the color
# margins used by the calibrated visibility checks.
LOW_LIGHT_PREPROCESS_ENABLED = False
LOW_LIGHT_PREPROCESS_GAMMA = 0.65
LOW_LIGHT_PREPROCESS_BRIGHTNESS = 1.0
LOW_LIGHT_PREPROCESS_CONTRAST = 1.0
# Fit on the available near-range S2 red masks.  The form is intentionally
# explicit so the PC calibration script can replace these values in a model
# artifact without changing the online image-only contract.
COLOR_X_COEFFICIENTS = (0.63521458, 0.15866379)
# The target grows substantially in the near field.  Keep the lower bound
# above isolated pixel noise, but do not discard a valid target merely because
# its image component is larger than the original far-field calibration crop.
COLOR_AREA_MIN = 0.0001
COLOR_AREA_MAX = 0.08


class ImageEntityPerceptionError(RuntimeError):
    """Raised when the image perception model or input is unusable."""


def validate_task_embedding(
    embedding: object,
    *,
    expected_dim: int = LANGUAGE_EMBEDDING_DIM,
) -> np.ndarray:
    """Validate the task condition; never create a fallback vector."""

    array = np.asarray(embedding, dtype=np.float32)
    if array.shape != (expected_dim,):
        raise ImageEntityPerceptionError(
            f"task embedding shape {array.shape}; expected ({expected_dim},)"
        )
    if not np.all(np.isfinite(array)):
        raise ImageEntityPerceptionError("task embedding contains NaN or Inf")
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ImageEntityPerceptionError("task embedding has zero or invalid norm")
    return np.ascontiguousarray(array, dtype=np.float32)


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value).strip()))


@dataclass(frozen=True)
class TaskSpec:
    """Small deterministic task contract used by image-only selection."""

    instruction: str
    instruction_id: str
    action: str
    color: str = ""
    bearing: str = ""
    valid: bool = False

    @property
    def is_stop(self) -> bool:
        return self.action == "stop"

    @property
    def is_follow(self) -> bool:
        return self.action == "follow"


def parse_task_instruction(instruction: object) -> TaskSpec:
    """Parse the minimal runtime task vocabulary without using UE truth.

    The parser deliberately accepts both the English test vocabulary and the
    Chinese task text used by the online demo. Unknown or empty text is
    invalid and therefore makes the perception output fail closed.
    """

    text = " ".join(str(instruction).strip().split())
    folded = text.casefold()
    if not text:
        return TaskSpec("", "unknown", "unknown")

    stop_tokens = ("stop", "halt", "hold", "emergency", "停", "停止", "急停")
    if any(token in folded for token in stop_tokens):
        return TaskSpec(text, "stop", "stop", valid=True)

    color = ""
    if any(token in folded for token in ("red", "红", "紅")):
        color = "red"
    elif any(token in folded for token in ("blue", "蓝", "藍")):
        color = "blue"

    bearing = ""
    if any(token in folded for token in ("left", "左")):
        bearing = "left"
    elif any(token in folded for token in ("right", "右")):
        bearing = "right"

    follow_tokens = (
        "follow",
        "track",
        "target",
        "跟随",
        "跟住",
        "跟踪",
        "追踪",
        "锁定",
        "鎖定",
        "驶向",
    )
    if (color or bearing) and any(token in folded for token in follow_tokens):
        selector = color or bearing
        if color and bearing:
            selector = f"{color}_{bearing}"
        return TaskSpec(
            text,
            f"follow_{selector}",
            "follow",
            color=color,
            bearing=bearing,
            valid=True,
        )
    return TaskSpec(text, "unknown", "unknown")


def _largest_color_component_in_area(
    mask: np.ndarray,
) -> tuple[int, float, float]:
    """Return the largest 8-connected component whose area is within
    [COLOR_AREA_MIN, COLOR_AREA_MAX].

    Background clutter (sky/water) often forms a huge component that exceeds
    the calibration area cap; picking the largest in-range component keeps
    the calibration robust to background while excluding noise.  Returns
    (0, nan, nan) when no component is in range.
    """

    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    best_count = 0
    best_sum_x = 0.0
    best_sum_y = 0.0
    for row, column in zip(*np.where(mask)):
        row = int(row)
        column = int(column)
        if visited[row, column]:
            continue
        visited[row, column] = True
        stack = [(row, column)]
        count = 0
        sum_x = 0.0
        sum_y = 0.0
        while stack:
            current_row, current_column = stack.pop()
            count += 1
            sum_x += current_column
            sum_y += current_row
            for delta_row in (-1, 0, 1):
                for delta_column in (-1, 0, 1):
                    if delta_row == 0 and delta_column == 0:
                        continue
                    neighbor_row = current_row + delta_row
                    neighbor_column = current_column + delta_column
                    if not (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                    ):
                        continue
                    if mask[neighbor_row, neighbor_column] and not visited[
                        neighbor_row, neighbor_column
                    ]:
                        visited[neighbor_row, neighbor_column] = True
                        stack.append((neighbor_row, neighbor_column))
        area = count / float(height * width)
        if COLOR_AREA_MIN <= area <= COLOR_AREA_MAX and count > best_count:
            best_count = count
            best_sum_x = sum_x
            best_sum_y = sum_y
    if best_count == 0:
        return 0, float("nan"), float("nan")
    return best_count, best_sum_x / best_count, best_sum_y / best_count


def calibrated_color_geometry(
    image: Image.Image | np.ndarray,
    color: str,
) -> tuple[bool, float, float, float, tuple[float, float]]:
    """Estimate red/blue-target geometry from RGB only.

    The returned area is the fraction of the fixed 320x180 grid occupied by
    the largest color component, and the centroid is in that same grid.  A
    missing or out-of-calibration component returns ``valid=False`` with NaN
    geometry; callers must fail closed rather than expose ridge geometry.
    """

    normalized_color = str(color).strip().casefold()
    if normalized_color not in {"red", "blue"}:
        raise ImageEntityPerceptionError(
            f"unsupported color calibration target {color!r}; use red or blue"
        )
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ImageEntityPerceptionError(
                f"expected HxWx3 image, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ImageEntityPerceptionError("image array contains non-finite values")
        if np.issubdtype(array.dtype, np.floating) and np.max(array) <= 1.0:
            array = array * 255.0
        image = Image.fromarray(
            np.clip(array, 0.0, 255.0).astype(np.uint8), mode="RGB"
        )
    if not isinstance(image, Image.Image):
        raise ImageEntityPerceptionError("image must be a PIL image or RGB array")
    grid = np.asarray(
        image.convert("RGB").resize(
            (COLOR_CALIBRATION_WIDTH, COLOR_CALIBRATION_HEIGHT),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    ) / 255.0
    red, green, blue = grid.transpose(2, 0, 1)
    primary = red if normalized_color == "red" else blue
    secondary = np.maximum(red if normalized_color == "blue" else blue, green)
    # Keep the red and blue contracts on the same component/geometry path;
    # only the selected primary channel changes. Area filtering below rejects
    # large background regions without relying on a scene-specific horizon or
    # camera crop.
    dominant_mask = (
        (primary >= 0.25)
        & (primary >= secondary * 1.25)
        & ((primary - secondary) >= 0.08)
    )
    if normalized_color == "blue":
        # Blue materials can be recorded as low-value cyan. This is a generic
        # RGB color-family fallback, not a location or scene assumption.
        cyan_mask = (
            (green >= red * 1.6)
            & (blue >= red * 1.6)
            & (green >= 0.08)
            & (blue >= 0.08)
            & (np.abs(blue - green) <= (32.0 / 255.0))
        )
        component_pixels, centroid_x, centroid_y = (
            _largest_color_component_in_area(cyan_mask)
        )
        if component_pixels == 0:
            component_pixels, centroid_x, centroid_y = (
                _largest_color_component_in_area(dominant_mask)
            )
    else:
        component_pixels, centroid_x, centroid_y = (
            _largest_color_component_in_area(dominant_mask)
        )
    area = component_pixels / float(COLOR_CALIBRATION_WIDTH * COLOR_CALIBRATION_HEIGHT)
    if component_pixels < 8 or not (COLOR_AREA_MIN <= area <= COLOR_AREA_MAX):
        return False, float("nan"), float("nan"), float(area), (
            float(centroid_x),
            float(centroid_y),
        )
    inverse_sqrt_area = 1.0 / math.sqrt(area)
    relative_x = COLOR_X_COEFFICIENTS[0] + COLOR_X_COEFFICIENTS[1] * inverse_sqrt_area
    # A centroid left of image centre is positive +Y in the ROS base_link
    # convention.  The 0.93 factor is the frozen camera horizontal scale.
    relative_y = (
        0.93
        * relative_x
        * (0.5 - centroid_x / COLOR_CALIBRATION_WIDTH)
        * 2.0
    )
    return (
        bool(math.isfinite(relative_x) and math.isfinite(relative_y)),
        float(relative_x),
        float(relative_y),
        float(area),
        (float(centroid_x), float(centroid_y)),
    )


def calibrated_red_geometry(
    image: Image.Image | np.ndarray,
) -> tuple[bool, float, float, float, tuple[float, float]]:
    """Backward-compatible red-only alias for the unified color calibrator."""

    return calibrated_color_geometry(image, "red")


def _resized_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL image or array to a finite, resized RGB float array."""
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ImageEntityPerceptionError(
                f"expected HxWx3 image, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ImageEntityPerceptionError("image array contains non-finite values")
        if np.issubdtype(array.dtype, np.floating) and np.max(array) <= 1.0:
            array = array * 255.0
        image = Image.fromarray(
            np.clip(array, 0.0, 255.0).astype(np.uint8), mode="RGB"
        )
    if not isinstance(image, Image.Image):
        raise ImageEntityPerceptionError("image must be a PIL image or RGB array")
    return np.asarray(
        image.convert("RGB").resize(
            (GRID_WIDTH, GRID_HEIGHT), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0


def _decoded_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return decoded RGB bytes before device-side feature construction."""

    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ImageEntityPerceptionError(
                f"expected HxWx3 image, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ImageEntityPerceptionError("image array contains non-finite values")
        if np.issubdtype(array.dtype, np.floating) and np.max(array) <= 1.0:
            array = array * 255.0
        return np.ascontiguousarray(
            np.clip(array, 0.0, 255.0).astype(np.uint8)
        )
    if not isinstance(image, Image.Image):
        raise ImageEntityPerceptionError("image must be a PIL image or RGB array")
    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def _extract_torch_image_features(
    image: Image.Image | np.ndarray,
    torch: Any,
    *,
    device: str,
    model_version: str,
):
    """Construct the feature map and moments on the requested CUDA device.

    The resize runs on the host with the same PIL BILINEAR path the trainer
    uses: torch interpolate differs numerically from PIL, and the tiny
    evidence-map differences get amplified by the moment denominators
    (observed up to 0.66 on one frame), which shifted the trained ridge
    outputs out of distribution at inference.
    """

    resized = torch.as_tensor(
        _resized_rgb(image), dtype=torch.float32, device=device
    )

    red = torch.clamp(
        resized[..., 0]
        - torch.maximum(resized[..., 1], resized[..., 2]),
        min=0.0,
    )
    blue = torch.clamp(
        resized[..., 2]
        - torch.maximum(resized[..., 0], resized[..., 1]),
        min=0.0,
    )
    brightness = resized.mean(dim=-1)
    saturation = resized.amax(dim=-1) - resized.amin(dim=-1)
    white = torch.clamp(brightness - 1.5 * saturation, min=0.0, max=1.0)
    bright = torch.clamp(brightness - 0.45, min=0.0, max=1.0)
    spatial = torch.cat(
        (
            resized,
            red.unsqueeze(-1),
            blue.unsqueeze(-1),
            white.unsqueeze(-1),
            bright.unsqueeze(-1),
        ),
        dim=-1,
    )
    if model_version == LEGACY_MODEL_VERSION:
        features = spatial.reshape(-1)
    else:
        x_coordinates = torch.linspace(
            -1.0,
            1.0,
            GRID_WIDTH,
            dtype=torch.float32,
            device=device,
        )[None, :]
        y_coordinates = torch.linspace(
            0.0,
            1.0,
            GRID_HEIGHT,
            dtype=torch.float32,
            device=device,
        )[:, None]
        moments = []
        for evidence in (red, blue, white, bright):
            total = evidence.sum()
            denominator = torch.clamp(total, min=1.0e-6)
            center_x = (evidence * x_coordinates).sum() / denominator
            center_y = (evidence * y_coordinates).sum() / denominator
            moments.extend(
                (
                    torch.log1p(total),
                    center_x,
                    center_y,
                    (evidence * (x_coordinates - center_x) ** 2).sum()
                    / denominator,
                    (evidence * (y_coordinates - center_y) ** 2).sum()
                    / denominator,
                    evidence.amax(),
                    evidence.mean(),
                    (evidence > 0.08).to(dtype=torch.float32).mean(),
                )
            )
        features = torch.cat((spatial.reshape(-1), torch.stack(moments)))
    # Language-conditioned models append the 256-D task embedding after this
    # helper returns. Validate only the image branch here; checking the fused
    # dimension at this stage rejects every real CUDA frame.
    expected = (
        BASE_FEATURE_DIM
        if model_version == LEGACY_MODEL_VERSION
        else FEATURE_DIM
    )
    if tuple(features.shape) != (expected,) or not bool(torch.isfinite(features).all()):
        raise ImageEntityPerceptionError(
            "CUDA image feature vector is invalid"
        )
    return features.contiguous()


def _extract_legacy_image_features(image: Image.Image | np.ndarray) -> np.ndarray:
    """Extract the v1 RGB/evidence vector for deployed model compatibility."""

    result = _resized_rgb(image)
    red = np.maximum(
        result[..., 0] - np.maximum(result[..., 1], result[..., 2]), 0.0
    )
    blue = np.maximum(
        result[..., 2] - np.maximum(result[..., 0], result[..., 1]), 0.0
    )
    brightness = np.mean(result, axis=-1)
    saturation = np.max(result, axis=-1) - np.min(result, axis=-1)
    white = np.clip(brightness - 1.5 * saturation, 0.0, 1.0)
    bright = np.clip(brightness - 0.45, 0.0, 1.0)
    result = np.concatenate(
        [result, red[..., None], blue[..., None], white[..., None], bright[..., None]],
        axis=-1,
    )
    result = np.ascontiguousarray(result.reshape(-1), dtype=np.float32)
    if result.shape != (BASE_FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ImageEntityPerceptionError("image feature vector is invalid")
    return result


def extract_image_features(image: Image.Image | np.ndarray) -> np.ndarray:
    """Extract spatial evidence plus color/area moments for v2."""

    result = _resized_rgb(image)
    red = np.maximum(
        result[..., 0] - np.maximum(result[..., 1], result[..., 2]), 0.0
    )
    blue = np.maximum(
        result[..., 2] - np.maximum(result[..., 0], result[..., 1]), 0.0
    )
    brightness = np.mean(result, axis=-1)
    saturation = np.max(result, axis=-1) - np.min(result, axis=-1)
    white = np.clip(brightness - 1.5 * saturation, 0.0, 1.0)
    bright = np.clip(brightness - 0.45, 0.0, 1.0)
    maps = (red, blue, white, bright)
    spatial = np.concatenate(
        [
            result,
            red[..., None],
            blue[..., None],
            white[..., None],
            bright[..., None],
        ],
        axis=-1,
    )

    x_coordinates = np.linspace(-1.0, 1.0, GRID_WIDTH, dtype=np.float32)[None, :]
    y_coordinates = np.linspace(0.0, 1.0, GRID_HEIGHT, dtype=np.float32)[:, None]
    moments: list[float] = []
    for evidence in maps:
        total = float(np.sum(evidence))
        denominator = max(total, 1.0e-6)
        center_x = float(np.sum(evidence * x_coordinates) / denominator)
        center_y = float(np.sum(evidence * y_coordinates) / denominator)
        moments.extend(
            (
                math.log1p(total),
                center_x,
                center_y,
                float(
                    np.sum(evidence * (x_coordinates - center_x) ** 2)
                    / denominator
                ),
                float(
                    np.sum(evidence * (y_coordinates - center_y) ** 2)
                    / denominator
                ),
                float(np.max(evidence)),
                float(np.mean(evidence)),
                float(np.mean(evidence > 0.08)),
            )
        )
    result = np.ascontiguousarray(
        np.concatenate(
            (spatial.reshape(-1), np.asarray(moments, dtype=np.float32))
        ),
        dtype=np.float32,
    )
    if result.shape != (FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ImageEntityPerceptionError("image feature vector is invalid")
    return result


def _feature_dim_for_model(model_version: str) -> int:
    if model_version == LEGACY_MODEL_VERSION:
        return BASE_FEATURE_DIM
    if model_version in IMAGE_ONLY_MODEL_VERSIONS:
        return FEATURE_DIM
    if model_version == MODEL_VERSION:
        return FEATURE_DIM + LANGUAGE_EMBEDDING_DIM
    raise ImageEntityPerceptionError(
        f"MODEL_SCHEMA_MISMATCH: unsupported model_version={model_version!r}"
    )


def _torch_for_device(device: str):
    """Load torch lazily and require the requested CUDA device explicitly."""

    normalized = str(device).strip().lower()
    if normalized in {"", "numpy", "cpu"}:
        return None
    if not normalized.startswith("cuda"):
        raise ImageEntityPerceptionError(
            f"unsupported perception device {device!r}; use cuda or numpy"
        )
    try:
        import torch
    except Exception as exc:
        raise ImageEntityPerceptionError(
            f"CUDA perception requested but torch is unavailable: {exc}"
        ) from exc
    if not bool(torch.cuda.is_available()):
        raise ImageEntityPerceptionError(
            "CUDA perception requested but torch.cuda.is_available() is false"
        )
    try:
        torch.device(device)
    except Exception as exc:
        raise ImageEntityPerceptionError(
            f"invalid CUDA perception device {device!r}: {exc}"
        ) from exc
    return torch


@dataclass(frozen=True)
class ImageEntityPrediction:
    entity_id: str
    visible: bool
    confidence: float
    relative_x: float
    relative_y: float
    relative_z: float

    @property
    def class_name(self) -> str:
        return "boat"

    @property
    def color(self) -> str:
        return {
            "target_red": "red",
            "target_blue": "blue",
            "target_left": "white",
            "target_right": "white",
        }.get(str(self.entity_id), "")

    @property
    def valid(self) -> bool:
        return bool(
            math.isfinite(float(self.confidence))
            and 0.0 <= float(self.confidence) <= 1.0
            and all(
                math.isfinite(float(value))
                for value in (self.relative_x, self.relative_y, self.relative_z)
            )
        )

    @property
    def relative_velocity(self) -> tuple[float, float, float]:
        """Velocity is intentionally unavailable to a single-frame head."""

        return (0.0, 0.0, 0.0)

    @property
    def velocity_valid(self) -> bool:
        return False


def _prediction_color(entity_id: str) -> str:
    return {
        "target_red": "red",
        "target_blue": "blue",
    }.get(str(entity_id), "")


def _prediction_bearing(entity: ImageEntityPrediction) -> str:
    entity_id = str(entity.entity_id)
    if entity_id == "target_left":
        return "left"
    if entity_id == "target_right":
        return "right"
    # The canonical bearing task is represented by the dedicated left/right
    # slots. Do not relabel a color target as a bearing target merely because
    # its current pixel happens to be on that side of the image.
    if entity_id in {"target_red", "target_blue"}:
        return ""
    if not math.isfinite(float(entity.relative_y)):
        return ""
    if float(entity.relative_y) > 0.0:
        return "left"
    if float(entity.relative_y) < 0.0:
        return "right"
    return ""


def task_matches_entity(
    entity: ImageEntityPrediction,
    task: TaskSpec | str,
) -> bool:
    """Return whether a model prediction is relevant to the parsed task."""

    spec = parse_task_instruction(task) if isinstance(task, str) else task
    if not spec.valid or not spec.is_follow:
        return False
    if spec.color and _prediction_color(entity.entity_id) != spec.color:
        return False
    if spec.bearing:
        bearing = _prediction_bearing(entity)
        if not bearing and spec.color and math.isfinite(float(entity.relative_y)):
            bearing = "left" if float(entity.relative_y) > 0.0 else "right"
        if bearing != spec.bearing:
            return False
    return bool(entity.visible)


def select_task_entities(
    predictions: Sequence[ImageEntityPrediction],
    task: TaskSpec | str,
) -> tuple[ImageEntityPrediction, ...]:
    """Return only visible predictions selected by the task instruction."""

    return tuple(
        prediction
        for prediction in predictions
        if task_matches_entity(prediction, task)
    )


@dataclass(frozen=True)
class ImageEntityModel:
    """Immutable inference weights loaded from the PC trainer output."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    model_version: str = MODEL_VERSION
    visibility_threshold: float = 0.0
    schema_version: str = MODEL_SCHEMA_VERSION
    input_contract: str = MODEL_INPUT_CONTRACT
    output_contract: str = STRUCTURED_ENTITY_OUTPUT_CONTRACT
    task_embedding_dim: int = LANGUAGE_EMBEDDING_DIM
    language_model_id: str = ""
    language_weights_sha256: str = ""
    velocity_output: bool = False

    @staticmethod
    def validate_device(device: str) -> None:
        """Validate an inference device before a node accepts the model."""

        _torch_for_device(device)

    def __post_init__(self) -> None:
        is_legacy = self.model_version in IMAGE_ONLY_MODEL_VERSIONS
        if not is_legacy:
            if self.schema_version != MODEL_SCHEMA_VERSION:
                raise ImageEntityPerceptionError(
                    "MODEL_SCHEMA_MISMATCH: expected "
                    f"{MODEL_SCHEMA_VERSION}, got {self.schema_version!r}"
                )
            if self.input_contract != MODEL_INPUT_CONTRACT:
                raise ImageEntityPerceptionError(
                    "MODEL_INPUT_CONTRACT_MISMATCH: expected "
                    f"{MODEL_INPUT_CONTRACT}, got {self.input_contract!r}"
                )
            if self.output_contract != STRUCTURED_ENTITY_OUTPUT_CONTRACT:
                raise ImageEntityPerceptionError(
                    "MODEL_OUTPUT_CONTRACT_MISMATCH: expected "
                    f"{STRUCTURED_ENTITY_OUTPUT_CONTRACT}, got {self.output_contract!r}"
                )
            if int(self.task_embedding_dim) != LANGUAGE_EMBEDDING_DIM:
                raise ImageEntityPerceptionError(
                    "LANGUAGE_EMBEDDING_DIM_MISMATCH: expected "
                    f"{LANGUAGE_EMBEDDING_DIM}, got {self.task_embedding_dim}"
                )
            if not str(self.language_model_id).strip():
                raise ImageEntityPerceptionError(
                    "LANGUAGE_MODEL_ID_MISSING: new perception models require "
                    "the identity of the embedding model"
                )
            if not _is_sha256(self.language_weights_sha256):
                raise ImageEntityPerceptionError(
                    "LANGUAGE_MODEL_HASH_INVALID: expected a 64-character "
                    "SHA-256 for the embedding model weights"
                )
        if bool(self.velocity_output):
            raise ImageEntityPerceptionError(
                "VELOCITY_OUTPUT_FORBIDDEN: velocity is estimated by the temporal tracker"
            )
        expected_feature_dim = _feature_dim_for_model(self.model_version)
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        scale = np.asarray(self.feature_scale, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32)
        if mean.shape != (expected_feature_dim,) or scale.shape != (expected_feature_dim,):
            raise ImageEntityPerceptionError("invalid feature normalization shape")
        if weights.shape != (expected_feature_dim, OUTPUT_DIM) or bias.shape != (OUTPUT_DIM,):
            raise ImageEntityPerceptionError("invalid perception weight shape")
        if (
            not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(bias))
            or np.any(scale <= 0.0)
        ):
            raise ImageEntityPerceptionError("perception weights contain invalid values")
        object.__setattr__(self, "feature_mean", np.ascontiguousarray(mean))
        object.__setattr__(self, "feature_scale", np.ascontiguousarray(scale))
        object.__setattr__(self, "weights", np.ascontiguousarray(weights))
        object.__setattr__(self, "bias", np.ascontiguousarray(bias))
        object.__setattr__(self, "task_embedding_dim", int(self.task_embedding_dim))

    @classmethod
    def load(
        cls, path: str | Path, *, allow_legacy: bool = False
    ) -> "ImageEntityModel":
        model_path = Path(path).expanduser()
        if not model_path.is_file():
            raise ImageEntityPerceptionError(f"model not found: {model_path}")
        try:
            with np.load(model_path, allow_pickle=False) as data:
                version = str(data["model_version"].item())
                if version in IMAGE_ONLY_MODEL_VERSIONS:
                    if not allow_legacy:
                        raise ImageEntityPerceptionError(
                            "MODEL_SCHEMA_MISMATCH: checkpoint is image-only "
                            f"({version}); load it only with explicit "
                            "allow_legacy=True image-only legacy mode"
                        )
                    return cls(
                        feature_mean=data["feature_mean"],
                        feature_scale=data["feature_scale"],
                        weights=data["weights"],
                        bias=data["bias"],
                        model_version=version,
                        visibility_threshold=float(
                            data["visibility_threshold"].item()
                        ),
                        schema_version="image_only_legacy_schema",
                        input_contract="(camera_image_rgb)->structured_entities",
                        output_contract=STRUCTURED_ENTITY_OUTPUT_CONTRACT,
                        task_embedding_dim=0,
                        language_model_id="legacy-image-only",
                        language_weights_sha256="",
                    )
                required = (
                    "model_schema_version",
                    "input_contract",
                    "output_contract",
                    "task_embedding_dim",
                    "language_model_id",
                    "language_weights_sha256",
                    "velocity_output",
                )
                missing = [key for key in required if key not in data]
                if missing:
                    raise ImageEntityPerceptionError(
                        "MODEL_SCHEMA_MISMATCH: new checkpoint is missing "
                        + ", ".join(missing)
                    )
                return cls(
                    feature_mean=data["feature_mean"],
                    feature_scale=data["feature_scale"],
                    weights=data["weights"],
                    bias=data["bias"],
                    model_version=version,
                    visibility_threshold=float(data["visibility_threshold"].item()),
                    schema_version=str(data["model_schema_version"].item()),
                    input_contract=str(data["input_contract"].item()),
                    output_contract=str(data["output_contract"].item()),
                    task_embedding_dim=int(data["task_embedding_dim"].item()),
                    language_model_id=str(data["language_model_id"].item()),
                    language_weights_sha256=str(
                        data["language_weights_sha256"].item()
                    ),
                    velocity_output=bool(data["velocity_output"].item()),
                )
        except ImageEntityPerceptionError:
            raise
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise ImageEntityPerceptionError(
                f"cannot load perception model {model_path}: {exc}"
            ) from exc

    def predict(
        self,
        image: Image.Image | np.ndarray,
        task: TaskSpec | str | None = None,
        *,
        device: str = "numpy",
        color_image: Image.Image | np.ndarray | None = None,
        task_embedding: object | None = None,
    ) -> tuple[ImageEntityPrediction, ...]:
        """Predict entities and optionally apply task selection at the boundary.

        New models require a real fixed-size task embedding and concatenate it
        with image features before the learned projection. ``task`` is only
        the parsed instruction metadata used for target selection; it is not a
        substitute for ``task_embedding``. ``color_image`` is the original
        decoded RGB image used only for the calibrated color contract. CUDA
        requests never fall back to NumPy.
        """

        is_legacy = self.model_version in IMAGE_ONLY_MODEL_VERSIONS
        validated_embedding = None
        if not is_legacy:
            validated_embedding = validate_task_embedding(
                task_embedding, expected_dim=self.task_embedding_dim
            )
        torch = _torch_for_device(device)
        if torch is None:
            feature_extractor = (
                _extract_legacy_image_features
                if self.model_version == LEGACY_MODEL_VERSION
                else extract_image_features
            )
            features = feature_extractor(image)
            if validated_embedding is not None:
                features = np.concatenate((features, validated_embedding))
            normalized = (features - self.feature_mean) / self.feature_scale
            output = normalized @ self.weights + self.bias
        else:
            # Decode RGB/PIL on the host, then keep the feature map,
            # normalization, and linear projection on the requested CUDA
            # device.  This branch has no NumPy fallback.
            try:
                feature_tensor = _extract_torch_image_features(
                    image,
                    torch,
                    device=device,
                    model_version=self.model_version,
                )
                if validated_embedding is not None:
                    feature_tensor = torch.cat(
                        (
                            feature_tensor,
                            torch.as_tensor(
                                validated_embedding,
                                dtype=torch.float32,
                                device=device,
                            ),
                        )
                    )
                mean_tensor = torch.as_tensor(
                    self.feature_mean, dtype=torch.float32, device=device
                )
                scale_tensor = torch.as_tensor(
                    self.feature_scale, dtype=torch.float32, device=device
                )
                weight_tensor = torch.as_tensor(
                    self.weights, dtype=torch.float32, device=device
                )
                bias_tensor = torch.as_tensor(
                    self.bias, dtype=torch.float32, device=device
                )
                output = (
                    ((feature_tensor - mean_tensor) / scale_tensor)
                    @ weight_tensor
                    + bias_tensor
                ).detach().cpu().numpy()
            except Exception as exc:
                raise ImageEntityPerceptionError(
                    f"CUDA perception matrix inference failed: {exc}"
                ) from exc
        if output.shape != (OUTPUT_DIM,) or not np.all(np.isfinite(output)):
            raise ImageEntityPerceptionError("perception output is non-finite")

        calibrated_colors: dict[
            str, tuple[bool, float, float, float, tuple[float, float]]
        ] = {}
        if color_image is not None:
            if self.model_version == COLOR_CALIBRATED_MODEL_VERSION:
                color_targets = ("red",)
            elif self.model_version in (
                COLOR_CALIBRATED_MODEL_VERSION_V2,
                MODEL_VERSION,
            ):
                color_targets = ("red", "blue")
            else:
                color_targets = ()
            calibrated_colors = {
                color: calibrated_color_geometry(color_image, color)
                for color in color_targets
            }

        predictions: list[ImageEntityPrediction] = []
        for index, entity_id in enumerate(ENTITY_IDS):
            offset = index * 4
            visible_logit = float(output[offset])
            visible = visible_logit >= self.visibility_threshold
            confidence = float(1.0 / (1.0 + math.exp(-np.clip(visible_logit, -30.0, 30.0))))
            geometry = output[offset + 1 : offset + 4] * POSITION_SCALE_M
            calibrated = calibrated_colors.get(_prediction_color(entity_id))
            if calibrated is not None:
                color_valid, color_x, color_y, _, _ = calibrated
                if color_valid:
                    visible = True
                    confidence = 1.0
                    geometry = np.asarray((color_x, color_y, 0.0), dtype=np.float32)
                else:
                    # Do not expose a ridge hallucination when the original
                    # RGB evidence for a calibrated target is absent.
                    visible = False
                    confidence = 0.0
                    geometry = np.zeros(3, dtype=np.float32)
            predictions.append(
                ImageEntityPrediction(
                    entity_id=entity_id,
                    visible=visible,
                    confidence=confidence,
                    relative_x=float(geometry[0]),
                    relative_y=float(geometry[1]),
                    relative_z=float(geometry[2]),
                )
            )
        result = tuple(predictions)
        if task is None:
            return result
        spec = parse_task_instruction(task) if isinstance(task, str) else task
        if not isinstance(spec, TaskSpec):
            raise ImageEntityPerceptionError(
                f"task must be TaskSpec, string, or None; got {type(task).__name__}"
            )
        return tuple(
            ImageEntityPrediction(
                entity_id=prediction.entity_id,
                visible=task_matches_entity(prediction, spec),
                confidence=prediction.confidence,
                relative_x=prediction.relative_x,
                relative_y=prediction.relative_y,
                relative_z=prediction.relative_z,
            )
            for prediction in result
        )


def save_model(
    path: str | Path,
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    visibility_threshold: float = 0.0,
    model_version: str = MODEL_VERSION,
    schema_version: str = MODEL_SCHEMA_VERSION,
    input_contract: str = MODEL_INPUT_CONTRACT,
    output_contract: str = STRUCTURED_ENTITY_OUTPUT_CONTRACT,
    task_embedding_dim: int = LANGUAGE_EMBEDDING_DIM,
    language_model_id: str = "",
    language_weights_sha256: str = "",
    velocity_output: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write an immutable model and optional JSON metadata next to it."""

    model_path = Path(path).expanduser()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_version.strip():
        raise ImageEntityPerceptionError("model_version must not be empty")
    if model_version not in IMAGE_ONLY_MODEL_VERSIONS:
        if schema_version != MODEL_SCHEMA_VERSION:
            raise ImageEntityPerceptionError(
                f"MODEL_SCHEMA_MISMATCH: expected {MODEL_SCHEMA_VERSION}, "
                f"got {schema_version!r}"
            )
        if int(task_embedding_dim) != LANGUAGE_EMBEDDING_DIM:
            raise ImageEntityPerceptionError(
                f"LANGUAGE_EMBEDDING_DIM_MISMATCH: expected {LANGUAGE_EMBEDDING_DIM}, "
                f"got {task_embedding_dim}"
            )
        if output_contract != STRUCTURED_ENTITY_OUTPUT_CONTRACT:
            raise ImageEntityPerceptionError("MODEL_OUTPUT_CONTRACT_MISMATCH")
        if not str(language_model_id).strip():
            raise ImageEntityPerceptionError("LANGUAGE_MODEL_ID_MISSING")
        if not _is_sha256(language_weights_sha256):
            raise ImageEntityPerceptionError("LANGUAGE_MODEL_HASH_INVALID")
        if velocity_output:
            raise ImageEntityPerceptionError("VELOCITY_OUTPUT_FORBIDDEN")
    arrays: dict[str, np.ndarray] = {
        "model_version": np.asarray(model_version),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_scale": np.asarray(feature_scale, dtype=np.float32),
        "weights": np.asarray(weights, dtype=np.float32),
        "bias": np.asarray(bias, dtype=np.float32),
        "visibility_threshold": np.asarray(
            float(visibility_threshold), dtype=np.float32
        ),
    }
    if model_version not in IMAGE_ONLY_MODEL_VERSIONS:
        arrays.update(
            {
                "model_schema_version": np.asarray(schema_version),
                "input_contract": np.asarray(input_contract),
                "output_contract": np.asarray(output_contract),
                "task_embedding_dim": np.asarray(int(task_embedding_dim)),
                "language_model_id": np.asarray(language_model_id),
                "language_weights_sha256": np.asarray(language_weights_sha256),
                "velocity_output": np.asarray(bool(velocity_output)),
            }
        )
    np.savez_compressed(model_path, **arrays)
    if metadata is not None:
        model_path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


