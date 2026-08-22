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

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from io import BytesIO
import re
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from PIL import Image


MODEL_VERSION = "perception_ridge_language"
MODEL_SCHEMA_VERSION = "perception_schema"
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
VELOCITY_SOURCE = "perception"
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
COLOR_CALIBRATION_WIDTH = 320
COLOR_CALIBRATION_HEIGHT = 180
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
    if tuple(features.shape) != (FEATURE_DIM,) or not bool(torch.isfinite(features).all()):
        raise ImageEntityPerceptionError(
            "CUDA image feature vector is invalid"
        )
    return features.contiguous()


def extract_image_features(image: Image.Image | np.ndarray) -> np.ndarray:
    """Extract spatial evidence plus color/area moments."""

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
    color_head_weights: np.ndarray | None = None
    color_head_bias: np.ndarray | None = None

    @staticmethod
    def validate_device(device: str) -> None:
        """Validate an inference device before a node accepts the model."""

        _torch_for_device(device)

    def __post_init__(self) -> None:
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
            raise ImageEntityPerceptionError("LANGUAGE_MODEL_ID_MISSING")
        if not _is_sha256(self.language_weights_sha256):
            raise ImageEntityPerceptionError(
                "LANGUAGE_MODEL_HASH_INVALID: expected a 64-character SHA-256"
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
        color_weights = None if self.color_head_weights is None else np.asarray(self.color_head_weights, dtype=np.float32)
        color_bias = None if self.color_head_bias is None else np.asarray(self.color_head_bias, dtype=np.float32)
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
        if (color_weights is None) != (color_bias is None):
            raise ImageEntityPerceptionError("color head weights and bias must be provided together")
        if color_weights is not None and (
            color_weights.shape != (2, expected_feature_dim, 4)
            or color_bias.shape != (2, 4)
            or not np.all(np.isfinite(color_weights))
            or not np.all(np.isfinite(color_bias))
        ):
            raise ImageEntityPerceptionError("invalid color head shape or values")
        object.__setattr__(self, "feature_mean", np.ascontiguousarray(mean))
        object.__setattr__(self, "feature_scale", np.ascontiguousarray(scale))
        object.__setattr__(self, "weights", np.ascontiguousarray(weights))
        object.__setattr__(self, "bias", np.ascontiguousarray(bias))
        object.__setattr__(self, "color_head_weights", None if color_weights is None else np.ascontiguousarray(color_weights))
        object.__setattr__(self, "color_head_bias", None if color_bias is None else np.ascontiguousarray(color_bias))
        object.__setattr__(self, "task_embedding_dim", int(self.task_embedding_dim))

    @classmethod
    def load(cls, path: str | Path) -> "ImageEntityModel":
        model_path = Path(path).expanduser()
        if not model_path.is_file():
            raise ImageEntityPerceptionError(f"model not found: {model_path}")
        try:
            with np.load(model_path, allow_pickle=False) as data:
                version = str(data["model_version"].item())
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
                    color_head_weights=data["color_head_weights"] if "color_head_weights" in data else None,
                    color_head_bias=data["color_head_bias"] if "color_head_bias" in data else None,
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

        validated_embedding = validate_task_embedding(
            task_embedding, expected_dim=self.task_embedding_dim
        )
        torch = _torch_for_device(device)
        head = self._color_head_for_task(task)
        if torch is None:
            features = np.concatenate(
                (extract_image_features(image), validated_embedding)
            )
            normalized = (features - self.feature_mean) / self.feature_scale
            output = normalized @ self.weights + self.bias
            if head is not None:
                head_index, head_weights, head_bias = head
                output[head_index * 4 : head_index * 4 + 4] = normalized @ head_weights + head_bias
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
                if head is not None:
                    head_index, head_weights, head_bias = head
                    output[head_index * 4 : head_index * 4 + 4] = (
                        ((feature_tensor - mean_tensor) / scale_tensor)
                        @ torch.as_tensor(head_weights, dtype=torch.float32, device=device)
                        + torch.as_tensor(head_bias, dtype=torch.float32, device=device)
                    ).detach().cpu().numpy()
            except Exception as exc:
                raise ImageEntityPerceptionError(
                    f"CUDA perception matrix inference failed: {exc}"
                ) from exc
        if output.shape != (OUTPUT_DIM,) or not np.all(np.isfinite(output)):
            raise ImageEntityPerceptionError("perception output is non-finite")

        predictions: list[ImageEntityPrediction] = []
        for index, entity_id in enumerate(ENTITY_IDS):
            offset = index * 4
            visible_logit = float(output[offset])
            visible = visible_logit >= self.visibility_threshold
            confidence = float(1.0 / (1.0 + math.exp(-np.clip(visible_logit, -30.0, 30.0))))
            geometry = output[offset + 1 : offset + 4] * POSITION_SCALE_M
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

    def _color_head_for_task(self, task: TaskSpec | str | None):
        if self.color_head_weights is None or task is None:
            return None
        spec = parse_task_instruction(task) if isinstance(task, str) else task
        if not isinstance(spec, TaskSpec) or spec.color not in {"red", "blue"}:
            return None
        index = 0 if spec.color == "red" else 1
        return index, self.color_head_weights[index], self.color_head_bias[index]


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
    color_head_weights: np.ndarray | None = None,
    color_head_bias: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write an immutable model and optional JSON metadata next to it."""

    model_path = Path(path).expanduser()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_version.strip():
        raise ImageEntityPerceptionError("model_version must not be empty")
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
    if color_head_weights is not None or color_head_bias is not None:
        if color_head_weights is None or color_head_bias is None:
            raise ImageEntityPerceptionError("color head weights and bias must be provided together")
        arrays["color_head_weights"] = np.asarray(color_head_weights, dtype=np.float32)
        arrays["color_head_bias"] = np.asarray(color_head_bias, dtype=np.float32)
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


TOKEN_COUNT = 2
VISUAL_FEATURE_DIM = 576
INPUT_SIZE = 224
BACKBONE_ID = "torchvision:mobilenet_v3_small:IMAGENET1K_V1"

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)



class VisualEncoderError(RuntimeError):
    """Base class for deterministic visual-encoder failures."""


class InvalidImageError(VisualEncoderError):
    """Raised when a camera payload cannot satisfy the frozen contract."""


class TargetSelectionError(VisualEncoderError):
    """Raised when no valid, visible target entity is available."""


class TargetProjectionError(VisualEncoderError):
    """Raised when the selected target cannot be projected into the image."""


class InvalidVisualFeaturesError(VisualEncoderError):
    """Raised when the backbone returns an unusable tensor."""


@dataclass(frozen=True)
class CameraProfile:
    width: int = 1280
    height: int = 720
    horizontal_fov_deg: float = 90.0
    mount_x_m: float = 0.42
    mount_y_m: float = 0.0
    mount_z_m: float = 0.20
    pitch_deg: float = -5.0
    crop_size_px: int = 224

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if not 0.0 < self.horizontal_fov_deg < 180.0:
            raise ValueError("horizontal_fov_deg must be between 0 and 180")
        if self.crop_size_px <= 0:
            raise ValueError("crop_size_px must be positive")


def decode_camera_image(data: bytes | bytearray, encoding: str) -> Image.Image:
    normalized_encoding = encoding.strip().lower()
    if normalized_encoding not in {"jpeg", "jpg"}:
        raise InvalidImageError(
            f"unsupported camera encoding {encoding!r}; expected jpeg"
        )
    if not data:
        raise InvalidImageError("camera payload is empty")
    try:
        with Image.open(BytesIO(bytes(data))) as source:
            source.load()
            return source.convert("RGB")
    except Exception as exc:
        raise InvalidImageError(
            f"failed to decode JPEG: {type(exc).__name__}: {exc}"
        ) from exc


def select_target(entities: Iterable[Any]) -> Any:
    candidates = []
    for entity in entities:
        coordinates = (
            float(entity.relative_x),
            float(entity.relative_y),
            float(entity.relative_z),
        )
        if (
            bool(entity.valid)
            and bool(entity.visible)
            and bool(entity.is_target)
            and all(math.isfinite(value) for value in coordinates)
        ):
            distance_squared = sum(value * value for value in coordinates)
            candidates.append(
                (distance_squared, str(entity.entity_id), entity)
            )
    if not candidates:
        raise TargetSelectionError(
            "no entity is simultaneously target, visible, valid, and finite"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def project_target_to_pixel(
    relative_x: float,
    relative_y: float,
    relative_z: float,
    profile: CameraProfile,
) -> tuple[float, float, float]:
    values = (relative_x, relative_y, relative_z)
    if not all(math.isfinite(float(value)) for value in values):
        raise TargetProjectionError("target coordinates contain NaN or Inf")

    # Entity positions use ROS base_link: +X forward, +Y left, +Z up.
    # The UE camera uses +X forward, +Y right, +Z up.  The frozen UE component
    # pitch is -5 degrees, so transform the base_link vector into the pitched
    # camera frame before applying the pinhole projection.
    dx = float(relative_x) - profile.mount_x_m
    dy_right = -(float(relative_y) - profile.mount_y_m)
    dz = float(relative_z) - profile.mount_z_m
    pitch = math.radians(profile.pitch_deg)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    depth = cos_pitch * dx + sin_pitch * dz
    camera_z = -sin_pitch * dx + cos_pitch * dz
    if depth <= 1.0e-6:
        raise TargetProjectionError(
            f"target is behind the camera or too close; depth={depth:.6f}"
        )

    focal_px = profile.width / (
        2.0 * math.tan(math.radians(profile.horizontal_fov_deg) / 2.0)
    )
    center_x = profile.width / 2.0
    center_y = profile.height / 2.0
    pixel_x = center_x + focal_px * dy_right / depth
    pixel_y = center_y - focal_px * camera_z / depth
    if not (
        0.0 <= pixel_x < profile.width
        and 0.0 <= pixel_y < profile.height
    ):
        raise TargetProjectionError(
            "target projects outside the image; "
            f"pixel=({pixel_x:.2f},{pixel_y:.2f})"
        )
    return pixel_x, pixel_y, depth


def crop_around_pixel(
    image: Image.Image,
    pixel_x: float,
    pixel_y: float,
    crop_size_px: int,
) -> Image.Image:
    if crop_size_px <= 0:
        raise ValueError("crop_size_px must be positive")
    if not (
        math.isfinite(pixel_x)
        and math.isfinite(pixel_y)
        and 0.0 <= pixel_x < image.width
        and 0.0 <= pixel_y < image.height
    ):
        raise TargetProjectionError("crop centre is outside the image")

    left = int(round(pixel_x)) - crop_size_px // 2
    top = int(round(pixel_y)) - crop_size_px // 2
    right = left + crop_size_px
    bottom = top + crop_size_px

    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(image.width, right)
    source_bottom = min(image.height, bottom)
    result = Image.new("RGB", (crop_size_px, crop_size_px), (0, 0, 0))
    region = image.crop(
        (source_left, source_top, source_right, source_bottom)
    )
    result.paste(region, (source_left - left, source_top - top))
    return result


def make_target_crop(
    image: Image.Image,
    target: Any,
    profile: CameraProfile,
) -> tuple[Image.Image, tuple[float, float, float]]:
    if image.size != (profile.width, profile.height):
        raise InvalidImageError(
            f"camera image shape {image.width}x{image.height} does not match "
            f"frozen profile {profile.width}x{profile.height}"
        )
    projection = project_target_to_pixel(
        target.relative_x,
        target.relative_y,
        target.relative_z,
        profile,
    )
    crop = crop_around_pixel(
        image,
        projection[0],
        projection[1],
        profile.crop_size_px,
    )
    return crop, projection


def _letterbox_and_normalize(
    image: Image.Image,
    output_size: int = INPUT_SIZE,
) -> np.ndarray:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(output_size / width, output_size / height)
    resized = rgb.resize(
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (output_size, output_size), (0, 0, 0))
    offset = (
        (output_size - resized.width) // 2,
        (output_size - resized.height) // 2,
    )
    canvas.paste(resized, offset)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))


class FrozenMobileNetEncoder:
    """Frozen MobileNetV3-small backbone producing normalized 576-D tokens."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        backbone: Any | None = None,
        feature_dim: int = VISUAL_FEATURE_DIM,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        try:
            import torch
        except ImportError as exc:
            raise VisualEncoderError("PyTorch is not installed") from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise VisualEncoderError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        self._torch = torch
        self.device = device
        self.feature_dim = int(feature_dim)

        if backbone is None:
            try:
                from torchvision.models import (
                    MobileNet_V3_Small_Weights,
                    mobilenet_v3_small,
                )

                model = mobilenet_v3_small(
                    weights=MobileNet_V3_Small_Weights.DEFAULT
                )
                backbone = model.features
            except Exception as exc:
                raise VisualEncoderError(
                    "failed to load torchvision MobileNetV3-small weights: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        self.backbone = backbone.eval().to(device)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    @property
    def frozen(self) -> bool:
        return all(
            not parameter.requires_grad
            for parameter in self.backbone.parameters()
        )

    def encode_pair(
        self,
        global_image: Image.Image,
        target_crop: Image.Image,
    ) -> np.ndarray:
        result = self.encode_images((global_image, target_crop))
        expected_shape = (TOKEN_COUNT, self.feature_dim)
        if result.shape != expected_shape:
            raise InvalidVisualFeaturesError(
                f"backbone returned shape {result.shape}; "
                f"expected {expected_shape}"
            )
        return result

    def encode_images(
        self,
        images: Iterable[Image.Image],
    ) -> np.ndarray:
        """Encode an arbitrary non-empty image batch.

        Day 6 used exactly two images (global plus one selected target).  Day
        13 reuses the same frozen backbone and preprocessing for one global
        image plus every projectable entity crop.  Keeping the batching here
        prevents the feature-cache builder from running one CUDA inference per
        entity.
        """

        image_batch = tuple(images)
        if not image_batch:
            raise InvalidVisualFeaturesError("image batch must not be empty")
        torch = self._torch
        batch_array = np.stack(
            tuple(_letterbox_and_normalize(image) for image in image_batch)
        )
        batch = torch.from_numpy(batch_array).to(self.device)
        try:
            with torch.inference_mode():
                raw = self.backbone(batch)
                pooled = torch.nn.functional.adaptive_avg_pool2d(
                    raw, 1
                ).flatten(1)
                features = torch.nn.functional.normalize(
                    pooled.float(), p=2.0, dim=1
                )
        except Exception as exc:
            raise VisualEncoderError(
                f"MobileNet inference failed: {type(exc).__name__}: {exc}"
            ) from exc

        result = np.ascontiguousarray(
            features.detach().cpu().numpy(), dtype=np.float32
        )
        expected_shape = (len(image_batch), self.feature_dim)
        if result.shape != expected_shape:
            raise InvalidVisualFeaturesError(
                f"backbone returned shape {result.shape}; "
                f"expected {expected_shape}"
            )
        if not np.all(np.isfinite(result)):
            raise InvalidVisualFeaturesError(
                "visual feature tensor contains NaN or Inf"
            )
        norms = np.linalg.norm(result, axis=1)
        if not np.allclose(norms, 1.0, atol=1.0e-5):
            raise InvalidVisualFeaturesError(
                f"visual token norms are invalid: {norms.tolist()}"
            )
        return result


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

    def as_entity_kwargs(self) -> dict[str, object]:
        """Return fields accepted by the ``Entity`` message."""

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


DEFAULT_DROPOUT_HOLD_FRAMES = 30
DEFAULT_DROPOUT_HOLD_SEC = 3.0


class _DropoutRecovery:
    """Bounded, identity-scoped recovery for short perception dropouts."""

    def __init__(
        self,
        *,
        dropout_hold_frames: int = DEFAULT_DROPOUT_HOLD_FRAMES,
        dropout_hold_sec: float = DEFAULT_DROPOUT_HOLD_SEC,
    ) -> None:
        if int(dropout_hold_frames) < 0:
            raise ValueError("dropout_hold_frames must be non-negative")
        if not math.isfinite(float(dropout_hold_sec)) or float(dropout_hold_sec) <= 0.0:
            raise ValueError("dropout_hold_sec must be finite and positive")
        self.dropout_hold_frames = int(dropout_hold_frames)
        self.dropout_hold_sec = float(dropout_hold_sec)
        self._identity: tuple[str, int] | None = None
        self._tracks: dict[str, TrackedEntity] = {}
        self.last_predicted_ids: tuple[str, ...] = ()

    def reset(self) -> None:
        self._identity = None
        self._tracks.clear()
        self.last_predicted_ids = ()

    def update(
        self,
        observed: tuple[TrackedEntity, ...],
        *,
        frame: FrameMetadata,
    ) -> tuple[TrackedEntity, ...]:
        if self._identity != frame.identity:
            self.reset()
            self._identity = frame.identity

        observed_ids = set()
        for item in observed:
            if item.run_id != frame.run_id or item.scene_seed != frame.scene_seed:
                raise TemporalEntityTrackerError(
                    "tracked entities must share the current run and scene"
                )
            observed_ids.add(item.entity_id)
            self._tracks[item.entity_id] = item

        predicted: list[TrackedEntity] = []
        expired: list[str] = []
        for entity_id, item in tuple(self._tracks.items()):
            if entity_id in observed_ids:
                continue
            frame_gap = frame.frame_index - item.frame_index
            elapsed_sec = (frame.stamp_us - item.stamp_us) / 1.0e6
            within_window = (
                frame_gap > 0
                and elapsed_sec >= 0.0
                and frame_gap <= self.dropout_hold_frames
                and elapsed_sec <= self.dropout_hold_sec
            )
            if not within_window:
                if frame_gap > self.dropout_hold_frames or elapsed_sec > self.dropout_hold_sec:
                    expired.append(entity_id)
                continue
            velocity = item.velocity if item.velocity_valid else (0.0, 0.0, 0.0)
            predicted_position = tuple(
                position + component * elapsed_sec
                for position, component in zip(item.position, velocity)
            )
            predicted.append(
                replace(
                    item,
                    relative_x=predicted_position[0],
                    relative_y=predicted_position[1],
                    relative_z=predicted_position[2],
                    frame_index=frame.frame_index,
                    stamp_us=frame.stamp_us,
                    frame_gap=frame_gap,
                    source="temporal_tracker",
                )
            )

        for entity_id in expired:
            self._tracks.pop(entity_id, None)
        self.last_predicted_ids = tuple(item.entity_id for item in predicted)
        return tuple(observed) + tuple(predicted)


__all__ = [
    "FrameMetadata",
    "GeometryObservation",
    "TemporalEntityTracker",
    "TemporalEntityTrackerError",
    "TrackedEntity",
]
