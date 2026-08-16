from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageEnhance


TOKEN_COUNT = 2
FEATURE_DIM = 576
INPUT_SIZE = 224
BACKBONE_ID = "torchvision:mobilenet_v3_small:IMAGENET1K_V1"

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

DEFAULT_LOW_LIGHT_GAMMA = 0.92
DEFAULT_LOW_LIGHT_BRIGHTNESS = 1.04
DEFAULT_LOW_LIGHT_CONTRAST = 1.03


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


def enhance_low_light_image(
    image: Image.Image,
    *,
    enabled: bool = True,
    gamma: float = DEFAULT_LOW_LIGHT_GAMMA,
    brightness: float = DEFAULT_LOW_LIGHT_BRIGHTNESS,
    contrast: float = DEFAULT_LOW_LIGHT_CONTRAST,
) -> Image.Image:
    """Return a separate RGB image with bounded low-light enhancement.

    The input is never modified.  Keeping this after JPEG decoding lets the
    episode recorder retain the original camera bytes while both online image
    consumers use the same deterministic transform.
    """

    if not isinstance(image, Image.Image):
        raise InvalidImageError("image must be a PIL image")
    values = {
        "gamma": float(gamma),
        "brightness": float(brightness),
        "contrast": float(contrast),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("low-light parameters must be finite")
    if values["gamma"] <= 0.0:
        raise ValueError("low-light gamma must be positive")
    if values["brightness"] <= 0.0:
        raise ValueError("low-light brightness must be positive")
    if values["contrast"] <= 0.0:
        raise ValueError("low-light contrast must be positive")

    result = image.convert("RGB").copy()
    if not enabled:
        return result

    # A gamma below 1 lifts shadows while preserving the RGB ordering that
    # the color-calibrated perception model relies on.
    gamma_lut = [
        int(round(255.0 * (index / 255.0) ** values["gamma"]))
        for index in range(256)
    ]
    result = result.point(gamma_lut * 3)
    result = ImageEnhance.Brightness(result).enhance(values["brightness"])
    return ImageEnhance.Contrast(result).enhance(values["contrast"])


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
        feature_dim: int = FEATURE_DIM,
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


