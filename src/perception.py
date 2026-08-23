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
MODEL_VERSION = 'perception_ridge_language'
MODEL_SCHEMA_VERSION = 'perception_schema'
MODEL_INPUT_CONTRACT = '(camera_image_rgb,task_embedding_float32[256])->structured_entities'
LANGUAGE_EMBEDDING_DIM = 256
LANGUAGE_EMBEDDING_CONTRACT = 'task_embedding:float32[256];finite;nonzero_l2;normalized_by_language_encoder'
STRUCTURED_ENTITY_OUTPUT_CONTRACT = 'entity_id,class_name,color,relative_position_m,visible_mask,bbox_px,confidence,valid,relative_velocity_mps,velocity_valid'
VELOCITY_SOURCE = 'perception'
GRID_WIDTH = 32
GRID_HEIGHT = 18
CHANNELS = 7
BASE_FEATURE_DIM = GRID_WIDTH * GRID_HEIGHT * CHANNELS
MOMENT_MAPS = 4
MOMENT_FEATURES_PER_MAP = 8
FEATURE_DIM = BASE_FEATURE_DIM + MOMENT_MAPS * MOMENT_FEATURES_PER_MAP
FUSED_FEATURE_DIM = FEATURE_DIM + LANGUAGE_EMBEDDING_DIM
ENTITY_IDS = ('target_red', 'target_blue', 'target_left', 'target_right')
ENTITY_COUNT = len(ENTITY_IDS)
OUTPUT_DIM = ENTITY_COUNT * 4
POSITION_SCALE_M = np.asarray((40.0, 40.0, 5.0), dtype=np.float32)

class ImageEntityPerceptionError(RuntimeError):
    """图像感知模型或输入不可用时抛出。"""

class InvalidImageError(RuntimeError):
    pass


class TargetProjectionError(RuntimeError):
    pass


def validate_task_embedding(embedding: object, *, expected_dim: int=LANGUAGE_EMBEDDING_DIM) -> np.ndarray:
    """校验任务条件，不创建回退向量。"""
    array = np.asarray(embedding, dtype=np.float32)
    if array.shape != (expected_dim,):
        raise ImageEntityPerceptionError(f'task embedding shape {array.shape}; expected ({expected_dim},)')
    if not np.all(np.isfinite(array)):
        raise ImageEntityPerceptionError('task embedding contains NaN or Inf')
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ImageEntityPerceptionError('task embedding has zero or invalid norm')
    return np.ascontiguousarray(array, dtype=np.float32)

def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch('[0-9a-fA-F]{64}', str(value).strip()))

def _rgb_uint8_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """Normalize camera payloads to contiguous RGB uint8."""
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ImageEntityPerceptionError(f'expected HxWx3 image, got {array.shape}')
        if not np.all(np.isfinite(array)):
            raise ImageEntityPerceptionError('image array contains non-finite values')
        if np.issubdtype(array.dtype, np.floating) and np.max(array) <= 1.0:
            array = array * 255.0
        return np.ascontiguousarray(np.clip(array, 0.0, 255.0).astype(np.uint8))
    if not isinstance(image, Image.Image):
        raise ImageEntityPerceptionError('image must be a PIL image or RGB array')
    return np.ascontiguousarray(np.asarray(image.convert('RGB'), dtype=np.uint8))

def _resized_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    """Resize RGB input to the fixed perception grid."""
    pil = Image.fromarray(_rgb_uint8_array(image), mode='RGB')
    return np.asarray(pil.resize((GRID_WIDTH, GRID_HEIGHT), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

def _extract_torch_image_features(image: Image.Image | np.ndarray, torch: Any, *, device: str, model_version: str):
    """在指定 CUDA 设备上构建特征图和矩。

    The resize runs on the host with the same PIL BILINEAR path the trainer
    uses: torch interpolate differs numerically from PIL, and the tiny
    evidence-map differences get amplified by the moment denominators
    (observed up to 0.66 on one frame), which shifted the trained ridge
    outputs out of distribution at inference.
    """
    resized = torch.as_tensor(_resized_rgb(image), dtype=torch.float32, device=device)
    red = torch.clamp(resized[..., 0] - torch.maximum(resized[..., 1], resized[..., 2]), min=0.0)
    blue = torch.clamp(resized[..., 2] - torch.maximum(resized[..., 0], resized[..., 1]), min=0.0)
    brightness = resized.mean(dim=-1)
    saturation = resized.amax(dim=-1) - resized.amin(dim=-1)
    white = torch.clamp(brightness - 1.5 * saturation, min=0.0, max=1.0)
    bright = torch.clamp(brightness - 0.45, min=0.0, max=1.0)
    spatial = torch.cat((resized, red.unsqueeze(-1), blue.unsqueeze(-1), white.unsqueeze(-1), bright.unsqueeze(-1)), dim=-1)
    x_coordinates = torch.linspace(-1.0, 1.0, GRID_WIDTH, dtype=torch.float32, device=device)[None, :]
    y_coordinates = torch.linspace(0.0, 1.0, GRID_HEIGHT, dtype=torch.float32, device=device)[:, None]
    moments = []
    for evidence in (red, blue, white, bright):
        total = evidence.sum()
        denominator = torch.clamp(total, min=1e-06)
        center_x = (evidence * x_coordinates).sum() / denominator
        center_y = (evidence * y_coordinates).sum() / denominator
        moments.extend((torch.log1p(total), center_x, center_y, (evidence * (x_coordinates - center_x) ** 2).sum() / denominator, (evidence * (y_coordinates - center_y) ** 2).sum() / denominator, evidence.amax(), evidence.mean(), (evidence > 0.08).to(dtype=torch.float32).mean()))
    features = torch.cat((spatial.reshape(-1), torch.stack(moments)))
    if tuple(features.shape) != (FEATURE_DIM,) or not bool(torch.isfinite(features).all()):
        raise ImageEntityPerceptionError('CUDA image feature vector is invalid')
    return features.contiguous()

def extract_image_features(image: Image.Image | np.ndarray) -> np.ndarray:
    """提取空间证据及颜色/面积矩。"""
    result = _resized_rgb(image)
    red = np.maximum(result[..., 0] - np.maximum(result[..., 1], result[..., 2]), 0.0)
    blue = np.maximum(result[..., 2] - np.maximum(result[..., 0], result[..., 1]), 0.0)
    brightness = np.mean(result, axis=-1)
    saturation = np.max(result, axis=-1) - np.min(result, axis=-1)
    white = np.clip(brightness - 1.5 * saturation, 0.0, 1.0)
    bright = np.clip(brightness - 0.45, 0.0, 1.0)
    maps = (red, blue, white, bright)
    spatial = np.concatenate([result, red[..., None], blue[..., None], white[..., None], bright[..., None]], axis=-1)
    x_coordinates = np.linspace(-1.0, 1.0, GRID_WIDTH, dtype=np.float32)[None, :]
    y_coordinates = np.linspace(0.0, 1.0, GRID_HEIGHT, dtype=np.float32)[:, None]
    moments: list[float] = []
    for evidence in maps:
        total = float(np.sum(evidence))
        denominator = max(total, 1e-06)
        center_x = float(np.sum(evidence * x_coordinates) / denominator)
        center_y = float(np.sum(evidence * y_coordinates) / denominator)
        moments.extend((math.log1p(total), center_x, center_y, float(np.sum(evidence * (x_coordinates - center_x) ** 2) / denominator), float(np.sum(evidence * (y_coordinates - center_y) ** 2) / denominator), float(np.max(evidence)), float(np.mean(evidence)), float(np.mean(evidence > 0.08))))
    result = np.ascontiguousarray(np.concatenate((spatial.reshape(-1), np.asarray(moments, dtype=np.float32))), dtype=np.float32)
    if result.shape != (FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ImageEntityPerceptionError('image feature vector is invalid')
    return result

def _feature_dim_for_model(model_version: str) -> int:
    if model_version == MODEL_VERSION:
        return FEATURE_DIM + LANGUAGE_EMBEDDING_DIM
    raise ImageEntityPerceptionError(f'MODEL_SCHEMA_MISMATCH: unsupported model_version={model_version!r}')

def _torch_for_device(device: str):
    """延迟加载 torch，并显式要求指定 CUDA 设备。"""
    normalized = str(device).strip().lower()
    if normalized in {'', 'numpy', 'cpu'}:
        return None
    if not normalized.startswith('cuda'):
        raise ImageEntityPerceptionError(f'unsupported perception device {device!r}; use cuda or numpy')
    try:
        import torch
    except Exception as exc:
        raise ImageEntityPerceptionError(f'CUDA perception requested but torch is unavailable: {exc}') from exc
    if not bool(torch.cuda.is_available()):
        raise ImageEntityPerceptionError('CUDA perception requested but torch.cuda.is_available() is false')
    try:
        torch.device(device)
    except Exception as exc:
        raise ImageEntityPerceptionError(f'invalid CUDA perception device {device!r}: {exc}') from exc
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
        return 'boat'

    @property
    def color(self) -> str:
        return {'target_red': 'red', 'target_blue': 'blue', 'target_left': 'white', 'target_right': 'white'}.get(str(self.entity_id), '')

    @property
    def valid(self) -> bool:
        return bool(math.isfinite(float(self.confidence)) and 0.0 <= float(self.confidence) <= 1.0 and all((math.isfinite(float(value)) for value in (self.relative_x, self.relative_y, self.relative_z))))

    @property
    def relative_velocity(self) -> tuple[float, float, float]:
        """单帧预测头不提供速度。"""
        return (0.0, 0.0, 0.0)

    @property
    def velocity_valid(self) -> bool:
        return False

@dataclass(frozen=True)
class ImageEntityModel:
    """从 PC 训练器输出加载的不可变推理权重。"""
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
    language_model_id: str = ''
    language_weights_sha256: str = ''
    velocity_output: bool = False

    @staticmethod
    def validate_device(device: str) -> None:
        """节点接受模型前校验推理设备。"""
        _torch_for_device(device)

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_SCHEMA_VERSION:
            raise ImageEntityPerceptionError(f'MODEL_SCHEMA_MISMATCH: expected {MODEL_SCHEMA_VERSION}, got {self.schema_version!r}')
        if self.input_contract != MODEL_INPUT_CONTRACT:
            raise ImageEntityPerceptionError(f'MODEL_INPUT_CONTRACT_MISMATCH: expected {MODEL_INPUT_CONTRACT}, got {self.input_contract!r}')
        if self.output_contract != STRUCTURED_ENTITY_OUTPUT_CONTRACT:
            raise ImageEntityPerceptionError(f'MODEL_OUTPUT_CONTRACT_MISMATCH: expected {STRUCTURED_ENTITY_OUTPUT_CONTRACT}, got {self.output_contract!r}')
        if int(self.task_embedding_dim) != LANGUAGE_EMBEDDING_DIM:
            raise ImageEntityPerceptionError(f'LANGUAGE_EMBEDDING_DIM_MISMATCH: expected {LANGUAGE_EMBEDDING_DIM}, got {self.task_embedding_dim}')
        if not str(self.language_model_id).strip():
            raise ImageEntityPerceptionError('LANGUAGE_MODEL_ID_MISSING')
        if not _is_sha256(self.language_weights_sha256):
            raise ImageEntityPerceptionError('LANGUAGE_MODEL_HASH_INVALID: expected a 64-character SHA-256')
        if bool(self.velocity_output):
            raise ImageEntityPerceptionError('VELOCITY_OUTPUT_FORBIDDEN: velocity is estimated by the temporal tracker')
        expected_feature_dim = _feature_dim_for_model(self.model_version)
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        scale = np.asarray(self.feature_scale, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32)
        if mean.shape != (expected_feature_dim,) or scale.shape != (expected_feature_dim,):
            raise ImageEntityPerceptionError('invalid feature normalization shape')
        if weights.shape != (expected_feature_dim, OUTPUT_DIM) or bias.shape != (OUTPUT_DIM,):
            raise ImageEntityPerceptionError('invalid perception weight shape')
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or (not np.all(np.isfinite(weights))) or (not np.all(np.isfinite(bias))) or np.any(scale <= 0.0):
            raise ImageEntityPerceptionError('perception weights contain invalid values')
        object.__setattr__(self, 'feature_mean', np.ascontiguousarray(mean))
        object.__setattr__(self, 'feature_scale', np.ascontiguousarray(scale))
        object.__setattr__(self, 'weights', np.ascontiguousarray(weights))
        object.__setattr__(self, 'bias', np.ascontiguousarray(bias))
        object.__setattr__(self, 'task_embedding_dim', int(self.task_embedding_dim))

    @classmethod
    def load(cls, path: str | Path) -> 'ImageEntityModel':
        model_path = Path(path).expanduser()
        if not model_path.is_file():
            raise ImageEntityPerceptionError(f'model not found: {model_path}')
        try:
            with np.load(model_path, allow_pickle=False) as data:
                version = str(data['model_version'].item())
                required = ('model_schema_version', 'input_contract', 'output_contract', 'task_embedding_dim', 'language_model_id', 'language_weights_sha256', 'velocity_output')
                missing = [key for key in required if key not in data]
                if missing:
                    raise ImageEntityPerceptionError('MODEL_SCHEMA_MISMATCH: new checkpoint is missing ' + ', '.join(missing))
                return cls(feature_mean=data['feature_mean'], feature_scale=data['feature_scale'], weights=data['weights'], bias=data['bias'], model_version=version, visibility_threshold=float(data['visibility_threshold'].item()), schema_version=str(data['model_schema_version'].item()), input_contract=str(data['input_contract'].item()), output_contract=str(data['output_contract'].item()), task_embedding_dim=int(data['task_embedding_dim'].item()), language_model_id=str(data['language_model_id'].item()), language_weights_sha256=str(data['language_weights_sha256'].item()), velocity_output=bool(data['velocity_output'].item()))
        except ImageEntityPerceptionError:
            raise
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise ImageEntityPerceptionError(f'cannot load perception model {model_path}: {exc}') from exc

    def predict(
        self,
        image: Image.Image | np.ndarray,
        *,
        device: str = "numpy",
        task_embedding: object | None = None,
    ) -> tuple[ImageEntityPrediction, ...]:
        """预测实体，并可在边界应用任务筛选。

        New models require a real fixed-size task embedding and concatenate it
        with image features before the learned projection. ``task`` is only
        the parsed instruction metadata used for target selection; it is not a
        substitute for ``task_embedding``. ``color_image`` is the original
        decoded RGB image used only for the calibrated color contract. CUDA
        requests never fall back to NumPy.
        """
        validated_embedding = validate_task_embedding(task_embedding, expected_dim=self.task_embedding_dim)
        torch = _torch_for_device(device)
        if torch is None:
            features = np.concatenate((extract_image_features(image), validated_embedding))
            normalized = (features - self.feature_mean) / self.feature_scale
            output = normalized @ self.weights + self.bias
        else:
            try:
                feature_tensor = _extract_torch_image_features(image, torch, device=device, model_version=self.model_version)
                if validated_embedding is not None:
                    feature_tensor = torch.cat((feature_tensor, torch.as_tensor(validated_embedding, dtype=torch.float32, device=device)))
                mean_tensor = torch.as_tensor(self.feature_mean, dtype=torch.float32, device=device)
                scale_tensor = torch.as_tensor(self.feature_scale, dtype=torch.float32, device=device)
                weight_tensor = torch.as_tensor(self.weights, dtype=torch.float32, device=device)
                bias_tensor = torch.as_tensor(self.bias, dtype=torch.float32, device=device)
                output = ((feature_tensor - mean_tensor) / scale_tensor @ weight_tensor + bias_tensor).detach().cpu().numpy()
            except Exception as exc:
                raise ImageEntityPerceptionError(f'CUDA perception matrix inference failed: {exc}') from exc
        if output.shape != (OUTPUT_DIM,) or not np.all(np.isfinite(output)):
            raise ImageEntityPerceptionError('perception output is non-finite')
        predictions: list[ImageEntityPrediction] = []
        for index, entity_id in enumerate(ENTITY_IDS):
            offset = index * 4
            visible_logit = float(output[offset])
            visible = visible_logit >= self.visibility_threshold
            confidence = float(1.0 / (1.0 + math.exp(-np.clip(visible_logit, -30.0, 30.0))))
            geometry = output[offset + 1:offset + 4] * POSITION_SCALE_M
            predictions.append(ImageEntityPrediction(entity_id=entity_id, visible=visible, confidence=confidence, relative_x=float(geometry[0]), relative_y=float(geometry[1]), relative_z=float(geometry[2])))
        return tuple(predictions)

def save_model(path: str | Path, *, feature_mean: np.ndarray, feature_scale: np.ndarray, weights: np.ndarray, bias: np.ndarray, visibility_threshold: float=0.0, model_version: str=MODEL_VERSION, schema_version: str=MODEL_SCHEMA_VERSION, input_contract: str=MODEL_INPUT_CONTRACT, output_contract: str=STRUCTURED_ENTITY_OUTPUT_CONTRACT, task_embedding_dim: int=LANGUAGE_EMBEDDING_DIM, language_model_id: str='', language_weights_sha256: str='', velocity_output: bool=False, metadata: dict[str, Any] | None=None) -> None:
    """写入不可变模型及可选 JSON 元数据。"""
    model_path = Path(path).expanduser()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_version.strip():
        raise ImageEntityPerceptionError('model_version must not be empty')
    if schema_version != MODEL_SCHEMA_VERSION:
        raise ImageEntityPerceptionError(f'MODEL_SCHEMA_MISMATCH: expected {MODEL_SCHEMA_VERSION}, got {schema_version!r}')
    if int(task_embedding_dim) != LANGUAGE_EMBEDDING_DIM:
        raise ImageEntityPerceptionError(f'LANGUAGE_EMBEDDING_DIM_MISMATCH: expected {LANGUAGE_EMBEDDING_DIM}, got {task_embedding_dim}')
    if output_contract != STRUCTURED_ENTITY_OUTPUT_CONTRACT:
        raise ImageEntityPerceptionError('MODEL_OUTPUT_CONTRACT_MISMATCH')
    if not str(language_model_id).strip():
        raise ImageEntityPerceptionError('LANGUAGE_MODEL_ID_MISSING')
    if not _is_sha256(language_weights_sha256):
        raise ImageEntityPerceptionError('LANGUAGE_MODEL_HASH_INVALID')
    if velocity_output:
        raise ImageEntityPerceptionError('VELOCITY_OUTPUT_FORBIDDEN')
    arrays: dict[str, np.ndarray] = {'model_version': np.asarray(model_version), 'feature_mean': np.asarray(feature_mean, dtype=np.float32), 'feature_scale': np.asarray(feature_scale, dtype=np.float32), 'weights': np.asarray(weights, dtype=np.float32), 'bias': np.asarray(bias, dtype=np.float32), 'visibility_threshold': np.asarray(float(visibility_threshold), dtype=np.float32)}
    arrays.update({'model_schema_version': np.asarray(schema_version), 'input_contract': np.asarray(input_contract), 'output_contract': np.asarray(output_contract), 'task_embedding_dim': np.asarray(int(task_embedding_dim)), 'language_model_id': np.asarray(language_model_id), 'language_weights_sha256': np.asarray(language_weights_sha256), 'velocity_output': np.asarray(bool(velocity_output))})
    np.savez_compressed(model_path, **arrays)
    if metadata is not None:
        model_path.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
BACKBONE_ID = 'torchvision:mobilenet_v3_small:IMAGENET1K_V1'

@dataclass(frozen=True)
class CameraProfile:
    width: int = 1280
    height: int = 720
    horizontal_fov_deg: float = 90.0
    mount_x_m: float = 0.42
    mount_y_m: float = 0.0
    mount_z_m: float = 0.2
    pitch_deg: float = -5.0
    crop_size_px: int = 224

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError('camera width and height must be positive')
        if not 0.0 < self.horizontal_fov_deg < 180.0:
            raise ValueError('horizontal_fov_deg must be between 0 and 180')
        if self.crop_size_px <= 0:
            raise ValueError('crop_size_px must be positive')

def decode_camera_image(data: bytes | bytearray, encoding: str) -> Image.Image:
    normalized_encoding = encoding.strip().lower()
    if normalized_encoding not in {'jpeg', 'jpg'}:
        raise InvalidImageError(f'unsupported camera encoding {encoding!r}; expected jpeg')
    if not data:
        raise InvalidImageError('camera payload is empty')
    try:
        with Image.open(BytesIO(bytes(data))) as source:
            source.load()
            return source.convert('RGB')
    except Exception as exc:
        raise InvalidImageError(f'failed to decode JPEG: {type(exc).__name__}: {exc}') from exc

def project_target_to_pixel(relative_x: float, relative_y: float, relative_z: float, profile: CameraProfile) -> tuple[float, float, float]:
    values = (relative_x, relative_y, relative_z)
    if not all((math.isfinite(float(value)) for value in values)):
        raise TargetProjectionError('target coordinates contain NaN or Inf')
    dx = float(relative_x) - profile.mount_x_m
    dy_right = -(float(relative_y) - profile.mount_y_m)
    dz = float(relative_z) - profile.mount_z_m
    pitch = math.radians(profile.pitch_deg)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    depth = cos_pitch * dx + sin_pitch * dz
    camera_z = -sin_pitch * dx + cos_pitch * dz
    if depth <= 1e-06:
        raise TargetProjectionError(f'target is behind the camera or too close; depth={depth:.6f}')
    focal_px = profile.width / (2.0 * math.tan(math.radians(profile.horizontal_fov_deg) / 2.0))
    center_x = profile.width / 2.0
    center_y = profile.height / 2.0
    pixel_x = center_x + focal_px * dy_right / depth
    pixel_y = center_y - focal_px * camera_z / depth
    if not (0.0 <= pixel_x < profile.width and 0.0 <= pixel_y < profile.height):
        raise TargetProjectionError(f'target projects outside the image; pixel=({pixel_x:.2f},{pixel_y:.2f})')
    return (pixel_x, pixel_y, depth)
Position3 = tuple[float, float, float]
BBox4 = tuple[float, float, float, float]
VelocityFilter = Literal['none', 'ema', 'alpha_beta']

class TemporalEntityTrackerError(ValueError):
    """几何帧不满足跟踪合同。"""

@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """单个观测帧附带的身份和时间。"""
    run_id: str
    scene_seed: int
    frame_index: int
    stamp_us: int

    def __post_init__(self) -> None:
        run_id = str(self.run_id).strip()
        if not run_id:
            raise TemporalEntityTrackerError('run_id must not be empty')
        scene_seed = int(self.scene_seed)
        frame_index = int(self.frame_index)
        stamp_us = int(self.stamp_us)
        if frame_index < 0:
            raise TemporalEntityTrackerError('frame_index must be non-negative')
        if stamp_us < 0:
            raise TemporalEntityTrackerError('stamp_us must be non-negative')
        object.__setattr__(self, 'run_id', run_id)
        object.__setattr__(self, 'scene_seed', scene_seed)
        object.__setattr__(self, 'frame_index', frame_index)
        object.__setattr__(self, 'stamp_us', stamp_us)

    @property
    def identity(self) -> tuple[str, int]:
        return (self.run_id, self.scene_seed)

@dataclass(frozen=True, slots=True)
class GeometryObservation:
    """包含几何和语义元数据的实体观测。"""
    entity_id: str
    relative_x: float
    relative_y: float
    relative_z: float
    class_name: str = ''
    color: str = ''
    is_target: bool = False
    visible: bool = True
    bbox: BBox4 | None = None
    confidence: float = 1.0
    run_id: str = ''
    scene_seed: int = 0
    frame_index: int = 0
    stamp_us: int = 0

    def __post_init__(self) -> None:
        entity_id = str(self.entity_id).strip()
        if not entity_id:
            raise TemporalEntityTrackerError('entity_id must not be empty')
        values = (float(self.relative_x), float(self.relative_y), float(self.relative_z))
        if not all((math.isfinite(value) for value in values)):
            raise TemporalEntityTrackerError(f'entity {entity_id!r} position must be finite')
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise TemporalEntityTrackerError(f'entity {entity_id!r} confidence must be in [0, 1]')
        bbox = None if self.bbox is None else tuple((float(v) for v in self.bbox))
        if bbox is not None and (len(bbox) != 4 or not all((math.isfinite(value) for value in bbox))):
            raise TemporalEntityTrackerError(f'entity {entity_id!r} bbox must contain four finite values')
        metadata = FrameMetadata(run_id=self.run_id, scene_seed=self.scene_seed, frame_index=self.frame_index, stamp_us=self.stamp_us)
        object.__setattr__(self, 'entity_id', entity_id)
        object.__setattr__(self, 'relative_x', values[0])
        object.__setattr__(self, 'relative_y', values[1])
        object.__setattr__(self, 'relative_z', values[2])
        object.__setattr__(self, 'class_name', str(self.class_name))
        object.__setattr__(self, 'color', str(self.color))
        object.__setattr__(self, 'is_target', bool(self.is_target))
        object.__setattr__(self, 'visible', bool(self.visible))
        object.__setattr__(self, 'bbox', bbox)
        object.__setattr__(self, 'confidence', confidence)
        object.__setattr__(self, 'run_id', metadata.run_id)
        object.__setattr__(self, 'scene_seed', metadata.scene_seed)
        object.__setattr__(self, 'frame_index', metadata.frame_index)
        object.__setattr__(self, 'stamp_us', metadata.stamp_us)

    @property
    def position(self) -> Position3:
        return (self.relative_x, self.relative_y, self.relative_z)

    @property
    def metadata(self) -> FrameMetadata:
        return FrameMetadata(self.run_id, self.scene_seed, self.frame_index, self.stamp_us)

@dataclass(frozen=True, slots=True)
class TrackedEntity:
    """当前几何及显式有效性门控的速度估计。"""
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
    source: str = 'temporal_tracker'

    @property
    def position(self) -> Position3:
        return (self.relative_x, self.relative_y, self.relative_z)

    @property
    def velocity(self) -> Position3:
        return (self.relative_velocity_x, self.relative_velocity_y, self.relative_velocity_z)

    def as_entity_kwargs(self) -> dict[str, object]:
        """返回 ``EntityState`` 消息接受的字段。"""
        bbox = self.bbox or (0.0, 0.0, 0.0, 0.0)
        return {'entity_id': self.entity_id, 'class_name': self.class_name, 'color': self.color, 'is_target': self.is_target, 'visible': self.visible, 'relative_x': self.relative_x, 'relative_y': self.relative_y, 'relative_z': self.relative_z, 'relative_velocity_x': self.relative_velocity_x, 'relative_velocity_y': self.relative_velocity_y, 'relative_velocity_z': self.relative_velocity_z, 'valid': self.valid, 'source': self.source, 'bbox_x_min': bbox[0], 'bbox_y_min': bbox[1], 'bbox_x_max': bbox[2], 'bbox_y_max': bbox[3], 'bbox_valid': self.bbox is not None, 'confidence': self.confidence, 'velocity_valid': self.velocity_valid}

@dataclass(slots=True)
class _TrackState:
    position: Position3
    filter_position: Position3
    frame_index: int
    stamp_us: int
    velocity: Position3 = (0.0, 0.0, 0.0)
    velocity_valid: bool = False

class TemporalEntityTracker:
    """跟踪几何观测并用有限差分估计速度。"""

    def __init__(self, *, ttl_frames: int=2, ttl_sec: float | None=None, velocity_filter: VelocityFilter='none', alpha: float=1.0, beta: float=0.85) -> None:
        if ttl_frames < 0:
            raise ValueError('ttl_frames must be non-negative')
        if ttl_sec is not None and ttl_sec <= 0.0:
            raise ValueError('ttl_sec must be positive when provided')
        if velocity_filter not in {'none', 'ema', 'alpha_beta'}:
            raise ValueError('velocity_filter must be none, ema, or alpha_beta')
        if not 0.0 < alpha <= 1.0:
            raise ValueError('alpha must be in (0, 1]')
        if not 0.0 < beta <= 2.0:
            raise ValueError('beta must be in (0, 2]')
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

    def update(self, observations: Iterable[GeometryObservation], *, frame: FrameMetadata | None=None) -> tuple[TrackedEntity, ...]:
        """消费一帧并返回其中实体记录。

        A frame with no observations can be represented by passing ``frame``;
        this advances TTL bookkeeping without fabricating entity positions.
        Frames with a non-increasing index are ignored.  A timestamp regression
        is accepted as a geometry update but invalidates velocity for that
        frame, rather than guessing a time interval.
        """
        items = tuple(observations)
        metadata = self._metadata_for(items, frame)
        if any((item.metadata != metadata for item in items)):
            raise TemporalEntityTrackerError('all observations in a frame must share run/scene/frame/stamp')
        ids = [item.entity_id for item in items]
        if len(ids) != len(set(ids)):
            raise TemporalEntityTrackerError('duplicate entity_id in frame')
        if self._identity != metadata.identity:
            self._tracks.clear()
            self._identity = metadata.identity
            self._last_frame_index = None
            self._last_stamp_us = None
        if self._last_frame_index is not None and metadata.frame_index <= self._last_frame_index:
            return ()
        monotonic_stamp = self._last_stamp_us is None or metadata.stamp_us > self._last_stamp_us
        self._expire_tracks(metadata)
        records = tuple((self._record_for(item, metadata, monotonic_stamp) for item in items))
        self._last_frame_index = metadata.frame_index
        self._last_stamp_us = metadata.stamp_us
        return records
    process_frame = update

    def _metadata_for(self, items: Sequence[GeometryObservation], frame: FrameMetadata | None) -> FrameMetadata:
        if frame is not None and (not isinstance(frame, FrameMetadata)):
            raise TypeError('frame must be FrameMetadata')
        if items:
            metadata = items[0].metadata
            if frame is not None and frame != metadata:
                raise TemporalEntityTrackerError('explicit frame metadata does not match observation')
            return metadata
        if frame is None:
            raise TemporalEntityTrackerError('an empty frame requires explicit FrameMetadata')
        return frame

    def _expire_tracks(self, metadata: FrameMetadata) -> None:
        expired = []
        for entity_id, state in self._tracks.items():
            frame_gap = metadata.frame_index - state.frame_index
            time_gap = (metadata.stamp_us - state.stamp_us) / 1000000.0
            too_many_frames = frame_gap > self.ttl_frames
            too_long = self.ttl_sec is not None and time_gap > self.ttl_sec
            if too_many_frames or (time_gap >= 0.0 and too_long):
                expired.append(entity_id)
        for entity_id in expired:
            del self._tracks[entity_id]

    def _record_for(self, item: GeometryObservation, metadata: FrameMetadata, monotonic_stamp: bool) -> TrackedEntity:
        state = self._tracks.get(item.entity_id)
        velocity = (0.0, 0.0, 0.0)
        velocity_valid = False
        frame_gap = 0
        if state is not None:
            frame_gap = metadata.frame_index - state.frame_index
            dt_sec = (metadata.stamp_us - state.stamp_us) / 1000000.0
            if monotonic_stamp and frame_gap > 0 and (dt_sec > 0.0):
                raw_velocity = tuple(((current - previous) / dt_sec for current, previous in zip(item.position, state.position)))
                velocity, filter_position = self._filter_velocity(state, raw_velocity, dt_sec, item)
                velocity_valid = all((math.isfinite(value) for value in velocity))
                if not velocity_valid:
                    velocity = (0.0, 0.0, 0.0)
                    filter_position = item.position
            else:
                filter_position = item.position
        else:
            filter_position = item.position
        self._tracks[item.entity_id] = _TrackState(position=item.position, filter_position=filter_position, frame_index=metadata.frame_index, stamp_us=metadata.stamp_us, velocity=velocity, velocity_valid=velocity_valid)
        return TrackedEntity(entity_id=item.entity_id, relative_x=item.relative_x, relative_y=item.relative_y, relative_z=item.relative_z, relative_velocity_x=velocity[0], relative_velocity_y=velocity[1], relative_velocity_z=velocity[2], velocity_valid=velocity_valid, class_name=item.class_name, color=item.color, is_target=item.is_target, visible=item.visible, bbox=item.bbox, confidence=item.confidence, run_id=metadata.run_id, scene_seed=metadata.scene_seed, frame_index=metadata.frame_index, stamp_us=metadata.stamp_us, frame_gap=frame_gap)

    def _filter_velocity(self, state: _TrackState, raw_velocity: Position3, dt_sec: float, item: GeometryObservation) -> tuple[Position3, Position3]:
        if not state.velocity_valid or self.velocity_filter == 'none':
            return (raw_velocity, item.position)
        if self.velocity_filter == 'ema':
            return (tuple((self.alpha * raw + (1.0 - self.alpha) * previous for raw, previous in zip(raw_velocity, state.velocity))), item.position)
        residual = tuple((current - (previous + velocity * dt_sec) for current, previous, velocity in zip(item.position, state.filter_position, state.velocity)))
        predicted_position = tuple((previous + velocity * dt_sec for previous, velocity in zip(state.filter_position, state.velocity)))
        corrected_position = tuple((predicted + self.alpha * error for predicted, error in zip(predicted_position, residual)))
        return (tuple((velocity + self.beta * error / dt_sec for velocity, error in zip(state.velocity, residual))), corrected_position)
__all__ = ['FrameMetadata', 'GeometryObservation', 'TemporalEntityTracker', 'TemporalEntityTrackerError', 'TrackedEntity']