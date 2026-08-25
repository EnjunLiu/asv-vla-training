"""Unified vision model: geometry + embedding from one MobileNet backbone."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torchvision

TASK_EMBEDDING_DIM = 256
ENTITY_EMBEDDING_DIM = 64
POSITION_SCALE_M = np.asarray((40.0, 40.0), dtype=np.float32)
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VISION_SCHEMA = "vision_v3_anon_slots"
BACKBONE_ID = "torchvision:mobilenet_v3_small:IMAGENET1K_V1"


def slot_entity_id(index: int) -> str:
    return f"slot_{int(index)}"


def infer_slot_count_from_head(
    *,
    head_weight: Tensor | None = None,
    out_per_slot: int,
    slot_count: int | None = None,
) -> int:
    if slot_count is not None:
        count = int(slot_count)
        if count <= 0:
            raise PerceptionError(f"slot_count must be positive, got {count}")
        return count
    if head_weight is None:
        raise PerceptionError("slot_count unavailable")
    rows = int(head_weight.shape[0])
    if rows <= 0 or rows % int(out_per_slot) != 0:
        raise PerceptionError(f"cannot infer slot_count from head shape {tuple(head_weight.shape)}")
    return int(rows // int(out_per_slot))

VelocityFilter = Literal["none", "ema", "alpha_beta"]
Position2 = tuple[float, float]
Velocity2 = tuple[float, float]


class PerceptionError(RuntimeError):
    pass


class InvalidImageError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraProfile:
    width: int = 1280
    height: int = 720
    horizontal_fov_deg: float = 90.0
    mount_x_m: float = 0.42
    mount_y_m: float = 0.0
    mount_z_m: float = 0.2
    pitch_deg: float = -5.0


def validate_task_embedding(embedding: object) -> np.ndarray:
    array = np.asarray(embedding, dtype=np.float32)
    if array.shape != (TASK_EMBEDDING_DIM,):
        raise PerceptionError(f"task embedding shape {array.shape}")
    return array


def decode_camera_image(data: bytes | bytearray, encoding: str) -> Image.Image:
    if encoding.strip().lower() not in {"jpeg", "jpg"}:
        raise InvalidImageError(f"unsupported encoding {encoding!r}")
    with Image.open(BytesIO(bytes(data))) as source:
        source.load()
        return source.convert("RGB")


def _prepare_image(image: Image.Image, device: torch.device) -> Tensor:
    resized = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


@dataclass(frozen=True)
class SlotPrediction:
    entity_id: str
    visible: bool
    confidence: float
    relative_x: float
    relative_y: float
    embedding: np.ndarray


class VisionModel(nn.Module):
    """MobileNet backbone + language-conditioned trunk + anonymous slot heads."""

    def __init__(
        self,
        *,
        slot_count: int,
        task_embedding_dim: int = TASK_EMBEDDING_DIM,
        embed_dim: int = ENTITY_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        if int(slot_count) <= 0:
            raise PerceptionError(f"slot_count must be positive, got {slot_count}")
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.backbone = torchvision.models.mobilenet_v3_small(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(
            nn.Linear(576 + task_embedding_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        # visible logit + relative_xy + embedding
        self.out_per_slot = 1 + 2 + embed_dim
        self.slot_count = int(slot_count)
        self.head = nn.Linear(256, self.slot_count * self.out_per_slot)
        self.embed_dim = embed_dim

    def forward(self, image: Tensor, task_embedding: Tensor) -> Tensor:
        features = self.pool(self.backbone(image)).flatten(1)
        fused = self.trunk(torch.cat((features, task_embedding), dim=-1))
        return self.head(fused).view(-1, self.slot_count, self.out_per_slot)

    def decode(self, raw: Tensor) -> list[SlotPrediction]:
        predictions: list[SlotPrediction] = []
        values = raw.detach().cpu().numpy()
        if values.ndim == 3:
            values = values[0]
        for index in range(self.slot_count):
            slot = values[index]
            visible_logit = float(slot[0])
            visible = visible_logit >= 0.0
            confidence = 1.0 / (1.0 + math.exp(-np.clip(visible_logit, -30.0, 30.0)))
            geometry = slot[1:3] * POSITION_SCALE_M
            embedding = slot[3 : 3 + self.embed_dim]
            norm = float(np.linalg.norm(embedding))
            if norm > 1e-8:
                embedding = embedding / norm
            predictions.append(
                SlotPrediction(
                    entity_id=slot_entity_id(index),
                    visible=visible,
                    confidence=confidence,
                    relative_x=float(geometry[0]),
                    relative_y=float(geometry[1]),
                    embedding=embedding.astype(np.float32),
                )
            )
        return predictions


@dataclass
class VisionRuntime:
    model: VisionModel
    device: torch.device

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cuda") -> "VisionRuntime":
        target = torch.device(device)
        checkpoint = torch.load(Path(path), map_location=target, weights_only=False)
        state = checkpoint["model_state_dict"]
        out_per_slot = 1 + 2 + ENTITY_EMBEDDING_DIM
        slot_count = infer_slot_count_from_head(
            head_weight=state.get("head.weight"),
            out_per_slot=out_per_slot,
            slot_count=checkpoint.get("slot_count"),
        )
        model = VisionModel(slot_count=slot_count)
        model.load_state_dict(state)
        model.to(target)
        model.eval()
        return cls(model=model, device=target)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": VISION_SCHEMA,
                "backbone_id": BACKBONE_ID,
                "slot_count": int(self.model.slot_count),
                "model_state_dict": self.model.state_dict(),
            },
            output,
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image, task_embedding: np.ndarray) -> list[SlotPrediction]:
        tensor = _prepare_image(image, self.device)
        task_embedding = torch.as_tensor(
            validate_task_embedding(task_embedding),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        raw = self.model(tensor, task_embedding)
        return self.model.decode(raw)


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    run_id: str
    scene_seed: int
    frame_index: int
    stamp_us: int


@dataclass(frozen=True, slots=True)
class EntityPhysicalObservation:
    entity_id: str
    relative_x: float
    relative_y: float
    visible: bool = True
    confidence: float = 1.0
    run_id: str = ""
    scene_seed: int = 0
    frame_index: int = 0
    stamp_us: int = 0


@dataclass(frozen=True, slots=True)
class ObservedEntity:
    entity_id: str
    relative_x: float
    relative_y: float
    relative_velocity_x: float
    relative_velocity_y: float
    velocity_valid: bool
    visible: bool
    confidence: float
    run_id: str
    scene_seed: int
    frame_index: int
    stamp_us: int
    embedding: np.ndarray
    valid: bool = True

    def as_entity_kwargs(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "visible": self.visible,
            "relative_x": self.relative_x,
            "relative_y": self.relative_y,
            "relative_velocity_x": self.relative_velocity_x,
            "relative_velocity_y": self.relative_velocity_y,
            "valid": self.valid,
            "confidence": self.confidence,
            "velocity_valid": self.velocity_valid,
            "entity_embedding": self.embedding.tolist(),
        }


@dataclass(slots=True)
class _EntityVelocityState:
    position: Position2
    frame_index: int
    stamp_us: int
    velocity: Velocity2 = (0.0, 0.0)
    velocity_valid: bool = False


class EntityVelocityObserver:
    """Finite-difference planar velocity with optional EMA smoothing."""

    def __init__(self, *, velocity_filter: VelocityFilter = "ema", alpha: float = 0.35) -> None:
        self.velocity_filter = velocity_filter
        self.alpha = alpha
        self._states: dict[str, _EntityVelocityState] = {}
        self._identity: tuple[str, int] | None = None
        self._last_frame_index: int | None = None

    def reset(self) -> None:
        self._states.clear()
        self._identity = None
        self._last_frame_index = None

    def update(
        self,
        observations: Iterable[EntityPhysicalObservation],
        *,
        frame: FrameMetadata,
    ) -> tuple[ObservedEntity, ...]:
        items = tuple(observations)
        if self._identity != (frame.run_id, frame.scene_seed):
            self.reset()
            self._identity = (frame.run_id, frame.scene_seed)
        if self._last_frame_index is not None and frame.frame_index <= self._last_frame_index:
            return ()
        records: list[ObservedEntity] = []
        for item in items:
            state = self._states.get(item.entity_id)
            velocity = (0.0, 0.0)
            velocity_valid = False
            if state is not None:
                dt_sec = (frame.stamp_us - state.stamp_us) / 1.0e6
                if dt_sec > 0.0:
                    raw = (
                        (item.relative_x - state.position[0]) / dt_sec,
                        (item.relative_y - state.position[1]) / dt_sec,
                    )
                    if self.velocity_filter == "ema" and state.velocity_valid:
                        velocity = (
                            self.alpha * raw[0] + (1.0 - self.alpha) * state.velocity[0],
                            self.alpha * raw[1] + (1.0 - self.alpha) * state.velocity[1],
                        )
                    else:
                        velocity = raw
                    velocity_valid = True
            self._states[item.entity_id] = _EntityVelocityState(
                position=(item.relative_x, item.relative_y),
                frame_index=frame.frame_index,
                stamp_us=frame.stamp_us,
                velocity=velocity,
                velocity_valid=velocity_valid,
            )
            records.append(
                ObservedEntity(
                    entity_id=item.entity_id,
                    relative_x=item.relative_x,
                    relative_y=item.relative_y,
                    relative_velocity_x=velocity[0],
                    relative_velocity_y=velocity[1],
                    velocity_valid=velocity_valid,
                    visible=item.visible,
                    confidence=item.confidence,
                    run_id=frame.run_id,
                    scene_seed=frame.scene_seed,
                    frame_index=frame.frame_index,
                    stamp_us=frame.stamp_us,
                    embedding=np.zeros(ENTITY_EMBEDDING_DIM, dtype=np.float32),
                )
            )
        self._last_frame_index = frame.frame_index
        return tuple(records)


def predict_frame(
    runtime: VisionRuntime,
    image: Image.Image,
    task_embedding: np.ndarray,
    frame: FrameMetadata,
    observer: EntityVelocityObserver,
    *,
    embeddings_by_id: Mapping[str, np.ndarray] | None = None,
) -> list[ObservedEntity]:
    predictions = runtime.predict(image, task_embedding)
    observations = [
        EntityPhysicalObservation(
            entity_id=pred.entity_id,
            relative_x=pred.relative_x,
            relative_y=pred.relative_y,
            visible=pred.visible,
            confidence=pred.confidence,
            run_id=frame.run_id,
            scene_seed=frame.scene_seed,
            frame_index=frame.frame_index,
            stamp_us=frame.stamp_us,
        )
        for pred in predictions
        if pred.visible
    ]
    tracked = observer.update(observations, frame=frame)
    embed_map = embeddings_by_id or {
        pred.entity_id: pred.embedding for pred in predictions if pred.visible
    }
    result: list[ObservedEntity] = []
    for item in tracked:
        embedding = embed_map.get(item.entity_id, item.embedding)
        result.append(
            ObservedEntity(
                entity_id=item.entity_id,
                relative_x=item.relative_x,
                relative_y=item.relative_y,
                relative_velocity_x=item.relative_velocity_x,
                relative_velocity_y=item.relative_velocity_y,
                velocity_valid=item.velocity_valid,
                visible=item.visible,
                confidence=item.confidence,
                run_id=item.run_id,
                scene_seed=item.scene_seed,
                frame_index=item.frame_index,
                stamp_us=item.stamp_us,
                embedding=embedding,
                valid=item.valid,
            )
        )
    return result


def build_entity_cache(
    records: Sequence[Any],
    runtime: VisionRuntime,
    embeddings: Mapping[str, np.ndarray],
) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    from data import language_for_record
    from PIL import Image as PILImage

    grouped: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for record in records:
        grouped[(record.run_id, record.scene_seed)].append(record)
    cache: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for group in grouped.values():
        group.sort(key=lambda item: (int(item.ego["stamp_us"]), int(item.frame_index)))
        observer = EntityVelocityObserver(velocity_filter="ema", alpha=0.35)
        for record in group:
            image = PILImage.open(record.image_path).convert("RGB")
            frame = FrameMetadata(
                run_id=record.run_id,
                scene_seed=record.scene_seed,
                frame_index=record.frame_index,
                stamp_us=int(record.ego["stamp_us"]),
            )
            tracked = predict_frame(
                runtime,
                image,
                language_for_record(record, embeddings),
                frame,
                observer,
            )
            key = (record.slot_id, record.run_id, record.scene_seed, record.frame_index)
            cache[key] = [item.as_entity_kwargs() for item in tracked]
    return cache
