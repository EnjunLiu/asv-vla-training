from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

try:
    from .entity_contract import ENTITY_EMBEDDING_DIM
    from .perception import (
        BACKBONE_ID,
        CameraProfile,
        TargetProjectionError,
        project_target_to_pixel,
    )
except ImportError:
    from entity_contract import ENTITY_EMBEDDING_DIM
    from perception import (
        BACKBONE_ID,
        CameraProfile,
        TargetProjectionError,
        project_target_to_pixel,
    )

ENTITY_EMBEDDING_SCHEMA = "entity_embedding_v1"
ENTITY_EMBEDDING_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ENTITY_SLOT_IDS = ("target_red", "target_blue", "target_left", "target_right")


class EntityEmbeddingError(RuntimeError):
    """实体 embedding 模型或输入不满足合同时抛出。"""


def crop_entity_patch(
    image: Image.Image,
    relative_x: float,
    relative_y: float,
    relative_z: float,
    profile: CameraProfile,
    *,
    crop_size_px: int = ENTITY_EMBEDDING_INPUT_SIZE,
) -> Image.Image:
    pixel_x, pixel_y, _ = project_target_to_pixel(
        relative_x, relative_y, relative_z, profile
    )
    half = float(crop_size_px) / 2.0
    left = int(max(0.0, pixel_x - half))
    top = int(max(0.0, pixel_y - half))
    right = int(min(float(profile.width), pixel_x + half))
    bottom = int(min(float(profile.height), pixel_y + half))
    if right <= left or bottom <= top:
        raise EntityEmbeddingError("entity crop has zero area")
    patch = image.crop((left, top, right, bottom))
    return patch.resize((crop_size_px, crop_size_px), Image.Resampling.BILINEAR)


def _prepare_tensor(patch: Image.Image, *, device: str, torch: Any) -> Any:
    array = np.asarray(patch.convert("RGB"), dtype=np.float32) / 255.0
    if array.shape != (ENTITY_EMBEDDING_INPUT_SIZE, ENTITY_EMBEDDING_INPUT_SIZE, 3):
        raise EntityEmbeddingError(f"unexpected crop shape {array.shape}")
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def encode_patch_batch(
    patches: Sequence[Image.Image],
    *,
    projector: Any,
    backbone: Any,
    torch: Any,
    device: str,
) -> np.ndarray:
    if not patches:
        return np.zeros((0, ENTITY_EMBEDDING_DIM), dtype=np.float32)
    tensors = [_prepare_tensor(patch, device=device, torch=torch) for patch in patches]
    batch = torch.cat(tensors, dim=0)
    with torch.inference_mode():
        features = backbone(batch)
        embeddings = torch.nn.functional.normalize(projector(features), dim=-1)
    values = embeddings.detach().cpu().numpy().astype(np.float32)
    if values.shape != (len(patches), ENTITY_EMBEDDING_DIM) or not np.all(np.isfinite(values)):
        raise EntityEmbeddingError("entity embedding batch is invalid")
    return values


@dataclass(frozen=True)
class EntityEmbeddingHead:
    backbone_id: str
    entity_embedding_dim: int
    input_size: int
    projector_state: Mapping[str, Any]
    classifier_state: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.backbone_id != BACKBONE_ID:
            raise EntityEmbeddingError(
                f"unsupported backbone_id={self.backbone_id!r}; expected {BACKBONE_ID!r}"
            )
        if int(self.entity_embedding_dim) != ENTITY_EMBEDDING_DIM:
            raise EntityEmbeddingError(
                f"entity_embedding_dim must be {ENTITY_EMBEDDING_DIM}, "
                f"got {self.entity_embedding_dim}"
            )
        if int(self.input_size) != ENTITY_EMBEDDING_INPUT_SIZE:
            raise EntityEmbeddingError(
                f"input_size must be {ENTITY_EMBEDDING_INPUT_SIZE}, got {self.input_size}"
            )

    @staticmethod
    def _build_modules(torch: Any, *, device: str) -> tuple[Any, Any, Any]:
        import torchvision

        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        backbone.classifier = torch.nn.Identity()
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone = backbone.to(device)
        feature_dim = int(
            backbone(
                torch.zeros(
                    1,
                    3,
                    ENTITY_EMBEDDING_INPUT_SIZE,
                    ENTITY_EMBEDDING_INPUT_SIZE,
                    device=device,
                )
            ).shape[-1]
        )
        projector = torch.nn.Linear(feature_dim, ENTITY_EMBEDDING_DIM).to(device)
        classifier = torch.nn.Linear(ENTITY_EMBEDDING_DIM, len(ENTITY_SLOT_IDS)).to(device)
        return backbone, projector, classifier

    @classmethod
    def create_trainable(cls, *, device: str = "cuda") -> "TrainableEntityEmbeddingHead":
        torch = _require_torch(device)
        backbone, projector, classifier = cls._build_modules(torch, device=device)
        return TrainableEntityEmbeddingHead(
            backbone=backbone,
            projector=projector,
            classifier=classifier,
            device=device,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cuda") -> "EntityEmbeddingRuntime":
        _require_torch(device)
        checkpoint_path = Path(path).expanduser()
        if not checkpoint_path.is_file():
            raise EntityEmbeddingError(f"entity embedding checkpoint not found: {checkpoint_path}")
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != ENTITY_EMBEDDING_SCHEMA:
            raise EntityEmbeddingError("entity embedding schema mismatch")
        head = cls(
            backbone_id=str(checkpoint["backbone_id"]),
            entity_embedding_dim=int(checkpoint["entity_embedding_dim"]),
            input_size=int(checkpoint["input_size"]),
            projector_state=checkpoint["projector_state_dict"],
            classifier_state=checkpoint.get("classifier_state_dict"),
        )
        return EntityEmbeddingRuntime(head=head, device=device)


@dataclass
class TrainableEntityEmbeddingHead:
    backbone: Any
    projector: Any
    classifier: Any
    device: str

    def encode_batch(self, patches: Sequence[Image.Image]) -> np.ndarray:
        torch = _require_torch(self.device)
        return encode_patch_batch(
            patches,
            projector=self.projector,
            backbone=self.backbone,
            torch=torch,
            device=self.device,
        )

    def fit_entity_classifier(
        self,
        patches: Sequence[Image.Image],
        labels: Sequence[int],
        *,
        epochs: int = 12,
        batch_size: int = 64,
        learning_rate: float = 3.0e-3,
    ) -> dict[str, float]:
        torch = _require_torch(self.device)
        if len(patches) != len(labels) or not patches:
            raise EntityEmbeddingError("embedding training set is empty")
        label_tensor = torch.tensor(list(labels), dtype=torch.long, device=self.device)
        optimizer = torch.optim.AdamW(
            list(self.projector.parameters()) + list(self.classifier.parameters()),
            lr=learning_rate,
            weight_decay=1.0e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        indices = torch.arange(len(patches), device=self.device)
        self.projector.train()
        self.classifier.train()
        generator = torch.Generator().manual_seed(20260823)
        last_loss = 0.0
        for _ in range(epochs):
            permutation = indices[torch.randperm(len(indices), generator=generator)]
            for start in range(0, len(permutation), batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_patches = [
                    patches[int(index)] for index in batch_indices.cpu().tolist()
                ]
                tensors = [
                    _prepare_tensor(patch, device=self.device, torch=torch)
                    for patch in batch_patches
                ]
                batch = torch.cat(tensors, dim=0)
                with torch.no_grad():
                    features = self.backbone(batch)
                embeddings = torch.nn.functional.normalize(self.projector(features), dim=-1)
                logits = self.classifier(embeddings)
                loss = loss_fn(logits, label_tensor[batch_indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())
        self.projector.eval()
        self.classifier.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for start in range(0, len(patches), batch_size):
                batch_patches = patches[start : start + batch_size]
                batch_labels = labels[start : start + batch_size]
                tensors = [
                    _prepare_tensor(patch, device=self.device, torch=torch)
                    for patch in batch_patches
                ]
                batch = torch.cat(tensors, dim=0)
                embeddings = torch.nn.functional.normalize(
                    self.projector(self.backbone(batch)),
                    dim=-1,
                )
                predictions = self.classifier(embeddings).argmax(dim=-1)
                correct += int((predictions.cpu() == torch.tensor(batch_labels)).sum())
                total += len(batch_labels)
        accuracy = float(correct / max(total, 1))
        return {"train_loss": last_loss, "train_accuracy": accuracy}

    def save_checkpoint(self, path: str | Path) -> None:
        import torch

        payload = {
            "schema_version": ENTITY_EMBEDDING_SCHEMA,
            "backbone_id": BACKBONE_ID,
            "entity_embedding_dim": ENTITY_EMBEDDING_DIM,
            "input_size": ENTITY_EMBEDDING_INPUT_SIZE,
            "projector_state_dict": self.projector.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
        }
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)


@dataclass
class EntityEmbeddingRuntime:
    head: EntityEmbeddingHead
    device: str
    _backbone: Any | None = None
    _projector: Any | None = None

    def _modules(self) -> tuple[Any, Any, Any]:
        if self._backbone is None or self._projector is None:
            torch = _require_torch(self.device)
            backbone, projector, _ = EntityEmbeddingHead._build_modules(torch, device=self.device)
            projector.load_state_dict(self.head.projector_state)
            projector.eval()
            self._backbone = backbone
            self._projector = projector
        return _require_torch(self.device), self._backbone, self._projector

    def encode_entity(
        self,
        image: Image.Image,
        relative_x: float,
        relative_y: float,
        relative_z: float,
        profile: CameraProfile,
    ) -> np.ndarray:
        patch = crop_entity_patch(image, relative_x, relative_y, relative_z, profile)
        torch, backbone, projector = self._modules()
        return encode_patch_batch(
            [patch],
            projector=projector,
            backbone=backbone,
            torch=torch,
            device=self.device,
        )[0]

    def encode_entities(
        self,
        image: Image.Image,
        entities: Iterable[Mapping[str, Any]],
        profile: CameraProfile,
    ) -> dict[str, np.ndarray]:
        encoded: dict[str, np.ndarray] = {}
        for item in entities:
            entity_id = str(item.get("entity_id", "")).strip()
            if not entity_id:
                continue
            if not bool(item.get("visible", False)) or not bool(item.get("valid", False)):
                encoded[entity_id] = np.zeros(ENTITY_EMBEDDING_DIM, dtype=np.float32)
                continue
            position = item.get("relative_position_m", (0.0, 0.0, 0.0))
            try:
                encoded[entity_id] = self.encode_entity(
                    image,
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    profile,
                )
            except (EntityEmbeddingError, TargetProjectionError, Exception):
                encoded[entity_id] = np.zeros(ENTITY_EMBEDDING_DIM, dtype=np.float32)
        return encoded


def _require_torch(device: str) -> Any:
    normalized = str(device).strip().lower()
    if not normalized.startswith("cuda"):
        raise EntityEmbeddingError(f"entity embedding requires cuda device, got {device!r}")
    try:
        import torch
    except Exception as exc:
        raise EntityEmbeddingError(f"torch is unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise EntityEmbeddingError("CUDA is unavailable for entity embedding")
    return torch


def entity_slot_index(entity_id: str) -> int:
    try:
        return ENTITY_SLOT_IDS.index(str(entity_id))
    except ValueError as exc:
        raise EntityEmbeddingError(f"unsupported entity_id {entity_id!r}") from exc


def build_training_crops(
    image_path: str | Path,
    entities: Sequence[Mapping[str, Any]],
    profile: CameraProfile,
) -> tuple[list[Image.Image], list[int]]:
    image = Image.open(image_path).convert("RGB")
    patches: list[Image.Image] = []
    labels: list[int] = []
    for item in entities:
        if not bool(item.get("visible", False)) or not bool(item.get("valid", False)):
            continue
        entity_id = str(item["entity_id"])
        position = item["relative_position_m"]
        try:
            patch = crop_entity_patch(
                image,
                float(position[0]),
                float(position[1]),
                float(position[2]),
                profile,
            )
        except (EntityEmbeddingError, TargetProjectionError):
            continue
        patches.append(patch)
        labels.append(entity_slot_index(entity_id))
    return patches, labels
