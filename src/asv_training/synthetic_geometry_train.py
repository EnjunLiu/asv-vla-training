"""PC-only synthetic geometry training for the single-point policy.

This module creates structured entity-geometry samples for the existing
``SmallActionPolicy``.  It deliberately uses the checked-in expert
implementation and entity tensor builder; it does not import ROS, UE5, or
start a Jetson workload.  Deployment training must use the real PC-exported
Qwen embeddings; a deterministic language vector remains available only for
dependency-free geometry smoke tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


from asv_vla.expert_trajectory import (
    ExpertActionResult,
    generate_expert_trajectory,
    task_from_labels,
)
from asv_vla.task_entity_tensor import build_entity_tensor


COLORS = ("red", "blue")
DISTANCE_BUCKETS = ("3m", "4m", "10m")
DISTANCE_METERS = {"3m": 3.0, "4m": 4.0, "10m": 10.0}

# The checked-in L7/S2 perception traces span approximately
# x=3.830759..4.840371 and y=-1.298636..0.497503.  Keep a small margin while
# retaining the actual runtime points used by the focused regression tests.
L7_RUNTIME_X_RANGE_M = (3.80, 4.85)
L7_RUNTIME_Y_RANGE_M = (-1.35, 0.55)
L7_RUNTIME_POINTS_M = ((4.74, 0.49), (3.94, -1.27))
LANGUAGE_DIM = 256
DATASET_SCHEMA_VERSION = "synthetic_geometry_dataset_v1"
CHECKPOINT_SCHEMA_VERSION = "synthetic_geometry_single_point_v2"


@dataclass(frozen=True)
class LanguageEmbeddingTable:
    """Validated Qwen task embeddings indexed by canonical task IDs."""

    instruction_ids: tuple[str, ...]
    instruction_texts: tuple[str, ...]
    embeddings: np.ndarray
    source_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != LANGUAGE_DIM:
            raise ValueError(
                "language embeddings must have shape [N, 256], "
                f"got {values.shape}"
            )
        if len(self.instruction_ids) != len(self.instruction_texts):
            raise ValueError("instruction IDs/texts must have equal lengths")
        if len(self.instruction_ids) != values.shape[0]:
            raise ValueError("instruction metadata and embeddings are misaligned")
        if not np.all(np.isfinite(values)):
            raise ValueError("language embeddings contain NaN or Inf")
        if not self.source_sha256:
            raise ValueError("language embedding source SHA256 is required")
        object.__setattr__(self, "embeddings", np.ascontiguousarray(values))

    def rows_for(self, color: str, distance_bucket: str) -> tuple[int, ...]:
        prefix = f"follow_{str(color).strip().casefold()}_{str(distance_bucket).strip().casefold()}_"
        rows = tuple(
            index
            for index, instruction_id in enumerate(self.instruction_ids)
            if instruction_id.startswith(prefix)
        )
        if not rows:
            raise ValueError(
                f"language table has no rows for {prefix.rstrip('_')!r}"
            )
        return rows

    def choose(
        self,
        color: str,
        distance_bucket: str,
        *,
        row_offset: int = 0,
    ) -> tuple[np.ndarray, str, str]:
        rows = self.rows_for(color, distance_bucket)
        row = rows[int(row_offset) % len(rows)]
        return (
            self.embeddings[row].copy(),
            self.instruction_ids[row],
            self.instruction_texts[row],
        )


def load_language_embeddings(path: str | Path) -> LanguageEmbeddingTable:
    """Load and validate the PC-exported Qwen embedding manifest."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"language embedding file not found: {source}")
    with np.load(source, allow_pickle=False) as arrays:
        required = {
            "instruction_ids",
            "instruction_texts",
            "embeddings",
        }
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(
                f"language embedding file is missing keys: {sorted(missing)}"
            )
        instruction_ids = tuple(
            str(value).strip() for value in np.asarray(arrays["instruction_ids"])
        )
        instruction_texts = tuple(
            str(value).strip()
            for value in np.asarray(arrays["instruction_texts"])
        )
        embeddings = np.asarray(arrays["embeddings"], dtype=np.float32).copy()
    if any(not value for value in instruction_ids + instruction_texts):
        raise ValueError("language embedding IDs and texts must not be empty")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    table = LanguageEmbeddingTable(
        instruction_ids=instruction_ids,
        instruction_texts=instruction_texts,
        embeddings=embeddings,
        source_path=str(source),
        source_sha256=source_sha256,
    )
    for color in COLORS:
        for distance_bucket in DISTANCE_BUCKETS:
            table.rows_for(color, distance_bucket)
    return table


@dataclass(frozen=True)
class SyntheticGeometryDataset:
    """Arrays matching the policy's online input and label contract."""

    language: np.ndarray
    entity_geometry: np.ndarray
    entity_geometry_mask: np.ndarray
    previous_action: np.ndarray
    previous_action_valid: np.ndarray
    target_action: np.ndarray
    target_stop: np.ndarray
    language_text: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]
    language_source: str = "synthetic_surrogate"
    language_source_sha256: str = ""

    def __post_init__(self) -> None:
        count = int(self.language.shape[0])
        expected = {
            "entity_geometry": (count, 16, 16),
            "entity_geometry_mask": (count, 16),
            "previous_action": (count, 2),
            "previous_action_valid": (count,),
            "target_action": (count, 2),
            "target_stop": (count,),
        }
        if self.language.ndim != 2:
            raise ValueError("language must have shape [N, language_dim]")
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape {value.shape} does not match {shape}")
        if len(self.language_text) != count or len(self.metadata) != count:
            raise ValueError("language_text and metadata must have one row per sample")
        if not str(self.language_source).strip():
            raise ValueError("language_source must not be empty")
        for name in (
            "language",
            "entity_geometry",
            "previous_action",
            "target_action",
        ):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} contains NaN or Inf")

    def as_policy_inputs(self) -> dict[str, np.ndarray]:
        """Return a batch mapping accepted by ``SmallActionPolicy``."""

        return {
            "language": self.language,
            "entity_geometry": self.entity_geometry,
            "language_valid": np.ones(len(self.language), dtype=np.bool_),
            "entity_geometry_mask": self.entity_geometry_mask,
            "previous_action": self.previous_action,
            "previous_action_valid": self.previous_action_valid,
            "policy_input_valid": np.ones(len(self.language), dtype=np.bool_),
        }


def _entity(
    entity_id: str,
    color: str,
    x: float,
    y: float,
    *,
    is_target: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        entity_id=entity_id,
        color=color,
        relative_x=float(x),
        relative_y=float(y),
        relative_z=0.0,
        relative_velocity_x=0.0,
        relative_velocity_y=0.0,
        relative_velocity_z=0.0,
        valid=True,
        visible=True,
        is_target=is_target,
    )


def _language_text(color: str, distance_bucket: str) -> str:
    distance = int(DISTANCE_METERS[distance_bucket])
    return f"跟随{color == 'red' and '红色' or '蓝色'}目标船，保持{distance}米距离"


def synthetic_language_embedding(
    color: str,
    distance_bucket: str,
    *,
    language_dim: int = LANGUAGE_DIM,
) -> np.ndarray:
    """Encode synthetic language deterministically without a model download.

    The first dimensions carry explicit task semantics so this remains useful
    for a small PC training smoke run.  The remaining values are a stable hash
    of the text and must not be mistaken for the runtime Qwen embedding.
    """

    normalized_color = str(color).strip().casefold()
    normalized_bucket = str(distance_bucket).strip().casefold()
    if normalized_color not in COLORS:
        raise ValueError(f"unsupported color: {color!r}")
    if normalized_bucket not in DISTANCE_BUCKETS:
        raise ValueError(f"unsupported distance bucket: {distance_bucket!r}")
    if language_dim < 8:
        raise ValueError("language_dim must be at least 8")

    text = _language_text(normalized_color, normalized_bucket)
    embedding = np.zeros(language_dim, dtype=np.float32)
    embedding[0] = 1.0 if normalized_color == "red" else -1.0
    embedding[1] = DISTANCE_METERS[normalized_bucket] / 10.0
    embedding[2] = 1.0  # FOLLOW task marker.
    embedding[3] = np.sin(DISTANCE_METERS[normalized_bucket])
    embedding[4] = np.cos(DISTANCE_METERS[normalized_bucket])
    embedding[5] = 1.0 if normalized_bucket == "3m" else 0.0
    embedding[6] = 1.0 if normalized_bucket == "4m" else 0.0
    embedding[7] = 1.0 if normalized_bucket == "10m" else 0.0

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    tail = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    tail = tail / 127.5 - 1.0
    repeat_count = int(np.ceil((language_dim - 8) / len(tail)))
    embedding[8:] = np.tile(tail, repeat_count)[: language_dim - 8] * 0.05
    return embedding


def expert_action_for_geometry(
    x: float,
    y: float,
    distance_bucket: str,
    *,
    color: str = "red",
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
) -> ExpertActionResult:
    """Generate one label through the production single-step expert."""

    task = task_from_labels("follow", f"color:{color}", distance_bucket)
    entity = _entity(
        f"target_{color}",
        color,
        x,
        y,
    )
    entity.relative_velocity_x = float(velocity_x)
    entity.relative_velocity_y = float(velocity_y)
    return generate_expert_trajectory(task, [entity])


def _sample_geometry_point(rng: np.random.Generator) -> tuple[float, float]:
    x = rng.uniform(*L7_RUNTIME_X_RANGE_M)
    y = rng.uniform(*L7_RUNTIME_Y_RANGE_M)
    return float(x), float(y)


def generate_synthetic_geometry_dataset(
    *,
    sample_count: int = 384,
    seed: int = 42,
    language_dim: int = LANGUAGE_DIM,
    language_embeddings: LanguageEmbeddingTable | None = None,
) -> SyntheticGeometryDataset:
    """Generate balanced red/blue, 3m/4m/10m L7-geometry samples.

    The first twelve rows are a deterministic coverage block containing both
    required runtime points for every color and distance bucket.  Remaining
    rows are uniformly sampled inside the bounded L7 runtime rectangle.  When
    ``language_embeddings`` is supplied, every language row is selected from
    the real Qwen table for its canonical color/distance task.
    """

    minimum_count = (
        len(COLORS) * len(DISTANCE_BUCKETS) * len(L7_RUNTIME_POINTS_M)
    )
    if sample_count < minimum_count:
        raise ValueError(f"sample_count must be at least {minimum_count}")
    if language_dim != LANGUAGE_DIM:
        raise ValueError(
            f"synthetic language contract currently requires {LANGUAGE_DIM} dimensions"
        )

    rng = np.random.default_rng(seed)
    coverage = [
        (color, bucket, point)
        for color in COLORS
        for bucket in DISTANCE_BUCKETS
        for point in L7_RUNTIME_POINTS_M
    ]
    languages = np.zeros((sample_count, language_dim), dtype=np.float32)
    geometries = np.zeros((sample_count, 16, 16), dtype=np.float32)
    geometry_masks = np.zeros((sample_count, 16), dtype=np.bool_)
    previous_actions = np.zeros((sample_count, 2), dtype=np.float32)
    previous_valid = np.zeros(sample_count, dtype=np.bool_)
    target_actions = np.zeros((sample_count, 2), dtype=np.float32)
    target_stop = np.zeros(sample_count, dtype=np.bool_)
    texts: list[str] = []
    metadata: list[dict[str, Any]] = []

    language_source = "synthetic_surrogate"
    language_source_sha256 = ""
    if language_embeddings is not None:
        if language_embeddings.embeddings.shape[1] != language_dim:
            raise ValueError("language table dimension does not match language_dim")
        language_source = language_embeddings.source_path
        language_source_sha256 = language_embeddings.source_sha256

    for index in range(sample_count):
        if index < len(coverage):
            color, bucket, (target_x, target_y) = coverage[index]
            target_vx, target_vy = 0.0, 0.0
        else:
            color = COLORS[index % len(COLORS)]
            bucket = DISTANCE_BUCKETS[index % len(DISTANCE_BUCKETS)]
            target_x, target_y = _sample_geometry_point(rng)
            # The tracker supplies relative velocity online.  Cover both
            # target motion and camera-induced relative motion instead of
            # silently training on a zero-velocity-only distribution.
            target_vx = float(rng.uniform(-1.2, 1.2))
            target_vy = float(rng.uniform(-1.2, 1.2))
        current_result = expert_action_for_geometry(
            target_x,
            target_y,
            bucket,
            color=color,
            velocity_x=target_vx,
            velocity_y=target_vy,
        )
        # Teacher-force an adjacent previous frame from the same task.  The
        # first frame of each seven-row group is the explicit invalid sentinel.
        previous_valid[index] = bool(index % 7)
        if previous_valid[index]:
            previous_result = expert_action_for_geometry(
                max(target_x - 0.12, 0.05),
                target_y - 0.04,
                bucket,
                color=color,
                velocity_x=target_vx,
                velocity_y=target_vy,
            )
            previous_actions[index] = np.asarray(
                previous_result.expert_action,
                dtype=np.float32,
            )
        target_actions[index] = np.asarray(
            current_result.expert_action,
            dtype=np.float32,
        )
        # Online image perception emits the instruction-selected target and
        # the runtime TaskFeatures trace shows one active entity.  Keep the
        # synthetic tensor identical instead of training attention on a
        # privileged second target that is absent online.
        target_entity = _entity(f"target_{color}", color, target_x, target_y)
        target_entity.relative_velocity_x = target_vx
        target_entity.relative_velocity_y = target_vy
        entities = [target_entity]
        tensor = build_entity_tensor(entities)
        if language_embeddings is None:
            languages[index] = synthetic_language_embedding(
                color,
                bucket,
                language_dim=language_dim,
            )
            instruction_id = f"synthetic_follow_{color}_{bucket}"
            instruction_text = _language_text(color, bucket)
        else:
            embedding, instruction_id, instruction_text = language_embeddings.choose(
                color,
                bucket,
                row_offset=index,
            )
            languages[index] = embedding
        geometries[index] = tensor.features
        geometry_masks[index] = tensor.mask
        texts.append(instruction_text)
        metadata.append(
            {
                "sample_index": index,
                "target_color": color,
                "distance_bucket": bucket,
                "target_x_m": target_x,
                "target_y_m": target_y,
                "target_vx_mps": target_vx,
                "target_vy_mps": target_vy,
                "selected_entity_id": current_result.selected_entity_id,
                "expert_detail": current_result.detail,
                "instruction_id": instruction_id,
                "language_source": language_source,
                "previous_action_valid": bool(previous_valid[index]),
                "runtime_geometry_source": (
                    "required_regression_point"
                    if index < len(coverage)
                    else "uniform_l7_runtime_range"
                ),
            }
        )

    return SyntheticGeometryDataset(
        language=languages,
        entity_geometry=geometries,
        entity_geometry_mask=geometry_masks,
        previous_action=previous_actions,
        previous_action_valid=previous_valid,
        target_action=target_actions,
        target_stop=target_stop,
        language_text=tuple(texts),
        metadata=tuple(metadata),
        language_source=language_source,
        language_source_sha256=language_source_sha256,
    )


def save_synthetic_dataset(path: str | Path, dataset: SyntheticGeometryDataset) -> None:
    """Persist arrays and human-readable metadata for PC inspection."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(DATASET_SCHEMA_VERSION),
        language=dataset.language,
        entity_geometry=dataset.entity_geometry,
        entity_geometry_mask=dataset.entity_geometry_mask,
        previous_action=dataset.previous_action,
        previous_action_valid=dataset.previous_action_valid,
        target_action=dataset.target_action,
        target_stop=dataset.target_stop,
        language_text=np.asarray(dataset.language_text),
        language_source=np.asarray(dataset.language_source),
        language_source_sha256=np.asarray(dataset.language_source_sha256),
        metadata_json=np.asarray(
            json.dumps(dataset.metadata, ensure_ascii=False, sort_keys=True)
        ),
    )


def checkpoint_contract(model_config: Any) -> dict[str, Any]:
    """Describe the exact decision-head tensors stored in the checkpoint."""

    batch = "B"
    entity_shape = [
        batch,
        int(model_config.entity_count),
        int(model_config.entity_geometry_dim),
    ]
    return {
        "decision_inputs": [
            "language",
            "entity_geometry",
            "previous_action",
            "language_valid",
            "entity_geometry_mask",
            "previous_action_valid",
            "policy_input_valid",
        ],
        "input_shapes": {
            "language": [batch, int(model_config.language_dim)],
            "entity_geometry": entity_shape,
            "previous_action": [batch, int(model_config.previous_action_dim)],
            "language_valid": [batch],
            "entity_geometry_mask": [batch, int(model_config.entity_count)],
            "previous_action_valid": [batch],
            "policy_input_valid": [batch],
        },
        "input_dtypes": {
            "language": "float32",
            "entity_geometry": "float32",
            "previous_action": "float32",
            "language_valid": "bool",
            "entity_geometry_mask": "bool",
            "previous_action_valid": "bool",
            "policy_input_valid": "bool",
        },
        "decision_input_exclusions": ["global_visual", "entity_visual", "ego"],
        "outputs": {
            "action": {
                "shape": [batch, int(model_config.action_dim)],
                "dtype": "float32",
                "frame": "base_link",
                "kind": "single_step_desired_displacement_m",
                "maximum_norm_m": float(model_config.maximum_action_m),
            },
            "stop_logit": {"shape": [batch, 1], "dtype": "float32"},
            "valid_mask": {"shape": [batch], "dtype": "bool"},
        },
        "temporal_context": {
            "previous_action_source": "same_run_same_instruction_previous_frame",
            "first_frame_action": [0.0, 0.0],
            "first_frame_valid": False,
        },
    }


def _load_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "synthetic policy training requires PC PyTorch; install torch on the PC"
        ) from exc
    return torch


def load_model_config(path: str | Path) -> Any:
    """Load the existing YAML model contract without importing Torch."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to read model_config YAML") from exc
    source = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise ValueError("model config must contain a mapping")
    from asv_training.model import SmallPolicyConfig

    return SmallPolicyConfig.from_mapping(source)


def train_synthetic_policy(
    dataset: SyntheticGeometryDataset,
    model_config: Any,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 3.0e-3,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[Any, list[dict[str, float]]]:
    """Train the existing policy architecture on the synthetic PC dataset."""

    torch = _load_torch()
    from asv_training.model import SmallActionPolicy

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if dataset.language.shape[1] != model_config.language_dim:
        raise ValueError("dataset language_dim does not match model_config")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable on this PC")

    torch.manual_seed(seed)
    target_device = torch.device(device)
    model = SmallActionPolicy(model_config).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    tensors = {
        "language": torch.from_numpy(dataset.language).to(target_device),
        "entity_geometry": torch.from_numpy(dataset.entity_geometry).to(target_device),
        "entity_geometry_mask": torch.from_numpy(dataset.entity_geometry_mask).to(
            target_device
        ),
        "previous_action": torch.from_numpy(dataset.previous_action).to(target_device),
        "previous_action_valid": torch.from_numpy(dataset.previous_action_valid).to(
            target_device
        ),
        "target_action": torch.from_numpy(dataset.target_action).to(target_device),
        "target_stop": torch.from_numpy(dataset.target_stop.astype(np.float32)).to(
            target_device
        ),
    }
    sample_count = len(dataset.language)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(sample_count, device=target_device)
        action_total = 0.0
        stop_total = 0.0
        batch_total = 0
        for start in range(0, sample_count, batch_size):
            indices = permutation[start : start + batch_size]
            output = model(
                language=tensors["language"][indices],
                entity_geometry=tensors["entity_geometry"][indices],
                previous_action=tensors["previous_action"][indices],
                language_valid=torch.ones(
                    len(indices), dtype=torch.bool, device=target_device
                ),
                entity_geometry_mask=tensors["entity_geometry_mask"][indices],
                previous_action_valid=tensors["previous_action_valid"][indices],
                policy_input_valid=torch.ones(
                    len(indices), dtype=torch.bool, device=target_device
                ),
            )
            action_loss = torch.nn.functional.smooth_l1_loss(
                output.action,
                tensors["target_action"][indices],
            )
            stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output.stop_logit.view(-1),
                tensors["target_stop"][indices],
            )
            loss = action_loss + 0.1 * stop_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            action_total += float(action_loss.detach().cpu())
            stop_total += float(stop_loss.detach().cpu())
            batch_total += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                "action_loss": action_total / batch_total,
                "stop_loss": stop_total / batch_total,
            }
        )
    model.eval()
    return model, history


def save_checkpoint(
    path: str | Path,
    model: Any,
    model_config: Any,
    *,
    dataset: SyntheticGeometryDataset,
    history: Sequence[Mapping[str, float]] = (),
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> None:
    """Save a strict-loadable Jetson policy checkpoint."""

    torch = _load_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_config": asdict(model_config),
        "contract": checkpoint_contract(model_config),
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "training": {
            "source": "pc_only_synthetic_geometry",
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "sample_count": len(dataset.language),
            "seed": int(seed),
            "dataset_seed": int(seed),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "device": str(device),
            "target_colors": list(COLORS),
            "distance_buckets": list(DISTANCE_BUCKETS),
            "l7_runtime_x_range_m": list(L7_RUNTIME_X_RANGE_M),
            "l7_runtime_y_range_m": list(L7_RUNTIME_Y_RANGE_M),
            "language_embedding": dataset.language_source,
            "language_embedding_sha256": dataset.language_source_sha256,
            "history": [dict(row) for row in history],
        },
    }
    torch.save(payload, output)


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Strictly reconstruct a saved checkpoint using the existing model class."""

    torch = _load_torch()
    from asv_training.model import SmallActionPolicy, SmallPolicyConfig

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    model_config = SmallPolicyConfig.from_mapping(checkpoint["model_config"])
    model = SmallActionPolicy(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, dict(checkpoint)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model_small_v3.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output .pt checkpoint",
    )
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument("--sample-count", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="PC torch device, default: cpu")
    parser.add_argument(
        "--language-embeddings",
        type=Path,
        help="PC-exported Qwen embedding .npz; required for deployment training",
    )
    parser.add_argument(
        "--synthetic-language",
        action="store_true",
        help="allow the deterministic language surrogate for smoke tests only",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.language_embeddings is None and not args.synthetic_language:
        raise SystemExit(
            "--language-embeddings is required for a deployable checkpoint; "
            "pass --synthetic-language only for a smoke test"
        )
    language_embeddings = (
        load_language_embeddings(args.language_embeddings)
        if args.language_embeddings is not None
        else None
    )
    dataset = generate_synthetic_geometry_dataset(
        sample_count=args.sample_count,
        seed=args.seed,
        language_embeddings=language_embeddings,
    )
    dataset_path = args.dataset_output or args.output.with_suffix(".npz")
    save_synthetic_dataset(dataset_path, dataset)
    model_config = load_model_config(args.model_config)
    model, history = train_synthetic_policy(
        dataset,
        model_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )
    save_checkpoint(
        args.output,
        model,
        model_config,
        dataset=dataset,
        history=history,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "dataset": str(dataset_path),
                "sample_count": len(dataset.language),
                "last_epoch": history[-1] if history else {},
                "device": args.device,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
