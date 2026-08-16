"""Day 15 three-seed training, baselines, and sealed test evaluation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Sampler
import yaml

from asv_training.dataset import (
    AnnotatedFeatureDataset,
    EpochSynonymDataset,
    FrozenFeatureDataset,
    discover_feature_caches,
    load_instruction_metadata,
    load_split_assignments,
    policy_inputs_from_batch,
)
from asv_training.losses import PolicyLossWeights, action_policy_loss
from asv_training.model import SmallActionPolicy, SmallPolicyConfig


TRAIN_SCHEMA_VERSION = "train_v1"
SUMMARY_SCHEMA_VERSION = "training_summary_v1"
CHECKPOINT_SCHEMA_VERSION = "policy_checkpoint_v1"
PROGRESS_SCHEMA_VERSION = "training_progress_v1"
SINGLE_POINT_SELECTION_METRIC = "action_error_m"


def _improvement_fraction(policy_value: float, baseline_value: float) -> float:
    if baseline_value <= 0.0:
        return 0.0
    return float((baseline_value - policy_value) / baseline_value)


def _require_pc_training_target(execution_target: str = "pc") -> None:
    if str(execution_target).strip().casefold() != "pc":
        raise RuntimeError(
            "training and evaluation are PC-only; Jetson is inference-only"
        )
    if Path("/etc/nv_tegra_release").is_file():
        raise RuntimeError(
            "training/evaluation cannot run on Jetson; use the PC training "
            "entry point and synchronize validated artifacts"
        )


def _binary_stop_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    true_positive = int(np.count_nonzero(predicted & target))
    false_positive = int(np.count_nonzero(predicted & ~target))
    false_negative = int(np.count_nonzero(~predicted & target))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def compute_action_metrics(
    prediction: Any,
    target: Any,
    stop_logits: Any,
    target_stop: Any,
    task_labels: Iterable[str],
    *,
    maximum_action_m: float = 0.3,
    stop_drift_limit_m: float = 0.10,
) -> dict[str, Any]:
    predicted_action = np.asarray(prediction, dtype=np.float32)
    target_action = np.asarray(target, dtype=np.float32)
    if predicted_action.ndim != 2 or predicted_action.shape[1:] != (2,):
        raise ValueError(f"prediction must have shape [N,2], got {predicted_action.shape}")
    if target_action.shape != predicted_action.shape:
        raise ValueError("prediction and target action shapes differ")
    logits = np.asarray(stop_logits, dtype=np.float32).reshape(-1)
    stop_target = np.asarray(target_stop, dtype=np.bool_).reshape(-1)
    labels = np.asarray([str(label) for label in task_labels], dtype=np.str_)
    sample_count = predicted_action.shape[0]
    if len(logits) != sample_count or len(stop_target) != sample_count or len(labels) != sample_count:
        raise ValueError("action, stop, and label arrays must have the same sample count")
    if not np.all(np.isfinite(predicted_action)) or not np.all(np.isfinite(logits)):
        raise ValueError("prediction contains NaN or Inf")
    if not np.all(np.isfinite(target_action)):
        raise ValueError("target action contains NaN or Inf")
    error = np.linalg.norm(predicted_action - target_action, axis=1)
    lateral_error = np.abs(predicted_action[:, 1] - target_action[:, 1])
    predicted_stop = logits >= 0.0
    stop_drift = np.linalg.norm(predicted_action[stop_target], axis=1)
    action_norm = np.linalg.norm(predicted_action, axis=1)
    per_label = {
        label: {
            "sample_count": int(np.count_nonzero(labels == label)),
            "action_error_m": float(np.mean(error[labels == label])),
            "lateral_action_error_m": float(
                np.mean(lateral_error[labels == label])
            ),
        }
        for label in sorted(set(labels.tolist()))
    }
    return {
        "sample_count": sample_count,
        "action_error_m": float(np.mean(error)) if sample_count else 0.0,
        "lateral_action_error_m": (
            float(np.mean(lateral_error)) if sample_count else 0.0
        ),
        "stop_drift": {
            "sample_count": int(len(stop_drift)),
            "mean_m": float(np.mean(stop_drift)) if len(stop_drift) else 0.0,
            "p95_m": float(np.percentile(stop_drift, 95)) if len(stop_drift) else 0.0,
            "maximum_m": float(np.max(stop_drift)) if len(stop_drift) else 0.0,
            "within_0_10m_rate": float(np.mean(stop_drift <= stop_drift_limit_m)) if len(stop_drift) else 0.0,
        },
        "stop_classification": _binary_stop_metrics(predicted_stop, stop_target),
        "action_bound": {
            "maximum_action_m": maximum_action_m,
            "observed_maximum_action_m": float(np.max(action_norm)) if sample_count else 0.0,
            "violation_count": int(np.count_nonzero(action_norm > maximum_action_m + 1.0e-6)),
            "violation_rate": float(np.mean(action_norm > maximum_action_m + 1.0e-6)) if sample_count else 0.0,
        },
        "invalid_count": 0,
        "per_label": per_label,
    }


def fit_label_mean_action_baseline(
    actions: Any, task_labels: Iterable[str]
) -> dict[str, np.ndarray]:
    target = np.asarray(actions, dtype=np.float32)
    if target.ndim != 2 or target.shape[1:] != (2,):
        raise ValueError(f"actions must have shape [N,2], got {target.shape}")
    labels = [str(label) for label in task_labels]
    if len(labels) != len(target):
        raise ValueError("task labels do not match action sample count")
    return {
        label: np.mean(target[np.asarray(labels) == label], axis=0).astype(np.float32)
        for label in sorted(set(labels))
    }


def predict_label_mean_action_baseline(
    means: Mapping[str, np.ndarray], task_labels: Iterable[str]
) -> tuple[np.ndarray, np.ndarray]:
    labels = [str(label) for label in task_labels]
    missing = sorted(set(labels) - set(means))
    if missing:
        raise ValueError(f"mean baseline has no labels: {missing}")
    actions = np.stack([means[label] for label in labels]).astype(np.float32)
    logits = np.asarray([20.0 if label.startswith("stop|") else -20.0 for label in labels], dtype=np.float32)
    return actions, logits


@dataclass(frozen=True)
class TrainSettings:
    epochs: int
    minimum_epochs: int
    early_stopping_patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    num_workers: int


@dataclass(frozen=True)
class DatasetBundle:
    train_base: FrozenFeatureDataset
    validation: AnnotatedFeatureDataset
    test: AnnotatedFeatureDataset | None
    instructions: Mapping[str, Any]
    manifest: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_progress(
    path: Path,
    *,
    output_root: Path,
    stage: str,
    **fields: Any,
) -> None:
    """Publish a crash-safe progress snapshot for detached PC training."""
    payload: dict[str, Any] = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "output_root": str(output_root),
        "stage": stage,
    }
    payload.update({key: value for key, value in fields.items() if value is not None})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    return device


def _parse_settings(config: Mapping[str, Any]) -> TrainSettings:
    value = config.get("training")
    if not isinstance(value, Mapping):
        raise ValueError("training configuration is missing")
    settings = TrainSettings(
        epochs=int(value["epochs"]),
        minimum_epochs=int(value["minimum_epochs"]),
        early_stopping_patience=int(value["early_stopping_patience"]),
        batch_size=int(value["batch_size"]),
        learning_rate=float(value["learning_rate"]),
        weight_decay=float(value["weight_decay"]),
        gradient_clip_norm=float(value["gradient_clip_norm"]),
        num_workers=int(value["num_workers"]),
    )
    if settings.epochs < settings.minimum_epochs or settings.minimum_epochs <= 0:
        raise ValueError("invalid epoch bounds")
    if settings.early_stopping_patience <= 0 or settings.batch_size <= 0:
        raise ValueError("patience and batch size must be positive")
    if settings.learning_rate <= 0.0 or settings.gradient_clip_norm <= 0.0:
        raise ValueError("learning rate and gradient clip must be positive")
    if settings.weight_decay < 0.0 or settings.num_workers < 0:
        raise ValueError("weight decay and num_workers must be non-negative")
    return settings


def _checkpoint_selection_eligible(
    metrics: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> bool:
    constraints = training_config.get("selection_constraints")
    if constraints is None:
        return True
    if not isinstance(constraints, Mapping):
        raise ValueError("training selection_constraints must be a mapping")
    unknown = set(constraints) - {
        "minimum_stop_f1",
        "minimum_stop_within_0_10m_rate",
    }
    if unknown:
        raise ValueError(
            f"unknown selection constraint keys: {sorted(unknown)}"
        )
    stop_classification = metrics.get("stop_classification")
    stop_drift = metrics.get("stop_drift")
    if not isinstance(stop_classification, Mapping) or not isinstance(
        stop_drift, Mapping
    ):
        raise ValueError("validation metrics have no STOP gate values")
    minimum_f1 = float(constraints.get("minimum_stop_f1", 0.0))
    minimum_drift_rate = float(
        constraints.get("minimum_stop_within_0_10m_rate", 0.0)
    )
    if not 0.0 <= minimum_f1 <= 1.0:
        raise ValueError("minimum_stop_f1 must be in [0, 1]")
    if not 0.0 <= minimum_drift_rate <= 1.0:
        raise ValueError(
            "minimum_stop_within_0_10m_rate must be in [0, 1]"
        )
    return (
        float(stop_classification["f1"]) >= minimum_f1
        and float(stop_drift["within_0_10m_rate"]) >= minimum_drift_rate
    )


def _checkpoint_selection_eligible_for_modality(
    metrics: Mapping[str, Any],
    training_config: Mapping[str, Any],
    modality: str,
) -> bool:
    if modality == "entity_only":
        return True
    if modality != "full":
        raise ValueError(f"unsupported training modality: {modality}")
    return _checkpoint_selection_eligible(metrics, training_config)


def _selection_score(
    metrics: Mapping[str, Any], training_config: Mapping[str, Any]
) -> float:
    """Return the metric used to select a single-step action checkpoint."""
    metric = str(
        training_config.get("selection_metric", SINGLE_POINT_SELECTION_METRIC)
    ).strip()
    if metric != SINGLE_POINT_SELECTION_METRIC:
        raise ValueError(
            "single-step training supports only selection_metric="
            f"{SINGLE_POINT_SELECTION_METRIC!r}; got {metric!r}"
        )
    score = float(metrics["action_error_m"])
    if not np.isfinite(score):
        raise ValueError("single-step selection metric is not finite")
    return score


def _build_dataset_bundle(
    *,
    feature_root: Path,
    split_path: Path,
    instructions_path: Path,
    train_config: Mapping[str, Any],
    include_test: bool,
) -> DatasetBundle:
    data_config = train_config.get("data")
    if not isinstance(data_config, Mapping):
        raise ValueError("data configuration is missing")
    runtime_limit_value = data_config.get("runtime_first_step_limit_m")
    if runtime_limit_value is None:
        runtime_first_step_limit_m = None
    else:
        try:
            runtime_first_step_limit_m = float(runtime_limit_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "data.runtime_first_step_limit_m must be finite and positive"
            ) from exc
        if (
            not np.isfinite(runtime_first_step_limit_m)
            or runtime_first_step_limit_m <= 0.0
        ):
            raise ValueError(
                "data.runtime_first_step_limit_m must be finite and positive"
            )
    caches = discover_feature_caches(feature_root)
    expected_run_count = int(data_config.get("expected_run_count", 12))
    if expected_run_count < 12:
        raise ValueError(f"expected_run_count must be >= 12, got {expected_run_count}")
    if len(caches) != expected_run_count:
        raise ValueError(
            f"Day 15 requires {expected_run_count} feature caches, "
            f"got {len(caches)}"
        )
    assignments = load_split_assignments(split_path)
    instructions = load_instruction_metadata(instructions_path)
    stride = int(data_config["frame_stride"])
    train_base = FrozenFeatureDataset(
        caches,
        selected_split=str(data_config["train_run_split"]),
        split_assignments=assignments,
        allowed_language_splits=data_config["train_language_splits"],
        frame_stride=stride,
        augment=bool(data_config.get("augment", False)),
        geometry_noise_std=float(data_config.get("geometry_noise_std", 0.02)),
        slot_dropout_prob=float(
            data_config.get("slot_dropout_prob", 0.1)
        ),
        mirror_prob=float(data_config.get("mirror_prob", 0.0)),
        instruction_swap_prob=float(
            data_config.get("instruction_swap_prob", 0.0)
        ),
        runtime_first_step_limit_m=runtime_first_step_limit_m,
        hold_band_m=data_config.get("hold_band_m", 0.0),
        hold_oversample_factor=data_config.get("hold_oversample_factor", 1),
    )
    validation_base = FrozenFeatureDataset(
        caches,
        selected_split=str(data_config["validation_run_split"]),
        split_assignments=assignments,
        allowed_language_splits=data_config["validation_language_splits"],
        frame_stride=stride,
        runtime_first_step_limit_m=runtime_first_step_limit_m,
        hold_band_m=0.0,
        hold_oversample_factor=1,
    )
    test_dataset: AnnotatedFeatureDataset | None = None
    if include_test:
        test_base = FrozenFeatureDataset(
            caches,
            selected_split=str(data_config["test_run_split"]),
            split_assignments=assignments,
            allowed_language_splits=data_config["test_language_splits"],
            frame_stride=stride,
            runtime_first_step_limit_m=runtime_first_step_limit_m,
            hold_band_m=0.0,
            hold_oversample_factor=1,
        )
        test_dataset = AnnotatedFeatureDataset(test_base, instructions)

    cache_manifests = {
        cache.name: _sha256_file(cache / "manifest.json") for cache in caches
    }
    feature_set_manifest = feature_root / "feature_set_manifest.json"
    if not feature_set_manifest.is_file():
        raise ValueError("feature_set_manifest.json is missing")
    manifest = {
        "schema_version": "dataset_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set_manifest_sha256": _sha256_file(feature_set_manifest),
        "split_sha256": _sha256_file(split_path),
        "instructions_sha256": _sha256_file(instructions_path),
        "cache_manifest_sha256": cache_manifests,
        "run_count": len(caches),
        "split_assignments": assignments,
        "frame_stride": stride,
        "runtime_first_step_limit_m": runtime_first_step_limit_m,
        "hold_band_m": train_base.hold_band_m,
        "hold_oversample_factor": train_base.hold_oversample_factor,
        "train_raw_sample_count": train_base.raw_sample_count,
        "train_resampled_sample_count": len(train_base),
        "validation_sample_count": len(validation_base),
        "test_sample_count": len(test_dataset) if test_dataset else None,
        "train_sampling": "one_synonym_per_frame_task_label_each_epoch",
    }
    return DatasetBundle(
        train_base=train_base,
        validation=AnnotatedFeatureDataset(validation_base, instructions),
        test=test_dataset,
        instructions=instructions,
        manifest=manifest,
    )


def _make_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


class _FrameGroupedBatchSampler(Sampler[list[int]]):
    """Shuffle observations while keeping all task labels in one batch."""

    def __init__(
        self,
        frame_groups: Sequence[Sequence[int]],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.frame_groups = tuple(tuple(group) for group in frame_groups)
        if not self.frame_groups or any(not group for group in self.frame_groups):
            raise ValueError("frame groups must be non-empty")
        if max(len(group) for group in self.frame_groups) > batch_size:
            raise ValueError("one frame group exceeds the batch size")
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.order = tuple(
            int(value)
            for value in torch.randperm(
                len(self.frame_groups), generator=generator
            )
        )
        batches: list[list[int]] = []
        current: list[int] = []
        for frame_index in self.order:
            group = self.frame_groups[frame_index]
            if current and len(current) + len(group) > batch_size:
                batches.append(current)
                current = []
            current.extend(group)
        if current:
            batches.append(current)
        self.batches = tuple(tuple(batch) for batch in batches)

    def __iter__(self) -> Iterable[list[int]]:
        return (list(batch) for batch in self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def _make_frame_grouped_loader(
    dataset: EpochSynonymDataset,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    sampler = _FrameGroupedBatchSampler(
        dataset.frame_group_indices,
        batch_size=batch_size,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _make_cross_run_loader(
    dataset: EpochSynonymDataset,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    """Loader that forces cross-run L3/L4 pairs into the same batch."""
    sampler = _FrameGroupedBatchSampler(
        dataset.cross_run_pair_indices,
        batch_size=batch_size,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _model_inputs(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    modality: str,
) -> dict[str, Tensor]:
    inputs = {
        key: tensor.to(device, non_blocking=True)
        for key, tensor in policy_inputs_from_batch(batch).items()
    }
    if modality == "full":
        return inputs
    if modality != "entity_only":
        raise ValueError(f"unknown modality={modality!r}")
    inputs["language"] = torch.zeros_like(inputs["language"])
    return inputs


def _evaluate_model(
    model: SmallActionPolicy,
    dataset: Dataset[Any],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    modality: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray | list[str]]]:
    loader = _make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        device=device,
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    stops: list[np.ndarray] = []
    labels: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(**_model_inputs(batch, device, modality=modality))
            if not bool(torch.all(output.valid_mask)):
                raise ValueError("evaluation dataset produced an invalid policy input")
            predictions.append(output.action.detach().cpu().numpy())
            targets.append(batch["target_action"].numpy())
            logits.append(output.stop_logit.detach().cpu().numpy())
            stops.append(batch["target_stop"].numpy())
            labels.extend(str(value) for value in batch["metadata"]["task_label"])
    arrays: dict[str, np.ndarray | list[str]] = {
        "prediction": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "stop_logits": np.concatenate(logits),
        "target_stop": np.concatenate(stops),
        "task_labels": labels,
    }
    metrics = compute_action_metrics(
        arrays["prediction"],
        arrays["target"],
        arrays["stop_logits"],
        arrays["target_stop"],
        labels,
    )
    return metrics, arrays


def _collect_targets(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    actions: list[np.ndarray] = []
    stops: list[np.ndarray] = []
    labels: list[str] = []
    for batch in loader:
        actions.append(batch["target_action"].numpy())
        stops.append(batch["target_stop"].numpy())
        labels.extend(str(value) for value in batch["metadata"]["task_label"])
    return np.concatenate(actions), np.concatenate(stops), labels


def _environment_report(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
        }
    return report


def _save_curves(path: Path, history: list[dict[str, Any]]) -> None:
    width, height = 1000, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 15), "Day 15 training curves", fill="black")
    panels = (
        ("train_loss", (50, 60, 950, 290), "blue"),
        ("validation_action_error_m", (50, 330, 950, 560), "red"),
    )
    for key, (left, top, right, bottom), color in panels:
        draw.rectangle((left, top, right, bottom), outline="black")
        values = [float(row[key]) for row in history]
        draw.text((left + 5, top + 5), key, fill="black")
        if len(values) < 2:
            continue
        minimum = min(values)
        maximum = max(values)
        span = max(maximum - minimum, 1.0e-12)
        points = []
        for index, value in enumerate(values):
            x = left + index * (right - left) / (len(values) - 1)
            y = bottom - (value - minimum) * (bottom - top) / span
            points.append((x, y))
        draw.line(points, fill=color, width=3)
        draw.text(
            (right - 250, top + 5),
            f"min={minimum:.6f} max={maximum:.6f}",
            fill="black",
        )
    image.save(path, format="PNG")


def _save_train_csv(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _checkpoint_payload(
    *,
    model: SmallActionPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    modality: str,
    git_sha: str,
    validation_metrics: Mapping[str, Any],
    model_config: SmallPolicyConfig,
    train_config_sha256: str,
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
        "seed": seed,
        "modality": modality,
        "git_sha": git_sha,
        "model_config": asdict(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": dict(validation_metrics),
        "train_config_sha256": train_config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
    }


def _train_one(
    *,
    experiment_dir: Path,
    modality: str,
    seed: int,
    bundle: DatasetBundle,
    train_config: Mapping[str, Any],
    train_config_path: Path,
    model_config: SmallPolicyConfig,
    model_config_source: Mapping[str, Any],
    device: torch.device,
    git_sha: str,
    progress_path: Path,
) -> dict[str, Any]:
    settings = _parse_settings(train_config)
    training_config = train_config.get("training")
    if not isinstance(training_config, Mapping):
        raise ValueError("training configuration is missing")
    loss_weights = PolicyLossWeights.from_mapping(train_config.get("loss"))
    if modality == "entity_only":
        loss_weights = replace(loss_weights, pairwise=0.0)
    _seed_everything(seed)
    experiment_dir.mkdir(parents=True, exist_ok=False)
    combined_config = {
        "training": dict(train_config),
        "model": dict(model_config_source),
        "seed": seed,
        "modality": modality,
        "git_sha": git_sha,
    }
    (experiment_dir / "config.yaml").write_text(
        yaml.safe_dump(combined_config, sort_keys=True),
        encoding="utf-8",
    )
    _write_json(experiment_dir / "dataset_manifest.json", bundle.manifest)
    _write_json(experiment_dir / "environment.json", _environment_report(device))
    _write_progress(
        progress_path,
        output_root=progress_path.parent,
        stage="model_initializing",
        seed=seed,
        modality=modality,
        epoch_completed=0,
        epochs_total=settings.epochs,
        train_samples=len(bundle.train_base),
        validation_samples=len(bundle.validation),
        device=str(device),
    )
    dataset_manifest_sha256 = _sha256_file(
        experiment_dir / "dataset_manifest.json"
    )
    train_config_sha256 = _sha256_file(train_config_path)

    train_dataset = EpochSynonymDataset(
        bundle.train_base,
        bundle.instructions,
        seed=seed,
    )
    model = SmallActionPolicy(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    _write_progress(
        progress_path,
        output_root=progress_path.parent,
        stage="training",
        seed=seed,
        modality=modality,
        epoch_completed=0,
        epochs_total=settings.epochs,
        train_samples=len(train_dataset),
        validation_samples=len(bundle.validation),
        device=str(device),
    )

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(settings.epochs):
        train_dataset.set_epoch(epoch)
        _write_progress(
            progress_path,
            output_root=progress_path.parent,
            stage="epoch_running",
            seed=seed,
            modality=modality,
            epoch_started=epoch + 1,
            epoch_completed=epoch,
            epochs_total=settings.epochs,
            device=str(device),
        )
        if loss_weights.pairwise > 0.0:
            loader = _make_cross_run_loader(
                train_dataset,
                batch_size=settings.batch_size,
                seed=seed * 10_000 + epoch,
                num_workers=settings.num_workers,
                device=device,
            )
        else:
            loader = _make_loader(
                train_dataset,
                batch_size=settings.batch_size,
                shuffle=True,
                seed=seed * 10_000 + epoch,
                num_workers=settings.num_workers,
                device=device,
            )
        model.train()
        weighted_loss = 0.0
        trained_samples = 0
        for batch in loader:
            inputs = _model_inputs(batch, device, modality=modality)
            target_action = batch["target_action"].to(
                device, non_blocking=True
            )
            target_stop = batch["target_stop"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs)
            losses = action_policy_loss(
                output,
                target_action,
                target_stop,
                previous_action=inputs["previous_action"],
                previous_action_valid=inputs["previous_action_valid"],
                weights=loss_weights,
                group_ids=[
                    ":".join(str(fk).split(":")[2:3])
                    + "_"
                    + str(batch["instruction_id"][idx])
                    for idx, fk in enumerate(batch["frame_key"])
                ],
            )
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            count = int(losses["valid_samples"].detach().cpu().item())
            weighted_loss += float(losses["total"].detach().cpu()) * count
            trained_samples += count
        validation_metrics, _ = _evaluate_model(
            model,
            bundle.validation,
            device=device,
            batch_size=settings.batch_size * 2,
            num_workers=settings.num_workers,
            modality=modality,
        )
        train_loss = weighted_loss / max(trained_samples, 1)
        selection_score = _selection_score(validation_metrics, training_config)
        minimum_checkpoint_epoch = int(
            training_config.get("minimum_checkpoint_epoch", 1)
        )
        if minimum_checkpoint_epoch <= 0:
            raise ValueError("minimum_checkpoint_epoch must be positive")
        selection_eligible = (
            epoch + 1 >= minimum_checkpoint_epoch
            and _checkpoint_selection_eligible_for_modality(
                validation_metrics, training_config, modality
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_action_error_m": validation_metrics["action_error_m"],
                "validation_stop_f1": validation_metrics[
                    "stop_classification"
                ]["f1"],
                "validation_stop_within_0_10m_rate": validation_metrics[
                    "stop_drift"
                ]["within_0_10m_rate"],
                "selection_eligible": selection_eligible,
                "selection_score": selection_score,
            }
        )
        # Keep a usable partial curve if a detached run is interrupted.
        _save_train_csv(experiment_dir / "train.csv", history)
        if selection_eligible and selection_score < best_score - 1.0e-9:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            _atomic_torch_save(
                experiment_dir / "best.pt",
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    seed=seed,
                    modality=modality,
                    git_sha=git_sha,
                    validation_metrics=validation_metrics,
                    model_config=model_config,
                    train_config_sha256=train_config_sha256,
                    dataset_manifest_sha256=dataset_manifest_sha256,
                ),
            )
        elif best_epoch >= 0:
            epochs_without_improvement += 1
        elapsed = time.perf_counter() - started
        per_epoch = elapsed / (epoch + 1)
        remaining = per_epoch * (settings.epochs - epoch - 1)
        print(
            f"EPOCH_PROGRESS epoch={epoch + 1}/{settings.epochs} "
            f"seed={seed} modality={modality} "
            f"train_loss={train_loss:.4f} "
            f"val_action_error={validation_metrics['action_error_m']:.3f} "
            f"best={best_score:.3f}@ep{best_epoch + 1} "
            f"elapsed={elapsed:.0f}s remaining~{remaining:.0f}s "
            f"patience={epochs_without_improvement}/{settings.early_stopping_patience}",
            flush=True,
        )
        _write_progress(
            progress_path,
            output_root=progress_path.parent,
            stage="training",
            seed=seed,
            modality=modality,
            epoch_completed=epoch + 1,
            epochs_total=settings.epochs,
            best_epoch=best_epoch + 1 if best_epoch >= 0 else None,
            best_score=best_score if np.isfinite(best_score) else None,
            selection_score=selection_score,
            elapsed_seconds=elapsed,
            device=str(device),
        )
        if (
            epoch + 1 >= settings.minimum_epochs
            and epochs_without_improvement >= settings.early_stopping_patience
        ):
            break

    if best_epoch < 0:
        raise RuntimeError(
            "no validation checkpoint satisfied selection_constraints"
        )

    last_validation, _ = _evaluate_model(
        model,
        bundle.validation,
        device=device,
        batch_size=settings.batch_size * 2,
        num_workers=settings.num_workers,
        modality=modality,
    )
    _atomic_torch_save(
        experiment_dir / "last.pt",
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=len(history) - 1,
            seed=seed,
            modality=modality,
            git_sha=git_sha,
            validation_metrics=last_validation,
            model_config=model_config,
            train_config_sha256=train_config_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
        ),
    )
    checkpoint = torch.load(
        experiment_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    best_validation, _ = _evaluate_model(
        model,
        bundle.validation,
        device=device,
        batch_size=settings.batch_size * 2,
        num_workers=settings.num_workers,
        modality=modality,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory_bytes = 0
    duration_seconds = time.perf_counter() - started
    metrics = {
        "schema_version": "experiment_metrics_v1",
        "modality": modality,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "duration_seconds": duration_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "trainable_parameter_count": model.trainable_parameter_count(),
        "checkpoint_size_bytes": (experiment_dir / "best.pt").stat().st_size,
        "validation": best_validation,
    }
    _write_json(experiment_dir / "metrics.json", metrics)
    _save_train_csv(experiment_dir / "train.csv", history)
    _save_curves(experiment_dir / "curves.png", history)
    _write_progress(
        progress_path,
        output_root=progress_path.parent,
        stage="seed_complete",
        seed=seed,
        modality=modality,
        epoch_completed=len(history),
        epochs_total=settings.epochs,
        best_epoch=best_epoch + 1,
        elapsed_seconds=duration_seconds,
        checkpoint=str(experiment_dir / "best.pt"),
        device=str(device),
    )
    return metrics


def _baseline_metrics(
    train_dataset: EpochSynonymDataset,
    evaluation_dataset: AnnotatedFeatureDataset,
    *,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    train_dataset.set_epoch(0)
    train_target, _, train_labels = _collect_targets(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    target, target_stop, labels = _collect_targets(
        evaluation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    label_means = fit_label_mean_action_baseline(train_target, train_labels)
    mean_prediction, mean_logits = predict_label_mean_action_baseline(
        label_means, labels
    )
    zero_prediction = np.zeros_like(target)
    zero_logits = np.full(len(target), 20.0, dtype=np.float32)
    expert_logits = np.where(
        target_stop.reshape(-1), 20.0, -20.0
    ).astype(np.float32)
    return {
        "zero_stop": compute_action_metrics(
            zero_prediction,
            target,
            zero_logits,
            target_stop,
            labels,
        ),
        "label_mean": compute_action_metrics(
            mean_prediction,
            target,
            mean_logits,
            target_stop,
            labels,
        ),
        "expert_upper_bound": compute_action_metrics(
            target,
            target,
            expert_logits,
            target_stop,
            labels,
        ),
    }


def _acceptance(
    metrics: Mapping[str, Any],
    label_mean: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    if "minimum_fde_improvement_over_label_mean" in acceptance:
        raise ValueError(
            "single-step acceptance cannot define an FDE gate; use "
            "minimum_action_error_improvement_over_label_mean"
        )
    action_improvement = _improvement_fraction(
        float(metrics["action_error_m"]), float(label_mean["action_error_m"])
    )
    minimum_action_improvement = float(
        acceptance.get(
            "minimum_action_error_improvement_over_label_mean",
            acceptance.get("minimum_ade_improvement_over_label_mean", 0.0),
        )
    )
    checks = {
        "action_error_improvement": action_improvement
        >= minimum_action_improvement,
        "stop_drift": float(metrics["stop_drift"]["within_0_10m_rate"])
        >= float(acceptance["minimum_stop_within_0_10m_rate"]),
        "stop_f1": float(metrics["stop_classification"]["f1"])
        >= float(acceptance["minimum_stop_f1"]),
        "action_bound": float(metrics["action_bound"]["violation_rate"])
        <= float(acceptance["maximum_speed_violation_rate"]),
        "finite": int(metrics["invalid_count"])
        <= int(acceptance["maximum_invalid_count"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "action_error_improvement_fraction": action_improvement,
    }


def _experiment_name(modality: str, seed: int) -> str:
    return f"{modality}_seed{seed}"


def train_validation_suite(args: argparse.Namespace) -> int:
    _require_pc_training_target(getattr(args, "execution_target", "pc"))
    train_config_path = args.config.resolve()
    model_config_path = args.model_config.resolve()
    train_config = _read_yaml(train_config_path)
    model_source = _read_yaml(model_config_path)
    if train_config.get("schema_version") != TRAIN_SCHEMA_VERSION:
        raise ValueError(f"config schema must be {TRAIN_SCHEMA_VERSION}")
    # The config pins the frozen feature provenance; any sha is accepted as
    # long as it matches the feature caches actually being trained on.
    config_frozen_sha = str(train_config.get("frozen_feature_git_sha", "")).strip()
    if not config_frozen_sha:
        raise ValueError("config must declare frozen_feature_git_sha")
    model_config = SmallPolicyConfig.from_mapping(model_source)
    settings = _parse_settings(train_config)
    seeds = tuple(int(seed) for seed in train_config.get("seeds", []))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("config must declare three distinct training seeds")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise ValueError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    progress_path = output_root / "progress.json"
    device = _resolve_device(args.device)
    _write_progress(
        progress_path,
        output_root=output_root,
        stage="dataset_loading",
        command="train",
        device=str(device),
    )
    bundle = _build_dataset_bundle(
        feature_root=args.features.resolve(),
        split_path=args.split.resolve(),
        instructions_path=args.instructions.resolve(),
        train_config=train_config,
        include_test=False,
    )
    _write_json(output_root / "dataset_manifest.json", bundle.manifest)
    _write_progress(
        progress_path,
        output_root=output_root,
        stage="dataset_ready",
        command="train",
        device=str(device),
        run_count=bundle.manifest["run_count"],
        train_samples=bundle.manifest["train_raw_sample_count"],
        validation_samples=bundle.manifest["validation_sample_count"],
    )
    train_for_baseline = EpochSynonymDataset(
        bundle.train_base,
        bundle.instructions,
        seed=seeds[0],
    )
    baselines = _baseline_metrics(
        train_for_baseline,
        bundle.validation,
        batch_size=settings.batch_size * 2,
        num_workers=settings.num_workers,
    )
    _write_json(output_root / "validation_baselines.json", baselines)
    _write_progress(
        progress_path,
        output_root=output_root,
        stage="baseline_complete",
        command="train",
        device=str(device),
        validation_samples=bundle.manifest["validation_sample_count"],
    )

    experiments: dict[str, Any] = {}
    for seed in seeds:
        name = _experiment_name("full", seed)
        _write_progress(
            progress_path,
            output_root=output_root,
            stage="seed_starting",
            command="train",
            seed=seed,
            modality="full",
            epochs_total=settings.epochs,
            device=str(device),
        )
        experiments[name] = _train_one(
            experiment_dir=output_root / name,
            modality="full",
            seed=seed,
            bundle=bundle,
            train_config=train_config,
            train_config_path=train_config_path,
            model_config=model_config,
            model_config_source=model_source,
            device=device,
            git_sha=args.git_sha,
            progress_path=progress_path,
        )
    entity_seed = int(train_config["baselines"]["entity_only_seed"])
    entity_name = _experiment_name("entity_only", entity_seed)
    _write_progress(
        progress_path,
        output_root=output_root,
        stage="seed_starting",
        command="train",
        seed=entity_seed,
        modality="entity_only",
        epochs_total=settings.epochs,
        device=str(device),
    )
    experiments[entity_name] = _train_one(
        experiment_dir=output_root / entity_name,
        modality="entity_only",
        seed=entity_seed,
        bundle=bundle,
        train_config=train_config,
        train_config_path=train_config_path,
        model_config=model_config,
        model_config_source=model_source,
        device=device,
        git_sha=args.git_sha,
        progress_path=progress_path,
    )

    acceptance_config = train_config.get("acceptance")
    if not isinstance(acceptance_config, Mapping):
        raise ValueError("acceptance configuration is missing")
    full_acceptance = {
        name: _acceptance(
            metrics["validation"],
            baselines["label_mean"],
            acceptance_config,
        )
        for name, metrics in experiments.items()
        if metrics["modality"] == "full"
    }
    validation_gate = all(
        result["passed"] for result in full_acceptance.values()
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": args.git_sha,
        "device": str(device),
        "seeds": list(seeds),
        "validation_gate_passed": validation_gate,
        "validation_baselines": baselines,
        "experiments": experiments,
        "validation_acceptance": full_acceptance,
    }
    _write_json(output_root / "summary.json", summary)
    _write_progress(
        progress_path,
        output_root=output_root,
        stage="suite_complete",
        command="train",
        device=str(device),
        validation_gate_passed=validation_gate,
        experiment_count=len(experiments),
    )
    status = "PASS" if validation_gate else "FAIL"
    print(
        f"VALIDATION_{status} "
        f"seeds={','.join(str(seed) for seed in seeds)} "
        f"label_mean_action_error={baselines['label_mean']['action_error_m']:.6f}"
    )
    return 0 if validation_gate else 2


def _load_best_model(
    experiment_dir: Path,
    *,
    model_config: SmallPolicyConfig,
    device: torch.device,
    git_sha: str,
) -> tuple[SmallActionPolicy, dict[str, Any]]:
    checkpoint = torch.load(
        experiment_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"{experiment_dir}: checkpoint schema mismatch")
    if checkpoint.get("git_sha") != git_sha:
        raise ValueError(f"{experiment_dir}: checkpoint Git SHA mismatch")
    model = SmallActionPolicy(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def evaluate_sealed_test(args: argparse.Namespace) -> int:
    _require_pc_training_target(getattr(args, "execution_target", "pc"))
    output_root = args.output_root.resolve()
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not bool(summary.get("validation_gate_passed")):
        raise ValueError("sealed test cannot run before validation gate passes")
    if summary.get("git_sha") != args.git_sha:
        raise ValueError("training summary Git SHA mismatch")
    train_config = _read_yaml(args.config.resolve())
    model_source = _read_yaml(args.model_config.resolve())
    model_config = SmallPolicyConfig.from_mapping(model_source)
    settings = _parse_settings(train_config)
    device = _resolve_device(args.device)
    bundle = _build_dataset_bundle(
        feature_root=args.features.resolve(),
        split_path=args.split.resolve(),
        instructions_path=args.instructions.resolve(),
        train_config=train_config,
        include_test=True,
    )
    if bundle.test is None:
        raise RuntimeError("test dataset was not built")
    seeds = tuple(int(seed) for seed in train_config["seeds"])
    baseline_train = EpochSynonymDataset(
        bundle.train_base,
        bundle.instructions,
        seed=seeds[0],
    )
    baselines = _baseline_metrics(
        baseline_train,
        bundle.test,
        batch_size=settings.batch_size * 2,
        num_workers=settings.num_workers,
    )
    _write_json(output_root / "test_baselines.json", baselines)

    test_experiments: dict[str, Any] = {}
    for name, experiment in summary["experiments"].items():
        model, checkpoint = _load_best_model(
            output_root / name,
            model_config=model_config,
            device=device,
            git_sha=args.git_sha,
        )
        metrics, _ = _evaluate_model(
            model,
            bundle.test,
            device=device,
            batch_size=settings.batch_size * 2,
            num_workers=settings.num_workers,
            modality=str(experiment["modality"]),
        )
        result = {
            "best_epoch": int(checkpoint["epoch"]),
            "modality": str(experiment["modality"]),
            "seed": int(experiment["seed"]),
            "test": metrics,
        }
        test_experiments[name] = result
        _write_json(output_root / name / "test_metrics.json", result)

    acceptance_config = train_config["acceptance"]
    full_acceptance = {
        name: _acceptance(
            value["test"],
            baselines["label_mean"],
            acceptance_config,
        )
        for name, value in test_experiments.items()
        if value["modality"] == "full"
    }
    full_action_error = [
        value["test"]["action_error_m"]
        for value in test_experiments.values()
        if value["modality"] == "full"
    ]
    test_gate = all(value["passed"] for value in full_acceptance.values())
    summary["test"] = {
        "opened_utc": datetime.now(timezone.utc).isoformat(),
        "gate_passed": test_gate,
        "baselines": baselines,
        "experiments": test_experiments,
        "acceptance": full_acceptance,
        "three_seed_test_action_error_mean_m": float(np.mean(full_action_error)),
        "three_seed_test_action_error_std_m": float(np.std(full_action_error)),
    }
    _write_json(summary_path, summary)
    status = "PASS" if test_gate else "FAIL"
    print(
        f"TRAINING_{status} "
        f"test_action_error_mean={np.mean(full_action_error):.6f} "
        f"test_action_error_std={np.std(full_action_error):.6f} "
        f"label_mean_action_error={baselines['label_mean']['action_error_m']:.6f}"
    )
    return 0 if test_gate else 2


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--execution-target",
        choices=("pc",),
        default="pc",
        help="PC-only training/evaluation; Jetson exposes inference only",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 15 training pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    _add_common_arguments(train_parser)
    test_parser = subparsers.add_parser("evaluate-test")
    _add_common_arguments(test_parser)
    args = parser.parse_args()
    if args.command == "train":
        try:
            return train_validation_suite(args)
        except Exception as exc:
            progress_path = args.output_root.resolve() / "progress.json"
            if progress_path.parent.is_dir():
                try:
                    _write_progress(
                        progress_path,
                        output_root=progress_path.parent,
                        stage="failed",
                        command="train",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                except OSError:
                    pass
            raise
    return evaluate_sealed_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
