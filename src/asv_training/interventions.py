"""Frozen-policy language and structured-entity intervention evaluation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

from asv_training.dataset import (
    AnnotatedFeatureDataset,
    FrozenFeatureDataset,
    discover_feature_caches,
    load_instruction_metadata,
    load_split_assignments,
    policy_inputs_from_batch,
)
from asv_training.metrics import compute_action_metrics
from asv_training.model import SmallActionPolicy, SmallPolicyConfig
from asv_training.train import CHECKPOINT_SCHEMA_VERSION


SCHEMA_VERSION = "intervention_report_v1"
CONFIG_SCHEMA_VERSION = "interventions_v1"
ABLATION_VARIANTS = frozenset(
    {
        "full",
        "no_language",
        "no_entity_geometry",
    }
)
FAIL_CLOSED_FAULTS = frozenset(
    {"missing_language", "missing_entity_geometry", "entity_alignment_error"}
)


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        output.append(value)
    return output


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


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    return device


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Day 16 configuration schema mismatch")
    if int(config.get("expected_run_count", 0)) not in (2, 8, 30):
        raise ValueError(
            "Day 16 requires the frozen 30-Run set or a 2/8-Run holdout"
        )
    seeds = tuple(int(seed) for seed in config.get("seeds", ()))
    if seeds != (17, 23, 42):
        raise ValueError("Day 16 requires frozen seeds [17, 23, 42]")
    language = config.get("language_interventions")
    color_swap = config.get("color_swap")
    ablations = config.get("ablations")
    fail_closed = config.get("fail_closed")
    if not all(
        isinstance(value, Mapping)
        for value in (language, color_swap, ablations, fail_closed)
    ):
        raise ValueError("Day 16 configuration sections are incomplete")
    variants = {str(value) for value in ablations.get("variants", ())}
    if not variants or not variants <= ABLATION_VARIANTS - {"full"}:
        raise ValueError(f"unsupported ablation variants: {sorted(variants)}")
    faults = {str(value) for value in fail_closed.get("faults", ())}
    if faults != FAIL_CLOSED_FAULTS:
        raise ValueError("all three fail-closed faults must be configured")
    for section in (language, color_swap):
        for key in (
            "minimum_directional_accuracy",
            "minimum_assignment_accuracy",
            "minimum_response_ratio",
        ):
            value = float(section[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be in [0, 1]")


def apply_intervention(
    inputs: Mapping[str, Tensor],
    intervention: str,
) -> dict[str, Tensor]:
    """Return a non-mutating policy-input intervention."""

    output = {key: value.clone() for key, value in inputs.items()}
    if intervention == "full":
        return output
    if intervention == "no_language":
        output["language"].zero_()
    elif intervention == "no_entity_geometry":
        output["entity_geometry"].zero_()
    elif intervention == "missing_language":
        output["language_valid"].zero_()
    elif intervention == "missing_entity_geometry":
        output["entity_geometry_mask"].zero_()
        output["policy_input_valid"].zero_()
    elif intervention == "entity_alignment_error":
        output["policy_input_valid"].zero_()
    else:
        raise ValueError(f"unknown intervention={intervention!r}")
    return output


def compute_pair_response(
    prediction_left: np.ndarray,
    prediction_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
) -> dict[str, Any]:
    """Score whether a paired output changes in the expert's direction."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            prediction_left,
            prediction_right,
            target_left,
            target_right,
        )
    )
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("paired action arrays must have equal shapes")
    if arrays[0].ndim != 2 or arrays[0].shape[-1] != 2:
        raise ValueError("paired actions must have shape [N,2]")
    pred_left, pred_right, true_left, true_right = arrays
    predicted_delta = pred_right - pred_left
    target_delta = true_right - true_left
    target_norm = np.linalg.norm(target_delta, axis=1)
    predicted_norm = np.linalg.norm(predicted_delta, axis=1)
    usable = target_norm > 1.0e-8
    if not bool(np.any(usable)):
        raise ValueError("paired expert actions never differ")
    signed_dot = np.sum(predicted_delta * target_delta, axis=1)
    direction_correct = signed_dot[usable] > 0.0
    response_ratio = predicted_norm[usable] / target_norm[usable]

    def _action_error(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.linalg.norm(first - second, axis=-1)

    correct_cost = _action_error(pred_left, true_left) + _action_error(
        pred_right, true_right
    )
    swapped_cost = _action_error(pred_left, true_right) + _action_error(
        pred_right, true_left
    )
    assignment_correct = correct_cost[usable] < swapped_cost[usable]
    return {
        "sample_count": int(np.sum(usable)),
        "directional_accuracy": float(np.mean(direction_correct)),
        "assignment_accuracy": float(np.mean(assignment_correct)),
        "median_response_ratio": float(np.median(response_ratio)),
        "mean_correct_cost_m": float(np.mean(correct_cost[usable])),
        "mean_swapped_cost_m": float(np.mean(swapped_cost[usable])),
        "_direction_correct": direction_correct,
        "_assignment_correct": assignment_correct,
        "_usable_indices": np.nonzero(usable)[0],
    }


def pair_response_passed(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> bool:
    return (
        float(metrics["directional_accuracy"])
        >= float(thresholds["minimum_directional_accuracy"])
        and float(metrics["assignment_accuracy"])
        >= float(thresholds["minimum_assignment_accuracy"])
        and float(metrics["median_response_ratio"])
        >= float(thresholds["minimum_response_ratio"])
    )


def _model_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in policy_inputs_from_batch(batch).items()
    }


def _predict_dataset(
    model: SmallActionPolicy,
    dataset: Dataset[Any],
    *,
    device: torch.device,
    batch_size: int,
    intervention: str,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    target_stops: list[np.ndarray] = []
    fields = {
        "run_id": [],
        "frame_key": [],
        "instruction_id": [],
        "task_label": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = apply_intervention(_model_inputs(batch, device), intervention)
            output = model(**inputs)
            if intervention in ABLATION_VARIANTS and not bool(
                torch.all(output.valid_mask)
            ):
                raise ValueError(
                    f"ablation {intervention} unexpectedly invalidated an input"
                )
            predictions.append(output.action.detach().cpu().numpy())
            targets.append(batch["target_action"].numpy())
            logits.append(output.stop_logit.detach().cpu().numpy())
            target_stops.append(batch["target_stop"].numpy())
            for key in ("run_id", "frame_key", "instruction_id"):
                fields[key].extend(str(value) for value in batch[key])
            fields["task_label"].extend(
                str(value) for value in batch["metadata"]["task_label"]
            )
    return {
        "prediction": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "stop_logits": np.concatenate(logits),
        "target_stop": np.concatenate(target_stops),
        **fields,
    }


def _policy_metrics(arrays: Mapping[str, Any]) -> dict[str, Any]:
    return compute_action_metrics(
        arrays["prediction"],
        arrays["target"],
        arrays["stop_logits"],
        arrays["target_stop"],
        arrays["task_label"],
    )


def _filter_arrays(
    arrays: Mapping[str, Any],
    predicate: Any,
) -> dict[str, Any]:
    indices = np.asarray(
        [index for index, label in enumerate(arrays["task_label"]) if predicate(label)],
        dtype=np.int64,
    )
    if not len(indices):
        raise ValueError("metric subset is empty")
    return {
        "prediction": arrays["prediction"][indices],
        "target": arrays["target"][indices],
        "stop_logits": arrays["stop_logits"][indices],
        "target_stop": arrays["target_stop"][indices],
        "task_label": [arrays["task_label"][index] for index in indices],
    }


def _index_predictions(arrays: Mapping[str, Any]) -> dict[tuple[str, str, str], int]:
    output: dict[tuple[str, str, str], int] = {}
    for index, values in enumerate(
        zip(arrays["run_id"], arrays["frame_key"], arrays["instruction_id"])
    ):
        key = tuple(str(value) for value in values)
        if key in output:
            raise ValueError(f"duplicate prediction key: {key}")
        output[key] = index
    return output


def _frame_index(frame_key: str) -> int:
    parts = frame_key.rsplit(":", 3)
    if len(parts) != 4:
        raise ValueError(f"invalid frame key: {frame_key!r}")
    return int(parts[2])


def _load_model(
    checkpoint_path: Path,
    *,
    model_config: SmallPolicyConfig,
    device: torch.device,
    expected_git_sha: str,
    expected_seed: int,
) -> tuple[SmallActionPolicy, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"{checkpoint_path}: checkpoint schema mismatch")
    if checkpoint.get("git_sha") != expected_git_sha:
        raise ValueError(f"{checkpoint_path}: checkpoint Git SHA mismatch")
    if int(checkpoint.get("seed", -1)) != expected_seed:
        raise ValueError(f"{checkpoint_path}: checkpoint seed mismatch")
    if checkpoint.get("modality") != "full":
        raise ValueError(f"{checkpoint_path}: expected full model")
    model = SmallActionPolicy(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def _clean_pair_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: current for key, current in value.items() if not key.startswith("_")}


def _pair_rows(
    *,
    arrays: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    seed: int,
    thresholds: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = _index_predictions(arrays)
    frames = sorted({(run_id, frame_key) for run_id, frame_key in zip(
        arrays["run_id"], arrays["frame_key"]
    )})
    rows: list[dict[str, Any]] = []
    plot_cases: list[dict[str, Any]] = []
    for pair in pairs:
        instruction_ids = tuple(str(value) for value in pair["instruction_ids"])
        left_indices: list[int] = []
        right_indices: list[int] = []
        pair_frames: list[tuple[str, str]] = []
        for run_id, frame_key in frames:
            left_key = (run_id, frame_key, instruction_ids[0])
            right_key = (run_id, frame_key, instruction_ids[1])
            if left_key in index and right_key in index:
                left_indices.append(index[left_key])
                right_indices.append(index[right_key])
                pair_frames.append((run_id, frame_key))
        if not left_indices:
            raise ValueError(f"pair {pair['pair_id']} has no matched observation")
        left = np.asarray(left_indices)
        right = np.asarray(right_indices)
        response = compute_pair_response(
            arrays["prediction"][left],
            arrays["prediction"][right],
            arrays["target"][left],
            arrays["target"][right],
        )
        passed = pair_response_passed(response, thresholds)
        row = {
            "kind": "language",
            "pair_id": str(pair["pair_id"]),
            "intervention_type": str(pair["intervention_type"]),
            "instruction_left": instruction_ids[0],
            "instruction_right": instruction_ids[1],
            "language_split": str(pair["split"]),
            "seed": seed,
            **_clean_pair_metrics(response),
            "passed": passed,
        }
        rows.append(row)
        plot_cases.append(
            {
                "name": f"language_{pair['pair_id']}_seed{seed}",
                "prediction_left": arrays["prediction"][left[0]],
                "prediction_right": arrays["prediction"][right[0]],
                "target_left": arrays["target"][left[0]],
                "target_right": arrays["target"][right[0]],
            }
        )
        usable = response["_usable_indices"]
        direction = response["_direction_correct"]
        assignment = response["_assignment_correct"]
        for local_index, direction_ok, assignment_ok in zip(
            usable, direction, assignment
        ):
            if bool(direction_ok) and bool(assignment_ok):
                continue
            run_id, frame_key = pair_frames[int(local_index)]
            failures.append(
                {
                    "category": (
                        "STOP"
                        if pair["intervention_type"] == "action"
                        else "visual"
                        if pair["intervention_type"] == "target_color"
                        else "grounding"
                    ),
                    "kind": "language_intervention",
                    "pair_id": str(pair["pair_id"]),
                    "seed": seed,
                    "run_id": run_id,
                    "frame_key": frame_key,
                    "direction_correct": bool(direction_ok),
                    "assignment_correct": bool(assignment_ok),
                }
            )
    return rows, plot_cases


def _registry_by_slot(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        slot = str(row.get("collection_slot", "")).strip()
        if not slot or slot in output:
            raise ValueError(f"invalid or duplicate collection slot: {slot!r}")
        output[slot] = row
    return output


def _subset_by_runs_and_instructions(
    dataset: AnnotatedFeatureDataset,
    run_ids: Iterable[str],
    instruction_ids: Iterable[str],
) -> Subset[Any]:
    wanted_runs = set(run_ids)
    wanted_instructions = set(instruction_ids)
    indices = [
        index
        for index in range(len(dataset.base))
        if (
            str(dataset.base.sample_metadata(index)["run_id"]) in wanted_runs
            and str(dataset.base.sample_metadata(index)["instruction_id"])
            in wanted_instructions
        )
    ]
    if not indices:
        raise ValueError("color-swap subset is empty")
    return Subset(dataset, indices)


def _color_swap_rows(
    *,
    arrays: Mapping[str, Any],
    seed: int,
    left_run: str,
    right_run: str,
    instruction_ids: Sequence[str],
    thresholds: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    plots: list[dict[str, Any]] = []
    for instruction_id in instruction_ids:
        left_index: dict[int, int] = {}
        right_index: dict[int, int] = {}
        for index, (run_id, frame_key, current_instruction) in enumerate(
            zip(arrays["run_id"], arrays["frame_key"], arrays["instruction_id"])
        ):
            if current_instruction != instruction_id:
                continue
            if run_id == left_run:
                left_index[_frame_index(frame_key)] = index
            elif run_id == right_run:
                right_index[_frame_index(frame_key)] = index
        matched = sorted(set(left_index) & set(right_index))
        if not matched:
            raise ValueError(
                f"no matched color-swap frames for instruction {instruction_id}"
            )
        left = np.asarray([left_index[value] for value in matched])
        right = np.asarray([right_index[value] for value in matched])
        response = compute_pair_response(
            arrays["prediction"][left],
            arrays["prediction"][right],
            arrays["target"][left],
            arrays["target"][right],
        )
        passed = pair_response_passed(response, thresholds)
        rows.append(
            {
                "kind": "color_swap",
                "pair_id": f"color_swap_{instruction_id}",
                "intervention_type": "held_out_L3_to_L4",
                "instruction_left": instruction_id,
                "instruction_right": instruction_id,
                "language_split": "test",
                "seed": seed,
                **_clean_pair_metrics(response),
                "passed": passed,
            }
        )
        plots.append(
            {
                "name": f"color_swap_{instruction_id}_seed{seed}",
                "prediction_left": arrays["prediction"][left[0]],
                "prediction_right": arrays["prediction"][right[0]],
                "target_left": arrays["target"][left[0]],
                "target_right": arrays["target"][right[0]],
            }
        )
        usable = response["_usable_indices"]
        for local_index, direction_ok, assignment_ok in zip(
            usable,
            response["_direction_correct"],
            response["_assignment_correct"],
        ):
            if bool(direction_ok) and bool(assignment_ok):
                continue
            failures.append(
                {
                    "category": "visual",
                    "kind": "held_out_color_swap",
                    "pair_id": f"color_swap_{instruction_id}",
                    "seed": seed,
                    "frame_index": matched[int(local_index)],
                    "left_run_id": left_run,
                    "right_run_id": right_run,
                    "direction_correct": bool(direction_ok),
                    "assignment_correct": bool(assignment_ok),
                }
            )
    return rows, plots


def _fail_closed_checks(
    model: SmallActionPolicy,
    dataset: Dataset[Any],
    *,
    device: torch.device,
    faults: Sequence[str],
    maximum_nonzero: float,
) -> dict[str, Any]:
    batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=False)))
    base = _model_inputs(batch, device)
    output: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for fault in faults:
            result = model(**apply_intervention(base, fault))
            maximum = float(torch.max(torch.linalg.vector_norm(result.action, dim=-1)).cpu())
            all_invalid = bool(torch.all(~result.valid_mask).cpu())
            all_stop = bool(torch.all(result.stop_logit >= 0.0).cpu())
            passed = maximum <= maximum_nonzero and all_invalid and all_stop
            output[fault] = {
                "maximum_action_norm_m": maximum,
                "all_invalid": all_invalid,
                "all_stop_selected": all_stop,
                "passed": passed,
            }
    return output


def _relative_degradation(ablated: float, full: float) -> float:
    if full <= 0.0:
        raise ValueError("full-model action error must be positive for ablation comparison")
    return float((ablated - full) / full)


def _draw_action_plot(path: Path, case: Mapping[str, Any]) -> None:
    width = height = 640
    margin = 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((15, 12), str(case["name"]), fill="black")
    actions = [
        ("pred A", np.asarray(case["prediction_left"]), (30, 100, 220)),
        ("pred B", np.asarray(case["prediction_right"]), (220, 60, 50)),
        ("expert A", np.asarray(case["target_left"]), (80, 160, 220)),
        ("expert B", np.asarray(case["target_right"]), (240, 150, 100)),
    ]
    points = np.stack([value.reshape(2) for _, value, _ in actions], axis=0)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    span = np.maximum(maximum - minimum, 1.0e-3)
    for ordinal, (name, values, color) in enumerate(actions):
        x_value, y_value = values.reshape(2)
        x = margin + (x_value - minimum[0]) / span[0] * (width - 2 * margin)
        y = height - margin - (y_value - minimum[1]) / span[1] * (height - 2 * margin)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        draw.text((15, 35 + ordinal * 18), name, fill=color)
    image.save(path, format="PNG")


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "seed",
        "variant",
        "scope",
        "task_label",
        "sample_count",
        "action_error_m",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_pairs_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "kind",
        "pair_id",
        "intervention_type",
        "instruction_left",
        "instruction_right",
        "language_split",
        "seed",
        "sample_count",
        "directional_accuracy",
        "assignment_accuracy",
        "median_response_ratio",
        "mean_correct_cost_m",
        "mean_swapped_cost_m",
        "passed",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = _read_yaml(config_path)
    _validate_config(config)
    summary_path = args.output.resolve() / "summary.json"
    expected_summary_sha256 = str(
        config.get(
            "expected_training_summary_sha256",
            config.get("expected_training_summary_sha256", ""),
        )
    )
    if not expected_summary_sha256:
        raise ValueError("expected training summary SHA-256 is missing")
    if _sha256_file(summary_path) != expected_summary_sha256:
        raise ValueError("frozen training summary SHA-256 mismatch")
    training_summary = _read_json(summary_path)
    require_test_gate = bool(config.get("require_sealed_test_gate", True))
    if require_test_gate and not bool(
        training_summary.get("test", {}).get("gate_passed")
    ):
        raise ValueError("Day 15 sealed test gate was not passed")
    if not bool(training_summary.get("validation_gate_passed")):
        raise ValueError("training validation gate was not passed")
    expected_git_sha = str(config["expected_checkpoint_git_sha"])
    if training_summary.get("git_sha") != expected_git_sha:
        raise ValueError("Day 15 training Git SHA mismatch")

    feature_root = args.features.resolve()
    registry_path = args.registry.resolve()
    split_path = args.split.resolve()
    provenance_files = {
        "expected_registry_sha256": registry_path,
        "expected_split_sha256": split_path,
        "expected_feature_set_manifest_sha256": (
            feature_root / "feature_set_manifest.json"
        ),
    }
    for config_key, source_path in provenance_files.items():
        expected_sha256 = str(config.get(config_key, "")).strip()
        if expected_sha256 and _sha256_file(source_path) != expected_sha256:
            raise ValueError(f"{config_key} mismatch")
    caches = discover_feature_caches(feature_root)
    if len(caches) != int(config["expected_run_count"]):
        raise ValueError(
            f"expected {int(config['expected_run_count'])} caches, "
            f"found {len(caches)}"
        )
    assignments = load_split_assignments(split_path)
    instructions = load_instruction_metadata(args.instructions.resolve())
    frame_stride = int(config["frame_stride"])
    evaluation_split = str(
        config["language_interventions"]["evaluation_run_split"]
    )
    test_all_base = FrozenFeatureDataset(
        caches,
        selected_split=evaluation_split,
        split_assignments=assignments,
        allowed_language_splits=config["language_interventions"][
            "evaluation_language_splits"
        ],
        frame_stride=frame_stride,
    )
    test_all = AnnotatedFeatureDataset(test_all_base, instructions)
    test_standard_base = FrozenFeatureDataset(
        caches,
        selected_split=evaluation_split,
        split_assignments=assignments,
        allowed_language_splits=["test"],
        frame_stride=frame_stride,
    )
    test_standard = AnnotatedFeatureDataset(test_standard_base, instructions)
    color_config = config["color_swap"]
    color_run_split = str(
        color_config.get("evaluation_run_split", evaluation_split)
    )
    color_all_base = FrozenFeatureDataset(
        caches,
        selected_split=color_run_split,
        split_assignments=assignments,
        allowed_language_splits=["train", "validation", "test"],
        frame_stride=frame_stride,
    )
    color_all = AnnotatedFeatureDataset(color_all_base, instructions)

    pairs = _read_jsonl(args.contrast_pairs.resolve())
    required_pairs = int(config["language_interventions"]["require_pair_count"])
    if len(pairs) != required_pairs:
        raise ValueError(f"expected {required_pairs} contrast pairs, found {len(pairs)}")
    registry = _registry_by_slot(registry_path)
    left_slot = str(color_config["left_slot"])
    right_slot = str(color_config["right_slot"])
    if left_slot not in registry or right_slot not in registry:
        raise ValueError("configured color-swap slots are absent from the registry")
    left_run = str(registry[left_slot]["run_id"])
    right_run = str(registry[right_slot]["run_id"])
    if (
        assignments[left_run] != color_run_split
        or assignments[right_run] != color_run_split
    ):
        raise ValueError(
            f"color-swap Runs must both be held-out {color_run_split} Runs"
        )
    color_instruction_ids = tuple(
        str(value) for value in color_config["instruction_ids"]
    )
    color_dataset = _subset_by_runs_and_instructions(
        color_all,
        (left_run, right_run),
        color_instruction_ids,
    )

    model_source = _read_yaml(args.model_config.resolve())
    model_config = SmallPolicyConfig.from_mapping(model_source)
    device = _resolve_device(args.device)
    batch_size = int(args.batch_size)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    plot_root = output_root / "action_plots"
    plot_root.mkdir()

    metrics_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    plot_cases: list[dict[str, Any]] = []
    seed_reports: dict[str, Any] = {}
    ablation_config = config["ablations"]
    variants = ("full", *tuple(str(v) for v in ablation_config["variants"]))
    for seed in (int(value) for value in config["seeds"]):
        model, checkpoint = _load_model(
            args.output.resolve() / f"full_seed{seed}" / "best.pt",
            model_config=model_config,
            device=device,
            expected_git_sha=expected_git_sha,
            expected_seed=seed,
        )
        variant_metrics: dict[str, Any] = {}
        for variant in variants:
            arrays = _predict_dataset(
                model,
                test_standard,
                device=device,
                batch_size=batch_size,
                intervention=variant,
            )
            overall = _policy_metrics(arrays)
            color = _policy_metrics(
                _filter_arrays(arrays, lambda label: "|color:" in label)
            )
            bearing = _policy_metrics(
                _filter_arrays(arrays, lambda label: "|bearing:" in label)
            )
            variant_metrics[variant] = {
                "overall": overall,
                "color": color,
                "bearing": bearing,
            }
            for label, values in overall["per_label"].items():
                metrics_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "scope": "per_label",
                        "task_label": label,
                        "sample_count": values["sample_count"],
                        "action_error_m": values["action_error_m"],
                    }
                )
            for scope, values in (
                ("overall", overall),
                ("color", color),
                ("bearing", bearing),
            ):
                metrics_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "scope": scope,
                        "task_label": "*",
                        "sample_count": values["sample_count"],
                        "action_error_m": values["action_error_m"],
                    }
                )

        pair_arrays = _predict_dataset(
            model,
            test_all,
            device=device,
            batch_size=batch_size,
            intervention="full",
        )
        current_pair_rows, current_plots = _pair_rows(
            arrays=pair_arrays,
            pairs=pairs,
            seed=seed,
            thresholds=config["language_interventions"],
            failures=failures,
        )
        pair_rows.extend(current_pair_rows)
        plot_cases.extend(current_plots)

        color_arrays = _predict_dataset(
            model,
            color_dataset,
            device=device,
            batch_size=batch_size,
            intervention="full",
        )
        current_color_rows, current_color_plots = _color_swap_rows(
            arrays=color_arrays,
            seed=seed,
            left_run=left_run,
            right_run=right_run,
            instruction_ids=color_instruction_ids,
            thresholds=color_config,
            failures=failures,
        )
        pair_rows.extend(current_color_rows)
        plot_cases.extend(current_color_plots)

        fail_closed = _fail_closed_checks(
            model,
            test_standard,
            device=device,
            faults=tuple(str(v) for v in config["fail_closed"]["faults"]),
            maximum_nonzero=float(
                config["fail_closed"]["maximum_nonzero_action_m"]
            ),
        )
        full_metrics = variant_metrics["full"]
        degradations = {
            "language_overall": _relative_degradation(
                variant_metrics["no_language"]["overall"]["action_error_m"],
                full_metrics["overall"]["action_error_m"],
            ),
            "entity_bearing": _relative_degradation(
                variant_metrics["no_entity_geometry"]["bearing"]["action_error_m"],
                full_metrics["bearing"]["action_error_m"],
            ),
        }
        seed_reports[str(seed)] = {
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "variants": variant_metrics,
            "degradations": degradations,
            "fail_closed": fail_closed,
        }

    mean_degradations = {
        name: float(
            np.mean(
                [
                    seed_reports[str(seed)]["degradations"][name]
                    for seed in config["seeds"]
                ]
            )
        )
        for name in ("language_overall", "entity_bearing")
    }
    ablation_gates = {
        "language": mean_degradations["language_overall"]
        >= float(
            ablation_config["minimum_language_relative_action_error_degradation"]
        ),
        "entity": mean_degradations["entity_bearing"]
        >= float(
            ablation_config["minimum_entity_bearing_relative_action_error_degradation"]
        ),
    }
    fail_closed_passed = all(
        fault["passed"]
        for seed in seed_reports.values()
        for fault in seed["fail_closed"].values()
    )
    language_rows = [row for row in pair_rows if row["kind"] == "language"]
    color_rows = [row for row in pair_rows if row["kind"] == "color_swap"]
    language_pair_pass = all(row["passed"] for row in language_rows)
    color_swap_pass = all(row["passed"] for row in color_rows)
    overall_passed = (
        language_pair_pass
        and color_swap_pass
        and all(ablation_gates.values())
        and fail_closed_passed
    )

    maximum_failures = int(config["report"]["maximum_failure_cases"])
    with (output_root / "failure_cases.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for value in failures[:maximum_failures]:
            stream.write(
                json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
                + "\n"
            )
    maximum_plots = int(config["report"]["maximum_action_plots"])
    for case in plot_cases[:maximum_plots]:
        _draw_action_plot(plot_root / f"{case['name']}.png", case)
    _write_metrics_csv(output_root / "metrics_by_label.csv", metrics_rows)
    _write_pairs_csv(output_root / "intervention_pairs.csv", pair_rows)
    ablation_summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "config_sha256": _sha256_file(config_path),
        "training_summary_sha256": _sha256_file(summary_path),
        "checkpoint_git_sha": expected_git_sha,
        "dataset": {
            "feature_root": str(feature_root),
            "feature_cache_count": len(caches),
            "evaluation_run_split": evaluation_split,
            "standard_test_sample_count": len(test_standard),
            "all_language_test_sample_count": len(test_all),
            "color_swap_left_slot": left_slot,
            "color_swap_right_slot": right_slot,
            "color_swap_left_run_id": left_run,
            "color_swap_right_run_id": right_run,
            "color_swap_run_split": color_run_split,
            "registry_sha256": _sha256_file(registry_path),
            "split_sha256": _sha256_file(split_path),
            "feature_set_manifest_sha256": _sha256_file(
                feature_root / "feature_set_manifest.json"
            ),
        },
        "language_interventions": {
            "configured_pair_count": len(pairs),
            "evaluated_seed_pair_count": len(language_rows),
            "all_passed": language_pair_pass,
        },
        "color_swap": {
            "configured_instruction_count": len(color_instruction_ids),
            "evaluated_seed_pair_count": len(color_rows),
            "all_passed": color_swap_pass,
        },
        "ablations": {
            "three_seed_mean_relative_action_error_degradation": mean_degradations,
            "gates": ablation_gates,
            "all_passed": all(ablation_gates.values()),
        },
        "fail_closed": {
            "all_passed": fail_closed_passed,
            "fault_count": sum(
                len(seed["fail_closed"]) for seed in seed_reports.values()
            ),
        },
        "failure_case_count": len(failures),
        "failure_case_count_written": min(len(failures), maximum_failures),
        "action_plot_count": min(len(plot_cases), maximum_plots),
        "seeds": seed_reports,
        "passed": overall_passed,
    }
    _write_json(output_root / "ablation_summary.json", ablation_summary)
    status = "PASS" if overall_passed else "FAIL"
    print(
        f"INTERVENTIONS_{status} "
        f"language={language_pair_pass} color_swap={color_swap_pass} "
        f"ablations={all(ablation_gates.values())} "
        f"fail_closed={fail_closed_passed}"
    )
    return 0 if overall_passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen policy under offline interventions."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--contrast-pairs", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
