"""Metrics and deterministic baselines for single-step actions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


MAXIMUM_ACTION_M = 0.3
STOP_DRIFT_LIMIT_M = 0.10


def _binary_metrics(
    predicted: np.ndarray, target: np.ndarray
) -> dict[str, float | int]:
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
    maximum_action_m: float = MAXIMUM_ACTION_M,
    stop_drift_limit_m: float = STOP_DRIFT_LIMIT_M,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float32)
    expected = np.asarray(target, dtype=np.float32)
    if predicted.ndim != 2 or predicted.shape[1:] != (2,):
        raise ValueError(f"prediction must have shape [N,2], got {predicted.shape}")
    if expected.shape != predicted.shape:
        raise ValueError("prediction and target action shapes differ")
    logits = np.asarray(stop_logits, dtype=np.float32).reshape(-1)
    stops = np.asarray(target_stop, dtype=np.bool_).reshape(-1)
    labels = np.asarray([str(label) for label in task_labels], dtype=np.str_)
    count = len(predicted)
    if len(logits) != count or len(stops) != count or len(labels) != count:
        raise ValueError("action, stop, and label arrays must have the same length")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(logits)):
        raise ValueError("prediction contains NaN or Inf")
    if not np.all(np.isfinite(expected)):
        raise ValueError("target action contains NaN or Inf")
    error = np.linalg.norm(predicted - expected, axis=1)
    norms = np.linalg.norm(predicted, axis=1)
    predicted_stop = logits >= 0.0
    stop_rows = norms[stops]
    per_label = {
        label: {
            "sample_count": int(np.count_nonzero(labels == label)),
            "action_error_m": float(np.mean(error[labels == label])),
        }
        for label in sorted(set(labels.tolist()))
    }
    violation = norms > maximum_action_m + 1.0e-6
    return {
        "sample_count": count,
        "action_error_m": float(np.mean(error)) if count else 0.0,
        "stop_drift": {
            "sample_count": int(len(stop_rows)),
            "mean_m": float(np.mean(stop_rows)) if len(stop_rows) else 0.0,
            "p95_m": float(np.percentile(stop_rows, 95)) if len(stop_rows) else 0.0,
            "maximum_m": float(np.max(stop_rows)) if len(stop_rows) else 0.0,
            "within_0_10m_rate": float(np.mean(stop_rows <= stop_drift_limit_m)) if len(stop_rows) else 0.0,
        },
        "stop_classification": _binary_metrics(predicted_stop, stops),
        "action_bound": {
            "maximum_action_m": maximum_action_m,
            "observed_maximum_action_m": float(np.max(norms)) if count else 0.0,
            "violation_count": int(np.count_nonzero(violation)),
            "violation_rate": float(np.mean(violation)) if count else 0.0,
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
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for label, action in zip(labels, target):
        grouped[label].append(action)
    return {
        label: np.mean(np.stack(rows), axis=0).astype(np.float32)
        for label, rows in sorted(grouped.items())
    }


def predict_label_mean_action_baseline(
    means: Mapping[str, np.ndarray], task_labels: Iterable[str]
) -> tuple[np.ndarray, np.ndarray]:
    labels = [str(label) for label in task_labels]
    missing = sorted(set(labels) - set(means))
    if missing:
        raise ValueError(f"mean baseline has no labels: {missing}")
    actions = np.stack([means[label] for label in labels]).astype(np.float32)
    stop_logits = np.asarray(
        [20.0 if label.startswith("stop|") else -20.0 for label in labels],
        dtype=np.float32,
    )
    return actions, stop_logits


def improvement_fraction(policy_value: float, baseline_value: float) -> float:
    if baseline_value <= 0.0:
        return 0.0
    return float((baseline_value - policy_value) / baseline_value)


__all__ = [
    "MAXIMUM_ACTION_M",
    "STOP_DRIFT_LIMIT_M",
    "compute_action_metrics",
    "fit_label_mean_action_baseline",
    "predict_label_mean_action_baseline",
    "improvement_fraction",
]
