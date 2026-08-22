from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_final import (
    build_policy_dataset,
    load_episode_records,
    load_language_embeddings,
    save_policy_checkpoint,
    split_moving_target_slots,
    split_slots,
    task_key,
    train_perception,
    validate_policy_checkpoint,
)
from decision import SmallActionPolicy, SmallPolicyConfig
from perception import ImageEntityModel


ROOT = Path("D:/asv-vla-training")
DATA = ROOT / "data/episodes/moving_target_valid"
RUN = ROOT / "experiments/moving_target"
EMBEDDING_PATH = ROOT / "data/qwen_final_embeddings.npz"


def moving_target_split(records):
    split = split_moving_target_slots([record.slot_id for record in records])
    owners = {}
    for record in records:
        partition = next(name for name, slots in split.items() if record.slot_id in slots)
        identity = (record.run_id, record.scene_seed)
        previous = owners.setdefault(identity, partition)
        if previous != partition:
            raise ValueError(f"run/seed leakage between {previous} and {partition}: {identity}")
    return split


def relative_entity_positions(
    entity_world: dict[str, np.ndarray], asv_world: np.ndarray
) -> dict[str, np.ndarray]:
    """Convert fixed world positions into the current ASV-relative frame."""

    return {
        entity_id: np.asarray(position, dtype=np.float32) - np.asarray(asv_world, dtype=np.float32)
        for entity_id, position in entity_world.items()
    }


def visibility_metrics(
    raw_visibility: dict[str, bool],
    truth_visibility: dict[str, bool],
    *,
    selected_target_id: str,
) -> dict[str, int]:
    """Report all-slot classification separately from selected-target recall."""

    return {
        "all_slot_correct": sum(
            bool(raw_visibility[entity_id]) == bool(truth_visibility[entity_id])
            for entity_id in truth_visibility
        ),
        "all_slot_total": len(truth_visibility),
        "selected_target_detected": int(
            bool(raw_visibility.get(selected_target_id, False))
            and bool(truth_visibility.get(selected_target_id, False))
        ),
        "selected_target_total": int(bool(truth_visibility.get(selected_target_id, False))),
    }


def tracking_metrics(
    time_s: np.ndarray,
    target_world_xy: np.ndarray,
    asv_world_xy: np.ndarray,
    *,
    desired_standoff_m: float,
    policy_driven: float,
) -> dict[str, float | bool]:
    time = np.asarray(time_s, dtype=np.float64).reshape(-1)
    target = np.asarray(target_world_xy, dtype=np.float64)
    asv = np.asarray(asv_world_xy, dtype=np.float64)
    if len(time) < 2 or target.shape != (len(time), 2) or asv.shape != target.shape:
        raise ValueError("tracking traces must have matching [N] and [N, 2] shapes")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(target)) or not np.all(np.isfinite(asv)):
        raise ValueError("tracking traces contain NaN or Inf")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("tracking time must be strictly increasing")
    signed_error = np.linalg.norm(target - asv, axis=1) - float(desired_standoff_m)
    duration = float(time[-1] - time[0])
    target_displacement = float(np.linalg.norm(target[-1] - target[0]))
    steady = signed_error[time >= time[-1] - 120.0]
    final = signed_error[time >= time[-1] - 10.0]
    post_warmup = signed_error[time >= time[0] + 10.0]
    steady_mae = float(np.mean(np.abs(steady)))
    final_mae = float(np.mean(np.abs(final)))
    max_abs = float(np.max(np.abs(post_warmup))) if len(post_warmup) else math.inf
    driven_ratio = float(policy_driven)
    diverged = bool(max_abs > 8.0)
    acceptance_ready = bool(
        duration >= 179.0
        and target_displacement >= 50.0
        and driven_ratio >= 0.90
        and not diverged
        and steady_mae <= 0.5
        and final_mae <= 0.3
    )
    return {
        "duration_s": duration,
        "target_displacement_m": target_displacement,
        "steady_state_mae_m": steady_mae,
        "final_window_mae_m": final_mae,
        "max_abs_error_after_warmup_m": max_abs,
        "policy_driven_ratio": driven_ratio,
        "diverged": diverged,
        "acceptance_ready": acceptance_ready,
    }


def perception_eval(model_path: Path, records, embeddings, slots):
    model = ImageEntityModel.load(model_path)
    errors = []
    visibility = {"all_slot_correct": 0, "all_slot_total": 0,
                  "selected_target_detected": 0, "selected_target_total": 0}
    for record in records:
        if record.slot_id not in slots:
            continue
        from PIL import Image
        predictions = model.predict(
            Image.open(record.image_path),
            task=record.task_text,
            task_embedding=embeddings[task_key(record.task_text)],
            color_image=Image.open(record.image_path),
        )
        truth = {item["entity_id"]: item for item in record.entities}
        raw_visibility = {prediction.entity_id: prediction.visible for prediction in predictions}
        truth_visibility = {entity_id: bool(item["visible"]) for entity_id, item in truth.items()}
        selected_color = "blue" if task_key(record.task_text).startswith("blue") else "red"
        selected_target_id = f"target_{selected_color}"
        current = visibility_metrics(
            raw_visibility, truth_visibility, selected_target_id=selected_target_id
        )
        for key in visibility:
            visibility[key] += current[key]
        for prediction in predictions:
            target = truth[prediction.entity_id]
            expected = np.asarray(target["relative_position_m"], dtype=np.float32)
            actual = np.asarray(
                (prediction.relative_x, prediction.relative_y, prediction.relative_z),
                dtype=np.float32,
            )
            errors.append(actual - expected)
    errors = np.asarray(errors)
    return {
        "geometry_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "geometry_xy_rmse_m": float(np.sqrt(np.mean(errors[:, :2] ** 2))),
        "visibility_accuracy": visibility["all_slot_correct"] / max(visibility["all_slot_total"], 1),
        "selected_target_visibility_recall": visibility["selected_target_detected"] / max(visibility["selected_target_total"], 1),
        "samples": int(len(errors)),
    }


def policy_eval(model_path: Path, records, embeddings, slots):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    validate_policy_checkpoint(checkpoint)
    policy = SmallActionPolicy(SmallPolicyConfig.from_mapping(checkpoint["model_config"]))
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    selected = [record for record in records if record.slot_id in slots]
    dataset = build_policy_dataset(selected, embeddings)
    with torch.inference_mode():
        output = policy(
            language=torch.from_numpy(dataset["language"]),
            entity_geometry=torch.from_numpy(dataset["entity_geometry"]),
            ego_state=torch.from_numpy(dataset["ego_state"]),
            language_valid=torch.ones(len(dataset["language"]), dtype=torch.bool),
            entity_geometry_mask=torch.from_numpy(dataset["entity_geometry_mask"]),
            ego_state_valid=torch.ones(len(dataset["language"]), dtype=torch.bool),
            policy_input_valid=torch.ones(len(dataset["language"]), dtype=torch.bool),
        )
    actual = output.action.numpy()
    target = dataset["action"]
    return {
        "action_rmse_m": float(np.sqrt(np.mean((actual - target) ** 2))),
        "action_max_norm_m": float(np.max(np.linalg.norm(actual, axis=1))),
        "samples": int(len(actual)),
        "policy_driven_ratio": float(np.mean(np.linalg.norm(actual, axis=1) > 0.01)),
    }


def rollout_plot(model_path: Path, records, embeddings, output_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    policy = SmallActionPolicy(SmallPolicyConfig.from_mapping(checkpoint["model_config"]))
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    held_out = [record for record in records if record.slot_id.endswith("_TEST")]
    chosen = []
    for title, key in (("RED 3m", "red_3m"), ("BLUE 3m", "blue_3m"), ("RED 4m", "red_4m")):
        matches = [record for record in held_out if task_key(record.task_text) == key]
        if not matches:
            raise ValueError(f"missing held-out task for plot: {key}")
        chosen.append((title, key, matches[0]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    summary = {}
    for column, (title, key, record) in enumerate(chosen):
        color = "blue" if key.startswith("blue") else "red"
        entity_world = {
            item["entity_id"]: np.asarray(item["relative_position_m"][:2], dtype=np.float32)
            for item in record.entities
        }
        desired = 4.0 if key.endswith("4m") else 3.0
        asv = np.zeros(2, dtype=np.float32)
        target_world = entity_world[f"target_{color}"]
        asv_trace = [asv.copy()]
        target_trace = [target_world.copy()]
        errors = []
        for _ in range(120):
            relative_positions = relative_entity_positions(entity_world, asv)
            scaled = []
            for item in record.entities:
                value = dict(item)
                relative = relative_positions[item["entity_id"]]
                value["relative_position_m"] = [float(relative[0]), float(relative[1]), 0.0]
                scaled.append(value)
            class R:
                pass
            temp = R()
            temp.entities = tuple(scaled)
            temp.ego = {"surge_velocity_mps": 0.0, "yaw_rate_radps": 0.0}
            temp.task_text = "follow the blue boat, keep 3 meters distance" if key.startswith("blue") else ("follow the red boat, keep 4 meters distance" if key.endswith("4m") else "follow the red boat, keep 3 meters distance")
            temp.slot_id = "rollout"
            temp.frame_index = 0
            temp.action = np.zeros(2, dtype=np.float32)
            data = build_policy_dataset([temp], embeddings, distance_scales=(1.0,))
            with torch.inference_mode():
                output = policy(
                    language=torch.from_numpy(data["language"]),
                    entity_geometry=torch.from_numpy(data["entity_geometry"]),
                    ego_state=torch.from_numpy(data["ego_state"]),
                    language_valid=torch.ones(1, dtype=torch.bool),
                    entity_geometry_mask=torch.from_numpy(data["entity_geometry_mask"]),
                    ego_state_valid=torch.ones(1, dtype=torch.bool),
                    policy_input_valid=torch.ones(1, dtype=torch.bool),
                )
            action = output.action[0].numpy()
            asv += action
            asv_trace.append(asv.copy())
            target_trace.append(target_world.copy())
            errors.append(float(np.linalg.norm(target_world - asv) - desired))
        asv_trace = np.asarray(asv_trace)
        target_trace = np.asarray(target_trace)
        axes[0, column].plot(target_trace[:, 0], target_trace[:, 1], color=color, label="target")
        axes[0, column].plot(asv_trace[:, 0], asv_trace[:, 1], color="black", label="ASV")
        axes[0, column].scatter([0], [0], color="black", s=20)
        axes[0, column].set_title(title)
        axes[0, column].set_aspect("equal", adjustable="box")
        axes[0, column].grid(alpha=0.25)
        axes[1, column].plot(errors, color=color)
        axes[1, column].axhline(0.0, color="black", linewidth=0.8)
        axes[1, column].set_xlabel("step")
        axes[1, column].set_ylabel("signed standoff error (m)")
        axes[1, column].grid(alpha=0.25)
        summary[key] = {
            "initial_signed_standoff_error_m": float(errors[0]),
            "final_signed_standoff_error_m": float(errors[-1]),
            "max_abs_signed_standoff_error_m": float(np.max(np.abs(errors))),
            "diverged": bool(np.max(np.abs(errors)) > max(2.0 * desired, 8.0)),
        }
    axes[0, 0].legend()
    fig.suptitle("Final policy closed-loop rollout on held-out task conditions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def main():
    records = load_episode_records(DATA)
    embeddings = load_language_embeddings(EMBEDDING_PATH)
    split = moving_target_split(records)
    RUN.mkdir(parents=True, exist_ok=True)
    perception_path = RUN / "perception.npz"
    policy_path = RUN / "policy.pt"
    perception_metrics = train_perception(
        records,
        embeddings,
        set(split["train"]),
        RUN,
        "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    )
    policy_metrics = save_policy_checkpoint(policy_path, [r for r in records if r.slot_id in split["train"]], embeddings)
    report = {
        "data_root": str(DATA),
        "split": split,
        "perception_train": perception_metrics,
        "perception_validation": perception_eval(perception_path, records, embeddings, set(split["validation"])),
        "perception_test": perception_eval(perception_path, records, embeddings, set(split["test"])),
        "policy_train": policy_metrics,
        "policy_validation": policy_eval(policy_path, records, embeddings, set(split["validation"])),
        "policy_test": policy_eval(policy_path, records, embeddings, set(split["test"])),
    }
    report["qwen_embeddings_sha256"] = hashlib.sha256(EMBEDDING_PATH.read_bytes()).hexdigest()
    (RUN / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
