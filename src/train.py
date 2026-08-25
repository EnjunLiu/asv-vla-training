#!/usr/bin/env python3
"""Windows-only training entry: vision + policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data import (
    CONTROL_DT_SEC,
    EpisodeRecord,
    control_window_actions,
    follow_entity_id,
    language_for_record,
    load_episode_records,
    load_language_embeddings,
    split_moving_target_slots,
    teacher_action,
)
from perception import (
    ENTITY_EMBEDDING_DIM,
    POSITION_SCALE_M,
    VisionModel,
    VisionRuntime,
    build_entity_cache,
    _prepare_image,
)
from decision import (
    ENTITY_FEATURE_DIM,
    ActionPolicy,
    PolicyConfig,
    build_entity_states,
)

DATA_ROOT = ROOT / "data/episodes/moving_target_valid"
EMBEDDING_PATH = ROOT / "data/qwen_final_embeddings.npz"
RUN_DIR = ROOT / "experiments/chase_standoff_tight_1m"
# Emphasize lateral corrections buried under ~0.3 m cruise surge.
ACTION_LOSS_WEIGHTS = (1.0, 60.0)
HARD_SAMPLE_COPIES = 8
HARD_SAMPLE_COPIES_4M = 8
HARD_LATERAL_COPIES = 8


class EntityObject:
    def __init__(self, value: dict) -> None:
        self.entity_id = str(value["entity_id"])
        self.visible = bool(value.get("visible", value.get("valid", False)))
        self.valid = bool(value.get("valid", self.visible))
        if "relative_position_m" in value:
            pos = value["relative_position_m"]
            self.relative_x = float(pos[0])
            self.relative_y = float(pos[1])
        else:
            self.relative_x = float(value["relative_x"])
            self.relative_y = float(value["relative_y"])
        if "relative_velocity_mps" in value:
            vel = value["relative_velocity_mps"]
            self.relative_velocity_x = float(vel[0])
            self.relative_velocity_y = float(vel[1])
        else:
            self.relative_velocity_x = float(value.get("relative_velocity_x", 0.0))
            self.relative_velocity_y = float(value.get("relative_velocity_y", 0.0))
        self.velocity_valid = bool(value.get("velocity_valid", False))
        raw = value.get("entity_embedding", [0.0] * ENTITY_EMBEDDING_DIM)
        self.entity_embedding = [float(v) for v in raw][:ENTITY_EMBEDDING_DIM]


def _slot_targets(record: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
    slot_count = record.vision.slot_count
    visible = np.zeros(slot_count, dtype=np.float32)
    geometry = np.zeros((slot_count, 2), dtype=np.float32)
    by_id = {str(item["entity_id"]): item for item in record.entities}
    for index, entity_id in enumerate(record.vision.slot_entity_ids):
        item = by_id.get(entity_id)
        if item is None or not item["visible"]:
            continue
        visible[index] = 1.0
        pos = np.asarray(item["relative_position_m"], dtype=np.float32)[:2]
        geometry[index] = pos / POSITION_SCALE_M
    return visible, geometry


def train_vision(
    records: list[EpisodeRecord],
    embeddings: dict[str, np.ndarray],
    train_slots: set[str],
    *,
    device: str,
    epochs: int,
    output: Path,
) -> dict[str, float]:
    train_records = [record for record in records if record.slot_id in train_slots]
    if not train_records:
        raise ValueError("no train records for vision")
    slot_count = train_records[0].vision.slot_count
    device_obj = torch.device(device)
    model = VisionModel(slot_count=slot_count).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(20260823)
    indices = torch.arange(len(train_records))
    last_loss = 0.0
    for _ in range(epochs):
        for start in range(0, len(indices), 16):
            batch_records = [train_records[int(i)] for i in indices[start : start + 16]]
            images = []
            languages = []
            visible_targets = []
            geometry_targets = []
            for record in batch_records:
                image = Image.open(record.image_path).convert("RGB")
                images.append(_prepare_image(image, device_obj)[0])
                languages.append(
                    torch.as_tensor(
                        language_for_record(record, embeddings),
                        dtype=torch.float32,
                        device=device_obj,
                    )
                )
                visible, geometry = _slot_targets(record)
                visible_targets.append(torch.as_tensor(visible, device=device_obj))
                geometry_targets.append(torch.as_tensor(geometry, device=device_obj))
            image_batch = torch.stack(images)
            language_batch = torch.stack(languages)
            visible_batch = torch.stack(visible_targets)
            geometry_batch = torch.stack(geometry_targets)
            raw = model(image_batch, language_batch)
            visible_logits = raw[..., 0]
            geometry_pred = raw[..., 1:3]
            loss_visible = bce(visible_logits, visible_batch)
            mask = visible_batch.unsqueeze(-1)
            loss_geometry = mse(geometry_pred * mask, geometry_batch * mask)
            loss = loss_visible + loss_geometry
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        indices = indices[torch.randperm(len(indices), generator=generator)]
    runtime = VisionRuntime(model=model, device=device_obj)
    runtime.save(output)
    return {
        "train_loss": last_loss,
        "epochs": float(epochs),
        "samples": float(len(train_records)),
        "slot_count": float(slot_count),
    }


def _gt_relative_xy(record: EpisodeRecord, follow_id: str) -> np.ndarray | None:
    by_id = {str(item["entity_id"]): item for item in record.entities}
    target = by_id.get(follow_id)
    if target is None or not bool(target.get("visible", target.get("valid", False))):
        return None
    if "relative_position_m" in target:
        rel = np.asarray(target["relative_position_m"][:2], dtype=np.float32)
    else:
        rel = np.asarray(
            [float(target["relative_x"]), float(target["relative_y"])],
            dtype=np.float32,
        )
    if not np.all(np.isfinite(rel)):
        return None
    return rel


def _gt_relative_velocities(records: list[EpisodeRecord]) -> list[np.ndarray | None]:
    velocities: list[np.ndarray | None] = [None] * len(records)
    groups: dict[tuple[str, str, int], list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault((record.slot_id, record.run_id, record.scene_seed), []).append(index)
    for indices in groups.values():
        indices.sort(key=lambda i: (int(records[i].ego["stamp_us"]), records[i].frame_index))
        for current_index, next_index in zip(indices, indices[1:]):
            follow_id = follow_entity_id(records[current_index])
            current_xy = _gt_relative_xy(records[current_index], follow_id)
            next_xy = _gt_relative_xy(records[next_index], follow_id)
            dt = (
                float(records[next_index].ego["stamp_us"])
                - float(records[current_index].ego["stamp_us"])
            ) / 1.0e6
            if current_xy is None or next_xy is None or dt <= 1.0e-3:
                continue
            velocities[current_index] = (next_xy - current_xy) / dt
    return velocities


def build_policy_dataset(
    records: list[EpisodeRecord],
    embeddings: dict[str, np.ndarray],
    entity_cache: dict[tuple[str, str, int, int], list[dict]],
    *,
    max_entities: int,
) -> dict[str, np.ndarray]:
    language_rows: list[np.ndarray] = []
    geometry_rows: list[np.ndarray] = []
    geometry_masks: list[np.ndarray] = []
    asv_rows: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    window_actions = control_window_actions(records)
    relative_velocities = _gt_relative_velocities(records)
    for record, window_action, relative_velocity in zip(
        records, window_actions, relative_velocities
    ):
        if window_action is None:
            continue
        key = (
            record.slot_id,
            record.run_id,
            record.scene_seed,
            record.frame_index,
        )
        items = entity_cache.get(key, [])
        standoff = float(record.teacher.standoff_m)
        follow_id = follow_entity_id(record)
        follow_index = int(record.teacher.follow_slot_index)
        by_id = {str(item["entity_id"]): item for item in items}
        if follow_id not in by_id and f"slot_{follow_index}" not in by_id:
            continue
        features = build_entity_states(
            [EntityObject(item) for item in items],
            max_entities=max_entities,
        )
        asv = np.asarray(
            [
                float(record.ego.get("surge_velocity_mps", 0.0)) / 5.0,
                float(record.ego.get("yaw_rate_radps", 0.0)),
            ],
            dtype=np.float32,
        )
        gt_by_id = {str(item["entity_id"]): item for item in record.entities}
        gt_target = gt_by_id.get(follow_id)
        if gt_target is None or not bool(gt_target.get("visible", gt_target.get("valid", False))):
            continue
        if "relative_position_m" in gt_target:
            rel = np.asarray(gt_target["relative_position_m"][:2], dtype=np.float32)
        else:
            rel = np.asarray(
                [float(gt_target["relative_x"]), float(gt_target["relative_y"])],
                dtype=np.float32,
            )
        action = teacher_action(
            rel,
            standoff,
            float(record.ego.get("surge_velocity_mps", 0.0)),
            float(record.ego.get("yaw_rate_radps", 0.0)),
            relative_velocity,
        )
        dist_err = float(np.linalg.norm(rel) - standoff)
        copies = 1
        if abs(float(action[1])) >= 0.01 or abs(dist_err) >= 0.75:
            copies = HARD_SAMPLE_COPIES
        if abs(float(action[1])) >= 0.02:
            copies = max(copies, HARD_LATERAL_COPIES)
        if standoff >= 3.5:
            copies = max(copies, HARD_SAMPLE_COPIES_4M)
            if abs(float(action[1])) >= 0.01 or abs(dist_err) >= 0.5:
                copies = max(copies, HARD_SAMPLE_COPIES_4M * 2)
        for _ in range(copies):
            language_rows.append(language_for_record(record, embeddings))
            geometry_rows.append(features.features.astype(np.float32))
            geometry_masks.append(features.mask.astype(bool))
            asv_rows.append(asv.copy())
            actions.append(action.copy())
        # Prefer follow_slot_index in vision order; fall back to online feature id.
        target_index = None
        slot_id = f"slot_{follow_index}"
        for index, entity_id in enumerate(features.entity_ids):
            if entity_id == slot_id or entity_id == follow_id:
                target_index = index
                break
        if target_index is not None:
            target_mask = np.zeros_like(features.mask, dtype=bool)
            target_mask[target_index] = True
            for _ in range(copies):
                language_rows.append(language_for_record(record, embeddings))
                geometry_rows.append(features.features.astype(np.float32))
                geometry_masks.append(target_mask)
                asv_rows.append(asv.copy())
                actions.append(action.copy())
    return {
        "task_embedding": np.stack(language_rows),
        "entity_states": np.stack(geometry_rows),
        "entity_states_mask": np.stack(geometry_masks),
        "asv_state": np.stack(asv_rows),
        "action": np.stack(actions),
    }


def train_policy(
    records: list[EpisodeRecord],
    embeddings: dict[str, np.ndarray],
    train_slots: set[str],
    vision_path: Path,
    *,
    device: str,
    epochs: int,
    output: Path,
) -> dict[str, float]:
    train_records = [record for record in records if record.slot_id in train_slots]
    runtime = VisionRuntime.load(vision_path, device=device)
    entity_cache = build_entity_cache(train_records, runtime, embeddings)
    config = PolicyConfig()
    max_entities = int(config.entity_count)
    dataset = build_policy_dataset(
        train_records,
        embeddings,
        entity_cache,
        max_entities=max_entities,
    )
    device_obj = torch.device(device)
    model = ActionPolicy(config).to(device_obj)
    with torch.no_grad():
        model.stop_head.weight.zero_()
        model.stop_head.bias.fill_(-5.0)
    model.stop_head.weight.requires_grad_(False)
    model.stop_head.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=1.0e-4)
    task_embedding = torch.from_numpy(dataset["task_embedding"])
    entity_states = torch.from_numpy(dataset["entity_states"])
    asv = torch.from_numpy(dataset["asv_state"])
    action = torch.from_numpy(dataset["action"])
    masks = torch.from_numpy(dataset["entity_states_mask"])
    batch_size = min(4096, len(task_embedding))
    model.train()
    generator = torch.Generator().manual_seed(20260819)
    for _ in range(epochs):
        for indices in torch.randperm(len(task_embedding), generator=generator).split(batch_size):
            valid = torch.ones(len(indices), dtype=torch.bool, device=device_obj)
            output_batch = model(
                task_embedding=task_embedding[indices].to(device_obj),
                entity_states=entity_states[indices].to(device_obj),
                asv_state=asv[indices].to(device_obj),
                task_embedding_valid=valid,
                entity_states_mask=masks[indices].to(device_obj),
                asv_state_valid=valid,
                policy_input_valid=valid,
            )
            pred = output_batch.action
            target = action[indices].to(device_obj)
            weights = torch.tensor(ACTION_LOSS_WEIGHTS, dtype=torch.float32, device=device_obj)
            loss = ((pred - target) ** 2 * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        preds = []
        for indices in torch.arange(len(task_embedding)).split(1024):
            valid = torch.ones(len(indices), dtype=torch.bool, device=device_obj)
            preds.append(
                model(
                    task_embedding=task_embedding[indices].to(device_obj),
                    entity_states=entity_states[indices].to(device_obj),
                    asv_state=asv[indices].to(device_obj),
                    task_embedding_valid=valid,
                    entity_states_mask=masks[indices].to(device_obj),
                    asv_state_valid=valid,
                    policy_input_valid=valid,
                ).action.cpu()
            )
        prediction = torch.cat(preds).numpy()
    rmse = float(np.sqrt(np.mean((prediction - dataset["action"]) ** 2)))
    rmse_y = float(np.sqrt(np.mean((prediction[:, 1] - dataset["action"][:, 1]) ** 2)))
    checkpoint = {
        "schema_version": "asv_policy_checkpoint",
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "training": {
            "rows": int(task_embedding.shape[0]),
            "epochs": epochs,
            "action_loss_weights": list(ACTION_LOSS_WEIGHTS),
            "hard_sample_copies": HARD_SAMPLE_COPIES,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "train_action_rmse_m": rmse,
        "train_action_y_rmse_m": rmse_y,
        "rows": float(task_embedding.shape[0]),
        "max_entities": float(max_entities),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train on Windows PC only.")
    parser.add_argument("command", choices=["vision", "policy", "all"])
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDING_PATH)
    parser.add_argument("--out", type=Path, default=RUN_DIR)
    parser.add_argument("--vision", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vision-epochs", type=int, default=20)
    parser.add_argument("--policy-epochs", type=int, default=400)
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.device == "cuda":
        raise SystemExit("CUDA required for Windows training")
    records = load_episode_records(args.data)
    embeddings = load_language_embeddings(args.embeddings)
    split = split_moving_target_slots([record.slot_id for record in records])
    train_slots = set(split["train"])
    args.out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {"split": split, "control_dt_sec": CONTROL_DT_SEC}
    vision_path = args.out / "vision.pt"
    policy_path = args.out / "policy.pt"
    if args.command in {"vision", "all"}:
        metrics["vision"] = train_vision(
            records,
            embeddings,
            train_slots,
            device=args.device,
            epochs=args.vision_epochs,
            output=vision_path,
        )
    if args.command in {"policy", "all"}:
        vision_source = args.vision or vision_path
        metrics["policy"] = train_policy(
            records,
            embeddings,
            train_slots,
            vision_source,
            device=args.device,
            epochs=args.policy_epochs,
            output=policy_path,
        )
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
