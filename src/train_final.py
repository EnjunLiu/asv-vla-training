from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge

from entity_embedding import EntityEmbeddingRuntime
from perception import CameraProfile

from decision import (
    ENTITY_EMBEDDING_DIM,
    ENTITY_FEATURE_DIM,
    ENTITY_COUNT,
    SmallActionPolicy,
    SmallPolicyConfig,
    build_entity_features,
)


class EntityObject:
    def __init__(self, value: Mapping[str, Any]) -> None:
        self.entity_id = str(value["entity_id"])
        self.class_name = str(value.get("class_name", ""))
        self.color = str(value.get("color", ""))
        self.is_target = bool(value.get("is_target", False))
        self.visible = bool(value["visible"])
        self.valid = bool(value["valid"])
        self.relative_x, self.relative_y, self.relative_z = map(
            float, value["relative_position_m"]
        )
        self.relative_velocity_x, self.relative_velocity_y, self.relative_velocity_z = map(
            float, value["relative_velocity_mps"]
        )
        self.velocity_valid = bool(value.get("velocity_valid", True))
        raw_embedding = value.get("entity_embedding")
        if raw_embedding is None:
            self.entity_embedding = [0.0] * ENTITY_EMBEDDING_DIM
        else:
            embedding = [float(item) for item in raw_embedding]
            if len(embedding) < ENTITY_EMBEDDING_DIM:
                embedding.extend([0.0] * (ENTITY_EMBEDDING_DIM - len(embedding)))
            self.entity_embedding = embedding[:ENTITY_EMBEDDING_DIM]


from perception import (
    FEATURE_DIM,
    LANGUAGE_EMBEDDING_DIM,
    OUTPUT_DIM,
    POSITION_SCALE_M,
    ImageEntityModel,
    MODEL_INPUT_CONTRACT,
    MODEL_SCHEMA_VERSION,
    MODEL_VERSION,
    STRUCTURED_ENTITY_OUTPUT_CONTRACT,
    extract_image_features,
    save_model,
)


FINAL_SCHEMA = "asv_policy_checkpoint"
CONTROL_DT_SEC = 0.5
RADIAL_GAIN = 0.35
FAR_ERROR_THRESHOLD_M = 1.0
FAR_RADIAL_GAIN = 0.75
LAG_TEACHER_ERROR_M = 0.40
CRUISE_SURGE_MPS = 0.60


@dataclass(frozen=True)
class EpisodeRecord:
    slot_id: str
    run_id: str
    scene_seed: int
    frame_index: int
    image_path: Path
    task_text: str
    ego: Mapping[str, Any]
    entities: tuple[Mapping[str, Any], ...]
    action: tuple[float, float] | None = None


def load_episode_records(root: str | Path) -> list[EpisodeRecord]:
    episode_root = Path(root)
    records: list[EpisodeRecord] = []
    for episode in sorted(path for path in episode_root.iterdir() if path.is_dir()):
        manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        slot_id = str(manifest["collection"]["slot_id"])
        for frame_path in sorted((episode / "frames").glob("*.json")):
            frame = json.loads(frame_path.read_text(encoding="utf-8"))
            if not frame.get("valid"):
                continue
            image_path = episode / str(frame["camera"]["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            records.append(
                EpisodeRecord(
                    slot_id=slot_id,
                    run_id=str(frame["run_id"]),
                    scene_seed=int(frame["scene_seed"]),
                    frame_index=int(frame["frame_index"]),
                    image_path=image_path,
                    task_text=str(frame["task"]["text"]),
                    ego=frame["ego"],
                    entities=tuple(frame["entities"]["items"]),
                    action=(
                        tuple(
                            float(value)
                            for value in frame["action"]["desired_displacement_m"]
                        )
                        if frame.get("action", {}).get("valid")
                        else None
                    ),
                )
            )
    return records


def load_language_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        ids = [str(value) for value in data["instruction_ids"]]
        values = np.asarray(data["embeddings"], dtype=np.float32)
    if values.shape != (len(ids), LANGUAGE_EMBEDDING_DIM):
        raise ValueError(f"invalid language table shape: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("language table contains NaN or Inf")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1.0e-8):
        raise ValueError("language table contains a zero vector")
    return {key: value / norm for key, value, norm in zip(ids, values, norms)}


def split_slots(slots: list[str]) -> dict[str, list[str]]:
    unique = sorted(set(slots))
    expected_test = {"L7_S2_R6", "L7B_S2_R6"}
    expected_validation = {"L7_S2_R5", "L7B_S2_R5"}
    return {
        "train": [slot for slot in unique if slot not in expected_test | expected_validation],
        "validation": sorted(expected_validation & set(unique)),
        "test": sorted(expected_test & set(unique)),
    }


def split_moving_target_slots(slots: list[str]) -> dict[str, list[str]]:
    unique = sorted(set(slots))
    split = {
        "train": [slot for slot in unique if "_TRAIN_" in slot],
        "validation": [slot for slot in unique if slot.endswith("_VALIDATION")],
        "test": [slot for slot in unique if slot.endswith("_TEST")],
    }
    assigned = set().union(*(set(values) for values in split.values()))
    if assigned != set(unique):
        raise ValueError(f"unrecognized moving-target slots: {sorted(set(unique) - assigned)}")
    if any(not values for values in split.values()):
        raise ValueError("moving-target split requires train, validation, and test slots")
    return split


def teacher_action(
    relative_xy: np.ndarray,
    standoff_m: float,
    surge_velocity_mps: float,
    yaw_rate_radps: float,
) -> np.ndarray:
    xy = np.asarray(relative_xy, dtype=np.float32)
    distance = float(np.linalg.norm(xy))
    if distance <= 1.0e-6 or not np.all(np.isfinite(xy)):
        return np.zeros(2, dtype=np.float32)
    error = distance - float(standoff_m)
    gain = FAR_RADIAL_GAIN if error > FAR_ERROR_THRESHOLD_M else RADIAL_GAIN
    radial = error * gain * (xy / distance)
    current = np.asarray((float(surge_velocity_mps) * CONTROL_DT_SEC, 0.0), dtype=np.float32)
    action = radial + current
    action[1] -= float(yaw_rate_radps) * 0.05
    norm = float(np.linalg.norm(action))
    return action * min(1.0, 0.50 / max(norm, 1.0e-8))


def control_window_actions(
    records: list[EpisodeRecord],
    *,
    control_dt_sec: float = CONTROL_DT_SEC,
) -> list[np.ndarray | None]:
    """Integrate recorded frame actions over one policy-command control window.

    Collected actions are applied once per camera frame (0.2 s in the moving
    target data), while the online policy publishes one displacement per 0.5 s
    control window. Each frame action is weighted by the fraction of its
    timestamp interval that overlaps the policy window. Incomplete terminal
    windows are omitted instead of teaching a shorter command, and the result
    keeps the existing 0.50 m single-command displacement bound.
    """

    labels: list[np.ndarray | None] = [None] * len(records)
    groups: dict[tuple[str, str, int], list[tuple[int, EpisodeRecord]]] = {}
    for index, record in enumerate(records):
        groups.setdefault((record.slot_id, record.run_id, record.scene_seed), []).append(
            (index, record)
        )
    for group in groups.values():
        group.sort(key=lambda item: (int(item[1].ego["stamp_us"]), item[1].frame_index))
        stamps = np.asarray([float(record.ego["stamp_us"]) / 1.0e6 for _, record in group])
        for local_index, (original_index, record) in enumerate(group):
            if record.action is None:
                continue
            end_time = stamps[local_index] + float(control_dt_sec)
            if stamps[-1] < end_time - 1.0e-6:
                continue
            action = np.zeros(2, dtype=np.float32)
            for candidate_index in range(local_index, len(group) - 1):
                interval_start = stamps[candidate_index]
                if interval_start >= end_time - 1.0e-6:
                    break
                interval_end = stamps[candidate_index + 1]
                interval_duration = interval_end - interval_start
                if interval_duration <= 0.0:
                    action = None
                    break
                _, candidate = group[candidate_index]
                if candidate.action is None:
                    action = None
                    break
                overlap_duration = min(interval_end, end_time) - interval_start
                overlap_fraction = overlap_duration / interval_duration
                action += np.asarray(candidate.action, dtype=np.float32) * overlap_fraction
            if action is not None:
                norm = float(np.linalg.norm(action))
                if norm > 0.50:
                    action *= 0.50 / norm
                labels[original_index] = action
    return labels


def task_key(text: str) -> str:
    folded = text.casefold()
    color = "blue" if "blue" in folded or "蓝" in text else "red"
    distance = "4m" if "4m" in folded or "4 m" in folded or "4米" in text else "3m"
    return f"{color}_{distance}"


def stamp_language_standoff(embedding: np.ndarray, standoff_m: float) -> np.ndarray:
    """Overwrite the last language dim so 3 m and 4 m are linearly separable.

    Qwen embeddings for the English 3 m/4 m prompts are nearly identical
    (cosine ~0.996), so the policy cannot learn two standoffs from language
    unless this explicit token is applied on both train and Jetson.
    """

    stamped = np.asarray(embedding, dtype=np.float32).copy()
    if stamped.ndim != 1 or stamped.size < 1:
        raise ValueError("language embedding must be a 1-D vector")
    stamped[-1] = (float(standoff_m) - 3.5) / 0.5
    return stamped


def _perception_rows(
    records: list[EpisodeRecord],
    embeddings: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for record in records:
        embedding = embeddings[task_key(record.task_text)]
        features = np.concatenate((extract_image_features(Image.open(record.image_path)), embedding))
        target = np.zeros(OUTPUT_DIM, dtype=np.float32)
        by_id = {str(item["entity_id"]): item for item in record.entities}
        for index, entity_id in enumerate(("target_red", "target_blue", "target_left", "target_right")):
            item = by_id[entity_id]
            offset = index * 4
            target[offset] = 3.0 if item["visible"] else -3.0
            position = np.asarray(item["relative_position_m"], dtype=np.float32)
            target[offset + 1:offset + 4] = position / POSITION_SCALE_M
        rows.append(features.astype(np.float32))
        labels.append(target)
    return np.stack(rows), np.stack(labels)


def train_perception(
    records: list[EpisodeRecord],
    embeddings: Mapping[str, np.ndarray],
    train_slots: set[str],
    output_dir: Path,
    language_weights_sha256: str,
) -> dict[str, float]:
    train_records = [record for record in records if record.slot_id in train_slots]
    all_x, all_y = _perception_rows(train_records, embeddings)
    mean = all_x.mean(axis=0)
    scale = all_x.std(axis=0)
    scale[scale < 1.0e-5] = 1.0
    normalized = (all_x - mean) / scale
    ridge = 1.0e-2
    estimator = Ridge(alpha=ridge, fit_intercept=True, solver="lsqr")
    estimator.fit(normalized, all_y)
    weights = np.asarray(estimator.coef_.T, dtype=np.float32)
    bias = np.asarray(estimator.intercept_, dtype=np.float32)
    color_weights = np.zeros((2, normalized.shape[1], 4), dtype=np.float32)
    color_bias = np.zeros((2, 4), dtype=np.float32)
    for head_index, color in enumerate(("red", "blue")):
        selected = [
            index for index, record in enumerate(train_records)
            if task_key(record.task_text).startswith(color)
        ]
        head = Ridge(alpha=100.0, fit_intercept=True, solver="lsqr").fit(
            normalized[selected], all_y[selected, head_index * 4 : head_index * 4 + 4]
        )
        color_weights[head_index] = np.asarray(head.coef_.T, dtype=np.float32)
        color_bias[head_index] = np.asarray(head.intercept_, dtype=np.float32)
    language_hash = str(language_weights_sha256)
    model_path = output_dir / "perception.npz"
    save_model(
        model_path,
        feature_mean=mean,
        feature_scale=scale,
        weights=weights,
        bias=bias,
        model_version=MODEL_VERSION,
        schema_version=MODEL_SCHEMA_VERSION,
        input_contract=MODEL_INPUT_CONTRACT,
        output_contract=STRUCTURED_ENTITY_OUTPUT_CONTRACT,
        task_embedding_dim=LANGUAGE_EMBEDDING_DIM,
        language_model_id="Qwen/Qwen3-Embedding-0.6B",
        language_weights_sha256=language_hash,
        metadata={"schema_version": MODEL_SCHEMA_VERSION, "language_hash": language_hash},
        color_head_weights=color_weights,
        color_head_bias=color_bias,
    )
    prediction = normalized @ weights + bias
    geometry_columns = [index for index in range(OUTPUT_DIM) if index % 4]
    geometry_scale = np.tile(POSITION_SCALE_M, 4)
    error_m = (
        prediction[:, geometry_columns] - all_y[:, geometry_columns]
    ) * geometry_scale
    return {"train_rmse_m": float(np.sqrt(np.mean(error_m ** 2)))}


def save_policy_checkpoint(
    output_path: str | Path,
    train_records: list[EpisodeRecord],
    embeddings: Mapping[str, np.ndarray],
    *,
    epochs: int = 80,
    entity_embedding_runtime: EntityEmbeddingRuntime | None = None,
    camera_profile: CameraProfile | None = None,
) -> dict[str, float]:
    dataset = build_policy_dataset(
        train_records,
        embeddings,
        distance_scales=(1.0, 1.25, 1.5, 2.0),
        entity_embedding_runtime=entity_embedding_runtime,
        camera_profile=camera_profile,
    )
    torch.manual_seed(20260819)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SmallPolicyConfig()
    model = SmallActionPolicy(config).to(device)
    with torch.no_grad():
        model.stop_head.weight.zero_()
        model.stop_head.bias.fill_(-5.0)
    model.stop_head.weight.requires_grad_(False)
    model.stop_head.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    language = torch.from_numpy(dataset["language"])
    geometry = torch.from_numpy(dataset["entity_geometry"])
    ego = torch.from_numpy(dataset["ego_state"])
    action = torch.from_numpy(dataset["action"])
    masks = torch.from_numpy(dataset["entity_geometry_mask"])
    batch_size = min(4096, len(language))
    model.train()
    generator = torch.Generator().manual_seed(20260819)
    for _ in range(epochs):
        for indices in torch.randperm(len(language), generator=generator).split(batch_size):
            batch_language = language[indices].to(device)
            batch_geometry = geometry[indices].to(device)
            batch_ego = ego[indices].to(device)
            batch_action = action[indices].to(device)
            batch_masks = masks[indices].to(device)
            valid = torch.ones(len(indices), dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                language=batch_language,
                entity_geometry=batch_geometry,
                ego_state=batch_ego,
                language_valid=valid,
                entity_geometry_mask=batch_masks,
                ego_state_valid=valid,
                policy_input_valid=valid,
            )
            loss = torch.nn.functional.mse_loss(output.action, batch_action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        predictions = []
        for indices in torch.arange(len(language)).split(1024):
            valid = torch.ones(len(indices), dtype=torch.bool, device=device)
            predictions.append(model(
                language=language[indices].to(device),
                entity_geometry=geometry[indices].to(device),
                ego_state=ego[indices].to(device),
                language_valid=valid,
                entity_geometry_mask=masks[indices].to(device),
                ego_state_valid=valid,
                policy_input_valid=valid,
            ).action.cpu())
    prediction = torch.cat(predictions).numpy()
    rmse = float(np.sqrt(np.mean((prediction - dataset["action"]) ** 2)))
    checkpoint = build_policy_checkpoint(model)
    checkpoint["training"] = {"rows": int(language.shape[0]), "epochs": epochs}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return {"train_action_rmse_m": rmse, "device": device.type}


def _entities_with_embeddings(
    record: EpisodeRecord,
    *,
    entity_embedding_runtime: EntityEmbeddingRuntime | None,
    camera_profile: CameraProfile,
) -> list[dict[str, Any]]:
    items = [dict(item) for item in record.entities]
    if entity_embedding_runtime is None:
        return items
    encoded = entity_embedding_runtime.encode_entities(
        Image.open(record.image_path).convert("RGB"),
        items,
        camera_profile,
    )
    for item in items:
        entity_id = str(item["entity_id"])
        item["entity_embedding"] = encoded.get(
            entity_id, np.zeros(ENTITY_EMBEDDING_DIM, dtype=np.float32)
        ).tolist()
    return items


def build_policy_dataset(
    records: list[EpisodeRecord],
    embeddings: Mapping[str, np.ndarray],
    *,
    distance_scales: tuple[float, ...] = (1.0,),
    entity_embedding_runtime: EntityEmbeddingRuntime | None = None,
    camera_profile: CameraProfile | None = None,
) -> dict[str, np.ndarray]:
    language_rows: list[np.ndarray] = []
    geometry_rows: list[np.ndarray] = []
    geometry_masks: list[np.ndarray] = []
    ego_rows: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    slot_ids: list[str] = []
    task_ids: list[str] = []
    window_actions = control_window_actions(records)
    profile = camera_profile or CameraProfile()
    for record, window_action in zip(records, window_actions):
        if record.action is None:
            raise ValueError(
                f"record {record.slot_id}/{record.frame_index} has no recorded action"
            )
        if window_action is None:
            continue
        if not distance_scales or any(scale <= 0.0 for scale in distance_scales):
            raise ValueError("distance scales must be positive")
        base_items = _entities_with_embeddings(
            record,
            entity_embedding_runtime=entity_embedding_runtime,
            camera_profile=profile,
        )
        for scale in distance_scales:
            scaled_items = [dict(item) for item in base_items]
            for item in scaled_items:
                position = np.asarray(item["relative_position_m"], dtype=np.float32)
                item["relative_position_m"] = (position * float(scale)).tolist()
            entities = [EntityObject(item) for item in scaled_items]
            features = build_entity_features(entities)
            surge = float(record.ego.get("surge_velocity_mps", 0.0))
            yaw_rate = float(record.ego.get("yaw_rate_radps", 0.0))
            by_id = {str(item["entity_id"]): item for item in scaled_items}
            key = task_key(record.task_text)
            color = "blue" if key.startswith("blue") else "red"
            target_id = f"target_{color}"
            if target_id not in by_id:
                raise ValueError(f"missing selected target for task {key}")
            relative_xy = np.asarray(
                by_id[target_id]["relative_position_m"][:2], dtype=np.float32
            )
            standoff_m = 4.0 if key.endswith("4m") else 3.0
            distance_error = float(np.linalg.norm(relative_xy)) - float(standoff_m)
            if float(scale) != 1.0 or distance_error > FAR_ERROR_THRESHOLD_M:
                # Cold-start / augmented far view: chase at full 0.50 m without
                # requiring the boat to already have surge.
                target_action = teacher_action(relative_xy, standoff_m, 0.0, 0.0)
                ego = np.zeros(2, dtype=np.float32)
            else:
                # Keep expert window labels inside the near-standoff band. The
                # mid-range teacher rewrite regressed RED/BLUE 3 m online while
                # language stamping alone fixed RED 4 m.
                target_action = np.asarray(window_action, dtype=np.float32)
                ego = np.asarray((surge / 5.0, yaw_rate), dtype=np.float32)
            language_rows.append(stamp_language_standoff(embeddings[key], standoff_m))
            geometry_rows.append(features.features.astype(np.float32))
            geometry_masks.append(features.mask.astype(bool))
            ego_rows.append(ego)
            actions.append(target_action)
            slot_ids.append(record.slot_id)
            task_ids.append(key)

            # Online perception may publish only the selected target. Keep a
            # second view of the same labeled state so the policy learns the
            # target-only mask without changing geometry or action semantics.
            target_index = next(
                (
                    index
                    for index, entity_id in enumerate(features.entity_ids)
                    if entity_id == f"target_{color}"
                ),
                None,
            )
            if target_index is None:
                raise ValueError(f"target feature missing for task {key}")
            target_only_mask = np.zeros_like(features.mask, dtype=bool)
            target_only_mask[target_index] = True
            geometry_masks.append(target_only_mask)
            language_rows.append(stamp_language_standoff(embeddings[key], standoff_m))
            geometry_rows.append(features.features.astype(np.float32))
            ego_rows.append(ego)
            actions.append(target_action.copy())
            slot_ids.append(record.slot_id)
            task_ids.append(key)

            # Keep expert labels above, but also teach a cruise-floor chase for
            # the common 0.4-1.0 m lag band that froze online at ~4.2 m.
            if (
                float(scale) == 1.0
                and LAG_TEACHER_ERROR_M < distance_error <= FAR_ERROR_THRESHOLD_M
            ):
                chase_surge = max(float(surge), CRUISE_SURGE_MPS)
                chase_action = teacher_action(
                    relative_xy, standoff_m, chase_surge, yaw_rate
                )
                chase_ego = np.asarray(
                    (chase_surge / 5.0, yaw_rate), dtype=np.float32
                )
                for mask in (features.mask.astype(bool), target_only_mask):
                    language_rows.append(
                        stamp_language_standoff(embeddings[key], standoff_m)
                    )
                    geometry_rows.append(features.features.astype(np.float32))
                    geometry_masks.append(mask)
                    ego_rows.append(chase_ego.copy())
                    actions.append(chase_action.copy())
                    slot_ids.append(record.slot_id)
                    task_ids.append(key)
    return {
        "language": np.stack(language_rows),
        "entity_geometry": np.stack(geometry_rows),
        "entity_geometry_mask": np.stack(geometry_masks),
        "ego_state": np.stack(ego_rows),
        "action": np.stack(actions),
        "slot_id": np.asarray(slot_ids),
        "task_id": np.asarray(task_ids),
    }


def build_policy_checkpoint(
    model: SmallActionPolicy | None = None,
) -> dict[str, Any]:
    policy = model or SmallActionPolicy(SmallPolicyConfig())
    return {
        "schema_version": FINAL_SCHEMA,
        "model_config": asdict(policy.config),
        "model_state_dict": policy.state_dict(),
    }


def validate_policy_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("schema_version") != FINAL_SCHEMA:
        raise ValueError("policy checkpoint schema mismatch")
    config = SmallPolicyConfig.from_mapping(checkpoint["model_config"])
    model = SmallActionPolicy(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
