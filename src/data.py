"""Episode loading and manifest-driven teacher / vision contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CONTROL_DT_SEC = 0.5


@dataclass(frozen=True)
class EpisodeTeacher:
    follow_slot_index: int
    standoff_m: float
    embedding_key: str


@dataclass(frozen=True)
class EpisodeVisionSpec:
    slot_entity_ids: tuple[str, ...]

    @property
    def slot_count(self) -> int:
        return len(self.slot_entity_ids)


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
    teacher: EpisodeTeacher
    vision: EpisodeVisionSpec
    action: tuple[float, float] | None = None


def require_manifest_contracts(manifest: Mapping[str, Any]) -> tuple[EpisodeTeacher, EpisodeVisionSpec]:
    teacher_raw = manifest.get("teacher")
    vision_raw = manifest.get("vision")
    if not isinstance(teacher_raw, Mapping):
        raise ValueError("manifest missing teacher contract")
    if not isinstance(vision_raw, Mapping):
        raise ValueError("manifest missing vision contract")
    slot_entity_ids = tuple(str(value) for value in vision_raw.get("slot_entity_ids", ()))
    if not slot_entity_ids:
        raise ValueError("manifest.vision.slot_entity_ids must be non-empty")
    follow_slot_index = int(teacher_raw["follow_slot_index"])
    if follow_slot_index < 0 or follow_slot_index >= len(slot_entity_ids):
        raise ValueError(
            f"follow_slot_index {follow_slot_index} out of range for "
            f"{len(slot_entity_ids)} vision slots"
        )
    embedding_key = str(teacher_raw.get("embedding_key", "")).strip()
    if not embedding_key:
        raise ValueError("manifest.teacher.embedding_key must be non-empty")
    teacher = EpisodeTeacher(
        follow_slot_index=follow_slot_index,
        standoff_m=float(teacher_raw["standoff_m"]),
        embedding_key=embedding_key,
    )
    vision = EpisodeVisionSpec(slot_entity_ids=slot_entity_ids)
    return teacher, vision


def load_episode_records(root: str | Path) -> list[EpisodeRecord]:
    episode_root = Path(root)
    records: list[EpisodeRecord] = []
    for episode in sorted(path for path in episode_root.iterdir() if path.is_dir()):
        manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        teacher, vision = require_manifest_contracts(manifest)
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
                    teacher=teacher,
                    vision=vision,
                    action=(
                        tuple(float(value) for value in frame["action"]["desired_displacement_m"])
                        if frame.get("action", {}).get("valid")
                        else None
                    ),
                )
            )
    return records


def load_language_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "instruction_texts" in data:
            keys = [str(value).strip() for value in data["instruction_texts"]]
        else:
            keys = [str(value) for value in data["instruction_ids"]]
        values = np.asarray(data["embeddings"], dtype=np.float32)
    return {
        key: value / float(np.linalg.norm(value))
        for key, value in zip(keys, values)
    }


def language_for_record(
    record: EpisodeRecord,
    embeddings: Mapping[str, np.ndarray],
) -> np.ndarray:
    key = record.teacher.embedding_key
    if key not in embeddings and record.task_text.strip() in embeddings:
        key = record.task_text.strip()
    if key not in embeddings:
        raise KeyError(f"no embedding for key={key!r} task_text={record.task_text!r}")
    return np.asarray(embeddings[key], dtype=np.float32).copy()


def follow_entity_id(record: EpisodeRecord) -> str:
    return record.vision.slot_entity_ids[record.teacher.follow_slot_index]


def split_moving_target_slots(slots: Sequence[str]) -> dict[str, list[str]]:
    unique = sorted(set(slots))
    return {
        "train": [slot for slot in unique if "_TRAIN_" in slot],
        "validation": [slot for slot in unique if slot.endswith("_VALIDATION")],
        "test": [slot for slot in unique if slot.endswith("_TEST")],
    }


def control_window_actions(
    records: list[EpisodeRecord],
    *,
    control_dt_sec: float = CONTROL_DT_SEC,
) -> list[np.ndarray | None]:
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
                action += np.asarray(candidate.action, dtype=np.float32) * (
                    overlap_duration / interval_duration
                )
            if action is not None:
                norm = float(np.linalg.norm(action))
                if norm > 0.55:
                    action *= 0.55 / norm
                labels[original_index] = action
    return labels


RADIAL_GAIN = 0.45
FAR_ERROR_THRESHOLD_M = 1.0
FAR_RADIAL_GAIN = 0.85


def teacher_action(
    relative_xy: np.ndarray,
    standoff_m: float,
    surge_velocity_mps: float,
    yaw_rate_radps: float,
    relative_velocity_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Training-only standoff chase label (not executed as an online rule)."""
    xy = np.asarray(relative_xy, dtype=np.float32)
    distance = float(np.linalg.norm(xy))
    if distance <= 1.0e-6 or not np.all(np.isfinite(xy)):
        return np.zeros(2, dtype=np.float32)
    error = distance - float(standoff_m)
    gain = FAR_RADIAL_GAIN if error > FAR_ERROR_THRESHOLD_M else RADIAL_GAIN
    radial = error * gain * (xy / distance)
    current = np.asarray((float(surge_velocity_mps) * CONTROL_DT_SEC, 0.0), dtype=np.float32)
    action = radial + current
    if relative_velocity_xy is not None:
        velocity = np.asarray(relative_velocity_xy, dtype=np.float32).reshape(-1)
        if velocity.size >= 2 and np.all(np.isfinite(velocity[:2])):
            action = action + velocity[:2] * CONTROL_DT_SEC
    action[1] -= float(yaw_rate_radps) * 0.05
    norm = float(np.linalg.norm(action))
    return action * min(1.0, 0.55 / max(norm, 1.0e-8))
