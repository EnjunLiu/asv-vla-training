from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


TOPICS = ("/ue/camera_frame", "/ue/asv_state", "/ue/entities")


def _identity(message: Any) -> tuple[str, int, int]:
    return str(message.run_id), int(message.scene_seed), int(message.frame_index)


def teacher_contract(
    *,
    follow_slot_index: int,
    standoff_m: float,
    embedding_key: str,
) -> dict[str, Any]:
    key = str(embedding_key).strip()
    if not key:
        raise ValueError("embedding_key must be non-empty")
    if int(follow_slot_index) < 0:
        raise ValueError("follow_slot_index must be non-negative")
    return {
        "follow_slot_index": int(follow_slot_index),
        "standoff_m": float(standoff_m),
        "embedding_key": key,
    }


def vision_contract(
    entity_items: Sequence[Mapping[str, Any]],
    *,
    slot_entity_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if slot_entity_ids is not None:
        ids = [str(value).strip() for value in slot_entity_ids if str(value).strip()]
    else:
        ids = []
        seen: set[str] = set()
        for item in entity_items:
            entity_id = str(item.get("entity_id", "")).strip()
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            ids.append(entity_id)
    if not ids:
        raise ValueError("vision.slot_entity_ids must be non-empty")
    return {"slot_entity_ids": ids}


def _entity(item: Any, ego: Any) -> dict[str, Any]:
    """Passthrough UE entity fields; do not derive color/is_target in Python."""

    yaw = float(ego.yaw)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    relative_x = float(item.relative_x)
    relative_y = float(item.relative_y)
    world_x = float(ego.position_x) + cosine * relative_x - sine * relative_y
    world_y = float(ego.position_y) + sine * relative_x + cosine * relative_y
    payload = {
        "entity_id": str(item.entity_id),
        "visible": bool(item.visible),
        "relative_position_m": [relative_x, relative_y],
        "world_position_m": [world_x, world_y],
        "relative_velocity_mps": [
            float(item.relative_velocity_x),
            float(item.relative_velocity_y),
        ],
        "valid": bool(item.valid),
    }
    if hasattr(item, "color") and item.color is not None and str(item.color) != "":
        payload["color"] = str(item.color)
    if hasattr(item, "is_target"):
        payload["is_target"] = bool(item.is_target)
    return payload


def _body_displacement(current: Any, following: Any) -> tuple[float, float]:
    world_x = float(following.position_x) - float(current.position_x)
    world_y = float(following.position_y) - float(current.position_y)
    yaw = float(current.yaw)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return cosine * world_x + sine * world_y, -sine * world_x + cosine * world_y


def export_decoded_frames(
    decoded: Mapping[tuple[str, int, int], Mapping[str, Any]],
    output_episode: str | Path,
    task_text: str,
    slot_id: str,
    layout_id: str,
    motion_state: str,
    *,
    source: str = "decoded_rosbag",
    follow_slot_index: int = 0,
    standoff_m: float = 3.0,
    embedding_key: str = "",
    slot_entity_ids: Sequence[str] | None = None,
) -> int:
    complete: list[tuple[tuple[str, int, int], Mapping[str, Any]]] = []
    for key, topics in decoded.items():
        if not set(TOPICS) <= set(topics):
            continue
        for topic in TOPICS:
            if _identity(topics[topic]) != key:
                raise ValueError(f"topic identity mismatch at {key}: {topic}")
        complete.append((key, topics))
    if len(complete) < 2:
        raise ValueError("need at least two complete synchronized frames")
    run_identities = {(key[0], key[1]) for key, _ in complete}
    if len(run_identities) != 1:
        raise ValueError("episode must contain a single run identity")
    complete.sort(key=lambda pair: pair[0][2])
    frame_indexes = [key[2] for key, _ in complete]
    if any(right <= left for left, right in zip(frame_indexes, frame_indexes[1:])):
        raise ValueError("frame indexes must be strictly increasing")

    out = Path(output_episode)
    (out / "camera").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for (key, topics), (_, next_topics) in zip(complete, complete[1:]):
        camera = topics["/ue/camera_frame"]
        ego = topics["/ue/asv_state"]
        entities = topics["/ue/entities"]
        following_ego = next_topics["/ue/asv_state"]
        frame_index = int(key[2])
        image_name = f"camera/{frame_index:012d}.jpg"
        image_bytes = bytes(camera.data)
        (out / image_name).write_bytes(image_bytes)
        action_x, action_y = _body_displacement(ego, following_ego)
        step_dt = float(following_ego.simulation_time) - float(ego.simulation_time)
        valid = bool(camera.valid and ego.valid and entities.valid and step_dt > 0.0)
        if not valid:
            # UE5 can publish a second state/entity sample while the next
            # camera capture is still pending.  It has the same simulation
            # timestamp and therefore no executable action interval.  Do not
            # write such a row into a training episode.
            continue
        stamp_us = int(ego.stamp_us)
        record = {
            "schema_version": "frame_record_v1",
            "run_id": str(key[0]),
            "scene_seed": int(key[1]),
            "frame_index": frame_index,
            "stamp_us": stamp_us,
            "frame_id": "base_link",
            "task": {"stamp_us": stamp_us, "text": task_text, "valid": True},
            "ego": {
                "stamp_us": stamp_us,
                "world_frame_id": "ue_world",
                "simulation_time_s": float(ego.simulation_time),
                "position_m": [
                    float(ego.position_x),
                    float(ego.position_y),
                    float(ego.position_z),
                ],
                "rpy_ue_rad": [float(ego.roll), float(ego.pitch), float(ego.yaw)],
                "surge_velocity_mps": float(ego.surge_velocity),
                "yaw_rate_radps": float(ego.yaw_rate),
                "valid": bool(ego.valid),
            },
            "camera": {
                "stamp_us": int(camera.stamp_us),
                "image_path": image_name,
                "encoding": str(camera.encoding or "jpeg"),
                "width_px": int(getattr(camera, "width_px", 1280)),
                "height_px": int(getattr(camera, "height_px", 720)),
                "fov_angle_deg": 90.0,
                "mount_frame_id": "base_link",
                "mount_position_m": [0.42, 0.0, 0.2],
                "mount_rpy_ue_deg": [0.0, -5.0, 0.0],
                "valid": bool(camera.valid),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
            },
            "entities": {
                "stamp_us": int(entities.stamp_us),
                "frame_id": str(entities.frame_id or "base_link"),
                "items": [_entity(item, ego) for item in entities.entities],
                "valid": bool(entities.valid),
            },
            "action": {
                "desired_displacement_m": [action_x, action_y],
                "step_dt_s": step_dt,
                "source": "ue_expert_executed_pose_delta",
                "next_frame_index": int(following_ego.frame_index),
                "valid": valid,
            },
            "modality_mask": {
                "task": True,
                "ego": bool(ego.valid),
                "camera": bool(camera.valid),
                "entities": bool(entities.valid),
                "action": valid,
            },
            "valid": valid,
            "detail": "ok",
        }
        (out / "frames" / f"{frame_index:012d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(str(frame_index))

    run_id, scene_seed = next(iter(run_identities))
    first_items = []
    if written:
        first_frame = json.loads(
            (out / "frames" / f"{int(written[0]):012d}.json").read_text(encoding="utf-8")
        )
        first_items = first_frame.get("entities", {}).get("items", [])
    key = embedding_key.strip() or task_text.strip()
    manifest = {
        "schema_version": "episode_manifest_v1",
        "run_id": run_id,
        "scene_seed": scene_seed,
        "frame_count": len(written),
        "status": "complete",
        "execution_mode": "ue_expert_closed_loop",
        "collection": {
            "slot_id": slot_id,
            "layout_id": layout_id,
            "motion_state": motion_state,
        },
        "source": source,
        "task_text": task_text,
        "teacher": teacher_contract(
            follow_slot_index=follow_slot_index,
            standoff_m=standoff_m,
            embedding_key=key,
        ),
        "vision": vision_contract(first_items, slot_entity_ids=slot_entity_ids),
        "frames": written,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(written)


def read_rosbag(input_bag: str | Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    from rclpy.serialization import deserialize_message
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message

    storage = rosbag2_py.StorageOptions(uri=str(input_bag), storage_id="sqlite3")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    message_types = {topic: get_message(type_name) for topic, type_name in types.items()}
    decoded: dict[tuple[str, int, int], dict[str, Any]] = {}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic not in TOPICS:
            continue
        message = deserialize_message(raw, message_types[topic])
        decoded.setdefault(_identity(message), {})[topic] = message
    return decoded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--motion", default="S2")
    parser.add_argument("--follow-slot-index", type=int, default=0)
    parser.add_argument("--standoff-m", type=float, default=3.0)
    parser.add_argument("--embedding-key", default="")
    parser.add_argument("--slot-entity-ids", nargs="*", default=None)
    args = parser.parse_args()
    count = export_decoded_frames(
        read_rosbag(args.bag),
        args.output,
        args.task,
        args.slot,
        args.layout,
        args.motion,
        source=str(args.bag),
        follow_slot_index=args.follow_slot_index,
        standoff_m=args.standoff_m,
        embedding_key=args.embedding_key,
        slot_entity_ids=args.slot_entity_ids,
    )
    print(f"MOVING_TARGET_EPISODE_PASS frames={count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
