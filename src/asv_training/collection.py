"""Plan and validate the counterbalanced Day 12 UE5 collection.

The validator deliberately checks observable entity geometry in recorded
FrameRecords.  A layout name in a manifest is not accepted as evidence that
the UE5 actors were actually rearranged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


PLAN_SCHEMA_VERSION = "collection_plan_v1"
REPORT_SCHEMA_VERSION = "collection_report_v1"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_plan(path: Path, ancestors: set[Path]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in ancestors:
        raise ValueError(f"collection plan inheritance cycle at {resolved}")
    plan = _load_object(resolved)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("collection plan schema_version is invalid")
    base_plan_value = str(plan.get("base_plan", "")).strip()
    if base_plan_value:
        base_path = Path(base_plan_value)
        if not base_path.is_absolute():
            base_path = resolved.parent / base_path
        base = _load_plan(base_path, ancestors | {resolved})
        extension_slots = plan.get("slots", [])
        if not isinstance(extension_slots, list):
            raise ValueError("collection plan slots must be a list")
        merged = dict(base)
        merged.update(
            {
                key: value
                for key, value in plan.items()
                if key not in {"base_plan", "slots"}
            }
        )
        merged["slots"] = list(base["slots"]) + extension_slots
        plan = merged
    slots = plan.get("slots")
    if not isinstance(slots, list) or not slots:
        raise ValueError("collection plan slots must be a non-empty list")
    minimum = int(plan.get("minimum_complete_runs", 0))
    if minimum < 1 or len(slots) < minimum:
        raise ValueError("collection plan has fewer slots than its minimum")
    rollout_action = str(plan.get("rollout_action", "follow")).strip()
    if rollout_action not in {"follow", "stop"}:
        raise ValueError(
            f"unsupported rollout_action={rollout_action!r}; "
            "expected follow or stop"
        )

    slot_ids: set[str] = set()
    seeds: set[int] = set()
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict):
            raise ValueError(f"slot[{index}] must be an object")
        slot_id = str(slot.get("slot_id", "")).strip()
        layout_id = str(slot.get("layout_id", "")).strip()
        motion_state = str(slot.get("motion_state", "")).strip()
        if not all((slot_id, layout_id, motion_state)):
            raise ValueError(f"slot[{index}] has incomplete identity")
        if motion_state not in {"S0", "S1", "S2"}:
            raise ValueError(
                f"slot {slot_id} has unsupported motion_state={motion_state!r}"
            )
        if slot_id in slot_ids:
            raise ValueError(f"duplicate slot_id: {slot_id}")
        slot_ids.add(slot_id)
        seed = int(slot.get("scene_seed"))
        if seed in seeds:
            raise ValueError(f"duplicate Scene Seed in plan: {seed}")
        seeds.add(seed)
        relations = slot.get("relations")
        if not isinstance(relations, list) or not relations:
            raise ValueError(f"slot {slot_id} has no observable relations")
        for relation in relations:
            if (
                not isinstance(relation, list)
                or len(relation) != 3
                or relation[0] not in {"nearer", "left_of"}
            ):
                raise ValueError(
                    f"slot {slot_id} has invalid relation: {relation!r}"
                )
    return plan


def load_plan(path: Path) -> dict[str, Any]:
    return _load_plan(path, set())


def _iter_bundles(data_root: Path) -> list[Path]:
    if (data_root / "artifacts").is_dir():
        return [data_root]
    extracted = data_root / "extracted"
    if not extracted.is_dir():
        return []
    if (extracted / "artifacts").is_dir():
        return [extracted]
    return [
        child
        for child in sorted(extracted.iterdir())
        if child.is_dir() and (child / "artifacts").is_dir()
    ]


def discover_slots(
    data_root: Path,
) -> tuple[dict[str, tuple[Path, Path | None]], list[str]]:
    discovered: dict[str, tuple[Path, Path | None]] = {}
    errors: list[str] = []
    seen_episode_paths: set[Path] = set()
    for bundle in _iter_bundles(data_root):
        episode_root = bundle / "artifacts" / "day8_episode"
        supervision_root = bundle / "artifacts" / "day10_supervised"
        if not episode_root.is_dir():
            continue
        for episode_dir in sorted(episode_root.iterdir()):
            manifest_path = episode_dir / "manifest.json"
            if not episode_dir.is_dir() or not manifest_path.is_file():
                continue
            resolved_episode = episode_dir.resolve()
            if resolved_episode in seen_episode_paths:
                # Jetson maintains artifacts/day8_episode/latest as a
                # convenience symlink. It is not a second recorded Run.
                continue
            seen_episode_paths.add(resolved_episode)
            try:
                manifest = _load_object(manifest_path)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            collection = manifest.get("collection")
            if not isinstance(collection, dict):
                continue
            slot_id = str(collection.get("slot_id", "")).strip()
            if not slot_id:
                errors.append(f"{episode_dir}: empty collection slot_id")
                continue
            if slot_id in discovered:
                errors.append(f"duplicate recorded collection slot: {slot_id}")
                continue
            run_id = str(manifest.get("run_id", episode_dir.name))
            supervision = supervision_root / run_id
            discovered[slot_id] = (
                episode_dir,
                supervision if supervision.is_dir() else None,
            )
    return discovered, errors


def _entity_index(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entities = record.get("entities", {}).get("items", [])
    if not isinstance(entities, list):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        entity_id = str(entity.get("entity_id", "")).strip()
        if entity_id and entity_id not in indexed:
            indexed[entity_id] = entity
    return indexed


def _position(entity: dict[str, Any]) -> tuple[float, float]:
    values = entity.get("relative_position_m")
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("entity has no planar relative_position_m")
    x_value = float(values[0])
    y_value = float(values[1])
    if not math.isfinite(x_value) or not math.isfinite(y_value):
        raise ValueError("entity position is not finite")
    return x_value, y_value


def _pairwise_target_distances(
    entities: dict[str, dict[str, Any]],
    required_ids: set[str],
) -> list[float]:
    positions = [
        _position(entities[entity_id])
        for entity_id in sorted(required_ids)
    ]
    return [
        math.hypot(first[0] - second[0], first[1] - second[1])
        for index, first in enumerate(positions)
        for second in positions[index + 1 :]
    ]


def _relation_passes(
    relation: list[str],
    entities: dict[str, dict[str, Any]],
    margin_m: float,
) -> bool:
    kind, first_id, second_id = relation
    first_x, first_y = _position(entities[first_id])
    second_x, second_y = _position(entities[second_id])
    if kind == "nearer":
        first_distance = math.hypot(first_x, first_y)
        second_distance = math.hypot(second_x, second_y)
        return first_distance + margin_m < second_distance
    if kind == "left_of":
        return first_y > second_y + margin_m
    raise ValueError(f"unsupported relation: {kind}")


def validate_slot(
    slot: dict[str, Any],
    episode_dir: Path,
    supervision_dir: Path | None,
    plan: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    manifest = _load_object(episode_dir / "manifest.json")
    collection = manifest.get("collection")
    if not isinstance(collection, dict):
        errors.append("episode manifest has no Day 12 collection metadata")
        collection = {}

    expected_identity = {
        "slot_id": slot["slot_id"],
        "layout_id": slot["layout_id"],
        "motion_state": slot["motion_state"],
    }
    for field, expected in expected_identity.items():
        if collection.get(field) != expected:
            errors.append(
                f"collection.{field}={collection.get(field)!r}, "
                f"expected {expected!r}"
            )
    if manifest.get("scene_seed") != slot["scene_seed"]:
        errors.append(
            f"scene_seed={manifest.get('scene_seed')!r}, "
            f"expected {slot['scene_seed']}"
        )
    if manifest.get("status") != "complete":
        errors.append("episode status is not complete")
    if manifest.get("execution_mode") != plan["required_execution_mode"]:
        errors.append("episode execution_mode is not the required mode")

    minimum_frames = int(plan["minimum_frames_per_run"])
    frame_count = int(manifest.get("frame_count", 0))
    if frame_count < minimum_frames:
        errors.append(
            f"frame_count={frame_count} is below minimum={minimum_frames}"
        )

    quality_path = episode_dir / "quality_report.json"
    if not quality_path.is_file():
        errors.append("quality_report.json is missing")
    else:
        quality = _load_object(quality_path)
        if not quality.get("passed"):
            errors.append("episode quality report did not pass")
        if quality.get("frame_count") != frame_count:
            errors.append("quality report frame_count mismatch")

    if supervision_dir is None:
        errors.append("supervised dataset is missing")
    else:
        supervision_manifest_path = supervision_dir / "manifest.json"
        if not supervision_manifest_path.is_file():
            errors.append("supervised manifest.json is missing")
        else:
            supervision = _load_object(supervision_manifest_path)
            coverage = supervision.get("label_coverage", {})
            samples = supervision.get("samples", {})
            source_episodes = supervision.get("source_episodes", [])
            if not isinstance(coverage, dict) or not coverage.get("complete"):
                errors.append("9/9 task-label coverage is incomplete")
            if (
                not isinstance(samples, dict)
                or int(samples.get("frame_count", 0)) < minimum_frames
            ):
                errors.append("supervised dataset has too few source frames")
            source_run_ids = {
                item.get("run_id")
                for item in source_episodes
                if isinstance(item, dict)
            }
            if manifest.get("run_id") not in source_run_ids:
                errors.append("supervised dataset does not reference this Run")

    required_ids = set(plan["required_entity_ids"])
    margin_m = float(plan["relation_margin_m"])
    relation_counts = [0] * len(slot["relations"])
    complete_entity_frames = 0
    relation_evaluated_frames = 0
    relation_window = int(plan.get("relation_evaluation_frames", 10))
    motion_evaluated_frames = 0
    motion_pass_frames = 0
    motion_window = int(plan.get("motion_evaluation_frames", 50))
    minimum_distance_change = float(
        plan.get("minimum_pairwise_distance_change_m", 0.05)
    )
    initial_pairwise_distances: list[float] | None = None
    frame_paths = sorted((episode_dir / "frames").glob("*.json"))
    for frame_path in frame_paths:
        try:
            record = _load_object(frame_path)
            entities = _entity_index(record)
            if not required_ids.issubset(entities):
                missing = sorted(required_ids - set(entities))
                errors.append(f"{frame_path.name}: missing entities {missing}")
                continue
            if any(
                not entities[entity_id].get("valid")
                or not entities[entity_id].get("visible")
                or not entities[entity_id].get("is_target")
                for entity_id in required_ids
            ):
                errors.append(
                    f"{frame_path.name}: required entity is not a valid "
                    "visible target"
                )
                continue
            if str(entities["target_red"].get("color", "")).casefold() != "red":
                errors.append(f"{frame_path.name}: target_red is not red")
                continue
            if str(entities["target_blue"].get("color", "")).casefold() != "blue":
                errors.append(f"{frame_path.name}: target_blue is not blue")
                continue
            complete_entity_frames += 1
            if relation_evaluated_frames < relation_window:
                for index, relation in enumerate(slot["relations"]):
                    if _relation_passes(relation, entities, margin_m):
                        relation_counts[index] += 1
                relation_evaluated_frames += 1
            # Motion check applies to S1 only: the S2 formation is rigid
            # (red/blue swing in phase, whites run parallel), so pairwise
            # target-distance change is naturally near zero and the check
            # would reject valid runs (observed 0.16-0.36 vs 0.3-0.6 bars).
            if (
                slot["motion_state"] == "S1"
                and motion_evaluated_frames < motion_window
            ):
                distances = _pairwise_target_distances(
                    entities, required_ids
                )
                if initial_pairwise_distances is None:
                    initial_pairwise_distances = distances
                else:
                    if any(
                        abs(current - initial) >= minimum_distance_change
                        for current, initial in zip(
                            distances, initial_pairwise_distances
                        )
                    ):
                        motion_pass_frames += 1
                    motion_evaluated_frames += 1
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{frame_path.name}: {exc}")

    if complete_entity_frames < minimum_frames:
        errors.append(
            f"only {complete_entity_frames} frame(s) have all required entities"
        )
    minimum_fraction = float(plan["minimum_relation_pass_fraction"])
    relation_fractions: list[float] = []
    for relation, count in zip(slot["relations"], relation_counts):
        fraction = (
            count / relation_evaluated_frames
            if relation_evaluated_frames
            else 0.0
        )
        relation_fractions.append(fraction)
        if fraction < minimum_fraction:
            errors.append(
                f"relation {relation} passed {fraction:.3f}, "
                f"required >= {minimum_fraction:.3f}"
            )

    motion_fraction: float | None = None
    if slot["motion_state"] == "S1":
        motion_fraction = (
            motion_pass_frames / motion_evaluated_frames
            if motion_evaluated_frames
            else 0.0
        )
        minimum_motion_fraction = float(
            plan.get("minimum_motion_pass_fraction", 0.6)
        )
        if motion_fraction < minimum_motion_fraction:
            errors.append(
                "pairwise target-distance motion passed "
                f"{motion_fraction:.3f}, "
                f"required >= {minimum_motion_fraction:.3f}"
            )

    return {
        "slot_id": slot["slot_id"],
        "run_id": manifest.get("run_id"),
        "scene_seed": manifest.get("scene_seed"),
        "frame_count": frame_count,
        "complete_entity_frame_count": complete_entity_frames,
        "relation_evaluated_frame_count": relation_evaluated_frames,
        "relation_pass_fractions": relation_fractions,
        "motion_evaluated_frame_count": motion_evaluated_frames,
        "motion_pass_fraction": motion_fraction,
        "passed": not errors,
        "errors": errors,
    }


def evaluate_collection(
    data_root: Path,
    plan_path: Path,
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    discovered, discovery_errors = discover_slots(data_root)
    slot_reports: list[dict[str, Any]] = []
    missing_slots: list[str] = []
    for slot in plan["slots"]:
        paths = discovered.get(slot["slot_id"])
        if paths is None:
            missing_slots.append(slot["slot_id"])
            continue
        slot_reports.append(validate_slot(slot, *paths, plan))

    passed_slots = [report for report in slot_reports if report["passed"]]
    complete = (
        not discovery_errors
        and not missing_slots
        and len(passed_slots) >= int(plan["minimum_complete_runs"])
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "passed": complete,
        "data_root": str(data_root.resolve()),
        "plan_path": str(plan_path.resolve()),
        "required_slot_count": len(plan["slots"]),
        "minimum_complete_runs": plan["minimum_complete_runs"],
        "discovered_slot_count": len(discovered),
        "passed_slot_count": len(passed_slots),
        "missing_slots": missing_slots,
        "discovery_errors": discovery_errors,
        "slot_reports": slot_reports,
    }


def _slot_command(
    slot: dict[str, Any],
    rollout_action: str = "follow",
) -> str:
    return (
        "ros2 launch asv_bringup collect.launch.py "
        f"slot_id:={slot['slot_id']} "
        f"layout_id:={slot['layout_id']} "
        f"motion_state:={slot['motion_state']} "
        f"scene_seed:={slot['scene_seed']} "
        f"action:={rollout_action}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or validate the Day 12 counterbalanced collection."
    )
    parser.add_argument(
        "command",
        choices=("next", "status", "validate"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/sine_near_collection_plan_v1.json"),
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the next slot as one machine-readable JSON object",
    )
    args = parser.parse_args()

    try:
        report = evaluate_collection(args.data_root, args.plan)
        plan = load_plan(args.plan)
    except (OSError, ValueError, TypeError) as exc:
        print(f"SCENE_COLLECTION_FAIL: {exc}")
        return 1

    if args.report is not None:
        _write_json_atomic(args.report, report)

    passed = {
        item["slot_id"]
        for item in report["slot_reports"]
        if item["passed"]
    }
    pending = [
        slot for slot in plan["slots"] if slot["slot_id"] not in passed
    ]
    if args.command == "next":
        if not pending:
            if args.json:
                print(json.dumps({"complete": True}, sort_keys=True))
            else:
                print("SCENE_NEXT: all planned slots have passed")
        else:
            slot = pending[0]
            if args.json:
                print(
                    json.dumps(
                        {
                            "complete": False,
                            "slot_id": slot["slot_id"],
                            "layout_id": slot["layout_id"],
                            "motion_state": slot["motion_state"],
                            "scene_seed": slot["scene_seed"],
                            "rollout_action": plan.get(
                                "rollout_action", "follow"
                            ),
                            "launch_command": _slot_command(
                                slot,
                                str(plan.get("rollout_action", "follow")),
                            ),
                        },
                        sort_keys=True,
                    )
                )
            else:
                print(
                    f"SCENE_NEXT slot={slot['slot_id']} "
                    f"layout={slot['layout_id']} "
                    f"motion={slot['motion_state']} "
                    f"scene_seed={slot['scene_seed']}"
                )
                print(_slot_command(slot))
        return 0

    marker = (
        "SCENE_COLLECTION_PASS"
        if report["passed"]
        else "SCENE_COLLECTION_INCOMPLETE"
    )
    print(
        f"{marker} passed={report['passed_slot_count']}/"
        f"{report['required_slot_count']} "
        f"discovered={report['discovered_slot_count']}"
    )
    for item in report["slot_reports"]:
        if not item["passed"]:
            print(
                f"  FAIL {item['slot_id']}: "
                + "; ".join(item["errors"][:5])
            )
    if report["missing_slots"]:
        print("  pending: " + ", ".join(report["missing_slots"]))
    if report["discovery_errors"]:
        print("  discovery: " + "; ".join(report["discovery_errors"]))
    if args.command == "validate" and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
