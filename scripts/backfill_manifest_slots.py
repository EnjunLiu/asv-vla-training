#!/usr/bin/env python3
"""One-time migration: write teacher/vision contracts into episode manifests.

Semantic color/slot mapping lives ONLY in this script — not in src/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

DEFAULT_SLOT_ENTITY_IDS = (
    "target_red",
    "target_blue",
    "target_left",
    "target_right",
)

_SLOT_PATTERN = re.compile(
    r"^(?P<color>RED|BLUE)_(?P<standoff>\d+)M_(?:TRAIN|VALIDATION|TEST)",
    re.IGNORECASE,
)


def teacher_from_slot_id(slot_id: str, *, task_text: str = "") -> dict[str, Any]:
    match = _SLOT_PATTERN.match(str(slot_id).strip())
    if match is None:
        raise ValueError(f"unrecognized collection.slot_id for teacher backfill: {slot_id!r}")
    color = match.group("color").upper()
    standoff_m = float(match.group("standoff"))
    follow_slot_index = 0 if color == "RED" else 1
    default_key = {
        ("RED", 3): "follow the red boat, keep 3 meters distance",
        ("BLUE", 3): "follow the blue boat, keep 3 meters distance",
        ("RED", 4): "follow the red boat, keep 4 meters distance",
        ("BLUE", 4): "follow the blue boat, keep 4 meters distance",
    }.get((color, int(standoff_m)), "")
    embedding_key = str(task_text).strip() or default_key
    if not embedding_key:
        embedding_key = f"{color.lower()}_{int(standoff_m)}m"
    return {
        "follow_slot_index": follow_slot_index,
        "standoff_m": standoff_m,
        "embedding_key": embedding_key,
    }


def slot_entity_ids_from_episode(episode: Path, layout_id: str) -> list[str]:
    del layout_id
    frames_dir = episode / "frames"
    frame_paths = sorted(frames_dir.glob("*.json"))
    if not frame_paths:
        return list(DEFAULT_SLOT_ENTITY_IDS)
    frame = json.loads(frame_paths[0].read_text(encoding="utf-8"))
    items = frame.get("entities", {}).get("items", [])
    present = [str(item.get("entity_id", "")) for item in items if item.get("entity_id")]
    ordered = [entity_id for entity_id in DEFAULT_SLOT_ENTITY_IDS if entity_id in present]
    extras = [entity_id for entity_id in present if entity_id not in DEFAULT_SLOT_ENTITY_IDS]
    resolved = ordered + extras
    return resolved or list(DEFAULT_SLOT_ENTITY_IDS)


def backfill_manifest(path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    collection = manifest.get("collection") or {}
    slot_id = str(collection.get("slot_id", ""))
    layout_id = str(collection.get("layout_id", ""))
    task_text = str(manifest.get("task_text", "")).strip()
    teacher = teacher_from_slot_id(slot_id, task_text=task_text)
    vision = {
        "slot_entity_ids": slot_entity_ids_from_episode(path.parent, layout_id),
    }
    updated = dict(manifest)
    updated["teacher"] = teacher
    updated["vision"] = vision
    if not dry_run:
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path), "teacher": teacher, "vision": vision}


def iter_manifests(root: Path) -> list[Path]:
    return sorted(root.rglob("manifest.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="Episode root (e.g. data/episodes/moving_target_valid)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifests = iter_manifests(args.episodes)
    if not manifests:
        raise SystemExit(f"no manifest.json under {args.episodes}")
    for path in manifests:
        result = backfill_manifest(path, dry_run=args.dry_run)
        print(
            f"BACKFILL path={result['path']} "
            f"follow={result['teacher']['follow_slot_index']} "
            f"standoff={result['teacher']['standoff_m']} "
            f"key={result['teacher']['embedding_key']} "
            f"slots={result['vision']['slot_entity_ids']}"
        )
    print(f"BACKFILL_DONE count={len(manifests)} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
