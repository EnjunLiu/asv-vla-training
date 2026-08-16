"""Day 11B: scan extracted episodes and supervision manifests, produce a
reproducible Run-level registry (``dataset_registry_v1.jsonl``).

The registry is a plain-text JSONL file — one line per Run — so it is
easy to diff, append to, and audit.  It never contains image bytes or
model weights.

Usage::

    PYTHONPATH=src \\
      python -m asv_training.dataset_registry \\
      --data-root data \\
      --output data/registry/dataset_registry_v1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA_VERSION = "dataset_registry_v1"
_EXPECTED_EXECUTION_MODES = {
    # Read-only compatibility for Day 8/10 manifests created before the
    # recorder renamed this mode to observation_only.
    "static",
    "observation_only",
    "ue5_kinematic_expert_v1",
    "legacy_thruster",
}
MINIMUM_TRAINING_RUNS = 12
MINIMUM_FRAMES_PER_RUN = 80
MINIMUM_SCENE_SEEDS = 3


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from *path*, raising on any failure."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a JSON object, got {type(value).__name__}")
    return value


def _iter_extracted_bundles(data_root: Path) -> list[Path]:
    """Yield every bundle under *data_root* that contains an ``artifacts/`` tree.

    Three layouts are supported (checked in order):

    1. **Jetson live** — ``data_root/artifacts/`` exists directly.
       Bundle is ``data_root``.
    2. **PC flat** — ``data_root/extracted/artifacts/`` exists.
       Bundle is ``data_root/extracted``.
    3. **PC nested** — ``data_root/extracted/<bundle>/artifacts/`` exists
       for one or more subdirectories.
    """
    # Jetson live layout.
    if (data_root / "artifacts").is_dir():
        return [data_root]

    extracted = data_root / "extracted"
    if not extracted.is_dir():
        return []

    # PC flat layout.
    if (extracted / "artifacts").is_dir():
        return [extracted]

    # PC nested layout.
    bundles: list[Path] = []
    for child in sorted(extracted.iterdir()):
        if child.is_dir() and (child / "artifacts").is_dir():
            bundles.append(child)
    return bundles


def _discover_episodes(data_root: Path) -> dict[str, Path]:
    """Walk every extracted bundle under *data_root* for Run directories.

    Returns a mapping ``{run_id: episode_dir}``.  Empty if no episodes are
    found.
    """
    episodes: dict[str, Path] = {}
    for bundle in _iter_extracted_bundles(data_root):
        episodes_root = bundle / "artifacts" / "day8_episode"
        if not episodes_root.is_dir():
            continue
        for entry in sorted(episodes_root.iterdir()):
            if not entry.is_dir():
                continue
            run_id = entry.name
            manifest = entry / "manifest.json"
            if not manifest.is_file():
                continue
            if run_id in episodes:
                raise ValueError(
                    f"duplicate episode run_id across bundles: {run_id}"
                )
            episodes[run_id] = entry
    return episodes


def _discover_supervisions(data_root: Path) -> dict[str, Path]:
    """Return ``{run_id: supervision_dir}`` for every Day-10 dataset
    found across all extracted bundles."""
    supervisions: dict[str, Path] = {}
    for bundle in _iter_extracted_bundles(data_root):
        sup_root = bundle / "artifacts" / "day10_supervised"
        if not sup_root.is_dir():
            continue
        for entry in sorted(sup_root.iterdir()):
            if not entry.is_dir():
                continue
            run_id = entry.name
            manifest = entry / "manifest.json"
            if not manifest.is_file():
                continue
            if run_id in supervisions:
                raise ValueError(
                    f"duplicate supervision run_id across bundles: {run_id}"
                )
            supervisions[run_id] = entry
    return supervisions


# ---------------------------------------------------------------------------
#  Single-Run scan
# ---------------------------------------------------------------------------

def scan_run(
    run_id: str,
    episode_dir: Path,
    data_root: Path,
    supervision_dir: Path | None = None,
) -> dict[str, Any]:
    """Produce one registry entry for *run_id*.

    The entry records identity, paths relative to *data_root*, hashes,
    frame / sample counts, label coverage, and execution mode.
    """
    entry: dict[str, Any] = {
        "run_id": run_id,
        "episode_path": str(episode_dir.relative_to(data_root)),
    }

    # -- episode manifest --
    ep_manifest_path = episode_dir / "manifest.json"
    try:
        ep_manifest = _load_json(ep_manifest_path)
    except ValueError:
        entry["episode_valid"] = False
        entry["episode_error"] = "missing or invalid manifest.json"
        return entry

    entry["episode_valid"] = True
    entry["scene_seed"] = ep_manifest.get("scene_seed")
    entry["frame_count"] = ep_manifest.get("frame_count", 0)
    entry["episode_manifest_sha256"] = _sha256_file(ep_manifest_path)
    if ep_manifest.get("run_id") != run_id:
        entry["episode_valid"] = False
        entry["episode_error"] = "manifest run_id does not match directory"
    if ep_manifest.get("status") != "complete":
        entry["episode_valid"] = False
        entry["episode_error"] = "episode status is not complete"

    execution_mode = str(ep_manifest.get("execution_mode", "static"))
    if execution_mode not in _EXPECTED_EXECUTION_MODES:
        raise ValueError(
            f"run_id={run_id}: unknown execution_mode={execution_mode!r}; "
            f"expected one of {sorted(_EXPECTED_EXECUTION_MODES)}"
        )
    entry["execution_mode"] = execution_mode
    collection = ep_manifest.get("collection")
    if isinstance(collection, dict):
        entry["collection_slot"] = str(
            collection.get("slot_id", "")
        ).strip()
        entry["layout_id"] = str(
            collection.get("layout_id", "")
        ).strip()
        entry["motion_state"] = str(
            collection.get("motion_state", "")
        ).strip()
    else:
        entry["collection_slot"] = ""
        entry["layout_id"] = ""
        entry["motion_state"] = ""

    quality_path = episode_dir / "quality_report.json"
    if quality_path.is_file():
        quality = _load_json(quality_path)
        entry["quality_report_sha256"] = _sha256_file(quality_path)
        entry["quality_passed"] = bool(quality.get("passed", False))
        if quality.get("run_id") != run_id:
            entry["quality_passed"] = False
            entry["quality_error"] = "quality report run_id mismatch"
    else:
        entry["quality_passed"] = False
        entry["quality_error"] = "quality_report.json is missing"

    # -- supervision manifest (optional for registry; required for training) --
    if supervision_dir is not None and supervision_dir.is_dir():
        sup_manifest_path = supervision_dir / "manifest.json"
        try:
            sup_manifest = _load_json(sup_manifest_path)
        except ValueError:
            entry["supervision_valid"] = False
            entry["supervision_error"] = "missing or invalid manifest.json"
            return entry

        entry["supervision_valid"] = True
        entry["supervision_path"] = str(supervision_dir.relative_to(data_root))
        entry["supervision_manifest_sha256"] = _sha256_file(sup_manifest_path)
        samples = sup_manifest.get("samples", {})
        entry["sample_count"] = samples.get("sample_count", 0)
        label_cov = sup_manifest.get("label_coverage", {})
        entry["coverage_complete"] = bool(label_cov.get("complete", False))
        entry["observed_labels"] = sorted(label_cov.get("observed_labels", []))
        entry["required_labels"] = sorted(label_cov.get("required_labels", []))
    else:
        entry["supervision_valid"] = False
        entry["supervision_error"] = "no supervision directory"

    entry["training_eligible"] = bool(
        entry.get("episode_valid")
        and entry.get("quality_passed")
        and entry.get("supervision_valid")
        and entry.get("coverage_complete")
        and int(entry.get("frame_count", 0)) >= MINIMUM_FRAMES_PER_RUN
        and entry.get("collection_slot")
        and entry.get("layout_id")
        and entry.get("motion_state")
    )
    return entry


def scan_all_runs(data_root: Path) -> list[dict[str, Any]]:
    """Discover and scan every episode under *data_root*.

    Returns one entry per discovered Run, sorted by run_id.
    """
    episodes = _discover_episodes(data_root)
    if not episodes:
        return []

    supervisions = _discover_supervisions(data_root)
    entries: list[dict[str, Any]] = []
    for run_id in sorted(episodes):
        entries.append(
            scan_run(
                run_id,
                episodes[run_id],
                data_root,
                supervision_dir=supervisions.get(run_id),
            )
        )
    return entries


# ---------------------------------------------------------------------------
#  Registry manifest
# ---------------------------------------------------------------------------

def build_registry(
    data_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Scan all runs and write ``dataset_registry_v1.jsonl``.

    Returns a summary dict suitable for reporting.
    """
    entries = scan_all_runs(data_root)

    lines: list[str] = []
    valid_episodes = 0
    valid_supervisions = 0
    total_frames = 0
    total_samples = 0
    scene_seeds: set[int] = set()
    eligible_run_count = 0
    eligible_scene_seeds: set[int] = set()
    eligible_slots: set[str] = set()

    for entry in entries:
        lines.append(
            json.dumps(entry, ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
        if entry.get("episode_valid"):
            valid_episodes += 1
            total_frames += entry.get("frame_count", 0)
            if entry.get("scene_seed") is not None:
                scene_seeds.add(entry["scene_seed"])
        if entry.get("supervision_valid"):
            valid_supervisions += 1
            total_samples += entry.get("sample_count", 0)
        if entry.get("training_eligible"):
            eligible_run_count += 1
            eligible_scene_seeds.add(int(entry["scene_seed"]))
            slot = str(entry["collection_slot"])
            if slot in eligible_slots:
                raise ValueError(f"duplicate eligible collection slot: {slot}")
            eligible_slots.add(slot)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "run_count": len(entries),
        "valid_episode_count": valid_episodes,
        "valid_supervision_count": valid_supervisions,
        "total_frame_count": total_frames,
        "total_sample_count": total_samples,
        "scene_seed_count": len(scene_seeds),
        "scene_seeds": sorted(scene_seeds),
        "eligible_run_count": eligible_run_count,
        "eligible_scene_seed_count": len(eligible_scene_seeds),
        "eligible_scene_seeds": sorted(eligible_scene_seeds),
        "minimum_runs_for_training": MINIMUM_TRAINING_RUNS,
        "minimum_frames_per_run": MINIMUM_FRAMES_PER_RUN,
        "min_scene_seeds_for_training": MINIMUM_SCENE_SEEDS,
        "training_ready": (
            eligible_run_count >= MINIMUM_TRAINING_RUNS
            and len(eligible_scene_seeds) >= MINIMUM_SCENE_SEEDS
        ),
        "registry_path": str(output_path.resolve()),
        "registry_sha256": _sha256_file(output_path),
    }

    manifest_path = Path(str(output_path).replace(".jsonl", "_manifest.json"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return manifest


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Day 11B: build the PC dataset registry from extracted data."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATASET_DATA_ROOT", "data")),
        help="External data root (default: $DATASET_DATA_ROOT or ./data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the JSONL registry (default: <data-root>/registry/dataset_registry_v1.jsonl).",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"DAY11_REGISTRY_FAIL: data root does not exist: {data_root}")
        return 1

    output = (
        args.output.resolve()
        if args.output
        else data_root / "registry" / "dataset_registry_v1.jsonl"
    )

    try:
        manifest = build_registry(data_root, output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"DAY11_REGISTRY_FAIL: {exc}")
        return 1

    ready = "ready" if manifest["training_ready"] else "not_ready"
    print(
        f"DAY11_REGISTRY_PASS "
        f"runs={manifest['run_count']} "
        f"frames={manifest['total_frame_count']} "
        f"samples={manifest['total_sample_count']} "
        f"scene_seeds={manifest['scene_seed_count']} "
        f"eligible_runs={manifest['eligible_run_count']} "
        f"training_ready={manifest['training_ready']}"
    )
    if not manifest["training_ready"]:
        print(
            f"TRAINING_NOT_READY: "
            f"need at least {MINIMUM_TRAINING_RUNS} eligible Runs and "
            f"{MINIMUM_SCENE_SEEDS} scene seeds; "
            f"eligible_runs={manifest['eligible_run_count']} "
            f"eligible_seeds={manifest['eligible_scene_seeds']}"
        )
    print(f"  registry: {manifest['registry_path']}")
    print(f"  sha256:   {manifest['registry_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
