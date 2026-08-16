"""Build and verify deterministic multimodal supervision manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
from typing import Any, Iterable

from .episode import evaluate_episode, load_episode_records, write_json_atomic
from .expert_trajectory import (
    MODEL_VERSION,
    ExpertTrajectoryError,
    generate_expert_trajectory,
    task_from_labels,
)
from .frame_record import read_frame_record
from .language_intervention_dataset import read_jsonl
from .trajectory_contract import DT_SEC


DATASET_SCHEMA_VERSION = "supervised_action_dataset_v2"
SAMPLE_SCHEMA_VERSION = "supervised_action_sample_v2"
GENERATOR_VERSION = "supervised_dataset_single_step_v2"
REQUIRED_LABELS = {
    "follow|color:red|3m",
    "follow|color:red|4m",
    "follow|color:red|10m",
    "follow|color:blue|3m",
    "follow|color:blue|4m",
    "follow|color:blue|10m",
    "follow|bearing:left|3m",
    "follow|bearing:left|4m",
    "follow|bearing:left|10m",
    "follow|bearing:right|3m",
    "follow|bearing:right|4m",
    "follow|bearing:right|10m",
    "stop|none|none",
}
VALID_LANGUAGE_SPLITS = {"train", "validation", "test"}


class SupervisedDatasetError(ValueError):
    """Raised when a dataset is incomplete or not reproducible."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SupervisedDatasetError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _relative_path(target: Path, anchor: Path) -> str:
    return Path(os.path.relpath(target.resolve(), anchor.resolve())).as_posix()


def _resolve_beneath(root: Path, relative: str, field: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise SupervisedDatasetError(f"{field} must be relative")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SupervisedDatasetError(f"{field} escapes its episode") from exc
    return resolved


def _label_key(record: dict[str, Any]) -> str:
    return "|".join(
        (
            str(record.get("action", "")).strip().casefold(),
            str(record.get("target_attribute", "")).strip().casefold(),
            str(record.get("distance_bucket", "")).strip().casefold(),
        )
    )


def _validate_instructions(
    instructions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not instructions:
        raise SupervisedDatasetError("instruction dataset is empty")
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(instructions):
        instruction_id = str(record.get("instruction_id", "")).strip()
        text = str(record.get("text", "")).strip()
        split = str(record.get("split", "")).strip().casefold()
        if not instruction_id:
            raise SupervisedDatasetError(
                f"instruction[{index}] has an empty instruction_id"
            )
        if instruction_id in indexed:
            raise SupervisedDatasetError(
                f"duplicate instruction_id: {instruction_id}"
            )
        if not text:
            raise SupervisedDatasetError(
                f"instruction {instruction_id} has empty text"
            )
        if split not in VALID_LANGUAGE_SPLITS:
            raise SupervisedDatasetError(
                f"instruction {instruction_id} has invalid split={split!r}"
            )
        try:
            task_from_labels(
                str(record.get("action", "")),
                str(record.get("target_attribute", "")),
                str(record.get("distance_bucket", "")),
            )
        except ExpertTrajectoryError as exc:
            raise SupervisedDatasetError(
                f"instruction {instruction_id}: {exc}"
            ) from exc
        indexed[instruction_id] = record
    return indexed


def _entity_objects(record: dict[str, Any]) -> list[SimpleNamespace]:
    entities: list[SimpleNamespace] = []
    for item in record["entities"]["items"]:
        position = item["relative_position_m"]
        velocity = item["relative_velocity_mps"]
        entities.append(
            SimpleNamespace(
                entity_id=item["entity_id"],
                class_name=item["class_name"],
                color=item["color"],
                is_target=item["is_target"],
                visible=item["visible"],
                relative_x=position[0],
                relative_y=position[1],
                relative_z=position[2],
                relative_velocity_x=velocity[0],
                relative_velocity_y=velocity[1],
                relative_velocity_z=velocity[2],
                valid=item["valid"],
            )
        )
    return entities


def _sample_id(
    *,
    run_id: str,
    scene_seed: int,
    frame_index: int,
    stamp_us: int,
    instruction_id: str,
) -> str:
    identity = (
        f"{run_id}\0{scene_seed}\0{frame_index}\0{stamp_us}\0"
        f"{instruction_id}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _expert_action(values: Iterable[float]) -> list[float]:
    action = [float(value) for value in values]
    if len(action) != 2 or not all(math.isfinite(value) for value in action):
        raise SupervisedDatasetError(
            "expert_action must be finite with shape [2]"
        )
    return action


def _make_sample(
    *,
    record: dict[str, Any],
    instruction: dict[str, Any],
    frame_record_path: str,
    frame_record_sha256: str,
    image_sha256: str,
) -> dict[str, Any] | None:
    task = task_from_labels(
        str(instruction["action"]),
        str(instruction["target_attribute"]),
        str(instruction["distance_bucket"]),
    )
    try:
        expert = generate_expert_trajectory(task, _entity_objects(record))
    except ExpertTrajectoryError as exc:
        if str(exc).startswith("no valid visible target matches"):
            return None
        raise

    instruction_id = str(instruction["instruction_id"])
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sample_id": _sample_id(
            run_id=record["run_id"],
            scene_seed=record["scene_seed"],
            frame_index=record["frame_index"],
            stamp_us=record["stamp_us"],
            instruction_id=instruction_id,
        ),
        "source": {
            "run_id": record["run_id"],
            "scene_seed": record["scene_seed"],
            "frame_index": record["frame_index"],
            "stamp_us": record["stamp_us"],
            "frame_id": record["frame_id"],
            "frame_record_path": frame_record_path,
            "frame_record_sha256": frame_record_sha256,
            "image_path": record["camera"]["image_path"],
            "image_sha256": image_sha256,
            "recorded_task_text": record["task"]["text"],
        },
        "instruction": {
            "instruction_id": instruction_id,
            "text": str(instruction["text"]),
            "language_split": str(instruction["split"]),
            "action": str(instruction["action"]),
            "target_attribute": str(instruction["target_attribute"]),
            "distance_bucket": str(instruction["distance_bucket"]),
        },
        "expert": {
            "model_version": MODEL_VERSION,
            "dt": DT_SEC,
            "expert_action": _expert_action(expert.expert_action),
            "safe_stop": expert.safe_stop,
            "selected_entity_id": expert.selected_entity_id,
            "valid": True,
            "detail": expert.detail,
        },
    }


def build_supervised_dataset(
    episode_dirs: Iterable[str | Path],
    instructions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Pair complete FrameRecords with every compatible instruction."""

    episodes = sorted({Path(path).resolve() for path in episode_dirs})
    if not episodes:
        raise SupervisedDatasetError("at least one episode is required")
    instructions_source = Path(instructions_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise SupervisedDatasetError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    instructions = read_jsonl(instructions_source)
    instruction_index = _validate_instructions(instructions)
    samples: list[dict[str, Any]] = []
    source_episodes: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    frame_count = 0

    for episode_dir in episodes:
        report = evaluate_episode(
            episode_dir,
            min_frames=1,
            write_report=False,
        )
        if not report["passed"]:
            raise SupervisedDatasetError(
                f"invalid episode {episode_dir}: "
                + "; ".join(report["errors"])
            )
        run_id = str(report["run_id"])
        if run_id in run_ids:
            raise SupervisedDatasetError(
                f"duplicate run_id across episodes: {run_id}"
            )
        run_ids.add(run_id)
        manifest_path = episode_dir / "manifest.json"
        records = load_episode_records(episode_dir)
        source_episodes.append(
            {
                "run_id": run_id,
                "scene_seed": report["scene_seed"],
                "path": _relative_path(episode_dir, output),
                "manifest_sha256": _sha256_file(manifest_path),
                "frame_count": len(records),
            }
        )
        frame_count += len(records)

        for record in records:
            frame_record_path = (
                f"frames/{int(record['frame_index']):012d}.json"
            )
            record_path = episode_dir / frame_record_path
            image_path = episode_dir / record["camera"]["image_path"]
            record_hash = _sha256_file(record_path)
            image_hash = _sha256_file(image_path)
            for instruction_id in sorted(instruction_index):
                try:
                    sample = _make_sample(
                        record=record,
                        instruction=instruction_index[instruction_id],
                        frame_record_path=frame_record_path,
                        frame_record_sha256=record_hash,
                        image_sha256=image_hash,
                    )
                except ExpertTrajectoryError as exc:
                    raise SupervisedDatasetError(
                        f"run_id={run_id} frame={record['frame_index']} "
                        f"instruction={instruction_id}: {exc}"
                    ) from exc
                if sample is not None:
                    samples.append(sample)

    if not samples:
        raise SupervisedDatasetError(
            "no instruction is compatible with the episode entities"
        )
    samples.sort(
        key=lambda sample: (
            sample["source"]["run_id"],
            sample["source"]["frame_index"],
            sample["instruction"]["instruction_id"],
        )
    )
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise SupervisedDatasetError("generated duplicate sample_id")

    label_counts = Counter(
        _label_key(sample["instruction"]) for sample in samples
    )
    split_counts = Counter(
        sample["instruction"]["language_split"] for sample in samples
    )
    compatible_instruction_ids = {
        sample["instruction"]["instruction_id"] for sample in samples
    }
    samples_text = "\n".join(_json_line(sample) for sample in samples) + "\n"

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        samples_path = temporary_dir / "samples.jsonl"
        _write_text_atomic(samples_path, samples_text)
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "source_episodes": source_episodes,
            "instruction_dataset": {
                "path": _relative_path(instructions_source, output),
                "sha256": _sha256_file(instructions_source),
                "instruction_count": len(instructions),
                "compatible_instruction_count": len(
                    compatible_instruction_ids
                ),
            },
            "samples": {
                "path": "samples.jsonl",
                "sha256": _sha256_file(samples_path),
                "sample_count": len(samples),
                "frame_count": frame_count,
                "language_split_counts": dict(sorted(split_counts.items())),
            },
            "expert_action_contract": {
                "frame_id": "base_link",
                "dt": DT_SEC,
                "action_shape": [2],
                "expert_model_version": MODEL_VERSION,
            },
            "label_coverage": {
                "required_labels": sorted(REQUIRED_LABELS),
                "observed_labels": sorted(label_counts),
                "label_sample_counts": dict(sorted(label_counts.items())),
                "complete": set(label_counts) == REQUIRED_LABELS,
            },
            "split_note": (
                "language_split evaluates held-out instruction templates; "
                "source frames are intentionally reused for paired language "
                "interventions and are not a visual-generalization split"
            ),
        }
        write_json_atomic(temporary_dir / "manifest.json", manifest)
        os.replace(temporary_dir, output)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return {
        "dataset_dir": str(output),
        "sample_count": len(samples),
        "frame_count": frame_count,
        "compatible_instruction_count": len(compatible_instruction_ids),
        "observed_label_count": len(label_counts),
        "required_label_count": len(REQUIRED_LABELS),
        "coverage_complete": set(label_counts) == REQUIRED_LABELS,
    }


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisedDatasetError(f"invalid {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise SupervisedDatasetError(f"{context} must be a JSON object")
    return value


def _assert_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise SupervisedDatasetError(
            f"{context} mismatch: expected {expected!r}, got {actual!r}"
        )


def evaluate_supervised_dataset(
    dataset_dir: str | Path,
    *,
    require_all_labels: bool = False,
) -> dict[str, Any]:
    """Verify source hashes and recompute every expert action."""

    root = Path(dataset_dir).resolve()
    manifest = _load_json(root / "manifest.json", "manifest.json")
    _assert_equal(
        manifest.get("schema_version"),
        DATASET_SCHEMA_VERSION,
        "manifest schema_version",
    )
    _assert_equal(
        manifest.get("expert_action_contract"),
        {
            "frame_id": "base_link",
            "dt": DT_SEC,
            "action_shape": [2],
            "expert_model_version": MODEL_VERSION,
        },
        "manifest expert_action_contract",
    )

    instruction_info = manifest.get("instruction_dataset")
    if not isinstance(instruction_info, dict):
        raise SupervisedDatasetError("instruction_dataset is missing")
    instructions_path = (root / str(instruction_info.get("path", ""))).resolve()
    _assert_equal(
        _sha256_file(instructions_path),
        instruction_info.get("sha256"),
        "instruction dataset sha256",
    )
    instructions = read_jsonl(instructions_path)
    instruction_index = _validate_instructions(instructions)

    episode_index: dict[str, Path] = {}
    for item in manifest.get("source_episodes", []):
        if not isinstance(item, dict):
            raise SupervisedDatasetError("invalid source_episodes entry")
        run_id = str(item.get("run_id", ""))
        episode = (root / str(item.get("path", ""))).resolve()
        if run_id in episode_index:
            raise SupervisedDatasetError(
                f"duplicate manifest run_id: {run_id}"
            )
        _assert_equal(
            _sha256_file(episode / "manifest.json"),
            item.get("manifest_sha256"),
            f"episode {run_id} manifest sha256",
        )
        episode_index[run_id] = episode

    samples_info = manifest.get("samples")
    if not isinstance(samples_info, dict):
        raise SupervisedDatasetError("samples metadata is missing")
    samples_path = _resolve_beneath(
        root, str(samples_info.get("path", "")), "samples.path"
    )
    _assert_equal(
        _sha256_file(samples_path),
        samples_info.get("sha256"),
        "samples sha256",
    )
    samples = read_jsonl(samples_path)
    _assert_equal(
        len(samples), samples_info.get("sample_count"), "sample_count"
    )

    frame_cache: dict[
        tuple[str, str], tuple[dict[str, Any], str, str]
    ] = {}
    sample_ids: set[str] = set()
    label_counts: Counter[str] = Counter()
    compatible_instructions: set[str] = set()
    frame_keys: set[tuple[str, int, int, int]] = set()

    for index, sample in enumerate(samples):
        context = f"sample[{index}]"
        _assert_equal(
            sample.get("schema_version"),
            SAMPLE_SCHEMA_VERSION,
            f"{context} schema_version",
        )
        source = sample.get("source")
        instruction = sample.get("instruction")
        expert = sample.get("expert")
        if not all(
            isinstance(value, dict)
            for value in (source, instruction, expert)
        ):
            raise SupervisedDatasetError(
                f"{context} source/instruction/expert must be objects"
            )

        run_id = str(source.get("run_id", ""))
        episode = episode_index.get(run_id)
        if episode is None:
            raise SupervisedDatasetError(
                f"{context} references unknown run_id={run_id!r}"
            )
        record_relative = str(source.get("frame_record_path", ""))
        cache_key = (run_id, record_relative)
        cached = frame_cache.get(cache_key)
        if cached is None:
            record_path = _resolve_beneath(
                episode, record_relative, f"{context} frame_record_path"
            )
            record = read_frame_record(record_path, image_root=episode)
            image_path = _resolve_beneath(
                episode,
                str(record["camera"]["image_path"]),
                f"{context} image_path",
            )
            cached = (
                record,
                _sha256_file(record_path),
                _sha256_file(image_path),
            )
            frame_cache[cache_key] = cached
        record, record_hash, image_hash = cached
        _assert_equal(
            record_hash,
            source.get("frame_record_sha256"),
            f"{context} frame_record_sha256",
        )
        _assert_equal(
            image_hash,
            source.get("image_sha256"),
            f"{context} image_sha256",
        )

        for field in ("run_id", "scene_seed", "frame_index", "stamp_us", "frame_id"):
            _assert_equal(
                source.get(field),
                record.get(field),
                f"{context} source.{field}",
            )
        _assert_equal(
            source.get("image_path"),
            record["camera"]["image_path"],
            f"{context} source.image_path",
        )

        instruction_id = str(instruction.get("instruction_id", ""))
        expected_instruction = instruction_index.get(instruction_id)
        if expected_instruction is None:
            raise SupervisedDatasetError(
                f"{context} unknown instruction_id={instruction_id!r}"
            )
        for field in (
            "text",
            "action",
            "target_attribute",
            "distance_bucket",
        ):
            _assert_equal(
                instruction.get(field),
                expected_instruction.get(field),
                f"{context} instruction.{field}",
            )
        _assert_equal(
            instruction.get("language_split"),
            expected_instruction.get("split"),
            f"{context} instruction.language_split",
        )

        expected_id = _sample_id(
            run_id=record["run_id"],
            scene_seed=record["scene_seed"],
            frame_index=record["frame_index"],
            stamp_us=record["stamp_us"],
            instruction_id=instruction_id,
        )
        _assert_equal(
            sample.get("sample_id"), expected_id, f"{context} sample_id"
        )
        if expected_id in sample_ids:
            raise SupervisedDatasetError(
                f"duplicate sample_id: {expected_id}"
            )
        sample_ids.add(expected_id)

        if "horizon" in expert or "delta_p_xy" in expert:
            raise SupervisedDatasetError(
                f"{context} contains legacy horizon/delta_p_xy fields"
            )
        action = expert.get("expert_action")
        if (
            not isinstance(action, list)
            or len(action) != 2
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in action
            )
        ):
            raise SupervisedDatasetError(
                f"{context} expert_action must be finite with shape [2]"
            )

        task = task_from_labels(
            str(expected_instruction["action"]),
            str(expected_instruction["target_attribute"]),
            str(expected_instruction["distance_bucket"]),
        )
        try:
            expected_result = generate_expert_trajectory(
                task, _entity_objects(record)
            )
        except ExpertTrajectoryError as exc:
            raise SupervisedDatasetError(
                f"{context} cannot recompute expert: {exc}"
            ) from exc
        expected_expert = {
            "model_version": MODEL_VERSION,
            "dt": DT_SEC,
            "expert_action": _expert_action(expected_result.expert_action),
            "safe_stop": expected_result.safe_stop,
            "selected_entity_id": expected_result.selected_entity_id,
            "valid": True,
            "detail": expected_result.detail,
        }
        _assert_equal(expert, expected_expert, f"{context} expert")

        label_counts[_label_key(instruction)] += 1
        compatible_instructions.add(instruction_id)
        frame_keys.add(
            (
                record["run_id"],
                record["scene_seed"],
                record["frame_index"],
                record["stamp_us"],
            )
        )

    observed_labels = set(label_counts)
    coverage_complete = observed_labels == REQUIRED_LABELS
    manifest_coverage = manifest.get("label_coverage", {})
    _assert_equal(
        sorted(observed_labels),
        manifest_coverage.get("observed_labels"),
        "observed label coverage",
    )
    _assert_equal(
        coverage_complete,
        manifest_coverage.get("complete"),
        "label coverage complete",
    )
    if require_all_labels and not coverage_complete:
        missing = sorted(REQUIRED_LABELS - observed_labels)
        raise SupervisedDatasetError(
            f"missing required task labels: {missing}"
        )

    return {
        "passed": True,
        "sample_count": len(samples),
        "frame_count": len(frame_keys),
        "compatible_instruction_count": len(compatible_instructions),
        "observed_label_count": len(observed_labels),
        "required_label_count": len(REQUIRED_LABELS),
        "coverage_complete": coverage_complete,
    }


def build_main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic supervised expert-action data."
    )
    parser.add_argument(
        "--episode",
        type=Path,
        action="append",
        required=True,
        help="Day 8 episode directory; repeat for multiple runs.",
    )
    parser.add_argument(
        "--instructions",
        type=Path,
        default=Path("dataset/language/instructions.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_supervised_dataset(
            args.episode, args.instructions, args.output
        )
    except (SupervisedDatasetError, OSError, ValueError) as exc:
        print(f"DATASET_BUILD_FAIL: {exc}")
        return 1
    coverage = (
        "complete" if report["coverage_complete"] else "partial"
    )
    print(
        "DATASET_BUILD_PASS "
        f"samples={report['sample_count']} "
        f"frames={report['frame_count']} "
        f"instructions={report['compatible_instruction_count']} "
        f"labels={report['observed_label_count']}/"
        f"{report['required_label_count']} "
        f"coverage={coverage}"
    )
    return 0


def evaluate_main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a deterministic supervised dataset."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--require-all-labels", action="store_true")
    args = parser.parse_args()
    try:
        report = evaluate_supervised_dataset(
            args.dataset_dir,
            require_all_labels=args.require_all_labels,
        )
    except (SupervisedDatasetError, OSError, ValueError) as exc:
        print(f"DAY10_SUPERVISED_DATASET_FAIL: {exc}")
        return 1
    coverage = (
        "complete" if report["coverage_complete"] else "partial"
    )
    print(
        "DAY10_SUPERVISED_DATASET_PASS "
        f"samples={report['sample_count']} "
        f"frames={report['frame_count']} "
        f"instructions={report['compatible_instruction_count']} "
        f"labels={report['observed_label_count']}/"
        f"{report['required_label_count']} "
        f"coverage={coverage}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate_main())


