"""Build the complete Day 15 feature set with one frozen-model load."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from asv_vla.language_encoder import USVLanguageEncoder
from asv_vla.image_entity_perception import ImageEntityModel
from asv_vla.language_intervention_dataset import read_jsonl
from asv_vla.visual_encoder import BACKBONE_ID, FrozenMobileNetEncoder
from asv_training.dataset import load_split_assignments
from asv_training.feature_cache import (
    DEFAULT_IMAGE_PREPROCESS_BRIGHTNESS,
    DEFAULT_IMAGE_PREPROCESS_CONTRAST,
    DEFAULT_IMAGE_PREPROCESS_ENABLED,
    DEFAULT_IMAGE_PREPROCESS_GAMMA,
    IMAGE_PERCEPTION_TRACKER_CONFIG,
    ModelFingerprint,
    build_feature_cache,
    encode_language_instructions,
    hash_torch_module_state,
    hash_weight_tree,
    make_image_preprocess_config,
    make_language_provenance,
    validate_feature_cache,
)


FEATURE_SET_SCHEMA_VERSION = "feature_set_v1"
FROZEN_FEATURE_GIT_SHA = "2ea3f77c8cf7"
SUPPORTED_SPLITS = {
    2: {"train": 0, "validation": 0, "test": 2},
    8: {"train": 0, "validation": 0, "test": 8},
    12: {"train": 8, "validation": 2, "test": 2},
    # Sine formation: 9/2/3 keeps the test split colour-paired
    # (red-left and red-right geometry both represented).
    14: {"train": 9, "validation": 2, "test": 3},
    # Combined near red-view + blue-view: 16/4/4, stratified by view.
    24: {"train": 16, "validation": 4, "test": 4},
    30: {"train": 18, "validation": 6, "test": 6},
    # Near-standoff extension (red/blue 2.5 m + blue 3 m): 67 runs.
    67: {"train": 45, "validation": 11, "test": 11},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registry(
    path: Path, required_run_count: int
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        if bool(value.get("training_eligible")):
            entries.append(value)
    if len(entries) != required_run_count:
        raise ValueError(
            f"Day 15 requires {required_run_count} eligible Runs, "
            f"found {len(entries)}"
        )
    run_ids = [str(entry.get("run_id", "")).strip() for entry in entries]
    if any(not run_id for run_id in run_ids):
        raise ValueError("registry contains an empty Run ID")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("registry contains duplicate Run IDs")
    return sorted(entries, key=lambda entry: str(entry["run_id"]))


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_complete_feature_set(
    *,
    data_root: str | Path,
    registry_path: str | Path,
    split_path: str | Path,
    instructions_path: str | Path,
    output_root: str | Path,
    language_model_path: str | Path,
    image_model_path: str | Path | None = None,
    device: str = "cuda",
    frozen_git_sha: str = FROZEN_FEATURE_GIT_SHA,
    required_run_count: int = 12,
    image_preprocess_enabled: bool = DEFAULT_IMAGE_PREPROCESS_ENABLED,
    image_preprocess_gamma: float = DEFAULT_IMAGE_PREPROCESS_GAMMA,
    image_preprocess_brightness: float = DEFAULT_IMAGE_PREPROCESS_BRIGHTNESS,
    image_preprocess_contrast: float = DEFAULT_IMAGE_PREPROCESS_CONTRAST,
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    registry_source = Path(registry_path).expanduser().resolve()
    split_source = Path(split_path).expanduser().resolve()
    instructions_source = Path(instructions_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    language_model_source = Path(language_model_path).expanduser().resolve()
    image_model_source = (
        None
        if image_model_path is None
        else Path(image_model_path).expanduser().resolve()
    )
    image_preprocess = make_image_preprocess_config(
        enabled=image_preprocess_enabled,
        gamma=image_preprocess_gamma,
        brightness=image_preprocess_brightness,
        contrast=image_preprocess_contrast,
    )
    if str(frozen_git_sha).strip() != FROZEN_FEATURE_GIT_SHA:
        raise ValueError(
            f"Day 15 feature provenance is frozen to {FROZEN_FEATURE_GIT_SHA}"
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA feature build requested but CUDA is unavailable")

    expected_split = SUPPORTED_SPLITS.get(int(required_run_count))
    if expected_split is None:
        raise ValueError(
            "Day 15 required_run_count must be one of "
            f"{sorted(SUPPORTED_SPLITS)}"
        )
    entries = _load_registry(registry_source, int(required_run_count))
    assignments = load_split_assignments(split_source)
    run_ids = {str(entry["run_id"]) for entry in entries}
    if set(assignments) != run_ids:
        raise ValueError("registry and split Run IDs differ")
    split_counts = {
        split: sum(value == split for value in assignments.values())
        for split in ("train", "validation", "test")
    }
    if split_counts != expected_split:
        raise ValueError(
            f"Day 15 split for {required_run_count} Runs must be "
            f"{expected_split}, got {split_counts}"
        )

    instructions = read_jsonl(instructions_source)
    if not instructions:
        raise ValueError("feature-cache build requires a non-empty instruction dataset")
    language_weights_sha256 = hash_weight_tree(language_model_source)
    language_encoder = USVLanguageEncoder(
        str(language_model_source),
        device=device,
        cache_size=128,
    )
    language_embeddings = encode_language_instructions(
        instructions, language_encoder
    )
    del language_encoder
    _release_cuda()

    visual_encoder = FrozenMobileNetEncoder(device=device)
    visual_weights_sha256 = hash_torch_module_state(visual_encoder.backbone)
    language_fingerprint = ModelFingerprint(
        "Qwen/Qwen3-Embedding-0.6B",
        language_weights_sha256,
    )
    visual_fingerprint = ModelFingerprint(
        BACKBONE_ID,
        visual_weights_sha256,
    )
    image_model = (
        None
        if image_model_source is None
        else ImageEntityModel.load(image_model_source)
    )
    image_model_weights_sha256 = (
        ""
        if image_model_source is None
        else _sha256_file(image_model_source)
    )
    run_reports: list[dict[str, Any]] = []
    for entry in entries:
        run_id = str(entry["run_id"])
        episode = root / str(entry["episode_path"])
        supervision = root / str(entry["supervision_path"])
        result = build_feature_cache(
            episode,
            supervision,
            instructions_source,
            output,
            language_encoder=None,
            visual_encoder=visual_encoder,
            language_model=language_fingerprint,
            visual_model=visual_fingerprint,
            git_sha=FROZEN_FEATURE_GIT_SHA,
            precomputed_language_embeddings=language_embeddings,
            image_model=image_model,
            image_model_path=image_model_source,
            image_preprocess_enabled=image_preprocess.enabled,
            image_preprocess_gamma=image_preprocess.gamma,
            image_preprocess_brightness=image_preprocess.brightness,
            image_preprocess_contrast=image_preprocess.contrast,
        )
        validation = validate_feature_cache(output / run_id)
        run_reports.append(
            {
                "run_id": run_id,
                "split": assignments[run_id],
                "cached": bool(result["cached"]),
                "frame_count": int(validation["frame_count"]),
                "instruction_count": int(validation["instruction_count"]),
                "sample_count": int(validation["sample_count"]),
                "cache_key_sha256": str(validation["cache_key_sha256"]),
                "manifest_sha256": _sha256_file(
                    output / run_id / "manifest.json"
                ),
            }
        )

    total_frames = sum(item["frame_count"] for item in run_reports)
    total_samples = sum(item["sample_count"] for item in run_reports)
    # The near-standoff collection records 200 frames per run (static start +
    # approach + hold), so the expected total comes from the registry's
    # episode manifests rather than a fixed 100 frames per run.
    expected_frames = sum(
        len(
            [
                p
                for p in (root / str(entry["episode_path"]) / "frames").glob(
                    "*.json"
                )
            ]
        )
        for entry in entries
    )
    if total_frames != expected_frames:
        raise ValueError(
            f"expected {expected_frames} cached frames, got {total_frames}"
        )
    if total_samples <= 0:
        raise ValueError("Day 15 feature set has no samples")
    report = {
        "schema_version": FEATURE_SET_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "frozen_feature_git_sha": FROZEN_FEATURE_GIT_SHA,
        "run_count": len(run_reports),
        "split_counts": split_counts,
        "frame_count": total_frames,
        "sample_count": total_samples,
        "language_weights_sha256": language_weights_sha256,
        "language_provenance": make_language_provenance(
            instructions,
            embedding_table_source="precomputed_language_embeddings",
            frame_perception_enabled=image_model is not None,
        ),
        "visual_weights_sha256": visual_weights_sha256,
        "image_perception": {
            "enabled": image_model is not None,
            "model_id": "asv_vla.image_entity_perception",
            "model_version": (
                str(image_model.model_version)
                if image_model is not None
                else "disabled"
            ),
            "weights_sha256": image_model_weights_sha256,
            "tracker": (
                dict(IMAGE_PERCEPTION_TRACKER_CONFIG)
                if image_model is not None
                else None
            ),
        },
        "image_preprocess": image_preprocess.as_manifest(),
        "registry_sha256": _sha256_file(registry_source),
        "split_sha256": _sha256_file(split_source),
        "instructions_sha256": _sha256_file(instructions_source),
        "runs": run_reports,
    }
    report_path = output / "feature_set_manifest.json"
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build all frozen Day 15 feature caches"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--language-model-path", type=Path, required=True)
    parser.add_argument(
        "--image-model-path",
        type=Path,
        default=None,
        help="Optional image-only entity model; enables prediction-only policy entities",
    )
    preprocess = parser.add_mutually_exclusive_group()
    preprocess.add_argument(
        "--image-preprocess-enabled",
        dest="image_preprocess_enabled",
        action="store_true",
        help="apply the fixed UE5-compatible transform to decoded original JPEGs",
    )
    preprocess.add_argument(
        "--no-image-preprocess",
        dest="image_preprocess_enabled",
        action="store_false",
        help="keep legacy raw-JPEG cache behavior (default)",
    )
    parser.set_defaults(
        image_preprocess_enabled=DEFAULT_IMAGE_PREPROCESS_ENABLED
    )
    parser.add_argument(
        "--image-preprocess-gamma",
        type=float,
        default=DEFAULT_IMAGE_PREPROCESS_GAMMA,
    )
    parser.add_argument(
        "--image-preprocess-brightness",
        type=float,
        default=DEFAULT_IMAGE_PREPROCESS_BRIGHTNESS,
    )
    parser.add_argument(
        "--image-preprocess-contrast",
        type=float,
        default=DEFAULT_IMAGE_PREPROCESS_CONTRAST,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frozen-git-sha", default=FROZEN_FEATURE_GIT_SHA)
    parser.add_argument(
        "--required-run-count",
        type=int,
        choices=sorted(SUPPORTED_SPLITS),
        default=12,
    )
    args = parser.parse_args()
    report = build_complete_feature_set(
        data_root=args.data_root,
        registry_path=args.registry,
        split_path=args.split,
        instructions_path=args.instructions,
        output_root=args.output_root,
        language_model_path=args.language_model_path,
        image_model_path=args.image_model_path,
        device=args.device,
        frozen_git_sha=args.frozen_git_sha,
        required_run_count=args.required_run_count,
        image_preprocess_enabled=args.image_preprocess_enabled,
        image_preprocess_gamma=args.image_preprocess_gamma,
        image_preprocess_brightness=args.image_preprocess_brightness,
        image_preprocess_contrast=args.image_preprocess_contrast,
    )
    print(
        "FEATURE_SET_PASS "
        f"runs={report['run_count']} "
        f"frames={report['frame_count']} "
        f"samples={report['sample_count']} "
        f"split={report['split_counts']['train']}/"
        f"{report['split_counts']['validation']}/"
        f"{report['split_counts']['test']} "
        f"git_sha={report['frozen_feature_git_sha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
