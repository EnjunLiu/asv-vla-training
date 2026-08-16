"""Day 14 loader for immutable Day 13 feature caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from asv_training.feature_cache import (
    FRAME_SHARD_NAME,
    LANGUAGE_FILE_NAME,
    validate_feature_cache,
)


POLICY_INPUT_KEYS = frozenset(
    {
        "language",
        "entity_geometry",
        "previous_action",
        "language_valid",
        "entity_geometry_mask",
        "previous_action_valid",
        "policy_input_valid",
    }
)
FORBIDDEN_POLICY_FIELDS = frozenset(
    {
        "action",
        "target_attribute",
        "distance_bucket",
        "task_label",
        "color",
        "color_red",
        "color_blue",
        "entity_ids",
        "expert_selected_entity_ids",
    }
)
METADATA_KEYS = frozenset(
    {"run_id", "frame_key", "sample_id", "instruction_id"}
)
_HOLD_DISTANCE_SCALE_M = 20.0
_HOLD_DISTANCE_RE = re.compile(
    r"(?<![0-9])(?:2\.5|3|4|10)\s*(?:m|米)",
    re.IGNORECASE,
)
_HOLD_TARGET_PATTERNS = (
    (
        "target_red",
        re.compile(
            r"\btarget[_ -]?red\b|\bred\b|(?:红色?)(?:目标)?(?:船|艇)?",
            re.IGNORECASE,
        ),
    ),
    (
        "target_blue",
        re.compile(
            r"\btarget[_ -]?blue\b|\bblue\b|(?:蓝色?)(?:目标)?(?:船|艇)?",
            re.IGNORECASE,
        ),
    ),
    (
        "target_left",
        re.compile(
            r"\btarget[_ -]?left\b|\bleft\b|左(?:侧)?(?:目标)?(?:船|艇)?",
            re.IGNORECASE,
        ),
    ),
    (
        "target_right",
        re.compile(
            r"\btarget[_ -]?right\b|\bright\b|右(?:侧)?(?:目标)?(?:船|艇)?",
            re.IGNORECASE,
        ),
    ),
)
_HOLD_STOP_RE = re.compile(r"\bstop\b|停止|停船|中止|终止", re.IGNORECASE)


def task_target_id_from_instruction(instruction_text: str) -> str | None:
    """Map task language to a loader-only canonical target ID."""

    text = str(instruction_text)
    for target_id, pattern in _HOLD_TARGET_PATTERNS:
        if pattern.search(text):
            return target_id
    return None


def is_stop_instruction(instruction_text: str) -> bool:
    """Return whether task language requests a stop/no-entity input."""

    return bool(_HOLD_STOP_RE.search(str(instruction_text)))


def mask_task_conditioned_entity_geometry(
    geometry: np.ndarray,
    geometry_mask: np.ndarray,
    entity_ids: Sequence[Any],
    instruction_text: str,
    *,
    force_stop: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Match the online task-selective entity tensor at sample granularity.

    Entity IDs are used only to choose the retained row.  The returned arrays
    contain no ID or colour field and are safe to pass as policy inputs.  A
    follow sample with a missing or ambiguous target is invalid; STOP remains
    a valid all-false entity input.
    """

    values = np.asarray(geometry, dtype=np.float32)
    masks = np.asarray(geometry_mask, dtype=np.bool_).reshape(-1)
    if values.ndim != 2 or values.shape[0] != masks.shape[0]:
        raise ValueError(
            "entity geometry/mask must have shapes [N, D] and [N]"
        )
    if len(entity_ids) < values.shape[0]:
        raise ValueError("entity_ids has fewer entries than entity geometry")

    masked_values = np.zeros_like(values)
    masked_mask = np.zeros_like(masks)
    if force_stop or is_stop_instruction(instruction_text):
        return masked_values, masked_mask, True

    target_id = task_target_id_from_instruction(instruction_text)
    if target_id is None:
        return masked_values, masked_mask, False
    matches = [
        slot
        for slot in range(values.shape[0])
        if bool(masks[slot])
        and str(entity_ids[slot]).strip().casefold() == target_id
    ]
    if len(matches) != 1 or not np.all(np.isfinite(values[matches[0]])):
        return masked_values, masked_mask, False

    # Compact to slot 0 to mirror task_entity_tensor's target-first output;
    # attention still sees exactly one active row either way.
    masked_values[0] = values[matches[0]]
    masked_mask[0] = True
    return masked_values, masked_mask, True


@dataclass(frozen=True)
class _SampleRef:
    cache_index: int
    sample_row: int


@dataclass(frozen=True)
class _RunCache:
    run_id: str
    instruction_ids: np.ndarray
    instruction_texts: np.ndarray
    entity_ids: np.ndarray
    language_splits: np.ndarray
    language: np.ndarray
    frame_indices: np.ndarray
    frame_keys: np.ndarray
    # These arrays are perception audit evidence only. They are deliberately
    # absent from policy_inputs_from_batch().
    global_visual: np.ndarray
    global_visual_mask: np.ndarray
    entity_visual: np.ndarray
    entity_visual_mask: np.ndarray
    entity_geometry: np.ndarray
    entity_geometry_mask: np.ndarray
    ego: np.ndarray
    ego_valid: np.ndarray
    policy_input_valid: np.ndarray
    sample_ids: np.ndarray
    sample_frame_rows: np.ndarray
    sample_instruction_rows: np.ndarray
    target_actions: np.ndarray
    target_safe_stop: np.ndarray
    previous_expert_actions: np.ndarray
    previous_action_valid: np.ndarray


def _normalize_hold_band_m(value: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "hold_band_m must be finite and non-negative"
        ) from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError("hold_band_m must be finite and non-negative")
    return normalized


def _normalize_hold_oversample_factor(value: int) -> int:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "hold_oversample_factor must be finite, positive, and integral"
        ) from exc
    if (
        not np.isfinite(normalized)
        or normalized <= 0.0
        or not normalized.is_integer()
    ):
        raise ValueError(
            "hold_oversample_factor must be finite, positive, and integral"
        )
    return int(normalized)


def _parse_hold_target_and_distance(
    instruction_text: str,
) -> tuple[str, float] | None:
    text = str(instruction_text)
    target_id: str | None = None
    for candidate, pattern in _HOLD_TARGET_PATTERNS:
        if pattern.search(text):
            target_id = candidate
            break
    distance_match = _HOLD_DISTANCE_RE.search(text)
    if target_id is None or distance_match is None:
        return None
    numeric_match = re.match(r"[0-9]+(?:\.[0-9]+)?", distance_match.group(0))
    if numeric_match is None:
        return None
    distance_m = float(numeric_match.group(0))
    if not np.isfinite(distance_m):
        return None
    return target_id, distance_m


def _hold_repeat_for_sample(
    cache: _RunCache,
    sample_row: int,
    *,
    hold_band_m: float,
    hold_oversample_factor: int,
) -> int:
    if hold_band_m <= 0.0 or hold_oversample_factor <= 1:
        return 1
    instruction_row = int(cache.sample_instruction_rows[sample_row])
    instruction_text = str(cache.instruction_texts[instruction_row])
    if bool(cache.target_safe_stop[sample_row]) or _HOLD_STOP_RE.search(
        instruction_text
    ):
        return 1
    parsed = _parse_hold_target_and_distance(instruction_text)
    if parsed is None:
        return 1
    target_id, desired_distance_m = parsed
    frame_row = int(cache.sample_frame_rows[sample_row])
    try:
        entity_ids = cache.entity_ids[frame_row]
        geometry = cache.entity_geometry[frame_row]
        geometry_mask = cache.entity_geometry_mask[frame_row]
        for slot, entity_id in enumerate(entity_ids):
            if slot >= len(geometry) or not bool(geometry_mask[slot]):
                continue
            if str(entity_id).strip().casefold() != target_id:
                continue
            xy = np.asarray(geometry[slot][:2], dtype=np.float64)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                return 1
            current_distance_m = float(
                np.linalg.norm(xy * _HOLD_DISTANCE_SCALE_M)
            )
            if abs(current_distance_m - desired_distance_m) <= hold_band_m:
                return hold_oversample_factor
            return 1
    except (IndexError, TypeError, ValueError):
        return 1
    return 1


@dataclass(frozen=True)
class InstructionMetadata:
    instruction_id: str
    intent_group: str
    action: str
    target_attribute: str
    distance_bucket: str
    split: str

    @property
    def task_label(self) -> str:
        return (
            f"{self.action}|{self.target_attribute}|{self.distance_bucket}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_split_assignments(path: str | Path) -> dict[str, str]:
    manifest = _read_json(Path(path).expanduser().resolve())
    assignments = manifest.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("split manifest assignments must be an object")
    normalized: dict[str, str] = {}
    for run_id, split in assignments.items():
        current_id = str(run_id).strip()
        current_split = str(split).strip().casefold()
        if not current_id:
            raise ValueError("split manifest contains an empty Run ID")
        if current_split not in {"train", "validation", "test"}:
            raise ValueError(
                f"run_id={current_id}: invalid split {current_split!r}"
            )
        normalized[current_id] = current_split
    return normalized


def discover_feature_caches(root: str | Path) -> list[Path]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"feature root does not exist: {base}")
    caches = sorted(
        path.parent
        for path in base.glob("*/manifest.json")
        if (path.parent / LANGUAGE_FILE_NAME).is_file()
        and (path.parent / FRAME_SHARD_NAME).is_file()
    )
    if not caches:
        raise ValueError(f"no feature caches found under {base}")
    return caches


def load_instruction_metadata(
    path: str | Path,
) -> dict[str, InstructionMetadata]:
    source = Path(path).expanduser().resolve()
    output: dict[str, InstructionMetadata] = {}
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{source}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number}: expected an object")
        instruction_id = str(value.get("instruction_id", "")).strip()
        metadata = InstructionMetadata(
            instruction_id=instruction_id,
            intent_group=str(value.get("intent_group", "")).strip(),
            action=str(value.get("action", "")).strip(),
            target_attribute=str(value.get("target_attribute", "")).strip(),
            distance_bucket=str(value.get("distance_bucket", "")).strip(),
            split=str(value.get("split", "")).strip().casefold(),
        )
        fields = (
            metadata.instruction_id,
            metadata.intent_group,
            metadata.action,
            metadata.target_attribute,
            metadata.distance_bucket,
        )
        if any(not field for field in fields):
            raise ValueError(
                f"{source}:{line_number}: incomplete instruction metadata"
            )
        if metadata.split not in {"train", "validation", "test"}:
            raise ValueError(
                f"{source}:{line_number}: invalid split={metadata.split!r}"
            )
        if instruction_id in output:
            raise ValueError(f"duplicate instruction ID: {instruction_id}")
        output[instruction_id] = metadata
    if not output:
        raise ValueError("instruction dataset must contain at least one instruction")
    return output


def _validate_frame_key(value: str, run_id: str) -> None:
    parts = value.rsplit(":", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"incomplete frame key: {value!r}")
    if parts[0] != run_id:
        raise ValueError(
            f"frame key Run ID {parts[0]!r} does not match {run_id!r}"
        )
    try:
        scene_seed, frame_index, stamp_us = map(int, parts[1:])
    except ValueError as exc:
        raise ValueError(f"frame key contains a non-integer field: {value!r}") from exc
    if scene_seed < 0 or frame_index < 0 or stamp_us < 0:
        raise ValueError(f"frame key contains a negative field: {value!r}")


def _load_cache(path: Path) -> _RunCache:
    validate_feature_cache(path)
    manifest = _read_json(path / "manifest.json")
    run_id = str(manifest.get("run_id", "")).strip()
    if not run_id:
        raise ValueError(f"{path}: manifest has no Run ID")
    try:
        with np.load(path / LANGUAGE_FILE_NAME, allow_pickle=False) as source:
            instruction_ids = np.asarray(source["instruction_ids"]).copy()
            instruction_texts = np.asarray(source["instruction_texts"]).copy()
            language_splits = np.asarray(source["language_splits"]).copy()
            language = np.asarray(source["embeddings"], dtype=np.float32).copy()
        with np.load(path / FRAME_SHARD_NAME, allow_pickle=False) as source:
            arrays = {
                name: np.asarray(source[name]).copy()
                for name in (
                    "frame_indices",
                    "frame_keys",
                    "global_visual",
                    "global_visual_mask",
                    "entity_visual",
                    "entity_visual_mask",
                    "entity_features",
                    "entity_ids",
                    "entity_mask",
                    "ego",
                    "ego_valid",
                    "policy_input_valid",
                    "sample_ids",
                    "sample_frame_rows",
                    "sample_instruction_rows",
                    "expert_actions",
                    "expert_safe_stop",
                    "previous_expert_actions",
                    "previous_action_valid",
                )
            }
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"cannot load feature cache {path}: {exc}") from exc

    for frame_key in arrays["frame_keys"]:
        _validate_frame_key(str(frame_key), run_id)
    if len(language_splits) != len(instruction_ids):
        raise ValueError(f"{path}: language split and ID counts differ")
    if any(
        str(split).casefold() not in {"train", "validation", "test"}
        for split in language_splits
    ):
        raise ValueError(f"{path}: invalid language template split")

    return _RunCache(
        run_id=run_id,
        instruction_ids=instruction_ids,
        instruction_texts=instruction_texts,
        entity_ids=np.asarray(arrays["entity_ids"], dtype=object),
        language_splits=language_splits,
        language=language,
        frame_indices=arrays["frame_indices"],
        frame_keys=arrays["frame_keys"],
        global_visual=np.asarray(arrays["global_visual"], dtype=np.float32),
        global_visual_mask=np.asarray(
            arrays["global_visual_mask"], dtype=np.bool_
        ),
        entity_visual=np.asarray(arrays["entity_visual"], dtype=np.float32),
        entity_visual_mask=np.asarray(
            arrays["entity_visual_mask"], dtype=np.bool_
        ),
        entity_geometry=np.asarray(
            arrays["entity_features"], dtype=np.float32
        ),
        entity_geometry_mask=np.asarray(arrays["entity_mask"], dtype=np.bool_),
        ego=np.asarray(arrays["ego"], dtype=np.float32),
        ego_valid=np.asarray(arrays["ego_valid"], dtype=np.bool_),
        policy_input_valid=np.asarray(
            arrays["policy_input_valid"], dtype=np.bool_
        ),
        sample_ids=arrays["sample_ids"],
        sample_frame_rows=np.asarray(
            arrays["sample_frame_rows"], dtype=np.int64
        ),
        sample_instruction_rows=np.asarray(
            arrays["sample_instruction_rows"], dtype=np.int64
        ),
        target_actions=np.asarray(
            arrays["expert_actions"], dtype=np.float32
        ),
        target_safe_stop=np.asarray(
            arrays["expert_safe_stop"], dtype=np.bool_
        ),
        previous_expert_actions=np.asarray(
            arrays["previous_expert_actions"], dtype=np.float32
        ),
        previous_action_valid=np.asarray(
            arrays["previous_action_valid"], dtype=np.bool_
        ),
    )


class FrozenFeatureDataset(Dataset[dict[str, Tensor | str]]):
    """Expose only permitted policy inputs, expert targets, and audit metadata."""

    def __init__(
        self,
        cache_dirs: Sequence[str | Path],
        *,
        selected_split: str | None = None,
        split_assignments: Mapping[str, str] | None = None,
        allowed_language_splits: Iterable[str] | None = None,
        frame_stride: int = 1,
        require_valid: bool = True,
        augment: bool = False,
        geometry_noise_std: float = 0.02,
        slot_dropout_prob: float = 0.1,
        mirror_prob: float = 0.0,
        instruction_swap_prob: float = 0.0,
        runtime_first_step_limit_m: float | None = None,
        hold_band_m: float = 0.0,
        hold_oversample_factor: int = 1,
    ) -> None:
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if geometry_noise_std < 0.0 or slot_dropout_prob < 0.0:
            raise ValueError("augmentation strengths must be non-negative")
        if not 0.0 <= mirror_prob <= 1.0:
            raise ValueError("mirror_prob must be in [0, 1]")
        if not 0.0 <= instruction_swap_prob <= 1.0:
            raise ValueError("instruction_swap_prob must be in [0, 1]")
        normalized_hold_band = _normalize_hold_band_m(hold_band_m)
        normalized_hold_factor = _normalize_hold_oversample_factor(
            hold_oversample_factor
        )
        if runtime_first_step_limit_m is None:
            normalized_runtime_limit = None
        else:
            try:
                normalized_runtime_limit = float(runtime_first_step_limit_m)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "runtime_first_step_limit_m must be finite and positive"
                ) from exc
            if (
                not np.isfinite(normalized_runtime_limit)
                or normalized_runtime_limit <= 0.0
            ):
                raise ValueError(
                    "runtime_first_step_limit_m must be finite and positive"
                )
        self._augment = augment
        self._geometry_noise_std = float(geometry_noise_std)
        self._slot_dropout_prob = float(slot_dropout_prob)
        self._mirror_prob = float(mirror_prob)
        self._instruction_swap_prob = float(instruction_swap_prob)
        self._runtime_first_step_limit_m = normalized_runtime_limit
        self._hold_band_m = normalized_hold_band
        self._hold_oversample_factor = normalized_hold_factor
        normalized_split = (
            str(selected_split).strip().casefold()
            if selected_split is not None
            else None
        )
        if normalized_split not in {None, "train", "validation", "test"}:
            raise ValueError(f"invalid selected_split={selected_split!r}")
        if normalized_split is not None and split_assignments is None:
            raise ValueError(
                "selected_split requires explicit Run-level split assignments"
            )
        allowed = (
            {str(value).strip().casefold() for value in allowed_language_splits}
            if allowed_language_splits is not None
            else None
        )
        if allowed is not None and (
            not allowed or not allowed <= {"train", "validation", "test"}
        ):
            raise ValueError(f"invalid allowed_language_splits={sorted(allowed)}")

        self._caches: list[_RunCache] = []
        self._samples: list[_SampleRef] = []
        seen_run_ids: set[str] = set()
        for candidate in sorted(Path(path).resolve() for path in cache_dirs):
            cache = _load_cache(candidate)
            if cache.run_id in seen_run_ids:
                raise ValueError(f"duplicate feature cache Run ID: {cache.run_id}")
            seen_run_ids.add(cache.run_id)
            if split_assignments is not None:
                assigned = split_assignments.get(cache.run_id)
                if assigned is None:
                    raise ValueError(
                        f"Run ID {cache.run_id} has no split assignment"
                    )
                if normalized_split is not None and assigned != normalized_split:
                    continue

            cache_index = len(self._caches)
            self._caches.append(cache)
            first_frame_index = int(np.min(cache.frame_indices))
            for sample_row, (frame_row, instruction_row) in enumerate(
                zip(
                    cache.sample_frame_rows,
                    cache.sample_instruction_rows,
                )
            ):
                frame_row_int = int(frame_row)
                instruction_row_int = int(instruction_row)
                if require_valid and not bool(
                    cache.policy_input_valid[frame_row_int]
                ):
                    continue
                if require_valid:
                    _, _, task_input_valid = (
                        mask_task_conditioned_entity_geometry(
                            cache.entity_geometry[frame_row_int],
                            cache.entity_geometry_mask[frame_row_int],
                            cache.entity_ids[frame_row_int],
                            str(cache.instruction_texts[instruction_row_int]),
                            force_stop=bool(cache.target_safe_stop[sample_row]),
                        )
                    )
                    if not task_input_valid:
                        continue
                if (
                    int(cache.frame_indices[frame_row_int]) - first_frame_index
                ) % frame_stride:
                    continue
                language_split = str(
                    cache.language_splits[instruction_row_int]
                ).casefold()
                if allowed is not None and language_split not in allowed:
                    continue
                if require_valid:
                    _, _, task_input_valid = (
                        mask_task_conditioned_entity_geometry(
                            cache.entity_geometry[frame_row_int],
                            cache.entity_geometry_mask[frame_row_int],
                            cache.entity_ids[frame_row_int],
                            str(cache.instruction_texts[instruction_row_int]),
                            force_stop=bool(
                                cache.target_safe_stop[sample_row]
                            ),
                        )
                    )
                    if not task_input_valid:
                        continue
                self._samples.append(_SampleRef(cache_index, sample_row))

        self._raw_sample_count = len(self._samples)
        if normalized_split in {None, "train"}:
            resampled: list[_SampleRef] = []
            for reference in self._samples:
                resampled.append(reference)
                cache = self._caches[reference.cache_index]
                repeat = _hold_repeat_for_sample(
                    cache,
                    reference.sample_row,
                    hold_band_m=self._hold_band_m,
                    hold_oversample_factor=self._hold_oversample_factor,
                )
                resampled.extend([reference] * (repeat - 1))
            self._samples = resampled

        # Instruction-swap support: build red/blue follow-row maps per cache
        # so __getitem__ can swap follow-red <-> follow-blue and regenerate
        # the expert label, teaching the policy to follow the commanded
        # colour even when it is not the nearest target.
        self._swap_rows: dict[int, tuple[list[int], list[int]]] = {}
        if self._augment and self._instruction_swap_prob > 0.0:
            for cache_index, cache in enumerate(self._caches):
                red_rows: list[int] = []
                blue_rows: list[int] = []
                for row, text in enumerate(cache.instruction_texts):
                    lowered = str(text).casefold()
                    if "红" in str(text):
                        red_rows.append(row)
                    elif "蓝" in str(text):
                        blue_rows.append(row)
                if red_rows and blue_rows:
                    self._swap_rows[cache_index] = (red_rows, blue_rows)

        if not self._caches:
            raise ValueError("no feature cache matches the selected Run split")
        if not self._samples:
            raise ValueError("no feature-cache samples match the loader filters")

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def raw_sample_count(self) -> int:
        return self._raw_sample_count

    @property
    def hold_band_m(self) -> float:
        return self._hold_band_m

    @property
    def hold_oversample_factor(self) -> int:
        return self._hold_oversample_factor

    def sample_metadata(self, index: int) -> dict[str, str | bool]:
        reference = self._samples[index]
        cache = self._caches[reference.cache_index]
        sample_row = reference.sample_row
        frame_row = int(cache.sample_frame_rows[sample_row])
        instruction_row = int(cache.sample_instruction_rows[sample_row])
        return {
            "run_id": cache.run_id,
            "frame_key": str(cache.frame_keys[frame_row]),
            "sample_id": str(cache.sample_ids[sample_row]),
            "instruction_id": str(cache.instruction_ids[instruction_row]),
            "target_stop": bool(cache.target_safe_stop[sample_row]),
        }

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        reference = self._samples[index]
        cache = self._caches[reference.cache_index]
        sample_row = reference.sample_row
        frame_row = int(cache.sample_frame_rows[sample_row])
        instruction_row = int(cache.sample_instruction_rows[sample_row])

        geometry = cache.entity_geometry[frame_row].copy()
        geometry_mask = cache.entity_geometry_mask[frame_row].copy()
        action = cache.target_actions[sample_row].copy()
        safe_stop = cache.target_safe_stop[sample_row].copy()
        previous_action = cache.previous_expert_actions[sample_row].copy()
        previous_action_valid = bool(cache.previous_action_valid[sample_row])
        language = cache.language[instruction_row].copy()

        if self._augment and self._instruction_swap_prob > 0.0:
            # Instruction swap: swap follow-red <-> follow-blue and
            # regenerate the expert label for the new instruction.  The
            # training set is red-near biased (~78% of follow-red frames),
            # so swapped samples teach the policy to follow the commanded
            # colour even when it is NOT the nearest target.
            swap_rows = self._swap_rows.get(reference.cache_index)
            if swap_rows is not None and not bool(safe_stop):
                red_rows, blue_rows = swap_rows
                rng = np.random.default_rng(
                    int(reference.sample_row) * 104729 + int(frame_row) * 19531
                )
                text = str(cache.instruction_texts[instruction_row])
                if "红" in str(text) and blue_rows and rng.random() < self._instruction_swap_prob:
                    new_rows = blue_rows
                    commanded = "blue"
                elif "蓝" in str(text) and red_rows and rng.random() < self._instruction_swap_prob:
                    new_rows = red_rows
                    commanded = "red"
                else:
                    new_rows = None
                    commanded = None
                if new_rows is not None:
                    new_row = int(new_rows[rng.integers(len(new_rows))])
                    language = cache.language[new_row].copy()
                    action, safe_stop = _regenerate_label(
                        cache, frame_row, geometry, geometry_mask,
                        commanded, text,
                    )
                    if previous_action_valid:
                        previous_frame_row = _previous_frame_row(cache, frame_row)
                        if previous_frame_row is None:
                            previous_action = np.zeros(2, dtype=np.float32)
                            previous_action_valid = False
                        else:
                            previous_action, previous_safe_stop = _regenerate_label(
                                cache,
                                previous_frame_row,
                                cache.entity_geometry[previous_frame_row],
                                cache.entity_geometry_mask[previous_frame_row],
                                commanded,
                                text,
                            )
                            if bool(previous_safe_stop):
                                previous_action = np.zeros(2, dtype=np.float32)
                                previous_action_valid = False

                    instruction_row = new_row

        instruction_text = str(cache.instruction_texts[instruction_row])
        geometry, geometry_mask, task_input_valid = (
            mask_task_conditioned_entity_geometry(
                geometry,
                geometry_mask,
                cache.entity_ids[frame_row],
                instruction_text,
                force_stop=bool(safe_stop),
            )
        )
        policy_input_valid = bool(cache.policy_input_valid[frame_row]) and (
            task_input_valid
        )

        if self._augment and self._mirror_prob > 0.0:
            # Geometric mirroring: negate the lateral axis (positions,
            # velocities, derived lateral columns) and mirror the expert
            # trajectory so the "follow red/blue" instruction is decoupled
            # from which side the commanded boat is on.  The training set
            # is red-near biased (~78%); mirroring halves that bias and
            # makes the model follow the commanded colour regardless of
            # which target is nearest.
            rng = np.random.default_rng(
                int(reference.sample_row) * 7919 + int(frame_row) * 104729
            )
            if rng.random() < self._mirror_prob:
                # Columns: 1=y, 4=vy, 7=bearing_sin, 9=closing_speed,
                # 11=cpa_distance keep signs consistent via explicit negate.
                mirrored = geometry.copy()
                for slot in range(geometry.shape[0]):
                    if not geometry_mask[slot]:
                        continue
                    mirrored[slot, 1] = -mirrored[slot, 1]
                    mirrored[slot, 4] = -mirrored[slot, 4]
                    mirrored[slot, 7] = -mirrored[slot, 7]
                geometry = mirrored
                action[1] = -action[1]
                if previous_action_valid:
                    previous_action[1] = -previous_action[1]

        if self._augment:
            # Training-time augmentation targeting online robustness:
            # 1) small absolute noise on the geometry tensor (position/velocity
            #    columns are normalised; 0.02 is ~0.4 m / ~0.1 m/s);
            # 2) random slot dropout, zeroing structured geometry of some entities
            #    to mimic occlusion or off-screen slots at inference time.
            rng = np.random.default_rng(
                int(reference.sample_row) + int(frame_row) * 1000003
            )
            noise = rng.normal(
                0.0, self._geometry_noise_std, size=geometry.shape
            ).astype(np.float32)
            geometry = geometry + noise * (
                np.asarray(geometry_mask, dtype=np.float32)[..., None]
            )
            drop = (
                rng.random(geometry.shape[0]) < self._slot_dropout_prob
            )
            drop = drop & np.asarray(geometry_mask, dtype=bool)
            if np.any(drop):
                geometry[drop] = 0.0
                geometry_mask[drop] = False
            if not bool(safe_stop) and not bool(np.any(geometry_mask)):
                policy_input_valid = False

        if self._runtime_first_step_limit_m is not None:
            action = np.clip(
                action,
                -self._runtime_first_step_limit_m,
                self._runtime_first_step_limit_m,
            )

        return {
            "language": torch.from_numpy(language),
            "entity_geometry": torch.from_numpy(geometry),
            "language_valid": torch.tensor(True, dtype=torch.bool),
            "entity_geometry_mask": torch.from_numpy(geometry_mask),
            "previous_action": torch.from_numpy(previous_action),
            "previous_action_valid": torch.tensor(
                previous_action_valid, dtype=torch.bool
            ),
            "policy_input_valid": torch.tensor(
                policy_input_valid, dtype=torch.bool
            ),
            "target_action": torch.from_numpy(action),
            "target_stop": torch.tensor(
                [float(safe_stop)],
                dtype=torch.float32,
            ),
            "run_id": cache.run_id,
            "frame_key": str(cache.frame_keys[frame_row]),
            "sample_id": str(cache.sample_ids[sample_row]),
            "instruction_id": str(cache.instruction_ids[instruction_row]),
        }


def _previous_frame_row(cache: _RunCache, frame_row: int) -> int | None:
    """Return the immediately preceding frame row, enforcing adjacency."""

    if frame_row <= 0 or frame_row >= len(cache.frame_indices):
        return None
    current_index = int(cache.frame_indices[frame_row])
    previous_index = int(cache.frame_indices[frame_row - 1])
    if previous_index != current_index - 1:
        return None
    return frame_row - 1


def _regenerate_label(
    cache: "_RunCache",
    frame_row: int,
    geometry: np.ndarray,
    geometry_mask: np.ndarray,
    commanded: str,
    original_text: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate the expert label for a swapped instruction.

    Reconstructs the visible entities from the cached geometry tensor and
    calls the deterministic expert generator with the new commanded colour,
    so the swapped sample is a consistent (instruction, label) pair.
    """
    from asv_vla.expert_trajectory import (
        ExpertTask,
        generate_expert_trajectory,
    )
    from types import SimpleNamespace

    distance_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*米", str(original_text))
    desired = float(distance_match.group(1)) if distance_match else 3.0
    distance = f"{desired:g}m" if distance_match else "3m"
    entities: list[Any] = []
    ids = cache.entity_ids[frame_row]
    for slot in range(min(len(ids), 16)):
        if not geometry_mask[slot] or not ids[slot]:
            continue
        row = geometry[slot]
        entity_id = str(ids[slot])
        entities.append(
            SimpleNamespace(
                entity_id=entity_id,
                relative_x=float(row[0]) * 20.0,
                relative_y=float(row[1]) * 20.0,
                relative_velocity_x=float(row[3]) * 5.0,
                relative_velocity_y=float(row[4]) * 5.0,
                valid=True,
                visible=True,
                is_target=True,
                color="red" if "red" in entity_id else "blue",
            )
        )
    task = ExpertTask(
        action="follow",
        target_attribute=f"color:{commanded}",
        desired_distance_m=desired,
    )
    result = generate_expert_trajectory(task, entities)
    action = np.asarray(result.expert_action, dtype=np.float32)
    if action.shape != (2,):
        raise ValueError(
            f"expert action shape {action.shape} is invalid; expected (2,)"
        )
    safe_stop = np.asarray([float(result.safe_stop)], dtype=np.float32)
    return action, safe_stop


def _annotate_item(
    item: dict[str, Tensor | str],
    metadata: InstructionMetadata,
) -> dict[str, Any]:
    output: dict[str, Any] = dict(item)
    output["metadata"] = {
        "task_label": metadata.task_label,
        "intent_group": metadata.intent_group,
        "action": metadata.action,
        "target_attribute": metadata.target_attribute,
        "distance_bucket": metadata.distance_bucket,
        "language_split": metadata.split,
    }
    return output


class AnnotatedFeatureDataset(Dataset[dict[str, Any]]):
    """Add evaluation-only labels under a nested metadata object."""

    def __init__(
        self,
        base: FrozenFeatureDataset,
        instructions: Mapping[str, InstructionMetadata],
    ) -> None:
        self.base = base
        self.instructions = dict(instructions)
        for index in range(len(base)):
            instruction_id = str(base.sample_metadata(index)["instruction_id"])
            if instruction_id not in self.instructions:
                raise ValueError(
                    f"missing metadata for instruction_id={instruction_id!r}"
                )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        instruction_id = str(item["instruction_id"])
        return _annotate_item(item, self.instructions[instruction_id])


class EpochSynonymDataset(Dataset[dict[str, Any]]):
    """Choose one deterministic synonym per frame/task label each epoch."""

    def __init__(
        self,
        base: FrozenFeatureDataset,
        instructions: Mapping[str, InstructionMetadata],
        *,
        seed: int,
    ) -> None:
        self.base = base
        self.instructions = dict(instructions)
        groups: dict[tuple[str, str, str], list[int]] = {}
        for index in range(len(base)):
            sample = base.sample_metadata(index)
            instruction_id = str(sample["instruction_id"])
            metadata = self.instructions.get(instruction_id)
            if metadata is None:
                raise ValueError(
                    f"missing metadata for instruction_id={instruction_id!r}"
                )
            key = (
                str(sample["run_id"]),
                str(sample["frame_key"]),
                metadata.task_label,
            )
            groups.setdefault(key, []).append(index)
        if not groups:
            raise ValueError("synonym dataset has no frame/task groups")
        self.seed = int(seed)
        self._groups = tuple(
            (key, tuple(indices)) for key, indices in sorted(groups.items())
        )
        frame_groups: list[list[int]] = []
        previous_frame: tuple[str, str] | None = None
        for group_index, (key, _) in enumerate(self._groups):
            current_frame = (key[0], key[1])
            if current_frame != previous_frame:
                frame_groups.append([])
                previous_frame = current_frame
            frame_groups[-1].append(group_index)
        self.frame_group_indices = tuple(
            tuple(indices) for indices in frame_groups
        )
        # Cross-run pair groups: group by (frame_index, task_label) across runs.
        pair_groups: dict[tuple[int, str], list[int]] = {}
        for group_index, (key, _) in enumerate(self._groups):
            frame_key = key[1]
            task_label = key[2]
            frame_index = frame_key.split(":")[2]
            pair_key = (int(frame_index), task_label)
            pair_groups.setdefault(pair_key, []).append(group_index)
        self.cross_run_pair_indices = tuple(
            tuple(indices) for indices in pair_groups.values()
            if len(indices) >= 2
        )
        self._selected: list[int] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        selected: list[int] = []
        for key, candidates in self._groups:
            payload = (
                f"{self.seed}:{int(epoch)}:{key[0]}:{key[1]}:{key[2]}"
            ).encode("utf-8")
            choice = int.from_bytes(
                hashlib.sha256(payload).digest()[:8], "big"
            ) % len(candidates)
            selected.append(candidates[choice])
        self._selected = selected

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index = self._selected[index]
        item = self.base[base_index]
        instruction_id = str(item["instruction_id"])
        return _annotate_item(item, self.instructions[instruction_id])


def policy_inputs_from_batch(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    missing = POLICY_INPUT_KEYS - set(batch)
    if missing:
        raise ValueError(f"policy batch is missing keys: {sorted(missing)}")
    forbidden = FORBIDDEN_POLICY_FIELDS & set(batch)
    if forbidden:
        raise ValueError(f"policy batch contains privileged fields: {sorted(forbidden)}")
    return {key: batch[key] for key in POLICY_INPUT_KEYS}
