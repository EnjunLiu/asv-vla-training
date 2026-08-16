from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable


MIN_INSTRUCTIONS = 80
MIN_CONTRAST_PAIRS = 20
REQUIRED_SPLITS = {"train", "validation", "test"}
REQUIRED_INTERVENTIONS = {
    "target_color",
    "target_bearing",
    "distance",
    "action",
}


class LanguageDatasetError(ValueError):
    """Raised when language intervention data violates its contract."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LanguageDatasetError(f"cannot read {source}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LanguageDatasetError(
                f"{source}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise LanguageDatasetError(
                f"{source}:{line_number}: record must be a JSON object"
            )
        records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    )
    destination.write_text(text + "\n", encoding="utf-8")


def default_dataset_dir() -> Path:
    workspace_candidate = Path.cwd() / "dataset" / "language"
    if workspace_candidate.is_dir():
        return workspace_candidate
    return Path(__file__).resolve().parents[3] / "dataset" / "language"


def _instruction_label(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("intent_group", "")),
        str(record.get("target_attribute", "")),
        str(record.get("distance_bucket", "")),
    )


def _require_text(
    record: dict[str, Any],
    field: str,
    context: str,
    errors: list[str],
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context}: {field} must be a non-empty string")
        return ""
    return value


def _pair_has_expected_difference(
    intervention_type: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    if intervention_type == "target_color":
        attributes = {
            str(left.get("target_attribute", "")),
            str(right.get("target_attribute", "")),
        }
        return attributes == {"color:red", "color:blue"}
    if intervention_type == "target_bearing":
        attributes = {
            str(left.get("target_attribute", "")),
            str(right.get("target_attribute", "")),
        }
        return attributes == {"bearing:left", "bearing:right"}
    if intervention_type == "distance":
        distances = {
            str(left.get("distance_bucket", "")),
            str(right.get("distance_bucket", "")),
        }
        return (
            len(distances) == 2
            and distances <= {"3m", "4m", "10m"}
        )
    if intervention_type == "action":
        actions = {
            str(left.get("action", "")),
            str(right.get("action", "")),
        }
        return actions == {"follow", "stop"}
    return False


def validate_language_dataset(
    instructions: list[dict[str, Any]],
    contrast_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if len(instructions) < MIN_INSTRUCTIONS:
        errors.append(
            f"instruction_count={len(instructions)} is below {MIN_INSTRUCTIONS}"
        )
    if len(contrast_pairs) < MIN_CONTRAST_PAIRS:
        errors.append(
            f"contrast_pair_count={len(contrast_pairs)} is below "
            f"{MIN_CONTRAST_PAIRS}"
        )

    instructions_by_id: dict[str, dict[str, Any]] = {}
    intent_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    template_families: dict[str, set[str]] = defaultdict(set)

    for index, record in enumerate(instructions):
        context = f"instruction[{index}]"
        instruction_id = _require_text(
            record, "instruction_id", context, errors
        )
        _require_text(record, "text", context, errors)
        intent_group = _require_text(
            record, "intent_group", context, errors
        )
        _require_text(record, "action", context, errors)
        _require_text(record, "target_attribute", context, errors)
        _require_text(record, "distance_bucket", context, errors)
        split = _require_text(record, "split", context, errors)
        template_family = _require_text(
            record, "template_family", context, errors
        )

        if instruction_id:
            if instruction_id in instructions_by_id:
                errors.append(f"duplicate instruction_id={instruction_id}")
            else:
                instructions_by_id[instruction_id] = record
        if intent_group:
            intent_counts[intent_group] += 1
        if split:
            split_counts[split] += 1
        if split and template_family:
            template_families[split].add(template_family)

    missing_splits = REQUIRED_SPLITS - set(split_counts)
    if missing_splits:
        errors.append(f"missing splits: {sorted(missing_splits)}")

    for intent_group, count in sorted(intent_counts.items()):
        if not 8 <= count <= 12:
            errors.append(
                f"intent_group={intent_group} has {count} instructions; "
                "expected 8 to 12"
            )

    for left_split in sorted(REQUIRED_SPLITS):
        for right_split in sorted(REQUIRED_SPLITS):
            if left_split >= right_split:
                continue
            overlap = (
                template_families[left_split]
                & template_families[right_split]
            )
            if overlap:
                errors.append(
                    f"template families overlap between {left_split} and "
                    f"{right_split}: {sorted(overlap)}"
                )

    pair_ids: set[str] = set()
    intervention_counts: Counter[str] = Counter()
    scene_labels: dict[int, set[tuple[str, str, str]]] = defaultdict(set)

    for index, pair in enumerate(contrast_pairs):
        context = f"contrast_pair[{index}]"
        pair_id = _require_text(pair, "pair_id", context, errors)
        intervention_type = _require_text(
            pair, "intervention_type", context, errors
        )
        pair_split = _require_text(pair, "split", context, errors)
        scene_seed = pair.get("scene_seed")
        if not isinstance(scene_seed, int):
            errors.append(f"{context}: scene_seed must be an integer")
            continue
        if pair_id:
            if pair_id in pair_ids:
                errors.append(f"duplicate pair_id={pair_id}")
            pair_ids.add(pair_id)

        instruction_ids = pair.get("instruction_ids")
        if (
            not isinstance(instruction_ids, list)
            or len(instruction_ids) != 2
            or not all(isinstance(item, str) for item in instruction_ids)
        ):
            errors.append(
                f"{context}: instruction_ids must contain exactly two strings"
            )
            continue
        if instruction_ids[0] == instruction_ids[1]:
            errors.append(f"{context}: instruction_ids must be different")
            continue

        left = instructions_by_id.get(instruction_ids[0])
        right = instructions_by_id.get(instruction_ids[1])
        if left is None or right is None:
            errors.append(
                f"{context}: references unknown instructions "
                f"{instruction_ids}"
            )
            continue

        if left.get("split") != pair_split or right.get("split") != pair_split:
            errors.append(
                f"{context}: pair and instruction splits must match"
            )
        if not _pair_has_expected_difference(
            intervention_type, left, right
        ):
            errors.append(
                f"{context}: labels do not match intervention_type="
                f"{intervention_type}"
            )
        intervention_counts[intervention_type] += 1
        scene_labels[scene_seed].add(_instruction_label(left))
        scene_labels[scene_seed].add(_instruction_label(right))

    missing_interventions = (
        REQUIRED_INTERVENTIONS - set(intervention_counts)
    )
    if missing_interventions:
        errors.append(
            f"missing intervention types: {sorted(missing_interventions)}"
        )

    nonconflicting_scenes = sorted(
        scene_seed
        for scene_seed, labels in scene_labels.items()
        if len(labels) < 2
    )
    if nonconflicting_scenes:
        errors.append(
            "scene seeds without conflicting instruction labels: "
            f"{nonconflicting_scenes}"
        )

    if errors:
        raise LanguageDatasetError("\n".join(errors))

    return {
        "instruction_count": len(instructions),
        "contrast_pair_count": len(contrast_pairs),
        "conflicting_scene_count": len(scene_labels),
        "intent_group_counts": dict(sorted(intent_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "intervention_counts": dict(sorted(intervention_counts.items())),
        "template_families": {
            split: sorted(families)
            for split, families in sorted(template_families.items())
        },
        "acceptance": {
            "minimum_instruction_count": (
                len(instructions) >= MIN_INSTRUCTIONS
            ),
            "minimum_contrast_pair_count": (
                len(contrast_pairs) >= MIN_CONTRAST_PAIRS
            ),
            "required_interventions_covered": (
                REQUIRED_INTERVENTIONS <= set(intervention_counts)
            ),
            "split_templates_disjoint": True,
            "all_scenes_have_conflicting_labels": True,
        },
    }


def load_and_validate(
    instructions_path: str | Path,
    pairs_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    instructions = read_jsonl(instructions_path)
    pairs = read_jsonl(pairs_path)
    report = validate_language_dataset(instructions, pairs)
    return instructions, pairs, report


