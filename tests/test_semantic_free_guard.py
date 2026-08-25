"""Static gate: src/ must stay free of semantic entity category tables."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

FORBIDDEN = (
    "ENTITY_SLOT_IDS",
    "ENTITY_IDS",
    "scenario_from_slot",
    "_color_from_entity_id",
    "target_red",
    "target_blue",
    "ENTITY_COUNT",
)


def test_src_has_no_semantic_entity_category_tables() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token == "ENTITY_COUNT":
                if re.search(r"\bENTITY_COUNT\s*=", text) or re.search(
                    r"\bimport\s+ENTITY_COUNT\b|\bENTITY_COUNT\b,", text
                ):
                    offenders.append(f"{path.relative_to(ROOT)}:{token}")
                continue
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}:{token}")
    assert not offenders, "semantic leftovers in src/:\n" + "\n".join(offenders)


def test_backfill_script_is_allowed_to_mention_legacy_ue_ids() -> None:
    script = ROOT / "scripts" / "backfill_manifest_slots.py"
    text = script.read_text(encoding="utf-8")
    assert "target_red" in text
    assert "target_blue" in text
