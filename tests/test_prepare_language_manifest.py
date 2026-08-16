from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from asv_training.prepare_language_manifest import ManifestError, build_language_manifest
from asv_training.train_image_entity_perception import _load_language_embeddings


def _write_sources(tmp_path: Path, *, embedding_ids: list[str]) -> tuple[Path, Path]:
    instructions_path = tmp_path / "instructions.jsonl"
    rows = [
        {"instruction_id": "instruction_a", "text": "follow red", "split": "train"},
        {"instruction_id": "instruction_b", "text": "follow blue", "split": "validation"},
    ]
    instructions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    row_by_id = {row["instruction_id"]: row for row in rows}
    values = np.zeros((len(embedding_ids), 256), dtype=np.float32)
    for row_index, instruction_id in enumerate(embedding_ids):
        values[row_index, 0] = float(row_index + 1)
    source_metadata = {
        instruction_id: row_by_id.get(
            instruction_id, {"text": "unknown", "split": "test"}
        )
        for instruction_id in embedding_ids
    }
    embeddings_path = tmp_path / "language.npz"
    np.savez(
        embeddings_path,
        embeddings=values,
        instruction_ids=np.asarray(embedding_ids, dtype=np.str_),
        instruction_texts=np.asarray(
            [
                source_metadata[instruction_id]["text"]
                for instruction_id in embedding_ids
            ],
            dtype=np.str_,
        ),
        language_splits=np.asarray(
            [
                source_metadata[instruction_id]["split"]
                for instruction_id in embedding_ids
            ],
            dtype=np.str_,
        ),
    )
    return embeddings_path, instructions_path


def test_build_language_manifest_aligns_metadata_by_id_and_loads(
    tmp_path: Path,
) -> None:
    embeddings_path, instructions_path = _write_sources(
        tmp_path, embedding_ids=["instruction_b", "instruction_a"]
    )
    output_path = tmp_path / "language_manifest.json"
    weights_sha256 = "a" * 64

    manifest = build_language_manifest(
        embeddings_path,
        instructions_path,
        output_path,
        model_id="test-language-model",
        weights_sha256=weights_sha256,
    )

    assert manifest["schema_version"] == "task_embedding_manifest_v1"
    assert manifest["model_id"] == "test-language-model"
    assert manifest["weights_sha256"] == weights_sha256
    assert manifest["embeddings_sha256"] == hashlib.sha256(
        embeddings_path.read_bytes()
    ).hexdigest()
    assert manifest["instruction_ids"] == ["instruction_b", "instruction_a"]
    assert manifest["instruction_texts"] == ["follow blue", "follow red"]
    assert manifest["language_splits"] == ["validation", "train"]
    assert manifest["source_paths"]["embeddings"] == str(embeddings_path.resolve())
    assert manifest["source_paths"]["instructions"] == str(instructions_path.resolve())

    table = _load_language_embeddings(embeddings_path, output_path)
    assert set(table["by_id"]) == {"instruction_a", "instruction_b"}
    np.testing.assert_array_equal(table["by_id"]["instruction_b"][0], 1.0)
    np.testing.assert_array_equal(table["by_id"]["instruction_a"][0], 2.0)


def test_build_language_manifest_rejects_id_set_mismatch(tmp_path: Path) -> None:
    embeddings_path, instructions_path = _write_sources(
        tmp_path, embedding_ids=["instruction_a", "instruction_missing"]
    )

    with pytest.raises(ManifestError, match="ID_SET_MISMATCH"):
        build_language_manifest(
            embeddings_path,
            instructions_path,
            tmp_path / "language_manifest.json",
            model_id="test-language-model",
            weights_sha256="b" * 64,
        )
