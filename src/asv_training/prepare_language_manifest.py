"""Build a language embedding manifest on a PC.

The manifest describes an existing NPZ embedding table without changing that
table.  NPZ row order is authoritative for the output; instruction metadata
is joined by instruction_id so the loader can safely pair each row with its
text and split.

This module is intentionally PC-only.  It must not be run on Jetson or any
aarch64 host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Sequence

import numpy as np


MANIFEST_SCHEMA_VERSION = "task_embedding_manifest_v1"
LANGUAGE_EMBEDDING_DIM = 256
_SHA256_LENGTH = 64


class ManifestError(ValueError):
    """Raised when source data cannot produce a safe language manifest."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_path_as_wsl(path_text: str) -> Path | None:
    windows_path = PureWindowsPath(path_text)
    if not windows_path.drive:
        return None
    drive = windows_path.drive.rstrip(":").lower()
    parts = windows_path.parts[1:]
    return Path("/mnt") / drive / Path(*parts)


def _wsl_path_as_windows(path_text: str) -> Path | None:
    posix_path = PurePosixPath(path_text)
    parts = posix_path.parts
    if len(parts) < 3 or parts[0] != "/" or parts[1].lower() != "mnt":
        return None
    drive = parts[2]
    if len(drive) != 1 or not drive.isalpha():
        return None
    return Path(f"{drive.upper()}:\\") / Path(*parts[3:])


def _resolve_path(
    raw_path: str | os.PathLike[str], *, label: str, must_exist: bool
) -> Path:
    """Resolve native Windows paths and the WSL /mnt/<drive> form."""

    raw_text = os.fspath(raw_path)
    expanded_text = os.path.expandvars(os.path.expanduser(raw_text))
    candidates = [Path(expanded_text)]
    if os.name != "nt":
        converted = _windows_path_as_wsl(expanded_text)
        if converted is not None:
            candidates.append(converted)
    else:
        converted = _wsl_path_as_windows(expanded_text)
        if converted is not None:
            candidates.append(converted)

    if not must_exist and len(candidates) == 2:
        return candidates[1].resolve()
    for candidate in candidates:
        if not must_exist or candidate.exists():
            return candidate.resolve()
    raise ManifestError(f"{label} not found: {raw_text}")


def _validate_sha256(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ManifestError(f"{label} must be a 64-character SHA-256 hex digest")
    return normalized


def _read_instruction_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ManifestError(f"cannot read instructions JSONL: {exc}") from exc

    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(
                f"invalid instructions JSONL at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise ManifestError(
                f"instructions JSONL line {line_number} must be a JSON object"
            )

        instruction_id = record.get("instruction_id")
        text = record.get("text")
        split = record.get("split")
        if not isinstance(instruction_id, str) or not instruction_id.strip():
            raise ManifestError(
                f"instructions JSONL line {line_number} has an invalid instruction_id"
            )
        if not isinstance(text, str) or not text:
            raise ManifestError(
                f"instructions JSONL line {line_number} has invalid text"
            )
        if not isinstance(split, str) or not split:
            raise ManifestError(
                f"instructions JSONL line {line_number} has invalid split"
            )
        instruction_id = instruction_id.strip()
        if instruction_id in ids:
            raise ManifestError(
                f"INSTRUCTION_ID_DUPLICATED in instructions JSONL: {instruction_id!r}"
            )
        ids.add(instruction_id)
        rows.append(
            {
                "instruction_id": instruction_id,
                "text": text,
                "split": split,
            }
        )

    if not rows:
        raise ManifestError("instructions JSONL is empty")
    return rows


def _read_string_array(
    archive: np.lib.npyio.NpzFile,
    key: str,
    *,
    expected_length: int | None = None,
) -> list[str]:
    if key not in archive.files:
        raise ManifestError(f"language NPZ is missing {key!r}")
    values = np.asarray(archive[key])
    if values.ndim != 1:
        raise ManifestError(f"language NPZ field {key!r} must be one-dimensional")
    if expected_length is not None and len(values) != expected_length:
        raise ManifestError(
            f"language NPZ field {key!r} has {len(values)} rows; "
            f"expected {expected_length}"
        )

    result: list[str] = []
    for index, value in enumerate(values.tolist()):
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ManifestError(
                    f"language NPZ field {key!r} row {index} is not UTF-8"
                ) from exc
        if not isinstance(value, str):
            raise ManifestError(
                f"language NPZ field {key!r} row {index} must be a string"
            )
        result.append(value.strip() if key == "instruction_ids" else value)
    return result


def _id_set_error(source_ids: set[str], dataset_ids: set[str]) -> ManifestError:
    missing = sorted(dataset_ids - source_ids)
    extra = sorted(source_ids - dataset_ids)
    return ManifestError(
        "ID_SET_MISMATCH: "
        f"missing_from_embeddings={missing!r}, extra_in_embeddings={extra!r}"
    )


def _load_embedding_source(
    path: Path,
) -> tuple[np.ndarray, list[str], list[str] | None, list[str] | None]:
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise ManifestError(f"cannot load language NPZ: {exc}") from exc
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ManifestError("language embeddings input must be an NPZ archive")

    try:
        if "embeddings" not in loaded.files:
            raise ManifestError("language NPZ is missing 'embeddings'")
        embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != LANGUAGE_EMBEDDING_DIM:
            raise ManifestError(
                "EMBEDDING_SHAPE_MISMATCH: got "
                f"{embeddings.shape}; expected [N,{LANGUAGE_EMBEDDING_DIM}]"
            )
        if not np.all(np.isfinite(embeddings)):
            raise ManifestError("language embeddings contain NaN or Inf")
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms <= 1.0e-12):
            raise ManifestError("language embeddings contain a zero-norm row")

        instruction_ids = _read_string_array(
            loaded, "instruction_ids", expected_length=embeddings.shape[0]
        )
        if any(not value for value in instruction_ids):
            raise ManifestError("language NPZ contains an empty instruction_id")
        if len(instruction_ids) != len(set(instruction_ids)):
            raise ManifestError("INSTRUCTION_ID_DUPLICATED in language NPZ")

        instruction_texts = None
        if "instruction_texts" in loaded.files:
            instruction_texts = _read_string_array(
                loaded, "instruction_texts", expected_length=embeddings.shape[0]
            )
        language_splits = None
        if "language_splits" in loaded.files:
            language_splits = _read_string_array(
                loaded, "language_splits", expected_length=embeddings.shape[0]
            )
        return embeddings, instruction_ids, instruction_texts, language_splits
    finally:
        loaded.close()


def build_language_manifest(
    embeddings_path: str | os.PathLike[str],
    instructions_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    model_id: str,
    weights_sha256: str,
    force: bool = False,
) -> dict[str, Any]:
    """Validate sources and write a manifest aligned to NPZ row order."""

    _reject_non_pc_host()
    embeddings_file = _resolve_path(
        embeddings_path, label="language embeddings", must_exist=True
    )
    instructions_file = _resolve_path(
        instructions_path, label="instructions JSONL", must_exist=True
    )
    output_file = _resolve_path(output_path, label="manifest output", must_exist=False)
    if output_file == embeddings_file or output_file == instructions_file:
        raise ManifestError("manifest output must be different from its source files")
    if output_file.exists() and not force:
        raise ManifestError(
            f"manifest output already exists; pass --force to replace it: {output_file}"
        )

    model_id = model_id.strip()
    if not model_id:
        raise ManifestError("model_id must not be empty")
    weights_sha256 = _validate_sha256(weights_sha256, label="weights_sha256")

    instruction_rows = _read_instruction_rows(instructions_file)
    dataset_by_id = {row["instruction_id"]: row for row in instruction_rows}
    embeddings, embedding_ids, npz_texts, npz_splits = _load_embedding_source(
        embeddings_file
    )
    embedding_id_set = set(embedding_ids)
    dataset_id_set = set(dataset_by_id)
    if embedding_id_set != dataset_id_set:
        raise _id_set_error(embedding_id_set, dataset_id_set)

    dataset_texts = [
        dataset_by_id[instruction_id]["text"] for instruction_id in embedding_ids
    ]
    dataset_splits = [
        dataset_by_id[instruction_id]["split"] for instruction_id in embedding_ids
    ]
    if npz_texts is not None and npz_texts != dataset_texts:
        mismatches = [
            instruction_id
            for instruction_id, source_text, dataset_text in zip(
                embedding_ids, npz_texts, dataset_texts
            )
            if source_text != dataset_text
        ]
        raise ManifestError(
            "TEXT_MISMATCH between language NPZ and instructions JSONL: "
            f"{mismatches!r}"
        )
    if npz_splits is not None and npz_splits != dataset_splits:
        mismatches = [
            instruction_id
            for instruction_id, source_split, dataset_split in zip(
                embedding_ids, npz_splits, dataset_splits
            )
            if source_split != dataset_split
        ]
        raise ManifestError(
            "LANGUAGE_SPLIT_MISMATCH between language NPZ and instructions JSONL: "
            f"{mismatches!r}"
        )

    text_to_id: dict[str, str] = {}
    for instruction_id, text in zip(embedding_ids, dataset_texts):
        previous_id = text_to_id.setdefault(text, instruction_id)
        if previous_id != instruction_id:
            raise ManifestError(
                "TEXT_AMBIGUOUS: the same instruction text maps to multiple IDs"
            )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_id": model_id,
        "weights_sha256": weights_sha256,
        "embeddings_sha256": _sha256_file(embeddings_file),
        "instruction_ids": embedding_ids,
        "instruction_texts": dataset_texts,
        "language_splits": dataset_splits,
        "source_paths": {
            "embeddings": str(embeddings_file),
            "instructions": str(instructions_file),
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_file.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ManifestError(f"cannot write language manifest: {exc}") from exc
    return manifest


def _reject_non_pc_host() -> None:
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64", "armv8l"} or Path(
        "/etc/nv_tegra_release"
    ).is_file():
        raise ManifestError(
            "PC_ONLY: Jetson/aarch64 hosts are rejected; run this tool on Windows or x86 PC"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create task_embedding_manifest_v1 from an existing language NPZ "
            "and instructions JSONL. PC-only; rejects Jetson/aarch64."
        )
    )
    parser.add_argument(
        "--embeddings",
        "--language-embeddings",
        dest="embeddings_path",
        required=True,
        help=(
            "Existing language embeddings NPZ containing embeddings and "
            "instruction_ids."
        ),
    )
    parser.add_argument(
        "--instructions",
        dest="instructions_path",
        required=True,
        help="Canonical dataset/language/instructions.jsonl path.",
    )
    parser.add_argument(
        "--output",
        "--manifest",
        dest="output_path",
        required=True,
        help="Output language_manifest.json path.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Language encoder model identifier recorded in the manifest.",
    )
    parser.add_argument(
        "--weights-sha256",
        required=True,
        help="SHA-256 digest of the language model weights.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_language_manifest(
            args.embeddings_path,
            args.instructions_path,
            args.output_path,
            model_id=args.model_id,
            weights_sha256=args.weights_sha256,
            force=args.force,
        )
    except ManifestError as exc:
        parser.error(str(exc))
    print(
        f"WROTE_LANGUAGE_MANIFEST path={args.output_path} "
        f"instructions={len(manifest['instruction_ids'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
