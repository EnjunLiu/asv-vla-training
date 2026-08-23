#!/usr/bin/env python3
"""Add a minimal entity-embedding contract test."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entity_contract import ENTITY_EMBEDDING_DIM, ENTITY_KINEMATIC_DIM
from entity_embedding import EntityEmbeddingHead, build_training_crops
from perception import CameraProfile
from train_final import EntityObject, load_episode_records, task_key
from decision import build_entity_features


def test_entity_embedding_checkpoint_loads_and_encodes() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    checkpoint = ROOT / "experiments/chase_standoff_entityfeat_v2/entity_embedding.pt"
    if not checkpoint.is_file():
        pytest.skip("entity embedding checkpoint not built yet")
    runtime = EntityEmbeddingHead.load(checkpoint, device="cuda")
    records = load_episode_records(ROOT / "data/episodes/moving_target_valid")
    record = next(item for item in records if item.slot_id == "RED_3M_TEST")
    entity = record.entities[0]
    position = entity["relative_position_m"]
    from PIL import Image

    values = runtime.encode_entity(
        Image.open(record.image_path).convert("RGB"),
        float(position[0]),
        float(position[1]),
        float(position[2]),
        CameraProfile(),
    )
    assert values.shape == (ENTITY_EMBEDDING_DIM,)
    assert np.linalg.norm(values) > 0.9


def test_policy_dataset_uses_nonzero_entity_embeddings() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    checkpoint = ROOT / "experiments/chase_standoff_entityfeat_v2/entity_embedding.pt"
    if not checkpoint.is_file():
        pytest.skip("entity embedding checkpoint not built yet")
    records = load_episode_records(ROOT / "data/episodes/moving_target_valid")
    record = next(item for item in records if item.slot_id == "RED_3M_TEST")
    runtime = EntityEmbeddingHead.load(checkpoint, device="cuda")
    from PIL import Image

    image = Image.open(record.image_path).convert("RGB")
    encoded = runtime.encode_entities(image, record.entities, CameraProfile())
    norms = [float(np.linalg.norm(value)) for value in encoded.values()]
    assert max(norms) > 0.1
