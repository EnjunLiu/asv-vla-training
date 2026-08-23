#!/usr/bin/env python3
"""Train and save the frozen-backbone entity embedding head."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from entity_embedding import EntityEmbeddingHead, build_training_crops
from online_perception import build_online_entity_cache
from perception import CameraProfile, ImageEntityModel
from train_final import load_episode_records, load_language_embeddings, split_moving_target_slots


DATA = ROOT / "data/episodes/moving_target_valid"
RUN = ROOT / "experiments/chase_standoff_entityfeat_v2"
PERCEPTION_PATH = RUN / "perception.npz"


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    if not PERCEPTION_PATH.is_file():
        raise SystemExit(f"missing perception checkpoint: {PERCEPTION_PATH}")
    records = load_episode_records(DATA)
    embeddings = load_language_embeddings(ROOT / "data/qwen_final_embeddings.npz")
    split = split_moving_target_slots([record.slot_id for record in records])
    train_records = [record for record in records if record.slot_id in set(split["train"])]
    perception_model = ImageEntityModel.load(PERCEPTION_PATH)
    entity_cache = build_online_entity_cache(
        train_records,
        model=perception_model,
        embeddings=embeddings,
    )
    patches = []
    labels = []
    profile = CameraProfile()
    for record in train_records:
        online_items = entity_cache.get(
            (
                str(record.slot_id),
                str(record.run_id),
                int(record.scene_seed),
                int(record.frame_index),
            ),
            [],
        )
        batch_patches, batch_labels = build_training_crops(
            record.image_path, online_items, profile
        )
        patches.extend(batch_patches)
        labels.extend(batch_labels)
    if not patches:
        raise SystemExit("no training crops were built from online perception geometry")

    trainer = EntityEmbeddingHead.create_trainable(device="cuda")
    metrics = trainer.fit_entity_classifier(patches, labels, epochs=12)
    checkpoint_path = RUN / "entity_embedding.pt"
    trainer.save_checkpoint(checkpoint_path)
    report = {
        "data_root": str(DATA),
        "split": split,
        "crop_count": len(patches),
        "crop_geometry": "online_perception",
        "checkpoint": str(checkpoint_path),
        **metrics,
    }
    (RUN / "entity_embedding_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
