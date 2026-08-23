#!/usr/bin/env python3
"""Train and save the frozen-backbone entity embedding head."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_training
from entity_embedding import EntityEmbeddingHead, build_training_crops
from perception import CameraProfile
from train_final import load_episode_records, split_moving_target_slots


DATA = ROOT / "data/episodes/moving_target_valid"
RUN = ROOT / "experiments/chase_standoff_entityfeat_v2"


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    records = load_episode_records(DATA)
    split = split_moving_target_slots([record.slot_id for record in records])
    train_records = [record for record in records if record.slot_id in set(split["train"])]
    profile = CameraProfile()
    patches = []
    labels = []
    for record in train_records:
        batch_patches, batch_labels = build_training_crops(
            record.image_path, record.entities, profile
        )
        patches.extend(batch_patches)
        labels.extend(batch_labels)
    if not patches:
        raise SystemExit("no training crops were built")

    trainer = EntityEmbeddingHead.create_trainable(device="cuda")
    metrics = trainer.fit_entity_classifier(patches, labels, epochs=12)
    checkpoint_path = RUN / "entity_embedding.pt"
    trainer.save_checkpoint(checkpoint_path)
    report = {
        "data_root": str(DATA),
        "split": split,
        "crop_count": len(patches),
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
