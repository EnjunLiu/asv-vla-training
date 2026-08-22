from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import sys

ROOT = Path("D:/asv-vla-training")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

import run_training
from decision import SmallActionPolicy, SmallPolicyConfig, build_entity_features, stamp_language_standoff
from train_final import (
    EntityObject,
    load_episode_records,
    load_language_embeddings,
    save_policy_checkpoint,
    teacher_action,
)


DATA = ROOT / "data/episodes/moving_target_valid"
EMBEDDING_PATH = ROOT / "data/qwen_final_embeddings.npz"
RUN = ROOT / "experiments/chase_standoff_v3_candidate"
PERCEPTION_SOURCE = ROOT / "experiments/chase_far_candidate"


def _probe(policy, embeddings, template, *, xy, standoff, key, surge=0.0):
    language = stamp_language_standoff(embeddings[key], standoff).reshape(1, 256)
    entity_id = "target_red" if key.startswith("red") else "target_blue"
    scaled = []
    for item in template.entities:
        value = dict(item)
        if item["entity_id"] == entity_id:
            value["relative_position_m"] = [float(xy[0]), float(xy[1]), 0.0]
        scaled.append(value)
    features = build_entity_features([EntityObject(item) for item in scaled])
    mask = np.zeros((1, 16), dtype=bool)
    target_index = next(
        index
        for index, name in enumerate(features.entity_ids)
        if name == entity_id
    )
    mask[0, target_index] = True
    ego = np.asarray([[surge / 5.0, 0.0]], dtype=np.float32)
    valid = torch.ones(1, dtype=torch.bool)
    with torch.inference_mode():
        output = policy(
            language=torch.from_numpy(language),
            entity_geometry=torch.from_numpy(features.features.reshape(1, 16, 16)),
            ego_state=torch.from_numpy(ego),
            language_valid=valid,
            entity_geometry_mask=torch.from_numpy(mask),
            ego_state_valid=valid,
            policy_input_valid=valid,
        )
    action = output.action[0].numpy()
    expected = teacher_action(np.asarray(xy, dtype=np.float32), standoff, surge, 0.0)
    return {
        "action": [float(action[0]), float(action[1])],
        "norm_m": float(np.linalg.norm(action)),
        "teacher": [float(expected[0]), float(expected[1])],
        "teacher_norm_m": float(np.linalg.norm(expected)),
    }


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    for name in ("perception.npz", "perception.json"):
        shutil.copy2(PERCEPTION_SOURCE / name, RUN / name)
    records = load_episode_records(DATA)
    embeddings = load_language_embeddings(EMBEDDING_PATH)
    split = run_training.moving_target_split(records)
    policy_path = RUN / "policy.pt"
    train_records = [record for record in records if record.slot_id in set(split["train"])]
    policy_train = save_policy_checkpoint(policy_path, train_records, embeddings)
    checkpoint = torch.load(policy_path, map_location="cpu", weights_only=True)
    policy = SmallActionPolicy(SmallPolicyConfig.from_mapping(checkpoint["model_config"]))
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    template_red = next(record for record in records if record.slot_id == "RED_3M_TEST")
    template_blue = next(record for record in records if record.slot_id == "BLUE_3M_TEST")
    report = {
        "data_root": str(DATA),
        "split": split,
        "perception_copied_from": str(PERCEPTION_SOURCE),
        "policy_train": policy_train,
        "policy_validation": run_training.policy_eval(
            policy_path, records, embeddings, set(split["validation"])
        ),
        "policy_test": run_training.policy_eval(
            policy_path, records, embeddings, set(split["test"])
        ),
        "qwen_embeddings_sha256": hashlib.sha256(EMBEDDING_PATH.read_bytes()).hexdigest(),
        "offline_probes": {
            "cold_start_blue_5p5_3m": _probe(
                policy, embeddings, template_blue, xy=(5.5, -0.34), standoff=3.0, key="blue_3m"
            ),
            "cold_start_red_5p2_4m": _probe(
                policy, embeddings, template_red, xy=(5.2, 0.0), standoff=4.0, key="red_4m"
            ),
            "midrange_red_3p8_3m": _probe(
                policy,
                embeddings,
                template_red,
                xy=(3.8, 0.0),
                standoff=3.0,
                key="red_3m",
                surge=0.6,
            ),
        },
    }
    (RUN / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
