import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perception import ImageEntityModel
from train_final import load_episode_records, load_language_embeddings, task_key


def evaluate(model_path: str) -> None:
    records = load_episode_records("data/episodes/moving_target_valid")
    embeddings = load_language_embeddings("data/qwen_final_embeddings.npz")
    model = ImageEntityModel.load(model_path)
    print(model_path)
    for slot, color in (("RED_3M_TEST", "red"), ("BLUE_3M_TEST", "blue"), ("RED_4M_TEST", "red")):
        errors = []
        for record in records:
            if record.slot_id != slot:
                continue
            image = Image.open(record.image_path)
            prediction = model.predict(
                image,
                task=record.task_text,
                task_embedding=embeddings[task_key(record.task_text)],
            )
            result = next(item for item in prediction if item.entity_id == f"target_{color}")
            truth = next(item for item in record.entities if item["entity_id"] == f"target_{color}")
            errors.append(
                np.asarray((result.relative_x, result.relative_y), dtype=np.float64)
                - np.asarray(truth["relative_position_m"][:2], dtype=np.float64)
            )
        errors = np.asarray(errors)
        print(
            slot,
            "xy_rmse=%.6f x_rmse=%.6f y_rmse=%.6f bias=(%.6f,%.6f)" % (
                np.sqrt(np.mean(errors**2)),
                np.sqrt(np.mean(errors[:, 0] ** 2)),
                np.sqrt(np.mean(errors[:, 1] ** 2)),
                np.mean(errors[:, 0]),
                np.mean(errors[:, 1]),
            ),
        )


if __name__ == "__main__":
    for path in sys.argv[1:]:
        evaluate(path)
