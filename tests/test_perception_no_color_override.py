import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perception import (  # noqa: E402
    FUSED_FEATURE_DIM,
    OUTPUT_DIM,
    ImageEntityModel,
)


def test_new_model_geometry_is_not_overridden_by_legacy_color_calibration() -> None:
    bias = np.zeros(OUTPUT_DIM, dtype=np.float32)
    bias[0] = 3.0
    bias[1:4] = np.asarray([0.25, 0.05, 0.0], dtype=np.float32)
    model = ImageEntityModel(
        feature_mean=np.zeros(FUSED_FEATURE_DIM, dtype=np.float32),
        feature_scale=np.ones(FUSED_FEATURE_DIM, dtype=np.float32),
        weights=np.zeros((FUSED_FEATURE_DIM, OUTPUT_DIM), dtype=np.float32),
        bias=bias,
        language_model_id="Qwen/Qwen3-Embedding-0.6B",
        language_weights_sha256="0" * 64,
    )
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    image[80:95, 150:165, 0] = 255
    embedding = np.ones(256, dtype=np.float32)
    without_color = model.predict(Image.fromarray(image), task_embedding=embedding)
    with_color = model.predict(
        Image.fromarray(image), color_image=Image.fromarray(image), task_embedding=embedding
    )
    assert with_color == without_color
