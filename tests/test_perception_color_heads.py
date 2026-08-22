import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perception import ImageEntityModel, ImageEntityPerceptionError


def _model(**kwargs):
    feature_dim = 32 * 18 * 7 + 4 * 8 + 256
    base = dict(
        feature_mean=np.zeros(feature_dim, dtype=np.float32),
        feature_scale=np.ones(feature_dim, dtype=np.float32),
        weights=np.zeros((feature_dim, 16), dtype=np.float32),
        bias=np.zeros(16, dtype=np.float32),
        language_model_id="Qwen/Qwen3-Embedding-0.6B",
        language_weights_sha256="0" * 64,
    )
    base.update(kwargs)
    return ImageEntityModel(**base)


def test_color_head_requires_complete_shapes():
    with pytest.raises(ImageEntityPerceptionError, match="color head"):
        _model(
            color_head_weights=np.zeros((2, 512, 3), dtype=np.float32),
            color_head_bias=np.zeros((2, 4), dtype=np.float32),
        )


def test_task_color_selects_color_head():
    feature_dim = 32 * 18 * 7 + 4 * 8 + 256
    red_w = np.zeros((feature_dim, 4), dtype=np.float32)
    blue_w = np.zeros((feature_dim, 4), dtype=np.float32)
    red_b = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float32)
    blue_b = np.asarray((0.0, 2.0, 0.0, 0.0), dtype=np.float32)
    model = _model(
        color_head_weights=np.stack((red_w, blue_w)),
        color_head_bias=np.stack((red_b, blue_b)),
    )
    assert model.color_head_bias.shape == (2, 4)
    assert model._color_head_for_task("follow red target")[2][1] == 1.0
    assert model._color_head_for_task("跟随蓝色目标船")[2][1] == 2.0
