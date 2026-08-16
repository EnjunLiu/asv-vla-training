from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
import pytest

from asv_training.synthetic_geometry_train import (
    COLORS,
    DISTANCE_BUCKETS,
    L7_RUNTIME_POINTS_M,
    L7_RUNTIME_X_RANGE_M,
    L7_RUNTIME_Y_RANGE_M,
    SyntheticGeometryDataset,
    expert_action_for_geometry,
    generate_synthetic_geometry_dataset,
    load_checkpoint,
    load_language_embeddings,
    load_model_config,
    save_checkpoint,
    save_synthetic_dataset,
    train_synthetic_policy,
)


def test_synthetic_dataset_covers_runtime_contract_and_input_shapes(tmp_path) -> None:
    dataset = generate_synthetic_geometry_dataset(sample_count=24, seed=7)

    assert isinstance(dataset, SyntheticGeometryDataset)
    assert dataset.language.shape == (24, 256)
    assert dataset.entity_geometry.shape == (24, 16, 16)
    assert dataset.entity_geometry_mask.shape == (24, 16)
    assert np.all(np.count_nonzero(dataset.entity_geometry_mask, axis=1) == 1)
    assert dataset.previous_action.shape == (24, 2)
    assert dataset.previous_action_valid.any()
    assert (~dataset.previous_action_valid).any()
    assert np.allclose(dataset.target_stop, False)
    assert np.all(np.linalg.norm(dataset.target_action, axis=1) <= 0.3 + 1.0e-6)

    coverage = {
        (row["target_color"], row["distance_bucket"])
        for row in dataset.metadata
    }
    assert coverage == {
        (color, bucket) for color in COLORS for bucket in DISTANCE_BUCKETS
    }
    xs = np.asarray([row["target_x_m"] for row in dataset.metadata])
    ys = np.asarray([row["target_y_m"] for row in dataset.metadata])
    assert xs.min() >= L7_RUNTIME_X_RANGE_M[0]
    assert xs.max() <= L7_RUNTIME_X_RANGE_M[1]
    assert ys.min() >= L7_RUNTIME_Y_RANGE_M[0]
    assert ys.max() <= L7_RUNTIME_Y_RANGE_M[1]
    required_points = {
        (row["target_x_m"], row["target_y_m"])
        for row in dataset.metadata[
            : len(COLORS) * len(DISTANCE_BUCKETS) * len(L7_RUNTIME_POINTS_M)
        ]
    }
    assert required_points == set(L7_RUNTIME_POINTS_M)

    for index, row in enumerate(dataset.metadata[:12]):
        expected = expert_action_for_geometry(
            row["target_x_m"],
            row["target_y_m"],
            row["distance_bucket"],
            color=row["target_color"],
        )
        np.testing.assert_allclose(
            dataset.target_action[index],
            expected.expert_action,
            rtol=0.0,
            atol=1.0e-7,
        )

    output = tmp_path / "synthetic_geometry.npz"
    save_synthetic_dataset(output, dataset)
    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["entity_geometry"].shape == (24, 16, 16)
        assert arrays["previous_action_valid"].dtype == np.bool_


def test_real_language_embedding_table_is_aligned_to_each_task(tmp_path) -> None:
    instruction_ids = np.asarray(
        [
            f"follow_{color}_{bucket}_01"
            for color in COLORS
            for bucket in DISTANCE_BUCKETS
        ]
    )
    instruction_texts = np.asarray(
        [
            f"follow {color} target at {bucket}"
            for color in COLORS
            for bucket in DISTANCE_BUCKETS
        ]
    )
    embeddings = np.arange(len(instruction_ids) * 256, dtype=np.float32).reshape(
        len(instruction_ids), 256
    )
    source = tmp_path / "language_embeddings.npz"
    np.savez(
        source,
        instruction_ids=instruction_ids,
        instruction_texts=instruction_texts,
        embeddings=embeddings,
    )
    table = load_language_embeddings(source)
    dataset = generate_synthetic_geometry_dataset(
        sample_count=24,
        seed=7,
        language_embeddings=table,
    )

    assert dataset.language_source == str(source)
    assert dataset.language_source_sha256
    assert np.allclose(dataset.language[0], embeddings[0])
    assert dataset.metadata[0]["instruction_id"] == instruction_ids[0]
    assert dataset.language_text[0] == instruction_texts[0]


def test_l7_runtime_points_have_expected_raw_expert_directions() -> None:
    red_action = expert_action_for_geometry(4.74, 0.49, "4m", color="red")
    blue_action = expert_action_for_geometry(3.94, -1.27, "3m", color="blue")

    assert red_action.safe_stop is False
    assert red_action.expert_action[0] > 0.0
    assert red_action.expert_action[1] > 0.0
    assert blue_action.safe_stop is False
    assert blue_action.expert_action[0] > 0.0
    assert blue_action.expert_action[1] < 0.0
    assert math.hypot(*red_action.expert_action) <= 0.3 + 1.0e-9
    assert math.hypot(*blue_action.expert_action) <= 0.3 + 1.0e-9


def test_checkpoint_round_trip_uses_existing_model_contract(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    dataset = generate_synthetic_geometry_dataset(sample_count=12, seed=3)
    config = load_model_config("configs/model_small_v3.yaml")
    model, history = train_synthetic_policy(
        dataset,
        config,
        epochs=1,
        batch_size=12,
        device="cpu",
        seed=3,
    )
    checkpoint_path = tmp_path / "synthetic_policy.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        config,
        dataset=dataset,
        history=history,
        seed=3,
        epochs=1,
        batch_size=12,
        learning_rate=0.003,
        device="cpu",
    )
    restored, payload = load_checkpoint(checkpoint_path, device="cpu")

    assert payload["model_config"] == asdict(config)
    assert payload["schema_version"] == "synthetic_geometry_single_point_v2"
    assert payload["contract"]["decision_inputs"] == [
        "language",
        "entity_geometry",
        "previous_action",
        "language_valid",
        "entity_geometry_mask",
        "previous_action_valid",
        "policy_input_valid",
    ]
    assert payload["contract"]["input_shapes"]["previous_action"] == ["B", 2]
    assert payload["contract"]["outputs"]["action"] == {
        "shape": ["B", 2],
        "dtype": "float32",
        "frame": "base_link",
        "kind": "single_step_desired_displacement_m",
        "maximum_norm_m": 0.3,
    }
    assert payload["training"]["dataset_schema_version"] == (
        "synthetic_geometry_dataset_v1"
    )
    assert payload["training"]["seed"] == 3
    assert payload["training"]["dataset_seed"] == 3
    assert payload["training"]["epochs"] == 1
    assert payload["training"]["batch_size"] == 12
    assert payload["training"]["learning_rate"] == 0.003
    assert payload["training"]["device"] == "cpu"
    assert restored.config == config
    assert payload["model_state_dict"]
    for name, value in model.state_dict().items():
        assert torch.equal(value.cpu(), restored.state_dict()[name].cpu())
    inputs = dataset.as_policy_inputs()
    with torch.no_grad():
        output = restored(
            language=torch.from_numpy(inputs["language"]),
            entity_geometry=torch.from_numpy(inputs["entity_geometry"]),
            previous_action=torch.from_numpy(inputs["previous_action"]),
            language_valid=torch.from_numpy(inputs["language_valid"]),
            entity_geometry_mask=torch.from_numpy(inputs["entity_geometry_mask"]),
            previous_action_valid=torch.from_numpy(inputs["previous_action_valid"]),
            policy_input_valid=torch.from_numpy(inputs["policy_input_valid"]),
        )
    assert tuple(output.action.shape) == (12, 2)
    assert torch.isfinite(output.action).all()
