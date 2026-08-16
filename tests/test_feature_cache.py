from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from asv_vla.episode import make_manifest, write_json_atomic
from asv_vla.frame_record import write_frame_record
from asv_vla.image_entity_perception import (
    ImageEntityPerceptionError,
    ImageEntityPrediction,
)
from asv_vla.language_intervention_dataset import write_jsonl
from asv_vla.supervised_dataset import build_supervised_dataset
from asv_vla.temporal_entity_tracker import TemporalEntityTracker
from asv_training.feature_cache import (
    DEFAULT_IMAGE_PREPROCESS_CONTRAST,
    DEFAULT_IMAGE_PREPROCESS_ENABLED,
    DEFAULT_IMAGE_PREPROCESS_GAMMA,
    FeatureCacheError,
    FeatureCacheMiss,
    IMAGE_PERCEPTION_TRACKER_CONFIG,
    ModelFingerprint,
    build_feature_cache,
    build_policy_entity_tensor,
    compare_feature_caches,
    encode_frame_visual,
    hash_weight_tree,
    make_image_preprocess_config,
    make_language_provenance,
    make_cache_key,
    preprocess_camera_image,
    _predict_image_entities,
    validate_feature_cache,
)


SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "frame_record_v1.json"
SHA_A = "a" * 64
SHA_B = "b" * 64


class FakeLanguageEncoder:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.calls = 0

    def encode(self, text: str) -> np.ndarray:
        self.calls += 1
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = np.resize(
            np.frombuffer(digest, dtype=np.uint8).astype(np.float32), 256
        )
        values = values + 1.0 + self.offset
        return values / np.linalg.norm(values)


class FakeVisualEncoder:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.calls = 0

    def encode_images(self, images) -> np.ndarray:
        self.calls += 1
        rows = []
        for image in images:
            pixels = np.asarray(image, dtype=np.float32)
            base = float(np.mean(pixels)) + 1.0 + self.offset
            row = np.linspace(base, base + 1.0, 576, dtype=np.float32)
            rows.append(row / np.linalg.norm(row))
        return np.stack(rows)


class FakeImageModel:
    model_version = "fake_image_entity_v1"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: Image.Image) -> tuple[ImageEntityPrediction, ...]:
        del image
        step = self.calls
        self.calls += 1
        displacement = 0.1 * (step >= 1) + 0.2 * (step >= 2)
        positions = {
            "target_red": (5.0 + displacement, 0.0, 0.0),
            "target_blue": (6.0 + displacement, 0.5, 0.0),
            "target_left": (7.0 + displacement, 1.5, 0.0),
            "target_right": (7.0 + displacement, -1.5, 0.0),
        }
        return tuple(
            ImageEntityPrediction(
                entity_id=entity_id,
                visible=True,
                confidence=0.9,
                relative_x=position[0],
                relative_y=position[1],
                relative_z=position[2],
            )
            for entity_id, position in positions.items()
        )


class FailingImageModel(FakeImageModel):
    def predict(self, image: Image.Image) -> tuple[ImageEntityPrediction, ...]:
        raise ImageEntityPerceptionError("synthetic perception failure")


def _instruction(index: int) -> dict:
    labels = [
        ("follow", "color:red", "3m"),
        ("follow", "color:red", "10m"),
        ("follow", "color:blue", "3m"),
        ("follow", "color:blue", "10m"),
        ("follow", "bearing:left", "3m"),
        ("follow", "bearing:left", "10m"),
        ("follow", "bearing:right", "3m"),
        ("follow", "bearing:right", "10m"),
        ("stop", "none", "none"),
    ]
    action, attribute, distance = labels[index]
    return {
        "instruction_id": f"instruction_{index:02d}",
        "text": f"instruction {index}",
        "action": action,
        "target_attribute": attribute,
        "distance_bucket": distance,
        "split": ("train", "validation", "test")[index % 3],
    }


def _entity(entity_id: str, color: str, x: float, y: float) -> dict:
    return {
        "entity_id": entity_id,
        "class_name": "boat",
        "color": color,
        "is_target": True,
        "visible": True,
        "relative_position_m": [x, y, 0.0],
        "relative_velocity_mps": [0.1, 0.0, 0.0],
        "valid": True,
    }


def _entity_object(item: dict):
    position = item["relative_position_m"]
    velocity = item["relative_velocity_mps"]
    return type(
        "Entity",
        (),
        {
            "entity_id": item["entity_id"],
            "class_name": item["class_name"],
            "color": item["color"],
            "is_target": item["is_target"],
            "visible": item["visible"],
            "relative_x": position[0],
            "relative_y": position[1],
            "relative_z": position[2],
            "relative_velocity_x": velocity[0],
            "relative_velocity_y": velocity[1],
            "relative_velocity_z": velocity[2],
            "valid": item["valid"],
        },
    )()


def _make_sources(
    tmp_path: Path,
    frame_count: int = 2,
    *,
    truth_offset: float = 0.0,
    truth_velocity_mps: float = 0.1,
):
    episode = tmp_path / "episode" / "RUN_02"
    template = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    entities = [
        _entity("target_red", "red", 5.0 + truth_offset, 0.0),
        _entity("target_blue", "blue", 6.0 + truth_offset, 0.5),
        _entity("target_left", "white", 7.0 + truth_offset, 1.5),
        _entity("target_right", "white", 7.0 + truth_offset, -1.5),
    ]
    for entity in entities:
        entity["relative_velocity_mps"] = [truth_velocity_mps, 0.0, 0.0]
    frame_indices = []
    stamps = []
    for frame_index in range(frame_count):
        stamp = 100_000 + frame_index * 100_000
        relative_image = f"camera/{frame_index:012d}.jpg"
        image_path = episode / relative_image
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (1280, 720),
            (10 + frame_index % 20, 20, 30),
        ).save(image_path, format="JPEG")

        record = copy.deepcopy(template)
        record["run_id"] = "RUN_02"
        record["scene_seed"] = 130001
        record["frame_index"] = frame_index
        record["stamp_us"] = stamp
        record["task"]["stamp_us"] = 0
        record["task"]["text"] = "feature cache"
        record["ego"]["stamp_us"] = stamp
        record["ego"]["simulation_time_s"] = stamp / 1_000_000.0
        record["camera"]["stamp_us"] = stamp
        record["camera"]["image_path"] = relative_image
        record["entities"]["stamp_us"] = stamp
        record["entities"]["items"] = copy.deepcopy(entities)
        write_frame_record(
            episode / "frames" / f"{frame_index:012d}.json",
            record,
            image_root=episode,
        )
        frame_indices.append(frame_index)
        stamps.append(stamp)

    manifest = make_manifest(
        run_id="RUN_02",
        scene_seed=130001,
        task_text="feature cache",
        frame_indices=frame_indices,
        stamp_values=stamps,
        status="complete",
    )
    write_json_atomic(episode / "manifest.json", manifest)
    instructions = tmp_path / "instructions.jsonl"
    write_jsonl(instructions, [_instruction(index) for index in range(9)])
    supervision = tmp_path / "supervision" / "RUN_02"
    build_supervised_dataset([episode], instructions, supervision)
    return episode, supervision, instructions


def _build(
    tmp_path: Path,
    *,
    output_name: str,
    frame_count: int = 2,
    language_offset: float = 0.0,
    visual_offset: float = 0.0,
):
    episode, supervision, instructions = _make_sources(
        tmp_path / output_name, frame_count=frame_count
    )
    language = FakeLanguageEncoder(language_offset)
    visual = FakeVisualEncoder(visual_offset)
    result = build_feature_cache(
        episode,
        supervision,
        instructions,
        tmp_path / output_name / "features",
        language_encoder=language,
        visual_encoder=visual,
        language_model=ModelFingerprint("fake-language", SHA_A),
        visual_model=ModelFingerprint("fake-visual", SHA_B),
        git_sha="deadbeef",
    )
    return result, language, visual, episode, supervision, instructions


def _build_image_only(
    tmp_path: Path,
    *,
    output_name: str,
    truth_offset: float,
    truth_velocity_mps: float,
    frame_count: int = 2,
):
    episode, supervision, instructions = _make_sources(
        tmp_path / output_name,
        frame_count=frame_count,
        truth_offset=truth_offset,
        truth_velocity_mps=truth_velocity_mps,
    )
    image_model = FakeImageModel()
    model_path = tmp_path / output_name / "image_model.npz"
    model_path.write_bytes(b"fake-image-model")
    visual = FakeVisualEncoder()
    result = build_feature_cache(
        episode,
        supervision,
        instructions,
        tmp_path / output_name / "features",
        language_encoder=FakeLanguageEncoder(),
        visual_encoder=visual,
        language_model=ModelFingerprint("fake-language", SHA_A),
        visual_model=ModelFingerprint("fake-visual", SHA_B),
        git_sha="deadbeef",
        image_model=image_model,
        image_model_path=model_path,
    )
    return result, image_model, visual


def test_policy_entity_tensor_preserves_image_derived_color_and_ids(
    tmp_path: Path,
) -> None:
    episode, _, _ = _make_sources(tmp_path)
    record = json.loads(
        (episode / "frames" / "000000000000.json").read_text(
            encoding="utf-8"
        )
    )
    entities = [_entity_object(item) for item in record["entities"]["items"]]

    result = build_policy_entity_tensor(entities)

    assert result.entity_ids[:4] == (
        "target_red",
        "target_blue",
        "target_left",
        "target_right",
    )
    assert np.array_equal(
        result.features[:4, 14:16],
        np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    assert result.mask[:4].all()


def test_missing_image_invalidates_entire_policy_visual_input(
    tmp_path: Path,
) -> None:
    episode, _, _ = _make_sources(tmp_path)
    frame_path = episode / "frames" / "000000000000.json"
    record = json.loads(frame_path.read_text(encoding="utf-8"))
    entities = build_policy_entity_tensor(
        [_entity_object(item) for item in record["entities"]["items"]]
    )
    (episode / record["camera"]["image_path"]).unlink()

    result = encode_frame_visual(
        record, episode, entities, FakeVisualEncoder()
    )

    assert result.global_valid is False
    assert np.count_nonzero(result.global_token) == 0
    assert np.count_nonzero(result.entity_visual_mask) == 0
    assert np.count_nonzero(result.entity_tokens) == 0


def test_cache_key_changes_for_weights_preprocess_or_source() -> None:
    source = [
        {
            "frame_key": "RUN:1:0:100",
            "source_frame_sha256": "1" * 64,
            "image_sha256": "2" * 64,
        }
    ]
    base = make_cache_key(
        source_frames=source,
        language_model=ModelFingerprint("language", SHA_A),
        visual_model=ModelFingerprint("visual", SHA_B),
        git_sha="abc",
    )
    changed = make_cache_key(
        source_frames=source,
        language_model=ModelFingerprint("language", "c" * 64),
        visual_model=ModelFingerprint("visual", SHA_B),
        git_sha="abc",
    )

    assert base != changed
    assert base["source_frame_sha256"]["RUN:1:0:100"] == "1" * 64
    assert base["image_sha256"]["RUN:1:0:100"] == "2" * 64

    image_provenance = {
        "enabled": True,
        "model_id": "image",
        "model_version": "v1",
        "weights_sha256": SHA_A,
        "tracker": dict(IMAGE_PERCEPTION_TRACKER_CONFIG),
    }
    tracker_changed = dict(image_provenance)
    tracker_changed["tracker"] = {
        **IMAGE_PERCEPTION_TRACKER_CONFIG,
        "alpha": 0.7,
    }
    image_key = make_cache_key(
        source_frames=source,
        language_model=ModelFingerprint("language", SHA_A),
        visual_model=ModelFingerprint("visual", SHA_B),
        git_sha="abc",
        image_perception=image_provenance,
    )
    changed_tracker_key = make_cache_key(
        source_frames=source,
        language_model=ModelFingerprint("language", SHA_A),
        visual_model=ModelFingerprint("visual", SHA_B),
        git_sha="abc",
        image_perception=tracker_changed,
    )
    assert image_key != changed_tracker_key


def test_fixed_image_preprocess_is_reproducible_and_manifested(
    tmp_path: Path,
) -> None:
    default = make_image_preprocess_config()
    assert default.enabled is DEFAULT_IMAGE_PREPROCESS_ENABLED
    assert default.gamma == pytest.approx(DEFAULT_IMAGE_PREPROCESS_GAMMA)
    assert default.contrast == pytest.approx(DEFAULT_IMAGE_PREPROCESS_CONTRAST)
    assert default.as_manifest()["random_brightness_augmentation"] is False

    image = Image.new("RGB", (8, 8), (12, 24, 48))
    bright = make_image_preprocess_config(
        enabled=True,
        gamma=0.65,
        brightness=1.0,
        contrast=1.0,
    )
    transformed = preprocess_camera_image(image, bright)
    assert transformed is not image
    assert np.asarray(transformed).mean() > np.asarray(image).mean()
    np.testing.assert_array_equal(
        np.asarray(transformed),
        np.asarray(preprocess_camera_image(image, bright)),
    )

    episode, supervision, instructions = _make_sources(tmp_path, frame_count=1)
    result = build_feature_cache(
        episode,
        supervision,
        instructions,
        tmp_path / "features",
        language_encoder=FakeLanguageEncoder(),
        visual_encoder=FakeVisualEncoder(),
        language_model=ModelFingerprint("fake-language", SHA_A),
        visual_model=ModelFingerprint("fake-visual", SHA_B),
        git_sha="deadbeef",
        image_preprocess_enabled=True,
        image_preprocess_gamma=0.65,
        image_preprocess_brightness=1.0,
        image_preprocess_contrast=1.0,
    )
    manifest = json.loads(
        (Path(result["output"]) / "manifest.json").read_text(encoding="utf-8")
    )
    metadata = manifest["image_preprocess"]
    assert metadata["contract"] == (
        "ue5_capture_gamma065_brightness100_contrast100_v2"
    )
    assert metadata["enabled"] is True
    assert metadata["gamma"] == pytest.approx(0.65)
    assert metadata["random_brightness_augmentation"] is False
    assert metadata["input"] == "decoded_original_jpeg"
    assert manifest["cache_key"]["image_preprocess"] == metadata
    assert manifest["source"]["frames"][0]["image_path"].endswith(
        ".jpg"
    )


def test_single_weight_file_uses_its_real_sha256(tmp_path: Path) -> None:
    weight = tmp_path / "model" / "model.safetensors"
    weight.parent.mkdir()
    weight.write_bytes(b"frozen-weights")

    assert hash_weight_tree(weight.parent) == hashlib.sha256(
        b"frozen-weights"
    ).hexdigest()


def test_build_validate_and_hit_immutable_cache(tmp_path: Path) -> None:
    result, language, visual, episode, supervision, instructions = _build(
        tmp_path, output_name="first"
    )
    cache = Path(result["output"])

    assert result["passed"]
    assert result["cached"] is False
    assert result["frame_count"] == 2
    assert result["instruction_count"] == 9
    assert result["sample_count"] == 18
    assert language.calls == 9
    assert visual.calls == 2
    assert validate_feature_cache(cache)["passed"]
    with np.load(cache / "frames_000.npz", allow_pickle=False) as frames:
        previous_actions = np.asarray(frames["previous_expert_actions"])
        previous_valid = np.asarray(frames["previous_action_valid"])
        expert_actions = np.asarray(frames["expert_actions"])
    assert previous_actions.shape == (18, 2)
    assert not np.any(previous_valid[:9])
    assert np.all(previous_valid[9:17])
    assert not bool(previous_valid[17])
    np.testing.assert_array_equal(previous_actions[9:17], expert_actions[:8])
    np.testing.assert_array_equal(previous_actions[17], np.zeros(2))

    cached = build_feature_cache(
        episode,
        supervision,
        instructions,
        cache.parent,
        language_encoder=FakeLanguageEncoder(),
        visual_encoder=FakeVisualEncoder(),
        language_model=ModelFingerprint("fake-language", SHA_A),
        visual_model=ModelFingerprint("fake-visual", SHA_B),
        git_sha="deadbeef",
    )
    assert cached["cached"] is True

    manifest = json.loads(
        (cache / "manifest.json").read_text(encoding="utf-8")
    )
    instruction_rows = [
        json.loads(line)
        for line in instructions.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["language_provenance"] == make_language_provenance(
        instruction_rows,
        embedding_table_source="language_encoder",
        frame_perception_enabled=False,
    )
    with np.load(cache / "frames_000.npz", allow_pickle=False) as frames:
        np.testing.assert_array_equal(
            frames["sample_instruction_rows"],
            np.tile(np.arange(9, dtype=np.int16), 2),
        )


def test_validate_rejects_missing_language_provenance(tmp_path: Path) -> None:
    result, *_ = _build(tmp_path, output_name="missing_provenance")
    manifest_path = Path(result["output"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["language_provenance"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FeatureCacheError, match="language_provenance"):
        validate_feature_cache(manifest_path.parent)


def test_precomputed_language_stage_does_not_keep_encoder(
    tmp_path: Path,
) -> None:
    episode, supervision, instructions = _make_sources(tmp_path)
    language = FakeLanguageEncoder()
    records = [
        json.loads(line)
        for line in instructions.read_text(encoding="utf-8").splitlines()
    ]
    precomputed = np.stack(
        [language.encode(record["text"]) for record in records]
    )

    result = build_feature_cache(
        episode,
        supervision,
        instructions,
        tmp_path / "features",
        language_encoder=None,
        visual_encoder=FakeVisualEncoder(),
        language_model=ModelFingerprint("fake-language", SHA_A),
        visual_model=ModelFingerprint("fake-visual", SHA_B),
        git_sha="deadbeef",
        precomputed_language_embeddings=precomputed,
    )

    assert result["passed"]
    assert language.calls == 9


def test_changed_weight_is_cache_miss_not_silent_reuse(tmp_path: Path) -> None:
    result, _, _, episode, supervision, instructions = _build(
        tmp_path, output_name="miss"
    )

    with pytest.raises(FeatureCacheMiss, match="cache key differs"):
        build_feature_cache(
            episode,
            supervision,
            instructions,
            Path(result["output"]).parent,
            language_encoder=FakeLanguageEncoder(),
            visual_encoder=FakeVisualEncoder(),
            language_model=ModelFingerprint("fake-language", "c" * 64),
            visual_model=ModelFingerprint("fake-visual", SHA_B),
            git_sha="deadbeef",
        )


def test_twenty_frame_consistency_passes_and_detects_drift(
    tmp_path: Path,
) -> None:
    left, *_ = _build(
        tmp_path, output_name="left", frame_count=20
    )
    right, *_ = _build(
        tmp_path, output_name="right", frame_count=20
    )
    passed = compare_feature_caches(
        left["output"], right["output"], sample_count=20
    )

    assert passed["passed"]
    assert passed["minimum_cosine"] == pytest.approx(1.0, abs=1.0e-7)

    drifted, *_ = _build(
        tmp_path,
        output_name="drifted",
        frame_count=20,
        language_offset=100.0,
        visual_offset=100.0,
    )
    failed = compare_feature_caches(
        left["output"], drifted["output"], sample_count=20
    )
    assert failed["passed"] is False
    assert failed["minimum_cosine"] < 0.999


def test_image_only_cache_uses_predictions_tracker_and_not_truth_geometry(
    tmp_path: Path,
) -> None:
    first, first_model, first_visual = _build_image_only(
        tmp_path,
        output_name="image_first",
        truth_offset=0.0,
        truth_velocity_mps=0.1,
        frame_count=3,
    )
    changed, changed_model, changed_visual = _build_image_only(
        tmp_path,
        output_name="image_changed_truth",
        truth_offset=100.0,
        truth_velocity_mps=99.0,
        frame_count=3,
    )
    assert first["passed"] and changed["passed"]
    assert first_model.calls == changed_model.calls == 3
    assert first_visual.calls == changed_visual.calls == 3

    first_cache = Path(first["output"])
    changed_cache = Path(changed["output"])
    with np.load(first_cache / "frames_000.npz", allow_pickle=False) as left:
        with np.load(changed_cache / "frames_000.npz", allow_pickle=False) as right:
            np.testing.assert_allclose(left["entity_features"], right["entity_features"])
            np.testing.assert_allclose(left["entity_visual"], right["entity_visual"])
            assert np.all(left["entity_features"][0, :, 3:6] == 0.0)
            np.testing.assert_allclose(
                left["entity_features"][1, :4, 3], 0.2, atol=1.0e-6
            )
            assert np.all(left["entity_features"][1, :4, 4:6] == 0.0)
            np.testing.assert_allclose(
                left["entity_features"][2, :4, 3], 0.32, atol=1.0e-6
            )

    quality = json.loads(
        (first_cache / "quality_report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (first_cache / "manifest.json").read_text(encoding="utf-8")
    )
    assert quality["perception_model_enabled"] is True
    assert quality["perception_model_id"] == "asv_vla.image_entity_perception"
    assert quality["perception_model_version"] == "fake_image_entity_v1"
    assert manifest["models"]["image_perception"]["model_version"] == (
        "fake_image_entity_v1"
    )
    assert manifest["models"]["image_perception"]["weights_sha256"] == hashlib.sha256(
        b"fake-image-model"
    ).hexdigest()
    assert manifest["cache_key"]["image_perception"]["model_version"] == (
        "fake_image_entity_v1"
    )
    assert manifest["models"]["image_perception"]["tracker"] == {
        "ttl_frames": 2,
        "ttl_sec": 0.5,
        "velocity_filter": "ema",
        "alpha": 0.6,
        "beta": 0.85,
    }
    assert manifest["language_provenance"]["embedding_table_source"] == (
        "language_encoder"
    )
    assert manifest["language_provenance"]["frame_perception"] == {
        "enabled": True,
        "embedding_source": "instructions_manifest",
        "embedding_strategy": "first_instruction_row",
        "instruction_row": 0,
        "instruction_id": "instruction_00",
    }
    assert manifest["language_provenance"]["decision_samples"] == {
        "embedding_source": "language_table",
        "embedding_strategy": "per_frame_instruction_id",
        "pairing_key": ["frame_index", "instruction_id"],
    }


def test_image_prediction_failure_is_fail_closed_without_truth_fallback(
    tmp_path: Path,
) -> None:
    episode, _, _ = _make_sources(tmp_path, frame_count=1)
    record = json.loads(
        (episode / "frames" / "000000000000.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(FeatureCacheError, match="image perception failed"):
        _predict_image_entities(
            record,
            episode,
            FailingImageModel(),
            TemporalEntityTracker(),
        )


def test_bad_image_produces_no_predicted_entities(tmp_path: Path) -> None:
    episode, _, _ = _make_sources(tmp_path, frame_count=1)
    frame_path = episode / "frames" / "000000000000.json"
    record = json.loads(frame_path.read_text(encoding="utf-8"))
    (episode / record["camera"]["image_path"]).unlink()
    tracked = _predict_image_entities(
        record,
        episode,
        FakeImageModel(),
        TemporalEntityTracker(),
    )
    assert tracked == ()
