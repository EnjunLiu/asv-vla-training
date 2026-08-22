from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import pytest

from export_moving_target_bag import export_decoded_frames


def message(
    frame_index: int,
    *,
    x: float,
    y: float,
    yaw: float = 0.0,
    run_id: str = "RUN-A",
):
    common = {
        "run_id": run_id,
        "scene_seed": 42,
        "frame_index": frame_index,
        "stamp_us": frame_index * 100_000,
        "valid": True,
    }
    camera = SimpleNamespace(
        **common,
        data=bytes((0xFF, 0xD8, frame_index, 0xFF, 0xD9)),
        encoding="jpeg",
    )
    ego = SimpleNamespace(
        **common,
        simulation_time=frame_index * 0.1,
        position_x=x,
        position_y=y,
        position_z=0.0,
        roll=0.0,
        pitch=0.0,
        yaw=yaw,
        surge_velocity=0.5,
        yaw_rate=0.0,
    )
    entity = SimpleNamespace(
        entity_id="target_red",
        class_name="boat",
        color="red",
        is_target=True,
        visible=True,
        relative_x=3.0,
        relative_y=1.0,
        relative_z=0.0,
        relative_velocity_x=0.6,
        relative_velocity_y=0.0,
        relative_velocity_z=0.0,
        valid=True,
    )
    entities = SimpleNamespace(
        **common,
        frame_id="base_link",
        entities=[entity],
    )
    return {
        "/ue/camera_frame": camera,
        "/ue/asv_state": ego,
        "/ue/entities": entities,
    }


def test_export_preserves_jpeg_and_uses_next_executed_pose_as_action(tmp_path: Path) -> None:
    decoded = {
        ("RUN-A", 42, 10): message(10, x=1.0, y=2.0),
        ("RUN-A", 42, 11): message(11, x=1.2, y=2.1),
        ("RUN-A", 42, 12): message(12, x=1.3, y=2.1),
    }

    count = export_decoded_frames(
        decoded,
        tmp_path,
        task_text="follow the red boat, keep 3 meters distance",
        slot_id="RED_3M_TRAIN_01",
        layout_id="L7",
        motion_state="S2",
    )

    assert count == 2
    record = json.loads((tmp_path / "frames/000000000010.json").read_text())
    assert (tmp_path / record["camera"]["image_path"]).read_bytes() == decoded[
        ("RUN-A", 42, 10)
    ]["/ue/camera_frame"].data
    np.testing.assert_allclose(
        record["action"]["desired_displacement_m"], [0.2, 0.1], atol=1e-6
    )
    np.testing.assert_allclose(
        record["entities"]["items"][0]["world_position_m"], [4.0, 3.0, 0.0]
    )


def test_export_rotates_world_delta_into_body_frame(tmp_path: Path) -> None:
    decoded = {
        ("RUN-A", 42, 1): message(1, x=0.0, y=0.0, yaw=np.pi / 2),
        ("RUN-A", 42, 2): message(2, x=0.0, y=0.2, yaw=np.pi / 2),
    }
    export_decoded_frames(decoded, tmp_path, "task", "slot", "L7", "S2")
    record = json.loads((tmp_path / "frames/000000000001.json").read_text())
    np.testing.assert_allclose(
        record["action"]["desired_displacement_m"], [0.2, 0.0], atol=1e-6
    )


def test_export_rejects_incomplete_or_mixed_identity_frames(tmp_path: Path) -> None:
    incomplete = message(1, x=0.0, y=0.0)
    del incomplete["/ue/entities"]
    with pytest.raises(ValueError, match="complete synchronized frames"):
        export_decoded_frames(
            {("RUN-A", 42, 1): incomplete}, tmp_path, "task", "slot", "L7", "S2"
        )

    mixed = {
        ("RUN-A", 42, 1): message(1, x=0.0, y=0.0),
        ("RUN-B", 42, 2): message(2, x=0.1, y=0.0, run_id="RUN-B"),
    }
    with pytest.raises(ValueError, match="single run identity"):
        export_decoded_frames(mixed, tmp_path, "task", "slot", "L7", "S2")


def test_export_drops_duplicate_zero_duration_samples(tmp_path: Path) -> None:
    first = message(1, x=0.0, y=0.0)
    duplicate = message(2, x=0.0, y=0.0)
    following = message(3, x=0.2, y=0.0)
    duplicate["/ue/asv_state"].simulation_time = 0.0
    first["/ue/asv_state"].simulation_time = 0.0
    following["/ue/asv_state"].simulation_time = 0.1
    export_decoded_frames(
        {
            ("RUN-A", 42, 1): first,
            ("RUN-A", 42, 2): duplicate,
            ("RUN-A", 42, 3): following,
        },
        tmp_path,
        "task",
        "slot",
        "L7",
        "S2",
    )
    rows = list((tmp_path / "frames").glob("*.json"))
    assert [path.stem for path in rows] == ["000000000002"]
