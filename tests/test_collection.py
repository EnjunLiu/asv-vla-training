from __future__ import annotations

import json
from pathlib import Path

import pytest

from asv_training.dataset_registry import build_registry
from asv_training.collection import discover_slots, load_plan, validate_slot


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _plan() -> dict:
    return {
        "minimum_frames_per_run": 2,
        "required_execution_mode": "ue5_kinematic_expert_v1",
        "required_entity_ids": [
            "target_red",
            "target_blue",
            "target_left",
            "target_right",
        ],
        "relation_margin_m": 0.25,
        "relation_evaluation_frames": 2,
        "minimum_relation_pass_fraction": 0.8,
        "motion_evaluation_frames": 2,
        "minimum_motion_pass_fraction": 0.6,
        "minimum_pairwise_distance_change_m": 0.05,
    }


def _slot() -> dict:
    return {
        "slot_id": "L1_S0_R1",
        "layout_id": "L1",
        "motion_state": "S0",
        "scene_seed": 120101,
        "relations": [
            ["nearer", "target_red", "target_blue"],
            ["left_of", "target_left", "target_right"],
        ],
    }


def _entity(
    entity_id: str,
    color: str,
    x_value: float,
    y_value: float,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
) -> dict:
    return {
        "entity_id": entity_id,
        "color": color,
        "relative_position_m": [x_value, y_value, 0.0],
        "relative_velocity_mps": [velocity_x, velocity_y, 0.0],
        "valid": True,
        "visible": True,
        "is_target": True,
    }


def _make_run(
    tmp_path: Path,
    *,
    swap_depth: bool = False,
    motion_state: str = "S0",
    distinct_motion: bool = False,
) -> tuple[Path, Path]:
    episode = tmp_path / "artifacts" / "day8_episode" / "RUN_001"
    supervision = (
        tmp_path / "artifacts" / "day10_supervised" / "RUN_001"
    )
    manifest = {
        "run_id": "RUN_001",
        "scene_seed": 120101,
        "frame_count": 2,
        "status": "complete",
        "execution_mode": "ue5_kinematic_expert_v1",
        "collection": {
            "slot_id": f"L1_{motion_state}_R1",
            "layout_id": "L1",
            "motion_state": motion_state,
        },
    }
    _write(episode / "manifest.json", manifest)
    _write(
        episode / "quality_report.json",
        {"passed": True, "run_id": "RUN_001", "frame_count": 2},
    )
    red_x, blue_x = ((4.0, 1.0) if swap_depth else (1.0, 4.0))
    for frame_index in range(2):
        _write(
            episode / "frames" / f"{frame_index:012d}.json",
            {
                "entities": {
                    "items": [
                        _entity(
                            "target_red",
                            "red",
                            red_x,
                            0.1 * frame_index if distinct_motion else 0.0,
                            velocity_y=0.08 if distinct_motion else 0.0,
                        ),
                        _entity("target_blue", "blue", blue_x, 0.0),
                        _entity("target_left", "white", 2.0, 1.0),
                        _entity("target_right", "white", 2.0, -1.0),
                    ]
                }
            },
        )
    _write(
        supervision / "manifest.json",
        {
            "source_episodes": [{"run_id": "RUN_001"}],
            "samples": {"frame_count": 2, "sample_count": 180},
            "label_coverage": {
                "complete": True,
                "observed_labels": [f"label_{index}" for index in range(9)],
                "required_labels": [f"label_{index}" for index in range(9)],
            },
        },
    )
    return episode, supervision


def test_near_range_sine_plan_has_mirrored_l7_slots() -> None:
    plan_path = (
        Path(__file__).parents[1]
        / "configs"
        / "sine_near_collection_plan_v1.json"
    )
    plan = load_plan(plan_path)

    assert plan["minimum_complete_runs"] == 12
    assert len(plan["slots"]) == 12
    assert {slot["layout_id"] for slot in plan["slots"]} == {"L7", "L7B"}
    assert {slot["motion_state"] for slot in plan["slots"]} == {"S2"}
    assert plan["minimum_frames_per_run"] == 80
    assert plan["target_frames_per_run"] == 100
    assert plan["required_execution_mode"] == "ue5_kinematic_expert_v1"
    assert plan["relation_evaluation_frames"] == 1
    assert plan["motion_evaluation_frames"] == 50
    assert plan["minimum_relation_pass_fraction"] == 1.0
    assert plan["minimum_motion_pass_fraction"] == 0.6
    assert "Final S2 near-range collection" in plan["note"]
    assert "4.5 m target-pair spawn distance" in plan["note"]
    assert "7 m white distractors" in plan["note"]
    assert {slot["scene_seed"] for slot in plan["slots"]} == set(
        range(220701, 220707)
    ) | set(range(220801, 220807))
    for slot in plan["slots"]:
        expected = (
            ["left_of", "target_red", "target_blue"]
            if slot["layout_id"] == "L7"
            else ["left_of", "target_blue", "target_red"]
        )
        assert expected in slot["relations"]
        assert ["left_of", "target_left", "target_right"] in slot["relations"]


def test_remote_collection_forwards_execution_endpoint() -> None:
    repository = Path(__file__).parents[2]
    if not (repository / "scripts" / "remote_collect.sh").is_file():
        pytest.skip("collection automation is outside this PC training import")
    remote_collect = (
        repository / "scripts" / "remote_collect.sh"
    ).read_text(encoding="utf-8")
    ue_collect = (
        repository / "tools" / "ue5" / "collect.ps1"
    ).read_text(encoding="utf-8")

    assert 'execution_address:="$execution_address"' in remote_collect
    assert 'execution_port:="$execution_port"' in remote_collect
    assert 'max_speed_mps:="$max_speed_mps"' in remote_collect
    assert "execution_address=${EXECUTION_ADDRESS:-}" in remote_collect
    assert "execution_port=${EXECUTION_PORT:-8081}" in remote_collect
    assert "max_speed_mps=${MAX_SPEED_MPS:-0.8}" in remote_collect
    assert '[string]$ExecutionAddress = "192.168.137.1"' in ue_collect
    assert "[int]$ExecutionPort = 8081" in ue_collect
    assert "EXECUTION_ADDRESS='$ExecutionAddress'" in ue_collect
    assert "EXECUTION_PORT='$ExecutionPort'" in ue_collect
    assert "[double]$MaxSpeedMps = 0.8" in ue_collect
    assert "MAX_SPEED_MPS='$MaxSpeedMps'" in ue_collect
    assert '"-SceneExecPort=$ExecutionPort"' in ue_collect

    completion = remote_collect.split(
        'if [[ $completion_seen == true ]] && kill -0 "$launch_pid"',
        1,
    )[1]
    assert "recorder_wait_deadline=$((SECONDS + 8))" in completion
    assert 'while kill -0 "$recorder_pid"' in completion
    wait_index = completion.index("recorder_wait_deadline")
    term_index = completion.index('kill -TERM "$recorder_pid"')
    assert wait_index < term_index
    assert 'if kill -0 "$recorder_pid" 2>/dev/null; then' in completion


def test_collection_plan_inheritance_cycle_is_rejected(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(
        first,
        {
            "schema_version": "collection_plan_v1",
            "base_plan": "second.json",
            "slots": [],
        },
    )
    _write(
        second,
        {
            "schema_version": "collection_plan_v1",
            "base_plan": "first.json",
            "slots": [],
        },
    )

    try:
        load_plan(first)
    except ValueError as exc:
        assert "inheritance cycle" in str(exc)
    else:
        raise AssertionError("expected inheritance cycle rejection")


def test_slot_validator_checks_observed_geometry(tmp_path: Path) -> None:
    episode, supervision = _make_run(tmp_path)

    report = validate_slot(_slot(), episode, supervision, _plan())

    assert report["passed"]
    assert report["relation_pass_fractions"] == [1.0, 1.0]


def test_slot_validator_rejects_manifest_claim_when_geometry_is_wrong(
    tmp_path: Path,
) -> None:
    episode, supervision = _make_run(tmp_path, swap_depth=True)

    report = validate_slot(_slot(), episode, supervision, _plan())

    assert not report["passed"]
    assert any("relation" in error for error in report["errors"])


def test_s1_slot_requires_observable_distinct_target_motion(
    tmp_path: Path,
) -> None:
    episode, supervision = _make_run(tmp_path, motion_state="S1")
    slot = _slot()
    slot["slot_id"] = "L1_S1_R1"
    slot["motion_state"] = "S1"

    report = validate_slot(slot, episode, supervision, _plan())

    assert not report["passed"]
    assert report["motion_pass_fraction"] == 0.0
    assert any(
        "pairwise target-distance motion" in error
        for error in report["errors"]
    )


def test_s1_slot_accepts_distinct_target_motion(tmp_path: Path) -> None:
    episode, supervision = _make_run(
        tmp_path,
        motion_state="S1",
        distinct_motion=True,
    )
    slot = _slot()
    slot["slot_id"] = "L1_S1_R1"
    slot["motion_state"] = "S1"

    report = validate_slot(slot, episode, supervision, _plan())

    assert report["passed"]
    assert report["motion_pass_fraction"] == 1.0


def test_registry_requires_collection_metadata_and_twelve_runs(
    tmp_path: Path,
) -> None:
    episode, _ = _make_run(tmp_path)
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_count"] = 80
    _write(manifest_path, manifest)
    quality_path = episode / "quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["frame_count"] = 80
    _write(quality_path, quality)
    supervision_path = (
        tmp_path
        / "artifacts"
        / "day10_supervised"
        / "RUN_001"
        / "manifest.json"
    )
    supervision = json.loads(supervision_path.read_text(encoding="utf-8"))
    supervision["samples"]["frame_count"] = 80
    _write(supervision_path, supervision)

    registry = tmp_path / "registry" / "dataset_registry_v1.jsonl"
    report = build_registry(tmp_path, registry)
    entry = json.loads(registry.read_text(encoding="utf-8"))

    assert entry["training_eligible"] is True
    assert report["eligible_run_count"] == 1
    assert report["training_ready"] is False
    assert report["minimum_runs_for_training"] == 12


def test_registry_reads_legacy_static_pilot_but_does_not_count_it(
    tmp_path: Path,
) -> None:
    episode, _ = _make_run(tmp_path)
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_mode"] = "static"
    manifest.pop("collection")
    _write(manifest_path, manifest)

    registry = tmp_path / "registry" / "legacy.jsonl"
    report = build_registry(tmp_path, registry)
    entry = json.loads(registry.read_text(encoding="utf-8"))

    assert entry["execution_mode"] == "static"
    assert entry["training_eligible"] is False
    assert report["eligible_run_count"] == 0


def test_latest_symlink_is_not_a_duplicate_collection_slot(
    tmp_path: Path,
) -> None:
    _make_run(tmp_path)
    latest = tmp_path / "artifacts" / "day8_episode" / "latest"
    try:
        latest.symlink_to("RUN_001", target_is_directory=True)
    except OSError as exc:
        # Windows requires Developer Mode or SeCreateSymbolicLinkPrivilege.
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("directory symlink privilege is unavailable")
        raise

    discovered, errors = discover_slots(tmp_path)

    assert set(discovered) == {"L1_S0_R1"}
    assert errors == []
