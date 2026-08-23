#!/usr/bin/env python3
"""RED 3m closed-loop smoke for entityfeat_v2 policy with real embeddings."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

LOG_NAME = "closed_loop_red3_entityfeat_v2_20260823.log"
JETSON = "jetson@192.168.137.100"
REMOTE_LOG = f"/home/jetson/jetson_asv_ws/logs/{LOG_NAME}"
LOCAL_LOG = Path(__file__).resolve().parents[1] / "experiments/chase_standoff_entityfeat_v2" / LOG_NAME
RESTART = Path(__file__).resolve().parents[1] / "scripts/jetson_restart_red3_entityfeat_v2.sh"
UE = Path(r"D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe")
UPROJECT = Path(r"D:\asv-unreal-simulation\VLA.uproject")
UE_LOG = LOCAL_LOG.with_suffix(".ue.log")

MARKERS = ("TASK_READY_VALID", "POLICY_READY", "entity_embedding=on")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def main() -> int:
    LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    run(["scp", "-o", "BatchMode=yes", str(RESTART), f"{JETSON}:/tmp/jetson_restart_red3_entityfeat_v2.sh"])
    run(["ssh", "-o", "BatchMode=yes", JETSON, "sed -i 's/\\r$//' /tmp/jetson_restart_red3_entityfeat_v2.sh && chmod +x /tmp/jetson_restart_red3_entityfeat_v2.sh"])
    launch = run(["ssh", "-o", "BatchMode=yes", JETSON, "bash /tmp/jetson_restart_red3_entityfeat_v2.sh"])
    print(launch.stdout)
    if launch.returncode != 0:
        print(launch.stderr, file=sys.stderr)
        return launch.returncode

    deadline = time.time() + 240
    ready = False
    while time.time() < deadline:
        text = run(["ssh", "-o", "BatchMode=yes", JETSON, f"tail -n 120 {REMOTE_LOG} 2>/dev/null || true"]).stdout
        hits = {marker: marker in text for marker in MARKERS}
        print(hits)
        if "traceback" in text.lower() or "policy_load_error" in text.lower():
            print(text[-3000:])
            return 1
        if all(hits.values()):
            ready = True
            break
        time.sleep(5)

    if not ready:
        print("TIMEOUT waiting for stack ready")
        return 2

    ue_args = [
        str(UE), str(UPROJECT), "Main_Map",
        "-game", "-SceneAuto",
        "-Slot=RED_3M_TEST", "-Layout=L7B", "-Motion=S2", "-Seed=231106",
        "-SceneExecPort=8081", "-MaxRuntimeSeconds=120", "-YawFixWholeRun",
        f"-Log={UE_LOG}",
    ]
    print("+ starting UE:", " ".join(ue_args))
    ue = subprocess.Popen(ue_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("STACK_READY; UE started; collecting 90s runtime...")
    time.sleep(90)

    run(["scp", "-o", "BatchMode=yes", f"{JETSON}:{REMOTE_LOG}", str(LOCAL_LOG)])
    text = LOCAL_LOG.read_text(encoding="utf-8", errors="replace") if LOCAL_LOG.exists() else ""
    policy_hits = sum(1 for line in text.splitlines() if "POLICY_INFERRED_SMOOTHED" in line)
    setpoints = sum(1 for line in text.splitlines() if "Sent kinematic setpoint to UE5" in line and "POLICY_INFERRED" in line)
    entities_ok = sum(1 for line in text.splitlines() if "entities=" in line)
    fail_closed = sum(1 for line in text.splitlines() if "fail_closed=1" in line or "FAIL_CLOSED" in line)
    summary = {
        "log": str(LOCAL_LOG),
        "entity_embedding_on": "entity_embedding=on" in text,
        "ue_connected": "UE5 connected" in text,
        "executor_connected": "Connected to UE5 kinematic executor" in text,
        "entity_valid_frames": entities_ok,
        "policy_inferred": policy_hits,
        "setpoints_sent": setpoints,
        "fail_closed_traces": fail_closed,
        "pass": (
            "entity_embedding=on" in text
            and "UE5 connected" in text
            and setpoints >= 5
            and entities_ok >= 3
            and fail_closed == 0
            and policy_hits >= 3
        ),
    }
    print("SUMMARY", summary)
    try:
        ue.wait(timeout=10)
    except subprocess.TimeoutExpired:
        ue.terminate()
    return 0 if summary["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
