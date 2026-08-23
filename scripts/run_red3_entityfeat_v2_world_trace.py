#!/usr/bin/env python3
"""RED 3m closed loop with UE world-coordinate log capture."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

LOG_NAME = "closed_loop_red3_entityfeat_v2.log"
JETSON = "jetson@192.168.137.100"
REMOTE_LOG = f"/home/jetson/jetson_asv_ws/logs/{LOG_NAME}"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments/chase_standoff_entityfeat_v2"
LOCAL_LOG = OUT_DIR / LOG_NAME
UE_LOG = OUT_DIR / "red_3m_ue.log"
PLOT_OUT = OUT_DIR / "red_3m_world_trace.png"
RESTART = ROOT / "scripts/jetson_restart_red3_entityfeat_v2.sh"
UE = Path(r"D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe")
UPROJECT = Path(r"D:\asv-unreal-simulation\VLA.uproject")
UE_SAVED_LOG = Path(r"D:\asv-unreal-simulation\Saved\Logs\VLA.log")
UE_RUNTIME_SEC = 180

MARKERS = ("TASK_READY_VALID", "POLICY_READY")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def extract_scene_lines(source: Path, destination: Path) -> int:
    if not source.is_file():
        return 0
    lines = [
        line
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines()
        if "SCENE_ASV_POS" in line
        or "SCENE_TARGET_POS" in line
        or "SCENE_UE_COMPLETE" in line
        or "SCENE_EXEC_APPLY" in line
    ]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["taskkill", "/IM", "UnrealEditor.exe", "/F"],
        capture_output=True,
        text=True,
    )
    time.sleep(3)
    run([
        "ssh", "-o", "BatchMode=yes", JETSON,
        "pkill -f 'ros2 launch bringup vla_closed_loop' || true; "
        "pkill -f '/install/lib/vla/' || true; "
        "pkill -f '/install/lib/bridge/bridge_node' || true; "
        "sleep 8",
    ])
    run(["scp", "-o", "BatchMode=yes", str(RESTART), f"{JETSON}:/tmp/jetson_restart_red3_entityfeat_v2.sh"])
    run(["ssh", "-o", "BatchMode=yes", JETSON, "sed -i 's/\\r$//' /tmp/jetson_restart_red3_entityfeat_v2.sh && chmod +x /tmp/jetson_restart_red3_entityfeat_v2.sh"])
    launch = run(["ssh", "-o", "BatchMode=yes", JETSON, "bash /tmp/jetson_restart_red3_entityfeat_v2.sh"])
    print(launch.stdout)
    if launch.returncode != 0:
        print(launch.stderr, file=sys.stderr)
        return launch.returncode

    deadline = time.time() + 240
    while time.time() < deadline:
        text = run(["ssh", "-o", "BatchMode=yes", JETSON, f"tail -n 120 {REMOTE_LOG} 2>/dev/null || true"]).stdout
        hits = {marker: marker in text for marker in MARKERS}
        print(hits)
        if "traceback" in text.lower() or "policy_load_error" in text.lower():
            print(text[-3000:])
            return 1
        if all(hits.values()):
            break
        time.sleep(5)
    else:
        print("TIMEOUT waiting for stack ready")
        return 2

    print("STACK_READY; waiting 90s for CUDA/task warmup...")
    time.sleep(90)

    ue_args = [
        str(UE), str(UPROJECT), "Main_Map",
        "-game", "-SceneAuto",
        "-Slot=RED_3M_TEST", "-Layout=L7B", "-Motion=S2", "-Seed=231106",
        "-SceneExecPort=8081", f"-MaxRuntimeSeconds={UE_RUNTIME_SEC}", "-YawFixWholeRun",
        "-log",
    ]
    print("+ starting UE:", " ".join(ue_args))
    UE_LOG.write_text("", encoding="utf-8")
    with UE_LOG.open("a", encoding="utf-8", errors="replace") as log_file:
        ue = subprocess.Popen(
            ue_args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(UPROJECT.parent),
        )
        print(f"STACK_READY; UE running up to {UE_RUNTIME_SEC + 15}s for full scene...")
        deadline = time.time() + UE_RUNTIME_SEC + 15
        while time.time() < deadline and ue.poll() is None:
            time.sleep(15)
            text = run(["ssh", "-o", "BatchMode=yes", JETSON, f"tail -n 200 {REMOTE_LOG} 2>/dev/null || true"]).stdout
            policy_hits = sum(1 for line in text.splitlines() if "POLICY_INFERRED_SMOOTHED" in line)
            print(f"policy_hits={policy_hits}")
        try:
            ue.wait(timeout=15)
        except subprocess.TimeoutExpired:
            ue.terminate()
            ue.wait(timeout=10)

    if UE_SAVED_LOG.is_file():
        shutil.copy2(UE_SAVED_LOG, OUT_DIR / "red_3m_ue_saved.log")
        saved_lines = extract_scene_lines(UE_SAVED_LOG, UE_LOG)
        print(f"extracted {saved_lines} scene lines from Saved/Logs/VLA.log")
    else:
        extract_scene_lines(UE_LOG, UE_LOG)

    run(["scp", "-o", "BatchMode=yes", f"{JETSON}:{REMOTE_LOG}", str(LOCAL_LOG)])
    scene_lines = sum(
        1
        for line in UE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        if "SCENE_ASV_POS" in line
    )
    if scene_lines < 5:
        print(f"WARNING: only {scene_lines} SCENE_ASV_POS lines in {UE_LOG}")

    plot = run([sys.executable, str(ROOT / "scripts/plot_red3_world_trace.py"), "--ue-log", str(UE_LOG), "--output", str(PLOT_OUT)])
    print(plot.stdout)
    if plot.returncode != 0:
        print(plot.stderr, file=sys.stderr)
        return plot.returncode

    text = LOCAL_LOG.read_text(encoding="utf-8", errors="replace") if LOCAL_LOG.exists() else ""
    summary = {
        "jetson_log": str(LOCAL_LOG),
        "ue_log": str(UE_LOG),
        "plot": str(PLOT_OUT),
        "scene_asv_samples": scene_lines,
        "policy_inferred": sum(1 for line in text.splitlines() if "POLICY_INFERRED_SMOOTHED" in line),
        "ue_connected": "UE5 connected" in text,
    }
    print("SUMMARY", summary)
    return 0 if scene_lines >= 100 and summary["policy_inferred"] >= 50 else 3


if __name__ == "__main__":
    raise SystemExit(main())
