#!/usr/bin/env python3
"""Closed-loop run + world trace plot for one scenario (Windows orchestrates UE + Jetson)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JETSON = "jetson@192.168.137.100"
UE = Path(r"D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe")
UPROJECT = Path(r"D:\asv-unreal-simulation\VLA.uproject")
UE_SAVED_LOG = UPROJECT.parent / "Saved/Logs/VLA.log"
MARKERS = ("TASK_READY_VALID", "POLICY_READY")

SCENARIOS = {
    "red3": {
        "slot": "RED_3M_TEST",
        "task_text": "follow the red boat, keep 3 meters distance",
        "target_id": "target_red",
        "desired_m": 3.0,
        "title": "RED 3m | scale",
    },
    "blue3": {
        "slot": "BLUE_3M_TEST",
        "task_text": "follow the blue boat, keep 3 meters distance",
        "target_id": "target_blue",
        "desired_m": 3.0,
        "title": "BLUE 3m | scale",
    },
    "red4": {
        "slot": "RED_4M_TEST",
        "task_text": "follow the red boat, keep 4 meters distance",
        "target_id": "target_red",
        "desired_m": 4.0,
        "title": "RED 4m | scale",
    },
}


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, text=True, capture_output=True, encoding="utf-8", errors="replace")


def extract_scene_lines(source: Path, destination: Path) -> int:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/chase_standoff_refactor")
    parser.add_argument("--runtime", type=int, default=180)
    args = parser.parse_args()
    cfg = SCENARIOS[args.scenario]
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"closed_loop_{args.scenario}.log"
    remote_log = f"/home/jetson/jetson_asv_ws/logs/{log_name}"
    local_log = out_dir / log_name
    ue_log = out_dir / f"{args.scenario}_ue.log"
    plot_out = out_dir / f"{args.scenario}_world_trace.png"

    subprocess.run(["taskkill", "/IM", "UnrealEditor.exe", "/F"], capture_output=True, text=True)
    time.sleep(3)
    restart = ROOT / "scripts/jetson_restart.sh"
    task_file = out_dir / f"{args.scenario}_task.txt"
    task_file.write_text(cfg["task_text"], encoding="utf-8")
    run(["scp", "-o", "BatchMode=yes", str(restart), f"{JETSON}:/tmp/jetson_restart.sh"])
    run(["scp", "-o", "BatchMode=yes", str(task_file), f"{JETSON}:/tmp/scenario_task.txt"])
    run([
        "ssh", "-o", "BatchMode=yes", JETSON,
        "sed -i 's/\\r$//' /tmp/jetson_restart.sh && chmod +x /tmp/jetson_restart.sh && "
        f"LOG_NAME={log_name} TASK_TEXT=\"$(cat /tmp/scenario_task.txt)\" bash /tmp/jetson_restart.sh",
    ])
    deadline = time.time() + 240
    while time.time() < deadline:
        text = run(["ssh", "-o", "BatchMode=yes", JETSON, f"tail -n 120 {remote_log} 2>/dev/null || true"]).stdout
        hits = {marker: marker in text for marker in MARKERS}
        print(hits)
        if all(hits.values()):
            break
        time.sleep(5)
    else:
        print("TIMEOUT waiting for stack ready", file=sys.stderr)
        return 2

    time.sleep(90)
    ue_args = [
        str(UE), str(UPROJECT), "Main_Map",
        "-game", "-SceneAuto",
        f"-Slot={cfg['slot']}", "-Layout=L7B", "-Motion=S2", "-Seed=231106",
        "-SceneExecPort=8081", f"-MaxRuntimeSeconds={args.runtime}", "-YawFixWholeRun", "-log",
    ]
    with ue_log.open("w", encoding="utf-8", errors="replace") as log_file:
        ue = subprocess.Popen(ue_args, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(UPROJECT.parent))
        deadline = time.time() + args.runtime + 15
        while time.time() < deadline and ue.poll() is None:
            time.sleep(15)
        if ue.poll() is None:
            ue.terminate()
            ue.wait(timeout=10)

    if UE_SAVED_LOG.is_file():
        extract_scene_lines(UE_SAVED_LOG, ue_log)
    run(["scp", "-o", "BatchMode=yes", f"{JETSON}:{remote_log}", str(local_log)])
    plot = run([
        sys.executable,
        str(ROOT / "scripts/plot_trace.py"),
        "--ue-log", str(ue_log),
        "--output", str(plot_out),
        "--title", cfg["title"],
        "--target-id", cfg["target_id"],
        "--desired-m", str(cfg["desired_m"]),
    ])
    print(plot.stdout)
    if plot.returncode != 0:
        print(plot.stderr, file=sys.stderr)
        return plot.returncode
    scene_lines = sum(1 for line in ue_log.read_text(encoding="utf-8", errors="replace").splitlines() if "SCENE_ASV_POS" in line)
    policy_hits = sum(1 for line in local_log.read_text(encoding="utf-8", errors="replace").splitlines() if "POLICY_INFERRED_SMOOTHED" in line)
    print({"plot": str(plot_out), "scene_asv_samples": scene_lines, "policy_inferred": policy_hits})
    return 0 if scene_lines >= 50 and policy_hits >= 30 else 3


if __name__ == "__main__":
    raise SystemExit(main())
