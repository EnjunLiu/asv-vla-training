from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOG_DIR = Path(r"D:\asv-vla-training\experiments\chase_standoff_candidate\closed_loop_logs")
OUTPUT = Path(r"C:\Users\LIU\Desktop\track_world_chase_standoff_2x3.png")
OUTPUT_COPY = Path(
    r"D:\asv-vla-training\experiments\chase_standoff_candidate\track_world_chase_standoff_2x3.png"
)

SCENES = (
    ("RED 4m", LOG_DIR / "red_4m_ue.log", "target_red", 4.0, "red"),
    ("BLUE 3m", LOG_DIR / "blue_3m_ue.log", "target_blue", 3.0, "blue"),
    ("RED 3m", LOG_DIR / "red_3m_ue.log", "target_red", 3.0, "red"),
)

ASV_RE = re.compile(r"SCENE_ASV_POS t=([0-9.]+) world=X=([-0-9.]+) Y=([-0-9.]+)")
TARGET_RE = re.compile(
    r"SCENE_TARGET_POS t=([0-9.]+) entity=(\S+) world=X=([-0-9.]+) Y=([-0-9.]+)"
)
APPLY_RE = re.compile(r"SCENE_EXEC_APPLY slot=(\S+) count=(\d+)")
COMPLETE_RE = re.compile(
    r"SCENE_UE_COMPLETE slot=(\S+) .* scene_seed=(\d+) runtime_seconds=([0-9.]+)"
)


def parse_log(path: Path) -> dict:
    asv: dict[float, np.ndarray] = {}
    targets: dict[str, dict[float, np.ndarray]] = defaultdict(dict)
    apply_count = 0
    complete = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ASV_RE.search(line)
        if match:
            t = round(float(match.group(1)), 1)
            asv[t] = np.array(
                [float(match.group(2)) / 100.0, float(match.group(3)) / 100.0],
                dtype=np.float32,
            )
            continue
        match = TARGET_RE.search(line)
        if match:
            t = round(float(match.group(1)), 1)
            targets[match.group(2)][t] = np.array(
                [float(match.group(3)) / 100.0, float(match.group(4)) / 100.0],
                dtype=np.float32,
            )
            continue
        match = APPLY_RE.search(line)
        if match:
            apply_count = max(apply_count, int(match.group(2)))
            continue
        match = COMPLETE_RE.search(line)
        if match:
            complete = {
                "slot": match.group(1),
                "scene_seed": int(match.group(2)),
                "runtime_seconds": float(match.group(3)),
            }
    return {
        "asv": asv,
        "targets": targets,
        "apply_count": apply_count,
        "complete": complete,
    }


def aligned_series(parsed: dict, target_id: str, desired: float):
    times = sorted(set(parsed["asv"]) & set(parsed["targets"][target_id]))
    asv = np.stack([parsed["asv"][t] for t in times])
    target = np.stack([parsed["targets"][target_id][t] for t in times])
    others = {
        name: np.stack([series[t] for t in times])
        for name, series in parsed["targets"].items()
        if times and all(t in series for t in times)
    }
    distance = np.linalg.norm(target - asv, axis=1)
    error = distance - float(desired)
    return np.asarray(times), asv, others, error


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    summary = {}
    styles = {
        "target_red": {"color": "tab:red", "linewidth": 1.8, "linestyle": "-"},
        "target_blue": {"color": "tab:blue", "linewidth": 1.8, "linestyle": "-"},
        "target_left": {"color": "0.45", "linewidth": 1.0, "linestyle": "--"},
        "target_right": {"color": "0.45", "linewidth": 1.0, "linestyle": ":"},
    }
    for column, (title, path, target_id, desired, color) in enumerate(SCENES):
        parsed = parse_log(path)
        times, asv, others, error = aligned_series(parsed, target_id, desired)
        ax_xy = axes[0, column]
        ax_err = axes[1, column]
        for name, series in others.items():
            ax_xy.plot(series[:, 0], series[:, 1], label=name, **styles.get(name, {}))
        ax_xy.plot(asv[:, 0], asv[:, 1], color="black", linewidth=2.4, label="ASV")
        ax_xy.set_title(title)
        ax_xy.set_xlabel("World X (m)")
        ax_xy.set_ylabel("World Y (m)")
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.grid(alpha=0.25)
        ax_err.plot(times, error, color=color, linewidth=1.6, label=f"{target_id} error")
        ax_err.axhline(0.0, color="black", linewidth=0.8)
        ax_err.set_xlabel("Runtime (s)")
        ax_err.set_ylabel("Standoff error (m)")
        ax_err.grid(alpha=0.25)
        ax_err.legend(loc="upper right", fontsize=8)
        summary[title] = {
            "log": str(path),
            "samples": int(len(times)),
            "apply_count": parsed["apply_count"],
            "complete": parsed["complete"],
            "mean_abs_error_m": float(np.mean(np.abs(error))),
            "final_error_m": float(error[-1]),
            "max_abs_error_m": float(np.max(np.abs(error))),
            "initial_error_m": float(error[0]),
        }
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "UE world tracks and signed standoff error | chase_standoff (language stamp)"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200)
    fig.savefig(OUTPUT_COPY, dpi=200)
    plt.close(fig)
    (LOG_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
