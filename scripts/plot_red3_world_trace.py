#!/usr/bin/env python3
"""Plot world-frame ASV/target tracks from UE LogSceneAutomation output."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UE_LOG = ROOT / "experiments/chase_standoff_entityfeat_v2/red_3m_ue_run2.log"
DEFAULT_OUTPUT = ROOT / "experiments/chase_standoff_entityfeat_v2/red_3m_world_trace_run2.png"

ASV_RE = re.compile(r"SCENE_ASV_POS t=([0-9.]+) world=X=([-0-9.]+) Y=([-0-9.]+)")
TARGET_RE = re.compile(
    r"SCENE_TARGET_POS t=([0-9.]+) entity=(\S+) world=X=([-0-9.]+) Y=([-0-9.]+)"
)
COMPLETE_RE = re.compile(
    r"SCENE_UE_COMPLETE slot=(\S+) .* scene_seed=(\d+) runtime_seconds=([0-9.]+)"
)


def parse_ue_log(path: Path) -> dict:
    asv: dict[float, np.ndarray] = {}
    targets: dict[str, dict[float, np.ndarray]] = defaultdict(dict)
    complete = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ASV_RE.search(line)
        if match:
            time_s = round(float(match.group(1)), 1)
            asv[time_s] = np.array(
                [float(match.group(2)) / 100.0, float(match.group(3)) / 100.0],
                dtype=np.float32,
            )
            continue
        match = TARGET_RE.search(line)
        if match:
            time_s = round(float(match.group(1)), 1)
            targets[match.group(2)][time_s] = np.array(
                [float(match.group(3)) / 100.0, float(match.group(4)) / 100.0],
                dtype=np.float32,
            )
            continue
        match = COMPLETE_RE.search(line)
        if match:
            complete = {
                "slot": match.group(1),
                "scene_seed": int(match.group(2)),
                "runtime_seconds": float(match.group(3)),
            }
    return {"asv": asv, "targets": targets, "complete": complete}


def aligned_series(parsed: dict, target_id: str, desired_m: float):
    times = sorted(set(parsed["asv"]) & set(parsed["targets"].get(target_id, {})))
    if not times:
        raise ValueError(f"no aligned samples for {target_id}")
    asv = np.stack([parsed["asv"][time_s] for time_s in times])
    target = np.stack([parsed["targets"][target_id][time_s] for time_s in times])
    others = {
        name: np.stack([series[time_s] for time_s in times])
        for name, series in parsed["targets"].items()
        if name != target_id and all(time_s in series for time_s in times)
    }
    distance = np.linalg.norm(target - asv, axis=1)
    error = distance - float(desired_m)
    return np.asarray(times), asv, target, others, distance, error


def plot_world_trace(
    ue_log: Path,
    output: Path,
    *,
    title: str = "RED 3m | entityfeat_v2",
    target_id: str = "target_red",
    desired_m: float = 3.0,
) -> dict:
    parsed = parse_ue_log(ue_log)
    times, asv, target, others, distance, error = aligned_series(
        parsed, target_id, desired_m
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_xy, ax_err = axes

    styles = {
        "target_red": {"color": "tab:red", "linewidth": 2.0, "linestyle": "-"},
        "target_blue": {"color": "tab:blue", "linewidth": 1.6, "linestyle": "-"},
        "target_left": {"color": "0.45", "linewidth": 1.0, "linestyle": "--"},
        "target_right": {"color": "0.45", "linewidth": 1.0, "linestyle": ":"},
    }
    for name, series in others.items():
        ax_xy.plot(series[:, 0], series[:, 1], label=name, **styles.get(name, {}))
    ax_xy.plot(target[:, 0], target[:, 1], label=target_id, **styles.get(target_id, {}))
    ax_xy.plot(asv[:, 0], asv[:, 1], color="black", linewidth=2.6, label="ASV")
    ax_xy.scatter(asv[0, 0], asv[0, 1], color="green", s=50, zorder=4)
    ax_xy.scatter(asv[-1, 0], asv[-1, 1], color="red", s=50, zorder=4)
    ax_xy.set_title(f"{title} | world tracks")
    ax_xy.set_xlabel("World X (m)")
    ax_xy.set_ylabel("World Y (m)")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(alpha=0.25)
    ax_xy.legend(loc="best", fontsize=8)

    ax_err.plot(times, error, color="tab:red", linewidth=1.8, label="standoff error")
    ax_err.axhline(0.0, color="black", linewidth=0.8)
    ax_err.fill_between(times, error, 0.0, color="tab:red", alpha=0.12)
    ax_err.set_title("Signed standoff error")
    ax_err.set_xlabel("Runtime (s)")
    ax_err.set_ylabel("distance - 3.0 m")
    ax_err.grid(alpha=0.25)
    ax_err.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{title} | samples={len(times)}", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    summary = {
        "ue_log": str(ue_log),
        "output": str(output),
        "samples": int(len(times)),
        "complete": parsed["complete"],
        "mean_abs_error_m": float(np.mean(np.abs(error))),
        "final_error_m": float(error[-1]),
        "max_abs_error_m": float(np.max(np.abs(error))),
        "initial_error_m": float(error[0]),
        "mean_distance_m": float(np.mean(distance)),
    }
    (output.with_suffix(".json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ue-log", type=Path, default=DEFAULT_UE_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.ue_log.is_file():
        raise SystemExit(f"UE log not found: {args.ue_log}")
    summary = plot_world_trace(args.ue_log, args.output)
    print(json.dumps(summary, indent=2))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
