#!/usr/bin/env python3
"""Plot world-coordinate closed-loop trace from UE log."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASV_RE = re.compile(r"SCENE_ASV_POS t=([0-9.]+) world=X=([-0-9.]+) Y=([-0-9.]+)")
TARGET_RE = re.compile(
    r"SCENE_TARGET_POS t=([0-9.]+) entity=(\S+) world=X=([-0-9.]+) Y=([-0-9.]+)"
)


def parse_ue_log(path: Path) -> dict:
    asv: dict[float, np.ndarray] = {}
    targets: dict[str, dict[float, np.ndarray]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ASV_RE.search(line)
        if match:
            time_s = round(float(match.group(1)), 1)
            asv[time_s] = np.array([float(match.group(2)), float(match.group(3))], dtype=np.float32) / 100.0
            continue
        match = TARGET_RE.search(line)
        if match:
            time_s = round(float(match.group(1)), 1)
            targets[match.group(2)][time_s] = np.array(
                [float(match.group(3)), float(match.group(4))], dtype=np.float32
            ) / 100.0
    return {"asv": asv, "targets": targets}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ue-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="closed loop")
    parser.add_argument("--target-id", default="target_red")
    parser.add_argument("--desired-m", type=float, default=3.0)
    args = parser.parse_args()
    parsed = parse_ue_log(args.ue_log)
    times = sorted(set(parsed["asv"]) & set(parsed["targets"].get(args.target_id, {})))
    asv = np.stack([parsed["asv"][t] for t in times])
    target = np.stack([parsed["targets"][args.target_id][t] for t in times])
    distance = np.linalg.norm(target - asv, axis=1)
    error = distance - args.desired_m

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.plot(asv[:, 0], asv[:, 1], label="ASV", linewidth=+2)
    ax.plot(target[:, 0], target[:, 1], label=args.target_id, linewidth=2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(args.title)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    axes[1].plot(times, error, color="tab:red", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("standoff error")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("error (m)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    summary = {
        "samples": len(times),
        "mean_abs_error_m": float(np.mean(np.abs(error))),
        "max_abs_error_m": float(np.max(np.abs(error))),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
