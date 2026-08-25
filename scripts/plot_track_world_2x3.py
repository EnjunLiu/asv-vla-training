#!/usr/bin/env python3
"""Plot 2x3 UE world tracks and signed standoff error (reference layout)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_trace import parse_ue_log

SCENARIOS = (
    {
        "key": "red4",
        "title": "RED 4m",
        "target_id": "target_red",
        "desired_m": 4.0,
        "error_color": "tab:red",
        "error_label": "target_red error",
    },
    {
        "key": "blue3",
        "title": "BLUE 3m",
        "target_id": "target_blue",
        "desired_m": 3.0,
        "error_color": "tab:blue",
        "error_label": "target_blue error",
    },
    {
        "key": "red3",
        "title": "RED 3m",
        "target_id": "target_red",
        "desired_m": 3.0,
        "error_color": "tab:red",
        "error_label": "target_red error",
    },
)


def _series(parsed: dict, target_id: str, desired_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = sorted(set(parsed["asv"]) & set(parsed["targets"].get(target_id, {})))
    asv = np.stack([parsed["asv"][t] for t in times])
    target = np.stack([parsed["targets"][target_id][t] for t in times])
    distance = np.linalg.norm(target - asv, axis=1)
    error = distance - desired_m
    return np.asarray(times), asv, target, error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closed-loop-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    summary: dict[str, object] = {}

    for col, scenario in enumerate(SCENARIOS):
        ue_log = args.closed_loop_dir / f"{scenario['key']}_ue.log"
        parsed = parse_ue_log(ue_log)
        times, asv, target, error = _series(parsed, scenario["target_id"], scenario["desired_m"])

        ax_top = axes[0, col]
        ax_top.plot(asv[:, 0], asv[:, 1], color="black", linewidth=2.2, label="ASV")
        if scenario["target_id"] in parsed["targets"]:
            ax_top.plot(target[:, 0], target[:, 1], color=scenario["error_color"], linewidth=2.0, label=scenario["target_id"])
        for extra_id, style in (("target_left", "--"), ("target_right", ":")):
            if extra_id in parsed["targets"]:
                extra_times = sorted(parsed["targets"][extra_id])
                if extra_times:
                    extra = np.stack([parsed["targets"][extra_id][t] for t in extra_times])
                    ax_top.plot(extra[:, 0], extra[:, 1], color="0.55", linestyle=style, linewidth=1.4, label=extra_id)
        ax_top.set_title(scenario["title"])
        ax_top.set_xlabel("World X (m)")
        ax_top.set_ylabel("World Y (m)")
        ax_top.set_aspect("equal", adjustable="box")
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(fontsize=8)

        ax_bot = axes[1, col]
        ax_bot.plot(times, error, color=scenario["error_color"], linewidth=1.8, label=scenario["error_label"])
        ax_bot.axhline(0.0, color="black", linewidth=0.8)
        ax_bot.set_xlabel("Runtime (s)")
        ax_bot.set_ylabel("Standoff error (m)")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(fontsize=8)

        summary[scenario["key"]] = {
            "samples": int(len(times)),
            "mean_abs_error_m": float(np.mean(np.abs(error))) if len(error) else None,
            "max_abs_error_m": float(np.max(np.abs(error))) if len(error) else None,
        }

    fig.suptitle("UE world tracks and signed standoff error", fontsize=14)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
