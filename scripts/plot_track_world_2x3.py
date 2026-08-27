#!/usr/bin/env python3
"""Plot 2x3 UE world tracks and signed standoff error."""

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
        "title": "RED 4 m",
        "target_id": "target_red",
        "desired_m": 4.0,
        "target_color": "#C0392B",
        "error_color": "#C0392B",
    },
    {
        "key": "blue3",
        "title": "BLUE 3 m",
        "target_id": "target_blue",
        "desired_m": 3.0,
        "target_color": "#2471A3",
        "error_color": "#2471A3",
    },
    {
        "key": "red3",
        "title": "RED 3 m",
        "target_id": "target_red",
        "desired_m": 3.0,
        "target_color": "#C0392B",
        "error_color": "#C0392B",
    },
)


def _series(parsed: dict, target_id: str, desired_m: float):
    times = sorted(set(parsed["asv"]) & set(parsed["targets"].get(target_id, {})))
    asv = np.stack([parsed["asv"][t] for t in times])
    target = np.stack([parsed["targets"][target_id][t] for t in times])
    distance = np.linalg.norm(target - asv, axis=1)
    error = distance - desired_m
    return np.asarray(times, dtype=np.float32), asv, target, error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closed-loop-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Closed-loop tracking  |  UE5 + Jetson")
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), constrained_layout=True)
    summary: dict[str, object] = {}
    legend_handles = []

    for col, scenario in enumerate(SCENARIOS):
        parsed = parse_ue_log(args.closed_loop_dir / f"{scenario['key']}_ue.log")
        times, asv, target, error = _series(parsed, scenario["target_id"], scenario["desired_m"])

        ax_top = axes[0, col]
        (h_asv,) = ax_top.plot(asv[:, 0], asv[:, 1], color="#1C1C1C", linewidth=2.0, zorder=3)
        (h_tgt,) = ax_top.plot(
            target[:, 0],
            target[:, 1],
            color=scenario["target_color"],
            linewidth=1.9,
            zorder=2,
        )
        extra_handles = []
        extra_styles = (("target_left", (0, (4, 3))), ("target_right", (0, (1, 2))))
        for extra_id, style in extra_styles:
            extra_map = parsed["targets"].get(extra_id, {})
            if not extra_map:
                continue
            extra_times = sorted(extra_map)
            extra = np.stack([extra_map[t] for t in extra_times])
            (h_extra,) = ax_top.plot(
                extra[:, 0],
                extra[:, 1],
                color="#8D99A6",
                linestyle=style,
                linewidth=1.15,
                zorder=1,
            )
            extra_handles.append(h_extra)
        if col == 0:
            legend_handles = [h_asv, h_tgt, *extra_handles]

        ax_top.set_title(scenario["title"], pad=6)
        ax_top.set_xlabel("World X (m)")
        if col == 0:
            ax_top.set_ylabel("World Y (m)")
        ax_top.set_aspect("equal", adjustable="datalim")
        ax_top.grid(True, color="#D5DBDB", linewidth=0.6)
        ax_top.tick_params(length=3)

        ax_bot = axes[1, col]
        ax_bot.plot(times, error, color=scenario["error_color"], linewidth=1.55)
        ax_bot.axhline(0.0, color="#2C3E50", linewidth=0.8, alpha=0.85)
        ax_bot.set_xlabel("Time (s)")
        if col == 0:
            ax_bot.set_ylabel("Standoff error (m)")
        ax_bot.grid(True, color="#D5DBDB", linewidth=0.6)
        ax_bot.tick_params(length=3)

        summary[scenario["key"]] = {
            "samples": int(len(times)),
            "mean_abs_error_m": float(np.mean(np.abs(error))) if len(error) else None,
            "max_abs_error_m": float(np.max(np.abs(error))) if len(error) else None,
        }

    labels = ["ASV", "Target", "Left boat", "Right boat"][: len(legend_handles)]
    fig.legend(
        legend_handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        columnspacing=1.6,
        handlelength=2.4,
    )
    fig.suptitle(args.title, y=1.08, fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
