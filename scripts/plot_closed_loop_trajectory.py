#!/usr/bin/env python3
"""Plot closed-loop trajectory from Jetson bridge log setpoints."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "experiments/chase_standoff_entityfeat_v2/closed_loop_red3_entityfeat_v2_20260823.log"
OUTPUT = ROOT / "experiments/chase_standoff_entityfeat_v2/closed_loop_red3_entityfeat_v2_trajectory.png"

PAYLOAD_RE = re.compile(r"payload=(\{.*\})__OD_END__")


def parse_setpoints(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PAYLOAD_RE.search(line)
        if not match:
            continue
        payload = json.loads(match.group(1))
        rows.append(
            {
                "stamp_us": int(payload["Source_Stamp_Us"]),
                "sequence": int(payload["Sequence"]),
                "frame_index": int(payload["Source_Frame_Index"]),
                "valid": bool(payload["Valid"]),
                "reason": str(payload["Reason"]),
                "hold": bool(payload.get("Hold_Position", False)),
                "dx_m": float(payload["Delta_X_Cm"]) / 100.0,
                "dy_m": float(payload["Delta_Y_Cm"]) / 100.0,
            }
        )
    return rows


def integrate_policy_path(policy_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    points = [np.zeros(2, dtype=np.float64)]
    for row in policy_rows:
        previous = points[-1]
        points.append(previous + np.array([row["dx_m"], row["dy_m"]], dtype=np.float64))
    trajectory = np.stack(points)
    times = np.asarray([row["stamp_us"] for row in policy_rows], dtype=np.float64) / 1.0e6
    return trajectory, times


def main() -> None:
    rows = parse_setpoints(LOG)
    if not rows:
        raise SystemExit(f"no setpoints found in {LOG}")

    policy_rows = [
        row
        for row in rows
        if row["valid"] and row["reason"] == "POLICY_INFERRED_SMOOTHED" and not row["hold"]
    ]
    trajectory, times = integrate_policy_path(policy_rows)
    step_norm = np.linalg.norm(
        np.stack([[row["dx_m"], row["dy_m"]] for row in policy_rows]), axis=1
    )

    reason_counts: dict[str, int] = {}
    for row in rows:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    ax_path = axes[0]
    ax_path.plot(trajectory[:, 0], trajectory[:, 1], color="black", linewidth=2.0, label="integrated path")
    ax_path.scatter(trajectory[0, 0], trajectory[0, 1], color="green", s=60, zorder=3, label="start")
    ax_path.scatter(trajectory[-1, 0], trajectory[-1, 1], color="red", s=60, zorder=3, label="end")
  # policy step arrows
    for index, row in enumerate(policy_rows[::4]):
        origin = trajectory[index]
        ax_path.arrow(
            origin[0],
            origin[1],
            row["dx_m"],
            row["dy_m"],
            width=0.02,
            head_width=0.08,
            head_length=0.05,
            length_includes_head=True,
            color="tab:blue",
            alpha=0.45,
        )
    ax_path.set_title("Policy command path\n(integrated ue_actor_local deltas)")
    ax_path.set_xlabel("Integrated X (m)")
    ax_path.set_ylabel("Integrated Y (m)")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(alpha=0.25)
    ax_path.legend(loc="best", fontsize=8)

    ax_step = axes[1]
    ax_step.plot(times - times[0], step_norm, color="tab:orange", linewidth=1.6)
    ax_step.axhline(0.5, color="0.4", linestyle="--", linewidth=1.0, label="0.50 m bound")
    ax_step.set_title("Policy step magnitude")
    ax_step.set_xlabel("Time since first policy cmd (s)")
    ax_step.set_ylabel("|delta| (m)")
    ax_step.grid(alpha=0.25)
    ax_step.legend(fontsize=8)

    ax_reason = axes[2]
    labels = list(reason_counts.keys())
    values = [reason_counts[label] for label in labels]
    colors = [
        "tab:green" if label == "POLICY_INFERRED_SMOOTHED" else "tab:red"
        for label in labels
    ]
    ax_reason.barh(labels, values, color=colors)
    ax_reason.set_title("Setpoint reasons")
    ax_reason.set_xlabel("count")
    ax_reason.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "RED 3m closed loop | entityfeat_v2 | "
        f"{len(policy_rows)} policy steps / {len(rows)} total setpoints",
        fontsize=12,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)

    summary = {
        "log": str(LOG),
        "output": str(OUTPUT),
        "policy_steps": len(policy_rows),
        "total_setpoints": len(rows),
        "path_length_m": float(np.sum(step_norm)),
        "max_step_m": float(np.max(step_norm)),
        "mean_step_m": float(np.mean(step_norm)),
        "reason_counts": reason_counts,
    }
    print(json.dumps(summary, indent=2))
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
