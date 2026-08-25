#!/usr/bin/env python3
"""Run three closed-loop scenarios and render the 2x3 acceptance figure."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/chase_standoff_tight_1m")
    parser.add_argument("--runtime", type=int, default=180)
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(r"C:\Users\LIU\Desktop\track_world_single_point_policy_dominant_2x3.png"),
    )
    args = parser.parse_args()
    closed_loop = args.out / "closed_loop"
    closed_loop.mkdir(parents=True, exist_ok=True)

    for scenario in ("red4", "blue3", "red3"):
        cmd = [
            sys.executable,
            str(ROOT / "scripts/run_closed_loop.py"),
            scenario,
            "--out",
            str(closed_loop),
            "--runtime",
            str(args.runtime),
        ]
        print("+", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            return result.returncode

    plot_cmd = [
        sys.executable,
        str(ROOT / "scripts/plot_track_world_2x3.py"),
        "--closed-loop-dir",
        str(closed_loop),
        "--output",
        str(args.figure),
    ]
    print("+", " ".join(plot_cmd))
    return subprocess.run(plot_cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
