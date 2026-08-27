#!/usr/bin/env python3
"""Run three closed-loop scenarios, render the 2x3 figure, and write acceptance.json."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_M = 1.0
SCENARIOS = ("red4", "blue3", "red3")


def _policy_inferred(log_path: Path) -> int:
    if not log_path.is_file():
        return 0
    return sum(
        1
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "POLICY_INFERRED_SMOOTHED" in line
    )


def write_acceptance(closed_loop: Path, checkpoint: Path, figure: Path, runtime: int) -> dict[str, object]:
    summary_path = figure.with_suffix(".json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    scenarios: dict[str, object] = {}
    passed = True
    for key in SCENARIOS:
        row = dict(summary.get(key) or {})
        mean = row.get("mean_abs_error_m")
        hits = _policy_inferred(closed_loop / f"closed_loop_{key}.log")
        ok = (
            isinstance(mean, (int, float))
            and mean < THRESHOLD_M
            and int(row.get("samples") or 0) >= 50
            and hits >= 30
        )
        row["policy_inferred"] = hits
        row["pass"] = ok
        scenarios[key] = row
        passed = passed and ok
    report = {
        "acceptance": "mean_abs_standoff_error_m < 1.0",
        "checkpoint": str(checkpoint),
        "figure": str(figure),
        "runtime_sec": runtime,
        "scenarios": scenarios,
        "passed": passed,
        "evidence": "现场验证" if passed else "未验证",
    }
    (closed_loop / "acceptance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


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

    for scenario in SCENARIOS:
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
    plot = subprocess.run(plot_cmd, cwd=str(ROOT))
    if plot.returncode != 0:
        return plot.returncode
    report = write_acceptance(closed_loop, args.out, args.figure, args.runtime)
    return 0 if report["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
