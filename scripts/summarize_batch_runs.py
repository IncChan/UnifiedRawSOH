#!/usr/bin/env python3
"""Aggregate scalar test metrics from a Paper-v1 multi-seed run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = (
    "mae",
    "mape",
    "mse",
    "rmse",
    "macro_rmse",
    "condition_macro_rmse",
    "battery_macro_rmse",
    "domain_macro_rmse",
    "loss",
)


def parse_args():
    parser = argparse.ArgumentParser("Summarize UnifiedRawSOH seed runs")
    parser.add_argument("--batch_root", required=True)
    parser.add_argument("--expected_seeds", nargs="*", type=int, default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    batch_root = Path(args.batch_root)
    rows = []
    missing = []
    seeds = args.expected_seeds or sorted(
        int(path.name.removeprefix("seed_"))
        for path in batch_root.glob("seed_*")
        if path.name.removeprefix("seed_").isdigit()
    )
    for seed in seeds:
        run_dir = batch_root / f"seed_{seed}"
        metrics_path = run_dir / "test_metrics.json"
        if not metrics_path.is_file():
            missing.append(seed)
            continue
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "metrics": {
                    name: float(metrics[name])
                    for name in METRICS
                    if isinstance(metrics.get(name), (int, float))
                    and math.isfinite(float(metrics[name]))
                },
            }
        )
    if missing:
        raise SystemExit(f"Missing completed test_metrics.json for seeds: {missing}")
    if not rows:
        raise SystemExit("No completed seed runs found.")

    summary = {}
    for metric in METRICS:
        values = [row["metrics"][metric] for row in rows if metric in row["metrics"]]
        if not values:
            continue
        summary[metric] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    batch_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_root": str(batch_root),
        "seed_count": len(rows),
        "runs": rows,
        "summary": summary,
    }
    with (batch_root / "summary_mean_std.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    fields = ["metric", "n", "mean", "std"]
    with (batch_root / "summary_mean_std.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric, values in summary.items():
            writer.writerow({"metric": metric, **values})
    print(f"summary saved: {batch_root / 'summary_mean_std.json'}")
    print(f"summary saved: {batch_root / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
