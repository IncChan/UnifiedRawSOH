#!/usr/bin/env python3
"""Aggregate total and per-domain test metrics from a Paper-v1 multi-seed run."""

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

DOMAIN_METRICS = (
    "mae",
    "mape",
    "mse",
    "rmse",
    "loss",
    "soh_loss",
)


def parse_args():
    parser = argparse.ArgumentParser("Summarize UnifiedRawSOH seed runs")
    parser.add_argument("--batch_root", required=True)
    parser.add_argument("--expected_seeds", nargs="*", type=int, default=[])
    return parser.parse_args()


def _finite_metrics(payload, metric_names):
    if not isinstance(payload, dict):
        return {}
    return {
        name: float(payload[name])
        for name in metric_names
        if isinstance(payload.get(name), (int, float))
        and math.isfinite(float(payload[name]))
    }


def _expected_domain_ids(batch_root):
    """Read the fixed experiment domain order from the batch manifest when present."""

    manifest_path = batch_root / "run_manifest.json"
    if not manifest_path.is_file():
        return []
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    experiment = manifest.get("experiment", {})
    if experiment.get("loader") == "leave_one_domain_out":
        target_domain_id = experiment.get("target_domain_id")
        if target_domain_id is None:
            target_domain_ids = experiment.get("target_domain_ids", [])
            if len(target_domain_ids) == 1:
                target_domain_id = target_domain_ids[0]
        return [str(target_domain_id)] if target_domain_id is not None else []
    domain_ids = experiment.get("domain_ids")
    if isinstance(domain_ids, str):
        domain_ids = [domain_ids]
    if isinstance(domain_ids, (list, tuple)):
        return list(dict.fromkeys(str(value) for value in domain_ids if str(value)))
    domain_id = experiment.get("domain_id")
    return [str(domain_id)] if domain_id is not None and str(domain_id) else []


def _per_domain_metrics(metrics):
    """Extract E1-aligned scalar metrics for each domain from one seed result."""

    per_domain = metrics.get("per_domain", {})
    if not isinstance(per_domain, dict):
        return {}
    result = {}
    for domain_id, values in sorted(per_domain.items()):
        entry = _finite_metrics(values, DOMAIN_METRICS)
        # Older completed runs did not store these two aliases per domain.
        # At test time both equal SOH MSE, so recover them without changing
        # the numerical interpretation of a historical result.
        if "mse" in entry:
            entry.setdefault("loss", entry["mse"])
            entry.setdefault("soh_loss", entry["mse"])
        sample_count = values.get("n_samples") if isinstance(values, dict) else None
        if isinstance(sample_count, (int, float)) and math.isfinite(float(sample_count)):
            entry["n_samples"] = int(sample_count)
        if entry:
            result[str(domain_id)] = entry
    return result


def _summary(rows, metric_names):
    summary = {}
    for metric in metric_names:
        values = [
            row["metrics"][metric]
            for row in rows
            if metric in row.get("metrics", {})
        ]
        if not values:
            continue
        summary[metric] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def _per_domain_summary(rows, expected_domain_ids):
    observed = {
        domain_id
        for row in rows
        for domain_id in row.get("per_domain", {})
    }
    domain_ids = list(expected_domain_ids)
    domain_ids.extend(sorted(observed - set(domain_ids)))

    summary = {}
    for domain_id in domain_ids:
        domain_rows = [
            {
                "seed": row["seed"],
                "run_dir": row["run_dir"],
                "metrics": row["per_domain"][domain_id],
            }
            for row in rows
            if domain_id in row.get("per_domain", {})
        ]
        values = {
            "seed_count": len(domain_rows),
            "metrics": _summary(domain_rows, DOMAIN_METRICS),
        }
        sample_counts = {
            str(row["seed"]): int(row["metrics"]["n_samples"])
            for row in domain_rows
            if "n_samples" in row["metrics"]
        }
        if sample_counts:
            unique_counts = sorted(set(sample_counts.values()))
            values["test_sample_counts"] = {
                "per_seed": sample_counts,
                "consistent": len(unique_counts) == 1,
                "value": unique_counts[0] if len(unique_counts) == 1 else None,
            }
        summary[domain_id] = values
    return summary


def _save_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main():
    args = parse_args()
    batch_root = Path(args.batch_root)
    expected_domains = _expected_domain_ids(batch_root)
    rows = []
    missing = []
    missing_domains = {}
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
        per_domain = _per_domain_metrics(metrics)
        absent = [domain_id for domain_id in expected_domains if domain_id not in per_domain]
        if absent:
            missing_domains[seed] = absent
        rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "metrics": _finite_metrics(metrics, METRICS),
                "per_domain": per_domain,
            }
        )
    if missing:
        raise SystemExit(f"Missing completed test_metrics.json for seeds: {missing}")
    if missing_domains:
        raise SystemExit(
            "Missing per-domain test metrics for completed seeds: "
            + ", ".join(
                f"seed_{seed}={domains}" for seed, domains in sorted(missing_domains.items())
            )
        )
    if not rows:
        raise SystemExit("No completed seed runs found.")

    summary = _summary(rows, METRICS)
    per_domain_summary = _per_domain_summary(rows, expected_domains)
    batch_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_root": str(batch_root),
        "seed_count": len(rows),
        "runs": rows,
        "summary": summary,
        "per_domain_summary": per_domain_summary,
        "per_domain_summary_file": "summary_per_domain_mean_std.json",
    }
    _save_json(batch_root / "summary_mean_std.json", payload)

    fields = ["metric", "n", "mean", "std"]
    with (batch_root / "summary_mean_std.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric, values in summary.items():
            writer.writerow({"metric": metric, **values})

    per_domain_payload = {
        "batch_root": str(batch_root),
        "seed_count": len(rows),
        "expected_domain_ids": expected_domains,
        "runs": [
            {
                "seed": row["seed"],
                "run_dir": row["run_dir"],
                "per_domain": row["per_domain"],
            }
            for row in rows
        ],
        "summary": per_domain_summary,
    }
    _save_json(batch_root / "summary_per_domain_mean_std.json", per_domain_payload)

    domain_fields = ["domain_id", "metric", "n", "mean", "std"]
    with (batch_root / "summary_per_domain_mean_std.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=domain_fields)
        writer.writeheader()
        for domain_id, domain_values in per_domain_summary.items():
            for metric, values in domain_values["metrics"].items():
                writer.writerow({"domain_id": domain_id, "metric": metric, **values})

    print(f"summary saved: {batch_root / 'summary_mean_std.json'}")
    print(f"summary saved: {batch_root / 'summary_mean_std.csv'}")
    print(f"per-domain summary saved: {batch_root / 'summary_per_domain_mean_std.json'}")
    print(f"per-domain summary saved: {batch_root / 'summary_per_domain_mean_std.csv'}")


if __name__ == "__main__":
    main()
