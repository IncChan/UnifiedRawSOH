"""Summarize one-cell support-group to test-group transfer matrices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from UnifiedRawSOH.utils.config import save_json


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stats(values):
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "values": values,
    }


def summarize_runtime(runtime_root):
    runtime_root = Path(runtime_root).resolve()
    manifest = json.loads(
        (runtime_root / "job_manifest.json").read_text(encoding="utf-8")
    )
    jobs = manifest["jobs"]
    incomplete = []
    for job in jobs:
        output = Path(job["output_dir"])
        status_path = output / "status.json"
        metrics_path = output / "metrics_overall.json"
        group_path = output / "metrics_by_test_group.csv"
        if not status_path.is_file():
            incomplete.append({"job_id": job["job_id"], "reason": "missing_status"})
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            incomplete.append(
                {"job_id": job["job_id"], "reason": status.get("status")}
            )
        elif not metrics_path.is_file() or not group_path.is_file():
            incomplete.append(
                {"job_id": job["job_id"], "reason": "missing_final_metrics"}
            )
    summary_root = runtime_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    if incomplete:
        payload = {
            "status": "incomplete",
            "expected_job_count": len(jobs),
            "incomplete_jobs": incomplete,
        }
        save_json(summary_root / "status.json", payload)
        return payload

    records = []
    overall_records = []
    for job in jobs:
        output = Path(job["output_dir"])
        overall = json.loads(
            (output / "metrics_overall.json").read_text(encoding="utf-8")
        )
        overall_records.append(
            {
                "target_domain": job["target_domain"],
                "support_group": job["support_group"],
                "support_choice": str(job["support_choice"]),
                "support_cell": job["support_cell"],
                "mape": float(overall["mape"]),
                "rmse": float(overall["rmse"]),
            }
        )
        for row in _read_csv(output / "metrics_by_test_group.csv"):
            records.append(
                {
                    "target_domain": job["target_domain"],
                    "support_group": job["support_group"],
                    "support_choice": str(job["support_choice"]),
                    "support_cell": job["support_cell"],
                    "test_group": row["test_group"],
                    "mape": float(row["mape"]),
                    "rmse": float(row["rmse"]),
                }
            )

    support_rows = []
    test_rows = []
    target_summary = {}
    for target in sorted({row["target_domain"] for row in records}):
        target_records = [row for row in records if row["target_domain"] == target]
        target_overall = [
            row for row in overall_records if row["target_domain"] == target
        ]
        support_groups = list(
            dict.fromkeys(
                job["support_group"]
                for job in jobs
                if job["target_domain"] == target
            )
        )
        test_groups = sorted({row["test_group"] for row in target_records})
        target_root = summary_root / target
        target_root.mkdir(parents=True, exist_ok=True)

        for metric in ("mape", "rmse"):
            matrix_rows = []
            for support_group in support_groups:
                row = {"support_group": support_group}
                for test_group in test_groups:
                    values = [
                        value[metric]
                        for value in target_records
                        if value["support_group"] == support_group
                        and value["test_group"] == test_group
                    ]
                    row[test_group] = _stats(values)["mean"]
                all_values = [
                    value[metric]
                    for value in target_overall
                    if value["support_group"] == support_group
                ]
                row["all_test_macro"] = _stats(all_values)["mean"]
                matrix_rows.append(row)
            _write_csv(
                target_root / f"{metric}_support_to_test_group.csv",
                matrix_rows,
            )

        for support_group in support_groups:
            current = [
                row for row in target_overall
                if row["support_group"] == support_group
            ]
            for metric in ("mape", "rmse"):
                stats = _stats(row[metric] for row in current)
                support_rows.append(
                    {
                        "target_domain": target,
                        "support_group": support_group,
                        "metric": metric,
                        **{key: stats[key] for key in ("n", "mean", "std")},
                    }
                )
        for test_group in test_groups:
            current = [
                row for row in target_records
                if row["test_group"] == test_group
            ]
            for metric in ("mape", "rmse"):
                stats = _stats(row[metric] for row in current)
                test_rows.append(
                    {
                        "target_domain": target,
                        "test_group": test_group,
                        "metric": metric,
                        **{key: stats[key] for key in ("n", "mean", "std")},
                    }
                )

        metric_summary = {}
        for metric in ("mape", "rmse"):
            support_means = [
                _stats(
                    row[metric]
                    for row in target_overall
                    if row["support_group"] == support_group
                )["mean"]
                for support_group in support_groups
            ]
            diagonal = [
                row[metric]
                for row in target_records
                if row["support_group"] == row["test_group"]
            ]
            off_diagonal = [
                row[metric]
                for row in target_records
                if row["support_group"] != row["test_group"]
            ]
            metric_summary[metric] = {
                "support_group_macro": _stats(support_means),
                "diagonal": _stats(diagonal),
                "off_diagonal": _stats(off_diagonal),
            }
        target_summary[target] = metric_summary

    _write_csv(summary_root / "summary_by_support_group.csv", support_rows)
    _write_csv(summary_root / "summary_by_test_group.csv", test_rows)
    payload = {
        "status": "completed",
        "job_count": len(jobs),
        "targets": target_summary,
    }
    save_json(summary_root / "summary_mean_std.json", payload)
    save_json(summary_root / "status.json", {"status": "completed"})
    return payload


def main():
    parser = argparse.ArgumentParser("Summarize one-cell head-only jobs")
    parser.add_argument("--runtime-root", required=True)
    args = parser.parse_args()
    result = summarize_runtime(args.runtime_root)
    print(json.dumps(result, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
