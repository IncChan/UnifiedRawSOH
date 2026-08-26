#!/usr/bin/env python3
"""Validate and aggregate one complete Paper-v2 seed batch.

The script is intentionally conservative: it validates every expected seed and
domain before writing any aggregate CSV/JSON.  An incomplete batch cannot
produce a partial or pseudo-complete summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


BOL_RULE_VERSION = "bol_peak_mean_top5_first100_v1"
METRICS = ("mae", "mape", "mse", "rmse")
REQUIRED_FILES = (
    "completed.status",
    "test_metrics.json",
    "metrics_by_cell.csv",
    "metrics_by_group.csv",
    "metrics_by_domain.csv",
)


class IncompleteBatchError(RuntimeError):
    """Raised when a requested batch is not complete enough to summarize."""


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IncompleteBatchError(f"Metric value is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise IncompleteBatchError(f"Metric value is not finite: {value!r}")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise IncompleteBatchError(f"Missing or empty required metric file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise IncompleteBatchError(f"Required metric file has no data rows: {path}")
    return rows


def _validate_seed(seed_dir: Path, expected_domains: list[str]) -> dict[str, object]:
    missing = [name for name in REQUIRED_FILES if not (seed_dir / name).is_file()]
    if missing:
        raise IncompleteBatchError(
            f"{seed_dir.name} is incomplete; missing {', '.join(missing)}"
        )
    status = (seed_dir / "completed.status").read_text(encoding="utf-8").strip()
    if status != "completed":
        raise IncompleteBatchError(
            f"{seed_dir} has status {status!r}; expected exactly 'completed'"
        )
    try:
        test_metrics = json.loads((seed_dir / "test_metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IncompleteBatchError(f"Invalid test_metrics.json: {seed_dir}") from exc
    if not isinstance(test_metrics, dict):
        raise IncompleteBatchError(f"test_metrics.json is not an object: {seed_dir}")

    tables = {
        name: _read_csv(seed_dir / name)
        for name in ("metrics_by_cell.csv", "metrics_by_group.csv", "metrics_by_domain.csv")
    }
    domain_rows = tables["metrics_by_domain.csv"]
    observed = {str(row.get("domain_id", "")).strip() for row in domain_rows}
    missing_domains = [domain for domain in expected_domains if domain not in observed]
    if missing_domains:
        raise IncompleteBatchError(
            f"{seed_dir} lacks expected domain metrics: {missing_domains}; observed={sorted(observed)}"
        )
    selected = {}
    for domain in expected_domains:
        matches = [row for row in domain_rows if str(row.get("domain_id", "")).strip() == domain]
        if len(matches) != 1:
            raise IncompleteBatchError(
                f"{seed_dir} must contain exactly one domain row for {domain!r}; got {len(matches)}"
            )
        row = matches[0]
        selected[domain] = {
            metric: _finite(row.get(metric))
            for metric in METRICS
        }
    return {
        "seed_dir": seed_dir,
        "test_metrics": test_metrics,
        "tables": tables,
        "domain_metrics": selected,
    }


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise IncompleteBatchError(f"Cannot summarize non-finite/empty values: {values!r}")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    return mean, math.sqrt(max(variance, 0.0))


def _stats(rows: list[Mapping[str, float]]) -> dict[str, object]:
    result = {}
    for metric in METRICS:
        mean, std = _mean_std(row[metric] for row in rows)
        result[metric] = {
            "mean": float(mean),
            "std": float(std),
            "n_seeds": int(len(rows)),
        }
    return result


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise IncompleteBatchError(f"Refusing to write an empty aggregate: {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def summarize_batch(
    batch_root: str | Path,
    expected_seeds: Iterable[int] = (42, 52, 62),
    expected_domains: Iterable[str] | None = None,
) -> dict:
    """Validate a complete batch, then write aggregate tables and summaries."""

    batch_root = Path(batch_root).resolve()
    seeds = [int(seed) for seed in expected_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise IncompleteBatchError(f"expected_seeds must be unique and non-empty: {seeds}")
    domains = [str(domain).strip() for domain in (expected_domains or []) if str(domain).strip()]
    if not domains:
        raise IncompleteBatchError("expected_domains must be supplied for a strict summary")
    if len(set(domains)) != len(domains):
        raise IncompleteBatchError(f"expected_domains must be unique: {domains}")
    if batch_root.name.startswith("Paper-v1") or "Paper-v1" in batch_root.parts:
        raise IncompleteBatchError(f"Refusing to summarize a Paper-v1 path: {batch_root}")

    validated = []
    for seed in seeds:
        seed_dir = batch_root / f"seed_{seed}"
        if not seed_dir.is_dir():
            raise IncompleteBatchError(f"Missing expected seed directory: {seed_dir}")
        validated.append(_validate_seed(seed_dir, domains))

    # All validation is complete before any aggregate output is touched.
    batch_root.mkdir(parents=True, exist_ok=True)
    for table_name in ("metrics_by_cell.csv", "metrics_by_group.csv", "metrics_by_domain.csv"):
        combined = []
        source_name = table_name
        for seed, entry in zip(seeds, validated):
            for row in entry["tables"][source_name]:
                combined.append({"seed": int(seed), **row})
        _write_csv_atomic(batch_root / table_name, combined)

    per_domain = {}
    for domain in domains:
        per_domain[domain] = _stats(
            [entry["domain_metrics"][domain] for entry in validated]
        )
    overall_seed_rows = []
    for entry in validated:
        overall_seed_rows.append(
            {
                metric: sum(entry["domain_metrics"][domain][metric] for domain in domains)
                / float(len(domains))
                for metric in METRICS
            }
        )
    summary = {
        "summary_version": "paper_v2_bol_mean_std_v1",
        "complete": True,
        "label_rule": BOL_RULE_VERSION,
        "label_mode": "bol_peak_relative",
        "batch_root": str(batch_root),
        "expected_seeds": seeds,
        "expected_domains": domains,
        "n_seeds": len(seeds),
        "aggregation": {
            "primary": "mean_std_across_seeds",
            "within_seed": "domain_macro_over_group_macro_over_physical_cell_metrics",
            "cell": "physical test cell independent",
            "group": "cell-macro within condition/strategy",
            "domain": "group-macro within domain",
        },
        "per_domain": per_domain,
        # metrics is a stable alias for small consumers and report builders.
        "metrics": per_domain,
        "overall_domain_macro": _stats(overall_seed_rows),
        "seed_test_metrics_files": {
            str(seed): str(entry["seed_dir"] / "test_metrics.json")
            for seed, entry in zip(seeds, validated)
        },
    }
    _write_json_atomic(batch_root / "summary_mean_std.json", summary)

    summary_rows = []
    for domain in domains:
        for metric in METRICS:
            values = per_domain[domain][metric]
            summary_rows.append(
                {
                    "scope": "domain",
                    "domain_id": domain,
                    "metric": metric,
                    "mean": values["mean"],
                    "std": values["std"],
                    "n_seeds": values["n_seeds"],
                }
            )
    for metric in METRICS:
        values = summary["overall_domain_macro"][metric]
        summary_rows.append(
            {
                "scope": "overall_domain_macro",
                "domain_id": "__all__",
                "metric": metric,
                "mean": values["mean"],
                "std": values["std"],
                "n_seeds": values["n_seeds"],
            }
        )
    _write_csv_atomic(batch_root / "summary_mean_std.csv", summary_rows)
    _write_csv_atomic(
        batch_root / "summary_per_domain_mean_std.csv",
        [row for row in summary_rows if row["scope"] == "domain"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_root", required=True)
    parser.add_argument("--expected_seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument(
        "--expected_domains",
        nargs="+",
        required=True,
        help="Strict expected domain IDs; all must be present for every seed.",
    )
    args = parser.parse_args(argv)
    try:
        summary = summarize_batch(
            args.batch_root,
            expected_seeds=args.expected_seeds,
            expected_domains=args.expected_domains,
        )
    except IncompleteBatchError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "batch_root": summary["batch_root"],
                "complete": summary["complete"],
                "expected_seeds": summary["expected_seeds"],
                "expected_domains": summary["expected_domains"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
