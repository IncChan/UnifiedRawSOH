#!/usr/bin/env python3
"""Aggregate E2 per-domain metric CSVs across the requested seed runs.

The input files are the seed-level ``metrics_by_domain.csv`` files.  The
output is one batch-level ``metrics_by_domain.csv`` containing one row per
domain, with only the metric columns averaged across seeds.  Seed outputs are
never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


TABLE_FIELDS = (
    "domain_id",
    "group_id",
    "cell_id",
    "aggregation",
    "n_samples",
    "n_cells",
    "n_groups",
    "mae",
    "mape",
    "mse",
    "rmse",
)
METRICS = ("mae", "mape", "mse", "rmse")
COUNT_FIELDS = ("n_samples", "n_cells", "n_groups")
METADATA_FIELDS = (
    "group_id",
    "cell_id",
    "aggregation",
    *COUNT_FIELDS,
)


class IncompleteBatchError(RuntimeError):
    """Raised when a seed batch cannot be aggregated safely."""


def _finite(value: object, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IncompleteBatchError(f"{context} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise IncompleteBatchError(f"{context} is not finite: {value!r}")
    return result


def _nonnegative_int(value: object, *, context: str) -> int:
    numeric = _finite(value, context=context)
    if numeric < 0.0 or numeric != float(int(numeric)):
        raise IncompleteBatchError(f"{context} must be a non-negative integer: {value!r}")
    return int(numeric)


def _read_domain_rows(path: Path, expected_domains: list[str]) -> dict[str, dict[str, object]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise IncompleteBatchError(f"Missing or empty metric file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_fields = [field for field in TABLE_FIELDS if field not in fieldnames]
        if missing_fields:
            raise IncompleteBatchError(
                f"{path} is missing required columns: {', '.join(missing_fields)}"
            )
        rows = list(reader)
    if not rows:
        raise IncompleteBatchError(f"Metric file has no data rows: {path}")

    selected: dict[str, dict[str, object]] = {}
    for row_number, row in enumerate(rows, start=2):
        domain = str(row.get("domain_id", "")).strip()
        if not domain:
            raise IncompleteBatchError(f"{path}:{row_number} has an empty domain_id")
        if domain in selected:
            raise IncompleteBatchError(
                f"{path} contains duplicate domain rows for {domain!r}"
            )
        if domain not in expected_domains:
            raise IncompleteBatchError(
                f"{path} contains unexpected domain {domain!r}; "
                f"expected exactly {expected_domains}"
            )

        normalized: dict[str, object] = {
            "domain_id": domain,
            "group_id": str(row.get("group_id", "")).strip(),
            "cell_id": str(row.get("cell_id", "")).strip(),
            "aggregation": str(row.get("aggregation", "")).strip(),
        }
        for field in COUNT_FIELDS:
            normalized[field] = _nonnegative_int(
                row.get(field), context=f"{path}:{row_number}:{field}"
            )
        for metric in METRICS:
            normalized[metric] = _finite(
                row.get(metric), context=f"{path}:{row_number}:{metric}"
            )
        selected[domain] = normalized

    missing_domains = [domain for domain in expected_domains if domain not in selected]
    if missing_domains:
        raise IncompleteBatchError(
            f"{path} is missing expected domains: {missing_domains}; "
            f"observed={sorted(selected)}"
        )
    return selected


def _validate_seed(seed_dir: Path, expected_domains: list[str]) -> dict[str, dict[str, object]]:
    status_path = seed_dir / "completed.status"
    if not seed_dir.is_dir() or not status_path.is_file():
        raise IncompleteBatchError(f"Missing completed seed directory: {seed_dir}")
    status = status_path.read_text(encoding="utf-8").strip()
    if status != "completed":
        raise IncompleteBatchError(
            f"{seed_dir} has status {status!r}; expected exactly 'completed'"
        )
    return _read_domain_rows(seed_dir / "metrics_by_domain.csv", expected_domains)


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise IncompleteBatchError(f"Refusing to write an empty aggregate: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TABLE_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate_metrics_by_domain(
    batch_root: str | Path,
    expected_seeds: Iterable[int] = (42, 52, 62),
    expected_domains: Iterable[str] = (),
) -> Path:
    """Validate all requested seeds and write the batch-level domain means."""

    batch_root = Path(batch_root).resolve()
    seeds = [int(seed) for seed in expected_seeds]
    domains = [str(domain).strip() for domain in expected_domains if str(domain).strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise IncompleteBatchError(f"expected_seeds must be unique and non-empty: {seeds}")
    if not domains or len(set(domains)) != len(domains):
        raise IncompleteBatchError(f"expected_domains must be unique and non-empty: {domains}")
    if "Paper-v1" in batch_root.parts:
        raise IncompleteBatchError(f"Refusing to aggregate a Paper-v1 path: {batch_root}")

    per_seed = []
    for seed in seeds:
        per_seed.append(
            _validate_seed(batch_root / f"seed_{seed}", domains)
        )

    rows: list[dict[str, object]] = []
    for domain in domains:
        domain_rows = [seed_rows[domain] for seed_rows in per_seed]
        reference = domain_rows[0]
        for metadata_field in METADATA_FIELDS:
            if any(row[metadata_field] != reference[metadata_field] for row in domain_rows[1:]):
                observed = [row[metadata_field] for row in domain_rows]
                raise IncompleteBatchError(
                    f"{batch_root}: metadata {metadata_field!r} for domain {domain!r} "
                    f"differs across seeds: {observed!r}"
                )
        row = {field: reference[field] for field in ("domain_id", *METADATA_FIELDS)}
        row.update(
            {
                metric: sum(float(seed_row[metric]) for seed_row in domain_rows)
                / float(len(domain_rows))
                for metric in METRICS
            }
        )
        rows.append(row)

    output_path = batch_root / "metrics_by_domain.csv"
    batch_root.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(output_path, rows)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_root", required=True)
    parser.add_argument("--expected_seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--expected_domains", nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        output_path = aggregate_metrics_by_domain(
            args.batch_root,
            expected_seeds=args.expected_seeds,
            expected_domains=args.expected_domains,
        )
    except IncompleteBatchError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "expected_seeds": [int(seed) for seed in args.expected_seeds],
                "expected_domains": [str(domain) for domain in args.expected_domains],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
