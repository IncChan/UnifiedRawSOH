"""Hierarchical Paper-v2 test metrics and CSV writers.

The main result is built in the order required by Paper-v2: sample predictions
are reduced to physical-cell metrics, cells to condition/group metrics, and
groups to domain metrics.  No variable-length trajectory is pooled into one
primary micro metric.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .metrics import compute_metrics

METRIC_NAMES = ("mae", "mape", "mse", "rmse")
_TABLE_FIELDS = (
    "domain_id",
    "group_id",
    "cell_id",
    "aggregation",
    "n_samples",
    "n_cells",
    "n_groups",
    *METRIC_NAMES,
)


def _mean_metric(rows: Iterable[Mapping], metric: str) -> float:
    values = [float(row[metric]) for row in rows if math.isfinite(float(row.get(metric, math.nan)))]
    return float(np.mean(values)) if values else float("nan")


def _sample_value(row: Mapping, *names: str, default: str = "unknown") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _metric_row(domain_id, group_id, cell_id, aggregation, n_samples, metrics, *, n_cells=0, n_groups=0):
    return {
        "domain_id": str(domain_id),
        "group_id": str(group_id),
        "cell_id": str(cell_id),
        "aggregation": str(aggregation),
        "n_samples": int(n_samples),
        "n_cells": int(n_cells),
        "n_groups": int(n_groups),
        **{name: float(metrics.get(name, math.nan)) for name in METRIC_NAMES},
    }


def build_hierarchical_metric_tables(prediction_rows: Iterable[Mapping]) -> dict[str, object]:
    """Build cell-, group-, domain-, and final macro-level metric tables."""

    samples = list(prediction_rows)
    by_cell = defaultdict(lambda: ([], []))
    for row in samples:
        domain_id = _sample_value(row, "domain_id", "dataset_id")
        group_id = _sample_value(row, "group_id", "condition", "batch_name")
        cell_id = _sample_value(row, "cell_id", "battery_id", "physical_cell_id")
        try:
            truth = float(row["y_true"])
            prediction = float(row["y_pred"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Paper-v2 prediction row: {row!r}") from exc
        if not (math.isfinite(truth) and math.isfinite(prediction)):
            continue
        by_cell[(domain_id, group_id, cell_id)][0].append(truth)
        by_cell[(domain_id, group_id, cell_id)][1].append(prediction)

    cells = []
    for (domain_id, group_id, cell_id), (truths, predictions) in sorted(by_cell.items()):
        cells.append(
            _metric_row(
                domain_id, group_id, cell_id, "physical_cell", len(truths),
                compute_metrics(truths, predictions), n_cells=1,
            )
        )

    by_group = defaultdict(list)
    for row in cells:
        by_group[(row["domain_id"], row["group_id"])].append(row)
    groups = []
    for (domain_id, group_id), group_cells in sorted(by_group.items()):
        groups.append(
            _metric_row(
                domain_id, group_id, "", "cell_macro",
                sum(row["n_samples"] for row in group_cells),
                {name: _mean_metric(group_cells, name) for name in METRIC_NAMES},
                n_cells=len(group_cells),
            )
        )

    by_domain = defaultdict(list)
    for row in groups:
        by_domain[row["domain_id"]].append(row)
    domains = []
    for domain_id, domain_groups in sorted(by_domain.items()):
        domain_cells = [row for row in cells if row["domain_id"] == domain_id]
        domains.append(
            _metric_row(
                domain_id, "", "", "group_macro",
                sum(row["n_samples"] for row in domain_cells),
                {name: _mean_metric(domain_groups, name) for name in METRIC_NAMES},
                n_cells=len(domain_cells), n_groups=len(domain_groups),
            )
        )

    overall = _metric_row(
        "__all__", "", "", "domain_macro",
        sum(row["n_samples"] for row in domains),
        {name: _mean_metric(domains, name) for name in METRIC_NAMES},
        n_cells=sum(row["n_cells"] for row in domains),
        n_groups=sum(row["n_groups"] for row in domains),
    )
    return {"by_cell": cells, "by_group": groups, "by_domain": domains, "overall": overall}


def write_metric_tables(output_dir: str | Path, tables: Mapping[str, object]) -> None:
    """Write the three required CSV tables into one seed output directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in (("by_cell", "metrics_by_cell.csv"), ("by_group", "metrics_by_group.csv"), ("by_domain", "metrics_by_domain.csv")):
        rows = list(tables.get(key, []))
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_TABLE_FIELDS), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def test_metrics_payload(tables: Mapping[str, object]) -> dict:
    """Return the JSON-compatible test payload used by V2 trainers."""

    overall = dict(tables["overall"])
    per_domain = {
        row["domain_id"]: {
            **{name: float(row[name]) for name in METRIC_NAMES},
            "n_samples": int(row["n_samples"]),
            "n_cells": int(row["n_cells"]),
            "n_groups": int(row["n_groups"]),
            "aggregation": row["aggregation"],
        }
        for row in tables["by_domain"]
    }
    return {
        **{name: float(overall[name]) for name in METRIC_NAMES},
        "aggregation": "domain_macro_over_group_macro_over_physical_cell_metrics",
        "per_domain": per_domain,
        "hierarchical_metrics": {
            "cell": "physical test cell independent",
            "group": "cell-macro within condition/strategy",
            "domain": "group-macro within domain",
            "overall": "domain-macro",
        },
    }


def read_metric_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


__all__ = [
    "METRIC_NAMES",
    "build_hierarchical_metric_tables",
    "read_metric_csv",
    "test_metrics_payload",
    "write_metric_tables",
]
