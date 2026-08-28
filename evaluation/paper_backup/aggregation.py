"""Paper-Backup battery/strategy-aware metric aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from ..metrics import compute_metrics


def _macro(per_group: Mapping[str, Mapping[str, Any]], metric: str) -> float:
    values = [float(item[metric]) for item in per_group.values() if np.isfinite(float(item.get(metric, np.nan)))]
    return float(np.mean(values)) if values else float("nan")


def _with_counts(rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(group_key, "unknown"))].append(row)
    output = {}
    for group, values in sorted(buckets.items()):
        metrics = compute_metrics(
            [float(item["y_true"]) for item in values],
            [float(item["y_pred"]) for item in values],
        )
        metrics["n_cycles"] = len(values)
        output[group] = metrics
    return output


def metrics_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute pooled, battery-macro and strategy-macro metrics.

    The macro values average each physical battery/strategy metric once, so a
    long-lived battery cannot dominate the reported primary metric.
    """

    rows = [dict(row) for row in rows]
    if not rows:
        return {
            "mae": float("nan"),
            "mape": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "n_cycles": 0,
            "per_battery": {},
            "per_strategy": {},
            "battery_macro": {"mae": float("nan"), "rmse": float("nan")},
            "strategy_macro": {"mae": float("nan"), "rmse": float("nan")},
            "worst_strategy": {"strategy_id": None, "rmse": float("nan")},
        }
    pooled = compute_metrics(
        [float(row["y_true"]) for row in rows],
        [float(row["y_pred"]) for row in rows],
    )
    per_battery = _with_counts(rows, "battery_id")
    per_strategy = _with_counts(rows, "strategy_id")
    pooled.update(
        {
            "n_cycles": len(rows),
            "n_batteries": len(per_battery),
            "n_strategies": len(per_strategy),
            "per_battery": per_battery,
            "per_strategy": per_strategy,
            "battery_macro": {
                "mae": _macro(per_battery, "mae"),
                "rmse": _macro(per_battery, "rmse"),
                "n_batteries": len(per_battery),
            },
            "strategy_macro": {
                "mae": _macro(per_strategy, "mae"),
                "rmse": _macro(per_strategy, "rmse"),
                "n_strategies": len(per_strategy),
            },
        }
    )
    if per_strategy:
        worst_strategy = max(per_strategy.items(), key=lambda item: float(item[1]["rmse"]))
        pooled["worst_strategy"] = {
            "strategy_id": worst_strategy[0],
            "rmse": float(worst_strategy[1]["rmse"]),
            "mae": float(worst_strategy[1]["mae"]),
        }
    else:
        pooled["worst_strategy"] = {"strategy_id": None, "rmse": float("nan")}
    return pooled


def aggregate_seed_metrics(seed_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Mean/std aggregation across completed seeds, excluding blocked runs."""

    fields = ("mae", "rmse", "battery_macro_rmse", "strategy_macro_rmse")
    values: dict[str, list[float]] = {field: [] for field in fields}
    for _, metrics in sorted(seed_metrics.items()):
        if str(metrics.get("status", "completed")) in {"blocked_by_data", "missing", "failed"}:
            continue
        battery_macro = metrics.get("battery_macro", {})
        strategy_macro = metrics.get("strategy_macro", {})
        candidates = {
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "battery_macro_rmse": battery_macro.get("rmse"),
            "strategy_macro_rmse": strategy_macro.get("rmse"),
        }
        for field, value in candidates.items():
            if value is not None and np.isfinite(float(value)):
                values[field].append(float(value))
    summary = {"seed_count": 0, "metrics": {}}
    counts = [len(items) for items in values.values()]
    summary["seed_count"] = max(counts) if counts else 0
    for field, items in values.items():
        summary["metrics"][field] = {
            "mean": float(np.mean(items)) if items else float("nan"),
            "std": float(np.std(items, ddof=1)) if len(items) > 1 else 0.0 if items else float("nan"),
            "n": len(items),
        }
    return summary


__all__ = ["aggregate_seed_metrics", "metrics_from_rows"]
