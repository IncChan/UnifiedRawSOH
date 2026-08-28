"""Paired E2/E3 comparisons for common physical cycles."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .aggregation import metrics_from_rows


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("battery_id", row.get("cell_id"))), str(row.get("cycle_id", row.get("cycle_id")))


def paired_comparison(
    left_rows: Iterable[Mapping[str, Any]],
    right_rows: Iterable[Mapping[str, Any]],
    *,
    left_name: str = "left",
    right_name: str = "right",
) -> dict[str, Any]:
    """Compare two predictions only on their common physical cycles."""

    left = {}
    for row in left_rows:
        key = _key(row)
        if key in left:
            raise ValueError(f"Duplicate physical cycle in {left_name}: {key}")
        left[key] = dict(row)
    right = {}
    for row in right_rows:
        key = _key(row)
        if key in right:
            raise ValueError(f"Duplicate physical cycle in {right_name}: {key}")
        right[key] = dict(row)
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError(f"No common physical cycles for paired comparison {left_name} vs {right_name}")
    rows = []
    for key in common:
        lhs, rhs = left[key], right[key]
        if not np.isclose(float(lhs["y_true"]), float(rhs["y_true"]), rtol=1e-5, atol=1e-6):
            raise ValueError(f"Paired labels differ for {key}: {lhs['y_true']} vs {rhs['y_true']}")
        rows.append(
            {
                "battery_id": key[0],
                "cycle_id": key[1],
                "strategy_id": str(rhs.get("strategy_id", rhs.get("group_id", "unknown"))),
                "y_true": float(rhs["y_true"]),
                "left_pred": float(lhs["y_pred"]),
                "right_pred": float(rhs["y_pred"]),
                "difference_right_minus_left": float(rhs["y_pred"]) - float(lhs["y_pred"]),
            }
        )
    per_battery = defaultdict(list)
    for row in rows:
        per_battery[row["battery_id"]].append(row)
    battery_differences = {}
    for battery, values in sorted(per_battery.items()):
        truth = [item["y_true"] for item in values]
        left_metrics = metrics_from_rows(
            [{"y_true": item["y_true"], "y_pred": item["left_pred"], "battery_id": battery, "strategy_id": item["strategy_id"]} for item in values]
        )
        right_metrics = metrics_from_rows(
            [{"y_true": item["y_true"], "y_pred": item["right_pred"], "battery_id": battery, "strategy_id": item["strategy_id"]} for item in values]
        )
        battery_differences[battery] = {
            "n_cycles": len(truth),
            "left_rmse": float(left_metrics["rmse"]),
            "right_rmse": float(right_metrics["rmse"]),
            "rmse_difference_right_minus_left": float(right_metrics["rmse"] - left_metrics["rmse"]),
            "left_mae": float(left_metrics["mae"]),
            "right_mae": float(right_metrics["mae"]),
            "mae_difference_right_minus_left": float(right_metrics["mae"] - left_metrics["mae"]),
        }
    return {
        "left": left_name,
        "right": right_name,
        "pair_key": "(battery_id, cycle_id)",
        "common_cycle_count": len(common),
        "common_battery_count": len(battery_differences),
        "battery_differences": battery_differences,
        "mean_rmse_difference_right_minus_left": float(np.mean([item["rmse_difference_right_minus_left"] for item in battery_differences.values()])),
        "mean_mae_difference_right_minus_left": float(np.mean([item["mae_difference_right_minus_left"] for item in battery_differences.values()])),
    }


def e2_comparisons(named_rows: Mapping[str, Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
    """Generate the three predeclared E2 system comparisons when available."""

    pairs = (
        ("full_vanilla", "terminal_vanilla", "Full Vanilla -> Terminal Vanilla"),
        ("terminal_vanilla", "terminal_ours", "Terminal Vanilla -> Terminal Ours"),
        ("full_vanilla", "terminal_ours", "Full Vanilla -> Terminal Ours"),
    )
    output = {}
    for left, right, label in pairs:
        if left in named_rows and right in named_rows:
            output[label] = paired_comparison(named_rows[left], named_rows[right], left_name=left, right_name=right)
    return output


def e3_strategy_comparison(
    specific_rows: Iterable[Mapping[str, Any]],
    pooled_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    return paired_comparison(
        specific_rows,
        pooled_rows,
        left_name="strategy_specific",
        right_name="dataset_pooled",
    )


def view_coverage(
    terminal_rows: Iterable[Mapping[str, Any]],
    full_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize terminal/full points and time-span ratios for common cycles."""

    terminal = {_key(row): dict(row) for row in terminal_rows}
    full = {_key(row): dict(row) for row in full_rows}
    common = sorted(set(terminal) & set(full))
    if not common:
        return {"common_cycle_count": 0, "ratios": []}
    ratios = []
    for key in common:
        terminal_points = float(terminal[key].get("raw_point_count", np.nan))
        full_points = float(full[key].get("raw_point_count", np.nan))
        terminal_duration = float(terminal[key].get("duration_min", np.nan))
        full_duration = float(full[key].get("duration_min", np.nan))
        if full_points <= 0 or full_duration <= 0:
            continue
        ratios.append(
            {
                "battery_id": key[0],
                "cycle_id": key[1],
                "terminal_point_ratio": terminal_points / full_points,
                "terminal_time_span_ratio": terminal_duration / full_duration,
                "terminal_duration_min": terminal_duration,
                "full_duration_min": full_duration,
            }
        )
    return {
        "common_cycle_count": len(common),
        "ratio_count": len(ratios),
        "terminal_point_ratio_mean": float(np.mean([item["terminal_point_ratio"] for item in ratios])) if ratios else float("nan"),
        "terminal_time_span_ratio_mean": float(np.mean([item["terminal_time_span_ratio"] for item in ratios])) if ratios else float("nan"),
        "rows": ratios,
    }


__all__ = ["e2_comparisons", "e3_strategy_comparison", "paired_comparison", "view_coverage"]
