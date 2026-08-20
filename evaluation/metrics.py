"""Metrics with battery-level aggregation as the default paper view."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {name: float("nan") for name in ("mae", "mape", "mse", "rmse")}
    error = y_pred - y_true
    mse = float(np.mean(error**2))
    denominator = np.maximum(np.abs(y_true), 1e-8)
    return {
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error) / denominator)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
    }


def grouped_metrics(y_true, y_pred, groups):
    buckets = defaultdict(lambda: ([], []))
    for truth, prediction, group in zip(y_true, y_pred, groups):
        buckets[str(group)][0].append(truth)
        buckets[str(group)][1].append(prediction)
    return {
        group: compute_metrics(values[0], values[1])
        for group, values in sorted(buckets.items())
    }


def macro_rmse_by_group(y_true, y_pred, groups):
    metrics = grouped_metrics(y_true, y_pred, groups)
    values = [entry["rmse"] for entry in metrics.values() if np.isfinite(entry["rmse"])]
    return float(np.mean(values)) if values else float("nan")

