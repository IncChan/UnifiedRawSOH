"""Evaluation helpers for Paper-Backup."""

from .aggregation import aggregate_seed_metrics, metrics_from_rows
from .comparisons import e2_comparisons, e3_strategy_comparison, paired_comparison, view_coverage

__all__ = [
    "aggregate_seed_metrics",
    "e2_comparisons",
    "e3_strategy_comparison",
    "metrics_from_rows",
    "paired_comparison",
    "view_coverage",
]
