"""Shared plotting and validation helpers for capacity trajectory audits.

The dataset-specific entry points in this directory deliberately keep their
CSV readers separate.  This module only owns the common representation and
the figure/summary rendering so that XJTU and MIT diagnostics use the same
visual conventions.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class CapacityPoint:
    """One validated capacity label at one canonical/source cycle."""

    cycle: int
    capacity_ah: float


@dataclass
class CellTrajectory:
    """Capacity points for one physical or source battery."""

    condition: str
    cell_id: str
    split_role: str
    source_file: str
    points: list[CapacityPoint] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def ordered_points(self) -> list[CapacityPoint]:
        ordered = sorted(self.points, key=lambda item: item.cycle)
        cycles = [item.cycle for item in ordered]
        if len(cycles) != len(set(cycles)):
            raise ValueError(f"{self.cell_id}: duplicate exported cycle labels")
        if not ordered:
            raise ValueError(f"{self.cell_id}: no capacity labels")
        return ordered

    def summary(self) -> dict[str, object]:
        ordered = self.ordered_points()
        capacity = np.asarray([item.capacity_ah for item in ordered], dtype=float)
        return {
            "cell_id": self.cell_id,
            "source_file": self.source_file,
            "split_role": self.split_role,
            "exported_cycles": len(ordered),
            "cycle_range": [int(ordered[0].cycle), int(ordered[-1].cycle)],
            "capacity_Ah_range": [float(np.min(capacity)), float(np.max(capacity))],
            **self.metadata,
        }


def finite(value: object, *, column: str, path: Path, line_number: int) -> float:
    """Parse one finite numeric CSV value with an actionable error."""

    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: invalid {column}={value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}:{line_number}: non-finite {column}={value!r}")
    return result


def integer_cycle(value: object, *, path: Path, line_number: int) -> int:
    cycle = finite(value, column="cycle", path=path, line_number=line_number)
    if not cycle.is_integer() or cycle <= 0:
        raise ValueError(f"{path}:{line_number}: invalid cycle={value!r}")
    return int(cycle)


def condition_sort_key(value: str) -> tuple[float, float, str]:
    """Sort common XJTU rate labels numerically, then fall back to text."""

    text = str(value)
    xjtu_order = {"2C": 0.0, "3C": 1.0, "R2.5": 2.0, "R3": 3.0, "RW": 4.0, "satellite": 5.0}
    if text in xjtu_order:
        return (xjtu_order[text], 0.0, text)
    try:
        year, month, day = (int(item) for item in text.split("-"))
        return (10.0 + year, month * 100.0 + day, text)
    except (TypeError, ValueError):
        return (math.inf, math.inf, text)


def group_by_condition(
    cells: Sequence[CellTrajectory],
) -> dict[str, list[CellTrajectory]]:
    grouped: dict[str, list[CellTrajectory]] = defaultdict(list)
    for cell in cells:
        if cell.points:
            grouped[cell.condition].append(cell)
    if not grouped:
        raise ValueError("No non-empty capacity trajectories were loaded")
    for condition_cells in grouped.values():
        condition_cells.sort(
            key=lambda item: (
                item.split_role != "test",
                item.cell_id,
                item.source_file,
            )
        )
    return dict(grouped)


def _capacity_limits(values: np.ndarray, nominal_capacity: float) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("Cannot determine capacity axis limits from empty data")
    if values.size >= 5:
        low, high = np.quantile(values, [0.005, 0.995])
    else:
        low, high = float(np.min(values)), float(np.max(values))
    low = min(float(low), float(nominal_capacity))
    high = max(float(high), float(nominal_capacity))
    span = max(float(high - low), 0.05 * max(abs(float(nominal_capacity)), 1.0), 0.05)
    margin = max(0.06 * span, 0.02 * max(abs(float(nominal_capacity)), 1.0), 0.02)
    return float(low - margin), float(high + margin)


def _plot_cell_legend(axis, cells: Sequence[CellTrajectory], *, show_cell_legend: bool) -> None:
    if show_cell_legend:
        axis.legend(
            fontsize=7.0 if len(cells) <= 20 else 5.8,
            frameon=False,
            loc="best",
            handlelength=2.6,
            ncol=1 if len(cells) <= 20 else 2,
        )
        return

    handles = [
        Line2D([0], [0], color="#333333", linewidth=1.1, linestyle="-", label="development cell"),
        Line2D([0], [0], color="#333333", linewidth=1.2, linestyle="--", label="test cell"),
        Line2D([0], [0], color="#444444", linewidth=0.8, linestyle=":", label="nominal capacity"),
    ]
    axis.legend(handles=handles, fontsize=7.2, frameon=False, loc="best")


def plot_trajectories(
    dataset_id: str,
    trajectories: Mapping[str, Sequence[CellTrajectory]],
    output_path: Path,
    *,
    nominal_capacity_ah: float,
    x_label: str,
    title_suffix: str,
    show_cell_legend: bool,
) -> None:
    """Render one condition-panel mosaic using the shared audit style."""

    conditions = sorted(trajectories, key=condition_sort_key)
    ncols = 2 if len(conditions) > 1 else 1
    nrows = math.ceil(len(conditions) / ncols)
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8.1 * ncols, max(4.8 * nrows, 5.6)),
        squeeze=False,
        constrained_layout=True,
    )
    palette = plt.get_cmap("turbo")

    for index, condition in enumerate(conditions):
        axis = axes.flat[index]
        cells = list(trajectories[condition])
        capacity_values: list[float] = []
        denominator = max(len(cells) - 1, 1)
        for cell_index, cell in enumerate(cells):
            points = cell.ordered_points()
            cycle = np.asarray([item.cycle for item in points], dtype=int)
            capacity = np.asarray([item.capacity_ah for item in points], dtype=float)
            capacity_values.extend(capacity.tolist())
            color = palette(cell_index / denominator)
            is_test = cell.split_role == "test"
            axis.plot(
                cycle,
                capacity,
                color=color,
                linewidth=1.2 if is_test else 0.95,
                linestyle="--" if is_test else "-",
                alpha=0.86 if len(cells) > 20 else 0.92,
                label=cell.cell_id,
            )

        axis.axhline(
            nominal_capacity_ah,
            color="#444444",
            linewidth=0.8,
            linestyle=":",
            label="nominal capacity",
        )
        values = np.asarray(capacity_values, dtype=float)
        axis.set_ylim(*_capacity_limits(values, nominal_capacity_ah))
        axis.set_xlabel(x_label)
        axis.set_ylabel("Capacity (Ah)")
        axis.set_title(
            f"{condition} | {len(cells)} cells",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
        )
        axis.grid(alpha=0.22, linewidth=0.5)
        _plot_cell_legend(axis, cells, show_cell_legend=show_cell_legend)

    for axis in axes.flat[len(conditions) :]:
        axis.set_visible(False)

    figure.suptitle(
        f"{dataset_id}: canonical capacity trajectories\n"
        "Solid = development; dashed = test; gray dotted line = fixed nominal capacity. "
        f"{title_suffix}",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def trajectories_summary(
    trajectories: Mapping[str, Sequence[CellTrajectory]],
) -> dict[str, object]:
    return {
        "conditions": {
            condition: {
                "cells": {
                    cell.cell_id: cell.summary()
                    for cell in trajectories[condition]
                }
            }
            for condition in sorted(trajectories, key=condition_sort_key)
        }
    }


def write_summary(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
