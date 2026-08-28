#!/usr/bin/env python3
"""Plot SmartHealth model-input labels and anchor-only calibrations by cycle.

This diagnostic reads the cycle-level provenance files rather than the
point-level CSVs.  Each domain produces one figure; each operating condition
is a panel and its cells are overlaid in that panel.  The graph deliberately
preserves the current canonical cycle ordering so that source-session/chunk
ordering problems remain visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPO_ROOT / "datasets" / "SmartHealth_raw"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "smarthealth_capacity_trajectories"

DOMAIN_AUDIT_TAG = {
    "smarthealth_lishen40": "LISHEN40",
    "smarthealth_catl280": "CATL280",
    "smarthealth_eve280": "EVE280",
}
NOMINAL_CAPACITY_AH = {
    "smarthealth_lishen40": 40.0,
    "smarthealth_catl280": 280.0,
    "smarthealth_eve280": 280.0,
}
DEFAULT_DOMAINS = tuple(DOMAIN_AUDIT_TAG)
REQUIRED_COLUMNS = {
    "domain_id",
    "logical_sequence_id",
    "source_serial",
    "condition",
    "cycle",
    "label_capacity_Ah",
    "label_source",
    "split_role",
    "selected_candidate",
    "output_status",
    "calibration_anchor_only",
    "model_input_role",
}


def condition_sort_key(value: str) -> tuple[float, float, str]:
    """Sort familiar ``0.3C-60%DOD`` labels numerically, with a fallback."""

    text = str(value)
    try:
        rate, dod = text.split("C-", maxsplit=1)
        return (float(rate.rstrip("C")), float(dod.split("%", maxsplit=1)[0]), text)
    except (ValueError, IndexError):
        return (math.inf, math.inf, text)


def truth(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def finite(value: str | None, *, column: str, path: Path, line_number: int) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: invalid {column}={value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}:{line_number}: non-finite {column}={value!r}")
    return result


@dataclass(frozen=True)
class CapacityPoint:
    cycle: int
    capacity_ah: float
    label_source: str


@dataclass
class CellTrajectory:
    condition: str
    logical_sequence_id: str
    source_serial: str
    split_role: str
    points: list[CapacityPoint] = field(default_factory=list)
    anchors: list[CapacityPoint] = field(default_factory=list)

    def ordered_points(self) -> list[CapacityPoint]:
        ordered = sorted(self.points, key=lambda item: item.cycle)
        cycles = [item.cycle for item in ordered]
        if len(cycles) != len(set(cycles)):
            raise ValueError(f"{self.logical_sequence_id}: duplicate exported cycle labels")
        return ordered

    def summary(self) -> dict[str, object]:
        ordered = self.ordered_points()
        capacity = np.asarray([item.capacity_ah for item in ordered], dtype=float)
        cycles = np.asarray([item.cycle for item in ordered], dtype=int)
        direct_cycles = np.asarray(
            [item.cycle for item in ordered if item.label_source == "capacity_direct"],
            dtype=int,
        )
        anchor_cycles = np.asarray(
            [item.cycle for item in sorted(self.anchors, key=lambda item: item.cycle)],
            dtype=int,
        )
        reference_cycles = anchor_cycles if anchor_cycles.size else direct_cycles
        cycle_gaps = np.diff(cycles) - 1
        calibration_gaps = (
            np.diff(reference_cycles) if reference_cycles.size >= 2 else np.asarray([])
        )
        return {
            "source_serial": self.source_serial,
            "split_role": self.split_role,
            "exported_cycles": len(ordered),
            "cycle_range": [int(ordered[0].cycle), int(ordered[-1].cycle)],
            "label_capacity_Ah_range": [float(np.min(capacity)), float(np.max(capacity))],
            "capacity_direct_cycles": sum(
                item.label_source == "capacity_direct" for item in ordered
            ),
            "calibration_interpolated_cycles": sum(
                item.label_source == "calibration_interpolated" for item in ordered
            ),
            "calibration_anchor_only_cycles": len(self.anchors),
            "longest_missing_export_run_cycles": (
                int(np.max(cycle_gaps)) if cycle_gaps.size else 0
            ),
            "largest_capacity_reference_interval_cycles": (
                int(np.max(calibration_gaps)) if calibration_gaps.size else None
            ),
        }


def audit_path(raw_root: Path, domain_id: str) -> Path:
    return raw_root / "audit" / f"SMARTHEALTH_{DOMAIN_AUDIT_TAG[domain_id]}_CYCLE_PROVENANCE.csv"


def load_domain(raw_root: Path, domain_id: str) -> dict[str, list[CellTrajectory]]:
    path = audit_path(raw_root, domain_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing SmartHealth capacity provenance for {domain_id}: {path}")

    cells: dict[str, CellTrajectory] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing required provenance columns {missing}")
        for line_number, row in enumerate(reader, start=2):
            if row.get("domain_id") != domain_id or not truth(row.get("selected_candidate")):
                continue
            exported = row.get("output_status") == "exported"
            anchor_only = truth(row.get("calibration_anchor_only"))
            if not exported and not anchor_only:
                continue
            cycle_value = finite(row.get("cycle"), column="cycle", path=path, line_number=line_number)
            if not cycle_value.is_integer() or cycle_value <= 0:
                raise ValueError(f"{path}:{line_number}: invalid cycle={row.get('cycle')!r}")
            logical_id = str(row["logical_sequence_id"])
            cell = cells.get(logical_id)
            if cell is None:
                cell = CellTrajectory(
                    condition=str(row["condition"]),
                    logical_sequence_id=logical_id,
                    source_serial=str(row.get("source_serial") or logical_id),
                    split_role=str(row.get("split_role") or "unknown"),
                )
                cells[logical_id] = cell
            label_source = str(row.get("label_source") or "unknown")
            expected_sources = (
                {"calibration_anchor_only"}
                if anchor_only
                else {"capacity_direct", "calibration_interpolated"}
            )
            if label_source not in expected_sources:
                raise ValueError(
                    f"{path}:{line_number}: unexpected label_source={label_source!r}"
                )
            point = CapacityPoint(
                cycle=int(cycle_value),
                capacity_ah=finite(
                    row.get("label_capacity_Ah"),
                    column="label_capacity_Ah",
                    path=path,
                    line_number=line_number,
                ),
                label_source=label_source,
            )
            (cell.anchors if anchor_only else cell.points).append(point)

    by_condition: dict[str, list[CellTrajectory]] = defaultdict(list)
    for cell in cells.values():
        if cell.points:
            by_condition[cell.condition].append(cell)
    if not by_condition:
        raise ValueError(f"{path}: no exported, finite capacity labels found")
    for cells_in_condition in by_condition.values():
        cells_in_condition.sort(
            key=lambda item: (item.split_role != "test", item.source_serial, item.logical_sequence_id)
        )
    return dict(by_condition)


def display_name(cell: CellTrajectory) -> str:
    role = "test" if cell.split_role == "test" else "development"
    return f"{cell.source_serial or cell.logical_sequence_id} ({role})"


def break_at_missing_cycles(
    cycle: np.ndarray, capacity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs so plotting never bridges absent canonical cycles."""

    if cycle.size != capacity.size or cycle.size == 0:
        raise ValueError("Cycle/capacity arrays must be non-empty and aligned")
    output_cycle: list[float] = [float(cycle[0])]
    output_capacity: list[float] = [float(capacity[0])]
    for previous, current, value in zip(cycle, cycle[1:], capacity[1:]):
        if int(current) != int(previous) + 1:
            output_cycle.append(math.nan)
            output_capacity.append(math.nan)
        output_cycle.append(float(current))
        output_capacity.append(float(value))
    return np.asarray(output_cycle), np.asarray(output_capacity)


def plot_domain(domain_id: str, trajectories: dict[str, list[CellTrajectory]], output_path: Path) -> None:
    conditions = sorted(trajectories, key=condition_sort_key)
    ncols = 2 if len(conditions) > 1 else 1
    nrows = math.ceil(len(conditions) / ncols)
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8.1 * ncols, max(4.6 * nrows, 5.6)),
        squeeze=False,
        constrained_layout=True,
    )
    palette = plt.get_cmap("tab20")

    for index, condition in enumerate(conditions):
        axis = axes.flat[index]
        cells = trajectories[condition]
        nominal_capacity = NOMINAL_CAPACITY_AH[domain_id]
        capacity_values: list[float] = []
        for cell_index, cell in enumerate(cells):
            points = cell.ordered_points()
            cycle = np.asarray([item.cycle for item in points], dtype=int)
            capacity = np.asarray([item.capacity_ah for item in points], dtype=float)
            capacity_values.extend(capacity.tolist())
            color = palette(cell_index % palette.N)
            is_test = cell.split_role == "test"
            line_cycle, line_capacity = break_at_missing_cycles(cycle, capacity)
            axis.plot(
                line_cycle,
                line_capacity,
                color=color,
                linewidth=1.2 if is_test else 0.95,
                linestyle="--" if is_test else "-",
                alpha=0.92,
                label=display_name(cell),
            )
            direct = np.asarray([item.label_source == "capacity_direct" for item in points])
            if np.any(direct):
                axis.scatter(cycle[direct], capacity[direct], color=color, s=11, alpha=0.95, zorder=4)
            interpolated = np.asarray(
                [item.label_source == "calibration_interpolated" for item in points]
            )
            if np.any(interpolated):
                axis.scatter(
                    cycle[interpolated],
                    capacity[interpolated],
                    facecolors="none",
                    edgecolors=[color],
                    s=5,
                    linewidths=0.35,
                    alpha=0.45,
                    zorder=3,
                )
            if cell.anchors:
                anchor_cycle = np.asarray([item.cycle for item in cell.anchors], dtype=int)
                anchor_capacity = np.asarray(
                    [item.capacity_ah for item in cell.anchors], dtype=float
                )
                capacity_values.extend(anchor_capacity.tolist())
                axis.scatter(
                    anchor_cycle,
                    anchor_capacity,
                    color=[color],
                    marker="x",
                    s=24,
                    linewidths=0.9,
                    alpha=0.95,
                    zorder=5,
                )
        axis.axhline(
            nominal_capacity,
            color="#444444",
            linewidth=0.8,
            linestyle=":",
            label=f"Nominal capacity = {nominal_capacity:g} Ah",
        )

        values = np.asarray(capacity_values, dtype=float)
        low, high = np.quantile(values, [0.002, 0.998])
        low = min(float(low), nominal_capacity)
        high = max(float(high), nominal_capacity)
        margin = max(0.06 * float(high - low), 0.05)
        axis.set_ylim(float(low - margin), float(high + margin))
        axis.set_xlabel("Canonical source cycle (current preprocessing order)")
        axis.set_ylabel("Label capacity (Ah)")
        axis.set_title(f"{condition} | {len(cells)} cells", loc="left", fontsize=10.5, fontweight="bold")
        axis.grid(alpha=0.22, linewidth=0.5)
        handles, labels = axis.get_legend_handles_labels()
        handles.extend(
            [
                Line2D([], [], color="#333333", marker="o", linestyle="None", markersize=3.5),
                Line2D(
                    [],
                    [],
                    color="#333333",
                    marker="o",
                    markerfacecolor="none",
                    linestyle="None",
                    markersize=3.5,
                    markeredgewidth=0.6,
                ),
                Line2D([], [], color="#333333", marker="x", linestyle="None", markersize=4),
            ]
        )
        labels.extend(["Direct normal-cycle label", "Interpolated normal-cycle label", "Calibration anchor only"])
        axis.legend(handles, labels, fontsize=7.1, frameon=False, loc="best", handlelength=2.6)

    for axis in axes.flat[len(conditions) :]:
        axis.set_visible(False)

    figure.suptitle(
        f"{domain_id}: canonical capacity trajectories\n"
        "Solid = development; dashed = test; filled dots = direct normal-cycle labels; "
        "hollow dots = interpolated labels; crosses = anchor-only calibrations; "
        "line breaks = missing model-input cycles. "
        "Gray dotted line = fixed nominal capacity. "
        "Cycle ordering is intentionally not reconstructed from absolute time.",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def domain_summary(domain_id: str, trajectories: dict[str, list[CellTrajectory]]) -> dict[str, object]:
    return {
        "domain_id": domain_id,
        "label": "label_capacity_Ah",
        "description": (
            "Exported model-input labels plus non-exported calibration anchors; "
            "no chronology reconstruction; lines break across missing canonical cycles."
        ),
        "conditions": {
            condition: {
                "cells": {
                    cell.logical_sequence_id: cell.summary() for cell in trajectories[condition]
                }
            }
            for condition in sorted(trajectories, key=condition_sort_key)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Canonical SmartHealth_raw root containing audit/*_CYCLE_PROVENANCE.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for the domain mosaics and summary.json.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DEFAULT_DOMAINS,
        default=list(DEFAULT_DOMAINS),
        help="One or more SmartHealth domains to plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Canonical SmartHealth raw root does not exist: {raw_root}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "raw_root": str(raw_root),
        "event_filter": "selected_candidate=true and (output_status=exported or calibration_anchor_only=true)",
        "label": "label_capacity_Ah",
        "domains": {},
    }
    for domain_id in args.domains:
        trajectories = load_domain(raw_root, domain_id)
        output_path = output_dir / f"{domain_id}_capacity_vs_cycle.png"
        plot_domain(domain_id, trajectories, output_path)
        summary["domains"][domain_id] = domain_summary(domain_id, trajectories)
        print(f"[{domain_id}] wrote {output_path}", flush=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
