#!/usr/bin/env python3
"""Plot full principal charge trajectories directly from SmartHealth source CSVs.

This diagnostic deliberately reads the original GB18030 source files rather
than ``datasets/SmartHealth_raw``.  It is therefore intended to answer a data
policy question such as whether 3.58 V remains inside the actual CC plateau.

Each source event is identified by ``(source file, source cycle)``.  The
script intentionally does *not* merge repeated/reset source-cycle numbers
between chunks, because source-file chronology is part of the diagnostic.  It
selects the longest contiguous ``恒流恒压充电`` event in every source cycle,
plots its full voltage/current trajectory, and separately records the existing
persistent-taper CC→CV inference outcome.

All accepted source events contribute to the mean ± one-standard-deviation
band.  Dashed curves are source-serial (logical-cell) means within the same
operating condition.  The source corpus is large (about 88 GB), so scanning is
parallelized by independent source file and retains only online curve moments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.smarthealth_common import (  # noqa: E402
    CATL280_CONFIG,
    CHARGE_STEP,
    EVE280_CONFIG,
    LISHEN40_CONFIG,
    DomainConfig,
    Point,
    SourceIdentity,
    duration_seconds,
    events,
    list_identities,
    pick_event,
    split_combined_charge,
    visit_cycles,
)


DOMAIN_CONFIGS: dict[str, DomainConfig] = {
    LISHEN40_CONFIG.domain_id: LISHEN40_CONFIG,
    CATL280_CONFIG.domain_id: CATL280_CONFIG,
    EVE280_CONFIG.domain_id: EVE280_CONFIG,
}
DEFAULT_DOMAINS = tuple(DOMAIN_CONFIGS)
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get("SMARTHEALTH_SOURCE_ROOT", "/data1/chenyanxi/lb_project/datasets/SmartHealth")
)


def _condition_sort_key(value: str) -> tuple[float, float, str]:
    """Sort familiar ``0.3C-60%DOD`` names numerically, with a safe fallback."""

    text = str(value)
    try:
        left, right = text.split("C-", maxsplit=1)
        return (float(left.rstrip("C")), float(right.split("%", maxsplit=1)[0]), text)
    except (ValueError, IndexError):
        return (math.inf, math.inf, text)


def _phase_arguments() -> SimpleNamespace:
    """The existing source-to-canonical phase-inference defaults, read-only."""

    return SimpleNamespace(
        min_cc_points=60,
        min_cv_points=60,
        cc_reference_fraction=0.20,
        cc_reference_min_points=120,
        cc_reference_quantile=0.90,
        cv_taper_fraction=0.01,
        cv_persistence_points=30,
        cv_voltage_tolerance_v=0.02,
    )


@dataclass
class CurveMoments:
    """Mergeable online mean/variance for a fixed-length source trajectory."""

    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if self.mean is None:
            self.count = 1
            self.mean = values.copy()
            self.m2 = np.zeros_like(values)
            return
        if values.shape != self.mean.shape:
            raise ValueError(f"Curve shape changed from {self.mean.shape} to {values.shape}")
        self.count += 1
        delta = values - self.mean
        self.mean += delta / float(self.count)
        assert self.m2 is not None
        self.m2 += delta * (values - self.mean)

    def merge(self, other: "CurveMoments") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = int(other.count)
            self.mean = None if other.mean is None else other.mean.copy()
            self.m2 = None if other.m2 is None else other.m2.copy()
            return
        if self.mean is None or self.m2 is None or other.mean is None or other.m2 is None:
            raise ValueError("Cannot merge incomplete curve moments")
        if self.mean.shape != other.mean.shape:
            raise ValueError(f"Curve shape mismatch: {self.mean.shape} vs {other.mean.shape}")
        left_count, right_count = self.count, other.count
        combined = left_count + right_count
        delta = other.mean - self.mean
        self.mean = self.mean + delta * (float(right_count) / float(combined))
        self.m2 = self.m2 + other.m2 + delta * delta * (left_count * right_count / float(combined))
        self.count = int(combined)

    def std(self) -> np.ndarray:
        if self.count <= 1 or self.m2 is None:
            return np.zeros_like(self.mean) if self.mean is not None else np.asarray([], dtype=float)
        return np.sqrt(np.maximum(self.m2 / float(self.count - 1), 0.0))


@dataclass
class ScalarMoments:
    """Small mergeable scalar summary; unlike curves it is JSON-friendly."""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += float(value)
        self.total_sq += float(value) * float(value)
        self.minimum = min(self.minimum, float(value))
        self.maximum = max(self.maximum, float(value))

    def merge(self, other: "ScalarMoments") -> None:
        if other.count == 0:
            return
        self.count += int(other.count)
        self.total += float(other.total)
        self.total_sq += float(other.total_sq)
        self.minimum = min(self.minimum, float(other.minimum))
        self.maximum = max(self.maximum, float(other.maximum))

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.total / float(self.count)
        variance = max(self.total_sq / float(self.count) - mean * mean, 0.0)
        return {
            "n": int(self.count),
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
            "min": float(self.minimum),
            "max": float(self.maximum),
        }


@dataclass
class SourceFileProfile:
    """Compact worker result for one original CSV chunk."""

    identity: SourceIdentity
    voltage: CurveMoments = field(default_factory=CurveMoments)
    current: CurveMoments = field(default_factory=CurveMoments)
    source_cycles: int = 0
    principal_charge_events: int = 0
    boundary_valid_events: int = 0
    cc_reaches_358: int = 0
    cc_reaches_360: int = 0
    boundary_fraction: ScalarMoments = field(default_factory=ScalarMoments)
    cc_max_voltage: ScalarMoments = field(default_factory=ScalarMoments)
    skips: Counter = field(default_factory=Counter)


@dataclass
class AggregateProfile:
    """Aggregate all source chunks for one condition or one source serial."""

    source_files: int = 0
    source_cycles: int = 0
    principal_charge_events: int = 0
    boundary_valid_events: int = 0
    cc_reaches_358: int = 0
    cc_reaches_360: int = 0
    voltage: CurveMoments = field(default_factory=CurveMoments)
    current: CurveMoments = field(default_factory=CurveMoments)
    boundary_fraction: ScalarMoments = field(default_factory=ScalarMoments)
    cc_max_voltage: ScalarMoments = field(default_factory=ScalarMoments)
    skips: Counter = field(default_factory=Counter)

    def merge_file(self, source: SourceFileProfile) -> None:
        self.source_files += 1
        self.source_cycles += int(source.source_cycles)
        self.principal_charge_events += int(source.principal_charge_events)
        self.boundary_valid_events += int(source.boundary_valid_events)
        self.cc_reaches_358 += int(source.cc_reaches_358)
        self.cc_reaches_360 += int(source.cc_reaches_360)
        self.voltage.merge(source.voltage)
        self.current.merge(source.current)
        self.boundary_fraction.merge(source.boundary_fraction)
        self.cc_max_voltage.merge(source.cc_max_voltage)
        self.skips.update(source.skips)

    def summary(self) -> dict:
        boundary_denominator = max(self.boundary_valid_events, 1)
        return {
            "source_files": int(self.source_files),
            "source_cycles": int(self.source_cycles),
            "principal_charge_events": int(self.principal_charge_events),
            "trajectory_events": int(self.voltage.count),
            "valid_cc_cv_boundary_events": int(self.boundary_valid_events),
            "valid_cc_cv_boundary_fraction": float(
                self.boundary_valid_events / max(self.principal_charge_events, 1)
            ),
            "inferred_cc_reaches_3_58_v": {
                "count": int(self.cc_reaches_358),
                "fraction_of_valid_boundaries": float(self.cc_reaches_358 / boundary_denominator),
            },
            "inferred_cc_reaches_3_60_v": {
                "count": int(self.cc_reaches_360),
                "fraction_of_valid_boundaries": float(self.cc_reaches_360 / boundary_denominator),
            },
            "inferred_cc_boundary_fraction_of_charge_time": self.boundary_fraction.summary(),
            "inferred_cc_max_voltage_v": self.cc_max_voltage.summary(),
            "skipped_source_cycles": dict(sorted(self.skips.items())),
        }


@dataclass
class ConditionProfile:
    condition: str
    aggregate: AggregateProfile = field(default_factory=AggregateProfile)
    cells: dict[str, AggregateProfile] = field(default_factory=dict)

    def merge_file(self, source: SourceFileProfile) -> None:
        self.aggregate.merge_file(source)
        cell = self.cells.setdefault(source.identity.logical_sequence_id, AggregateProfile())
        cell.merge_file(source)

    def summary(self) -> dict:
        return {
            **self.aggregate.summary(),
            "source_serial_cells": {
                cell_id: profile.summary() for cell_id, profile in sorted(self.cells.items())
            },
        }


def _resample_full_charge_event(event: list[Point], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Resample the complete original charge event by its actual source time."""

    time = np.asarray([duration_seconds(point.time_text) for point in event], dtype=np.float64)
    voltage = np.asarray([point.voltage_v for point in event], dtype=np.float64)
    current = np.abs(np.asarray([point.current_a for point in event], dtype=np.float64))
    valid = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current)
    time, voltage, current = time[valid], voltage[valid], current[valid]
    if time.size < 2:
        return None
    order = np.argsort(time, kind="stable")
    time, voltage, current = time[order], voltage[order], current[order]
    unique_time, unique_index = np.unique(time, return_index=True)
    if unique_time.size < 2 or unique_time[-1] <= unique_time[0]:
        return None
    voltage, current = voltage[unique_index], current[unique_index]
    target_time = unique_time[0] + grid * (unique_time[-1] - unique_time[0])
    return (
        np.interp(target_time, unique_time, voltage).astype(np.float64),
        np.interp(target_time, unique_time, current).astype(np.float64),
    )


def _scan_one_source_file(identity: SourceIdentity, resample_points: int) -> SourceFileProfile:
    """Worker: scan one source chunk without source-cycle deduplication."""

    profile = SourceFileProfile(identity=identity)
    grid = np.linspace(0.0, 1.0, int(resample_points), dtype=np.float64)
    phase_args = _phase_arguments()

    def callback(_source_cycle: int, points: list[Point], _has_temperature: bool) -> None:
        selected = pick_event(
            events(points, CHARGE_STEP), "charge_capacity_ah", prefer_capacity_span=False
        )
        if selected is None:
            profile.skips["no_combined_charge_event"] += 1
            return
        _event_index, charge_event = selected
        resampled = _resample_full_charge_event(charge_event, grid)
        if resampled is None:
            profile.skips["invalid_full_charge_time_or_signal"] += 1
            return
        voltage, current = resampled
        profile.voltage.update(voltage)
        profile.current.update(current)
        profile.principal_charge_events += 1

        # This uses the already established source policy only to locate the
        # CC→CV boundary for the plot annotation; no canonical sample is read.
        phase = split_combined_charge(points, phase_args)
        if phase.status != "ok" or not phase.cc or not phase.cv:
            profile.skips[f"boundary_{phase.reason or phase.status}"] += 1
            return
        full_time = np.asarray([duration_seconds(point.time_text) for point in charge_event], dtype=float)
        cc_end_time = duration_seconds(phase.cc[-1].time_text)
        valid_time = full_time[np.isfinite(full_time)]
        if valid_time.size >= 2 and math.isfinite(cc_end_time) and valid_time[-1] > valid_time[0]:
            profile.boundary_fraction.update(
                (cc_end_time - float(valid_time[0])) / float(valid_time[-1] - valid_time[0])
            )
        cc_max = float(max(point.voltage_v for point in phase.cc))
        profile.cc_max_voltage.update(cc_max)
        profile.boundary_valid_events += 1
        profile.cc_reaches_358 += int(cc_max >= 3.58)
        profile.cc_reaches_360 += int(cc_max >= 3.60)

    source_info = visit_cycles(identity, callback)
    profile.source_cycles = int(source_info["source_cycle_count"])
    return profile


def _load_domain(
    config: DomainConfig,
    source_root: Path,
    *,
    resample_points: int,
    workers: int,
    max_source_files: int | None,
) -> dict[str, ConditionProfile]:
    identities = list_identities(source_root, config, max_source_files)
    print(
        f"[{config.domain_id}] scanning {len(identities):,} original source chunks "
        f"with {workers} worker(s)",
        flush=True,
    )
    by_path: dict[str, SourceFileProfile] = {}
    if workers == 1:
        for index, identity in enumerate(identities, start=1):
            by_path[identity.relative_path] = _scan_one_source_file(identity, resample_points)
            print(f"[{config.domain_id}] {index:,}/{len(identities):,}: {identity.relative_path}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {
                executor.submit(_scan_one_source_file, identity, resample_points): identity
                for identity in identities
            }
            completed = 0
            report_step = max(1, len(identities) // 20)
            for future in as_completed(pending):
                identity = pending[future]
                by_path[identity.relative_path] = future.result()
                completed += 1
                if completed == len(identities) or completed % report_step == 0:
                    print(
                        f"[{config.domain_id}] completed {completed:,}/{len(identities):,} source chunks",
                        flush=True,
                    )

    conditions: dict[str, ConditionProfile] = {}
    # Deterministic merge order makes repeated serial/parallel diagnostic runs
    # numerically stable up to ordinary floating-point roundoff.
    for path in sorted(by_path):
        source = by_path[path]
        condition = conditions.setdefault(source.identity.condition, ConditionProfile(source.identity.condition))
        condition.merge_file(source)
    return conditions


def _safe_y_limits(mean: np.ndarray, std: np.ndarray, *, floor: float | None = None) -> tuple[float, float]:
    values = np.concatenate([mean - std, mean + std])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    low, high = np.quantile(values, [0.002, 0.998])
    span = max(float(high - low), 1e-5)
    low = float(low - max(0.08 * span, 1e-4))
    high = float(high + max(0.08 * span, 1e-4))
    if floor is not None:
        low = min(low, float(floor))
    return low, high


def _fraction_text(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{100.0 * numerator / denominator:.1f}%"


def _plot_quantity(
    ax,
    aggregate: AggregateProfile,
    cells: dict[str, AggregateProfile],
    *,
    x: np.ndarray,
    quantity: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    moments = aggregate.voltage if quantity == "voltage" else aggregate.current
    if moments.count == 0 or moments.mean is None:
        raise ValueError("Cannot plot an empty source-event aggregate")
    mean, std = moments.mean, moments.std()
    ax.fill_between(x, mean - std, mean + std, color="#9ecae1", alpha=0.48, linewidth=0, label="all source events: mean ± 1 SD")
    ax.plot(x, mean, color="#252525", linewidth=1.5, label="all source events: mean")
    for cell_id in sorted(cells):
        profile = cells[cell_id]
        cell_moments = profile.voltage if quantity == "voltage" else profile.current
        if cell_moments.count and cell_moments.mean is not None:
            ax.plot(x, cell_moments.mean, color="#636363", linewidth=0.62, alpha=0.75, linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Normalized principal charge-event time")
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.18, linewidth=0.45)
    return mean, std


def _plot_domain(domain_id: str, conditions: dict[str, ConditionProfile], output_path: Path) -> None:
    ordered_conditions = sorted(conditions, key=_condition_sort_key)
    if not ordered_conditions:
        raise ValueError(f"No principal charge events were found for {domain_id}")
    figure, axes = plt.subplots(
        nrows=len(ordered_conditions),
        ncols=2,
        figsize=(15.5, max(3.25 * len(ordered_conditions), 6.0)),
        constrained_layout=True,
    )
    if len(ordered_conditions) == 1:
        axes = np.asarray([axes])
    first_mean = conditions[ordered_conditions[0]].aggregate.voltage.mean
    if first_mean is None:
        raise ValueError(f"No resampled source trajectory was found for {domain_id}")
    x = np.linspace(0.0, 1.0, first_mean.size)

    for row_index, condition_name in enumerate(ordered_conditions):
        profile = conditions[condition_name]
        aggregate = profile.aggregate
        voltage_ax, current_ax = axes[row_index]
        voltage_mean, voltage_std = _plot_quantity(
            voltage_ax,
            aggregate,
            profile.cells,
            x=x,
            quantity="voltage",
            label="Voltage (V)",
        )
        current_mean, current_std = _plot_quantity(
            current_ax,
            aggregate,
            profile.cells,
            x=x,
            quantity="current",
            label="|Current| (A)",
        )
        for voltage, color, text in ((3.45, "#969696", "3.45 V"), (3.58, "#238b45", "3.58 V"), (3.60, "#d95f0e", "3.60 V")):
            voltage_ax.axhline(voltage, color=color, linestyle=":", linewidth=0.85)
        if aggregate.boundary_fraction.count:
            boundary = aggregate.boundary_fraction.total / float(aggregate.boundary_fraction.count)
            voltage_ax.axvline(boundary, color="#6a51a3", linestyle="--", linewidth=0.9)
            current_ax.axvline(boundary, color="#6a51a3", linestyle="--", linewidth=0.9)
        voltage_ax.set_ylim(*_safe_y_limits(voltage_mean, voltage_std))
        current_ax.set_ylim(*_safe_y_limits(current_mean, current_std, floor=0.0))

        title = (
            f"{condition_name} | {aggregate.principal_charge_events:,} source charge events, "
            f"{len(profile.cells)} cells, {aggregate.source_files} chunks"
        )
        voltage_ax.set_title(title, loc="left", fontsize=9.2, fontweight="bold")
        current_ax.set_title(
            "CC boundary valid: "
            f"{_fraction_text(aggregate.boundary_valid_events, aggregate.principal_charge_events)}; "
            f"CC reaches 3.58 V: {_fraction_text(aggregate.cc_reaches_358, aggregate.boundary_valid_events)}; "
            f"3.60 V: {_fraction_text(aggregate.cc_reaches_360, aggregate.boundary_valid_events)}",
            loc="left",
            fontsize=8.1,
        )

    axes[0, 0].legend(loc="best", fontsize=7.2, frameon=False)
    figure.suptitle(
        f"{domain_id}: original SmartHealth full principal charge events\n"
        "Every source-file/source-cycle event is retained without canonical deduplication. "
        "Dashed curves are source-serial means; purple line is mean inferred CC→CV boundary.",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _domain_summary(domain_id: str, conditions: dict[str, ConditionProfile]) -> dict:
    ordered = sorted(conditions.items(), key=lambda item: _condition_sort_key(item[0]))
    source_files = sum(profile.aggregate.source_files for _, profile in ordered)
    source_cycles = sum(profile.aggregate.source_cycles for _, profile in ordered)
    charge_events = sum(profile.aggregate.principal_charge_events for _, profile in ordered)
    return {
        "domain_id": domain_id,
        "source_files": int(source_files),
        "source_cycles": int(source_cycles),
        "principal_charge_events": int(charge_events),
        "conditions": {name: profile.summary() for name, profile in ordered},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Original SmartHealth source root containing LISHEN/CATL/EVE, not canonical SmartHealth_raw.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "smarthealth_source_charge_profiles",
        help="Directory for source-trajectory mosaics and summary.json.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DEFAULT_DOMAINS,
        default=list(DEFAULT_DOMAINS),
        help="One or more SmartHealth battery domains to scan.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Independent source-file workers; default is a conservative four I/O workers.",
    )
    parser.add_argument(
        "--resample-points",
        type=int,
        default=256,
        help="Common points for each full source charge event (default: 256).",
    )
    parser.add_argument(
        "--max-source-files",
        type=int,
        default=None,
        help="Optional per-domain limit for a quick source-schema smoke only.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.resample_points < 16:
        parser.error("--resample-points must be at least 16")
    if args.max_source_files is not None and args.max_source_files < 1:
        parser.error("--max-source-files must be positive")
    return args


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Original SmartHealth source root does not exist: {source_root}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": (
            "Original, full principal combined-charge events from SmartHealth source CSVs. "
            "No canonical raw files, source-cycle deduplication, SOH labels, or model windows are used."
        ),
        "source_root": str(source_root),
        "event_identity": "source_file + source_cycle; repeated/reset cycle numbers remain separate events",
        "principal_charge_selection": "longest contiguous 恒流恒压充电 event per source cycle",
        "cc_cv_annotation": "existing persistent-taper source policy, used only for diagnostics",
        "resample_points_per_full_charge_event": int(args.resample_points),
        "workers": int(args.workers),
        "domains": {},
    }
    for domain_id in args.domains:
        conditions = _load_domain(
            DOMAIN_CONFIGS[domain_id],
            source_root,
            resample_points=int(args.resample_points),
            workers=int(args.workers),
            max_source_files=args.max_source_files,
        )
        output_path = output_dir / f"{domain_id}_source_charge_profiles.png"
        _plot_domain(domain_id, conditions, output_path)
        summary["domains"][domain_id] = _domain_summary(domain_id, conditions)
        print(f"[{domain_id}] wrote {output_path}", flush=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
