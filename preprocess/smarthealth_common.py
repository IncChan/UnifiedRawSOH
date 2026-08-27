#!/usr/bin/env python3
"""Common, auditable building blocks for the three SmartHealth families.

This module deliberately has no all-domain command-line entry point.  The
three family-specific RAW entry points and the three family-specific FEATURE
entry points import the appropriate immutable :class:`DomainConfig` below.

The source ``恒流恒压充电`` step is first divided into a full CC and CV phase by
a persistent-taper detector.  Only after that decision do we select the model
windows.  Labels come only from periodic full-capacity calibration discharges;
partial-DOD discharge capacities are never normalized by nominal capacity or
used as SOH labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:
    from .smarthealth_bol import (
        BOL_REFERENCE_CONTRACT_VERSION,
        BOL_REFERENCE_SOURCE,
        BOL_RULE_VERSION,
        build_frozen_smarthealth_bol_reference,
    )
except ImportError:  # Family entry points also support direct script execution.
    from smarthealth_bol import (  # type: ignore[no-redef]
        BOL_REFERENCE_CONTRACT_VERSION,
        BOL_REFERENCE_SOURCE,
        BOL_RULE_VERSION,
        build_frozen_smarthealth_bol_reference,
    )


SOURCE_ENCODING = "gb18030"
# v5 retains v4's absolute-time chronology and overlap reconciliation. It also
# separates model-window eligibility from calibration eligibility: partial-DOD
# charge traces may start above the common CC lower bound, while a valid
# full-capacity discharge can calibrate labels even when its charge trace is
# not a model input.
POLICY_VERSION = "smarthealth_cccv_calibration_v5"
SPLIT_STRATEGY_VERSION = "smarthealth_condition_cell_split_2development_1test_v3"
DEFAULT_MAX_SOURCE_CYCLE_DURATION_HOURS = 24.0
# CV boundary detection is dataset/DoD dependent because the source devices do
# not all record the same length of the current-taper tail.  These are detection
# thresholds only: they do not change the canonical RAW schema, label policy,
# split strategy, or either version identifier above.  An omitted domain/DoD
# keeps the historical parser default (60 points).
CV_MIN_POINTS_BY_DOMAIN_DOD: dict[str, dict[int, int]] = {
    "smarthealth_eve280": {
        20: 30,
        60: 30,
        100: 30,
    },
}
CHARGE_STEP = "恒流恒压充电"
DISCHARGE_STEP = "恒流放电"
SOURCE_POINT_REQUIRED_COLUMNS = (
    "循环号",
    "工步号",
    "工步类型",
    "时间",
    "绝对时间",
    "电流(A)",
    "电压(V)",
    "充电容量(Ah)",
    "放电容量(Ah)",
)
SOURCE_REQUIRED_COLUMNS = set(SOURCE_POINT_REQUIRED_COLUMNS)


@dataclass(frozen=True)
class DomainConfig:
    """Immutable source and protocol facts for exactly one battery family."""

    domain_id: str
    source_top_level: str
    manufacturer: str
    paper_alias: str
    nominal_capacity_ah: float
    cc_voltage_low_v: float = 3.45
    cc_voltage_high_v: float = 3.58
    cc_coverage_tolerance_v: float = 0.01
    cv_c_rate_low: float = 0.05
    cv_c_rate_high: float = 0.25
    cv_selection_tolerance_c: float = 0.002
    cv_high_coverage_c_rate: float = 0.252
    cv_low_coverage_c_rate: float = 0.052
    calibration_min_nominal_fraction: float = 0.90


LISHEN40_CONFIG = DomainConfig(
    domain_id="smarthealth_lishen40",
    source_top_level="LISHEN",
    manufacturer="LISHEN",
    paper_alias="C1",
    nominal_capacity_ah=40.0,
)
CATL280_CONFIG = DomainConfig(
    domain_id="smarthealth_catl280",
    source_top_level="CATL",
    manufacturer="CATL",
    paper_alias="C2",
    nominal_capacity_ah=280.0,
)
EVE280_CONFIG = DomainConfig(
    domain_id="smarthealth_eve280",
    source_top_level="EVE",
    manufacturer="EVE",
    paper_alias="C3",
    nominal_capacity_ah=280.0,
)


# This is intentionally versioned evidence, rather than a runtime choice made
# from a particular train/test subset.  It documents the small, read-only
# representative-source audit that preceded the canonical implementation.
CC_UPPER_BOUND_AUDIT = {
    "decision": "3.58 V",
    "rejected_candidate": "3.60 V",
    "reason": (
        "Across representative LISHEN, CATL and EVE cycles at 20/60/100%DOD, "
        "3.58 V remained consistently inside the inferred high-current plateau; "
        "3.60 V was frequently the transition point or was not stably reached "
        "before persistent CV taper."
    ),
    "read_only_sample_summary": {
        "LISHEN": "20-cycle samples: inferred CC maxima 3.5841–3.6015 V; all sampled cycles reached 3.58 V.",
        "CATL": "representative 20-cycle samples: inferred CC maxima 3.5972–3.6002 V; all sampled cycles reached 3.58 V.",
        "EVE": "representative 20-cycle samples: inferred CC maxima 3.5941–3.5996 V; all sampled cycles reached 3.58 V.",
        "CV": "All accepted representative cycles covered nominal-current 0.25C down to 0.05C within the configured tolerance.",
    },
}


RAW_COLUMNS = [
    "dataset",
    "dataset_id",
    "domain_id",
    "manufacturer",
    "cell",
    "battery_id",
    "source_serial",
    "logical_sequence_id",
    "source_series",
    "condition",
    "cycle",
    "SOH",
    "label_source",
    "cycle_discharge_capacity_Ah",
    "label_capacity_Ah",
    "reference_calibration_capacity_Ah",
    "bol_q_ref_Ah",
    "bol_q_ref_rule",
    "bol_q_ref_source",
    "split_role",
    "split_status",
    "split_issue",
    "split_strategy_version",
    "segment",
    "cycle_point_index",
    "segment_point_index",
    "source_row_index",
    "relative_time",
    "relative_time_min",
    "relative_time_unit",
    "voltage_V",
    "current_A",
    "c_rate",
    "charge_capacity_Ah",
    "discharge_capacity_Ah",
    "temperature_C",
    "power_W",
    "source_file",
    "chunk_id",
    "source_file_part",
    "source_cycle",
    "source_absolute_start_time",
    "source_absolute_end_time",
    "source_step_id",
    "strategy_version",
    "phase_policy_version",
    "cc_current_reference_A",
    "cv_start_source_row_index",
    "cv_start_voltage_V",
    "cv_start_current_A",
    "cc_voltage_low_V",
    "cc_voltage_high_V",
    "cv_c_rate_low",
    "cv_c_rate_high",
]

FEATURE_ELECTRICAL_COLUMNS = [
    "voltage mean",
    "voltage std",
    "voltage kurtosis",
    "voltage skewness",
    "CC Q",
    "CC charge time",
    "voltage slope",
    "voltage entropy",
    "current mean",
    "current std",
    "current kurtosis",
    "current skewness",
    "CV Q",
    "CV charge time",
    "current slope",
    "current entropy",
]
FEATURE_TEMPERATURE_COLUMNS = [
    "T_CC_mean",
    "T_CC_max",
    "T_CC_delta",
    "T_CC_slope",
    "T_CV_mean",
    "T_CV_max",
    "T_CV_delta",
    "T_CV_slope",
]
FEATURE_COLUMNS = [
    *FEATURE_ELECTRICAL_COLUMNS,
    *FEATURE_TEMPERATURE_COLUMNS,
    "capacity",
]
FEATURE_PREFIX_COLUMNS = [
    "dataset",
    "dataset_id",
    "domain_id",
    "manufacturer",
    "cell",
    "battery_id",
    "source_serial",
    "logical_sequence_id",
    "source_series",
    "condition",
    "cycle",
    "SOH",
    "label_source",
    "cycle_discharge_capacity_Ah",
    "label_capacity_Ah",
    "reference_calibration_capacity_Ah",
    "bol_q_ref_Ah",
    "bol_q_ref_rule",
    "bol_q_ref_source",
    "split_role",
    "split_status",
    "split_issue",
    "split_strategy_version",
    "source_file",
    "chunk_id",
    "source_file_part",
    "source_cycle",
    "source_absolute_start_time",
    "source_absolute_end_time",
    "strategy_version",
    "phase_policy_version",
]

CYCLE_PROVENANCE_COLUMNS = [
    "dataset",
    "domain_id",
    "manufacturer",
    "source_serial",
    "logical_sequence_id",
    "source_series",
    "condition",
    "cycle",
    "source_cycle",
    "source_file",
    "chunk_id",
    "source_file_part",
    "source_file_size_bytes",
    "source_file_mtime_ns",
    "source_absolute_start_time",
    "source_absolute_end_time",
    "source_cycle_duration_hours",
    "source_rows",
    "source_charge_event_count",
    "source_discharge_event_count",
    "source_temperature_column_present",
    "selected_charge_event_index",
    "selected_charge_step_id",
    "selected_charge_points",
    "inferred_cc_points",
    "inferred_cv_points",
    "selected_cc_points",
    "selected_cv_points",
    "cc_current_reference_A",
    "cv_start_source_row_index",
    "cv_start_voltage_V",
    "cv_start_current_A",
    "charge_voltage_max_V",
    "selected_cc_voltage_min_V",
    "selected_cc_voltage_max_V",
    "selected_cv_c_rate_min",
    "selected_cv_c_rate_max",
    "cc_window_lower_covered",
    "cc_window_upper_covered",
    "cc_window_complete",
    "cc_window_accepted",
    "cv_window_high_covered",
    "cv_window_low_covered",
    "cv_window_complete",
    "temperature_complete",
    "boundary_detection_status",
    "boundary_detection_reason",
    "candidate_eligible",
    "candidate_eligibility_reason",
    "candidate_count_for_event",
    "selected_candidate",
    "selection_reason",
    "cycle_discharge_capacity_Ah",
    "calibration_direct_candidate",
    "calibration_reason",
    "reference_calibration_capacity_Ah",
    "bol_q_ref_Ah",
    "bol_q_ref_rule",
    "bol_q_ref_source",
    "label_capacity_Ah",
    "SOH",
    "label_source",
    "split_role",
    "split_status",
    "split_issue",
    "split_strategy_version",
    "output_status",
    "output_reason",
    "raw_rows_written",
    "strategy_version",
]

CELL_PROVENANCE_COLUMNS = [
    "dataset",
    "domain_id",
    "manufacturer",
    "paper_alias",
    "source_serial",
    "logical_sequence_id",
    "source_series",
    "condition",
    "dod_percent",
    "nominal_capacity_Ah",
    "source_chunk_count",
    "source_cycle_candidates",
    "unique_source_events",
    "duplicate_timestamp_candidates",
    "overlapping_source_cycle_candidates",
    "phase_eligible_cycles",
    "direct_calibration_cycles",
    "interpolated_label_cycles",
    "selected_output_cycles",
    "excluded_selected_cycles",
    "reference_calibration_capacity_Ah",
    "reference_calibration_cycle_count",
    "first_calibration_cycle",
    "last_calibration_cycle",
    "bol_q_ref_Ah",
    "bol_q_ref_rule",
    "bol_q_ref_source",
    "bol_reference_candidate_count",
    "bol_reference_valid_candidate_count_after_outlier_filter",
    "bol_reference_window_start_cycle",
    "bol_reference_window_end_cycle",
    "bol_reference_window_observation_count",
    "bol_reference_window_initial_size",
    "bol_reference_window_expanded_after_mad",
    "bol_reference_source_observation_count",
    "bol_reference_source_direct_calibration_count",
    "bol_reference_source_model_ineligible_direct_calibration_count",
    "bol_reference_selected_cycle_ids_json",
    "bol_reference_selected_capacities_Ah_json",
    "bol_reference_rejected_outliers_json",
    "raw_rows_written",
    "split_role",
    "split_status",
    "split_issue",
    "split_rank_sha256",
    "split_file",
    "split_strategy_version",
    "strategy_version",
]

SOURCE_FILE_AUDIT_COLUMNS = [
    "dataset",
    "domain_id",
    "manufacturer",
    "source_serial",
    "logical_sequence_id",
    "condition",
    "source_file",
    "chunk_id",
    "source_file_part",
    "source_file_size_bytes",
    "source_file_mtime_ns",
    "header_columns",
    "temperature_column_present",
    "source_cycle_count",
    "source_row_count",
    "malformed_rows_skipped",
    "malformed_row_reason_counts",
    "malformed_row_examples",
    "nul_characters_removed",
    "boundary_status_counts",
    "candidate_eligible_cycles",
]


PART_RE = re.compile(r"^(?P<series>.+)-(?P<part>[0-9]+)$")
SERIES_RE = re.compile(
    r"^(?P<serial>.+)-(?P<rate>[0-9]+(?:[.][0-9]+)?C)-(?P<dod>[0-9]+%DOD)$",
    flags=re.IGNORECASE,
)
DOD_RE = re.compile(r"(?P<dod>[0-9]+)%DOD$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class SourceIdentity:
    path: Path
    relative_path: str
    config: DomainConfig
    source_series: str
    source_serial: str
    condition: str
    dod_percent: int
    chunk_id: int
    logical_sequence_id: str
    file_size_bytes: int
    file_mtime_ns: int

    @property
    def battery_id(self) -> str:
        """Compatibility alias; logical sequence is the only model cell ID."""

        return self.logical_sequence_id


@dataclass(frozen=True)
class Point:
    source_row_index: int
    cycle: int
    step_id: str
    step_type: str
    time_text: str
    absolute_time: datetime
    current_a: float
    voltage_v: float
    charge_capacity_ah: float
    discharge_capacity_ah: float
    temperature_c: float


@dataclass
class PhaseResult:
    status: str
    reason: str
    charge_event_count: int = 0
    discharge_event_count: int = 0
    charge_event_index: int | None = None
    charge_step_id: str = ""
    charge_points: int = 0
    inferred_cc_points: int = 0
    inferred_cv_points: int = 0
    selected_cc_points: int = 0
    selected_cv_points: int = 0
    cc_current_reference_a: float = math.nan
    cv_start_source_row_index: int | None = None
    cv_start_voltage_v: float = math.nan
    cv_start_current_a: float = math.nan
    charge_voltage_max_v: float = math.nan
    selected_cc_voltage_min_v: float = math.nan
    selected_cc_voltage_max_v: float = math.nan
    selected_cv_c_rate_min: float = math.nan
    selected_cv_c_rate_max: float = math.nan
    cc_window_lower_covered: bool = False
    cc_window_upper_covered: bool = False
    cc_window_complete: bool = False
    # ``cc_window_complete`` records physical coverage of both configured
    # endpoints. ``cc_window_accepted`` records the protocol decision: a
    # partial-DOD cycle may be accepted without the lower endpoint when it has
    # enough selected CC points and reaches the upper endpoint.
    cc_window_accepted: bool = False
    cv_window_high_covered: bool = False
    cv_window_low_covered: bool = False
    cv_window_complete: bool = False
    temperature_complete: bool = False
    cc: list[Point] = field(default_factory=list)
    cv: list[Point] = field(default_factory=list)
    selected_cc: list[Point] = field(default_factory=list)
    selected_cv: list[Point] = field(default_factory=list)


@dataclass
class CycleCandidate:
    identity: SourceIdentity
    source_cycle: int
    source_absolute_start_time: datetime
    source_absolute_end_time: datetime
    source_rows: int
    source_temperature_column_present: bool
    phase: PhaseResult
    cycle_discharge_capacity_ah: float
    candidate_eligible: bool
    candidate_eligibility_reason: str
    # ``cycle`` is assigned only after source events have been ordered by
    # absolute timestamp.  Source-local cycle numbers are not globally unique
    # across continuation chunks.
    canonical_cycle: int | None = None
    candidate_count_for_event: int = 0
    selected_candidate: bool = False
    selection_reason: str = ""
    calibration_direct_candidate: bool = False
    calibration_reason: str = ""
    reference_calibration_capacity_ah: float = math.nan
    bol_q_ref_ah: float = math.nan
    bol_q_ref_rule: str = ""
    bol_q_ref_source: str = ""
    label_capacity_ah: float = math.nan
    soh: float = math.nan
    label_source: str = ""
    split_role: str = ""
    split_status: str = ""
    split_issue: str = ""
    output_status: str = "not_selected"
    output_reason: str = ""
    raw_rows_written: int = 0


# A source-local ``循环号`` is not a physical identity: it is reused after
# continuation chunks.  An event is therefore identified by its sequence plus
# its complete source wall-clock interval.  The interval avoids collapsing two
# distinct events should a source export reuse the same start timestamp.
SourceEventKey = tuple[str, datetime, datetime]


@dataclass
class CellSummary:
    identity: SourceIdentity
    source_chunk_count: int = 0
    source_cycle_candidates: int = 0
    unique_source_events: int = 0
    duplicate_timestamp_candidates: int = 0
    overlapping_source_cycle_candidates: int = 0
    phase_eligible_cycles: int = 0
    direct_calibration_cycles: int = 0
    interpolated_label_cycles: int = 0
    selected_output_cycles: int = 0
    excluded_selected_cycles: int = 0
    reference_calibration_capacity_ah: float = math.nan
    reference_calibration_cycle_count: int = 0
    first_calibration_cycle: int | None = None
    last_calibration_cycle: int | None = None
    bol_reference: dict[str, object] = field(default_factory=dict)
    raw_rows_written: int = 0
    split_role: str = ""
    split_status: str = ""
    split_issue: str = ""
    split_rank_sha256: str = ""
    split_file: str = ""


@dataclass
class SourceScanResult:
    """Pickle-safe result of scanning one independent source CSV."""

    identity: SourceIdentity
    candidates: list[CycleCandidate]
    audit_row: dict[str, object]


@dataclass
class LogicalSequenceExportResult:
    """Pickle-safe result of exporting one logical sequence into one CSV."""

    logical_sequence_id: str
    raw_rows_by_cycle: dict[int, int]
    source_files_with_selected_cycles: int


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lstrip("-")).strip("_")
    return value.lower() or "unknown"


def to_float(value: str) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else math.nan
    except (TypeError, ValueError):
        return math.nan


def source_cycle(value: str, path: Path, source_row_index: int) -> int:
    number = to_float(value)
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise ValueError(
            f"Invalid source cycle {value!r} in {path} at source row {source_row_index}"
        )
    return int(number)


def source_absolute_time(value: str, path: Path, source_row_index: int) -> datetime:
    """Parse a source ``绝对时间`` cell into a timezone-naive local datetime.

    SmartHealth files use local wall-clock timestamps such as
    ``2022-10-26 21:04:23``. A few exports use slash separators, ``T``, or
    non-zero-padded month/day/hour/minute fields (for example
    ``2022/8/4 8:27``). A missing/unparseable timestamp is unsafe because it
    would reintroduce the old local-cycle ordering bug, so fail the source scan
    explicitly.
    """

    text = str(value).strip().replace("/", "-").replace("T", " ")
    if not text:
        raise ValueError(
            f"Missing source absolute time in {path} at source row {source_row_index}"
        )
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
        # ``strptime`` accepts unpadded numeric fields for these directives,
        # whereas ``fromisoformat`` deliberately requires ISO zero-padding.
        for layout in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ):
            try:
                parsed = datetime.strptime(text, layout)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError(
                f"Invalid source absolute time {value!r} in {path} at source row {source_row_index}"
            )
    if parsed.tzinfo is not None:
        # All current source files are local-time strings.  Keeping a timezone
        # here would make naive/aware datetime ordering fail unpredictably.
        parsed = parsed.replace(tzinfo=None)
    return parsed


def format_source_absolute_time(value: datetime) -> str:
    """Stable CSV/provenance serialization for a parsed source timestamp."""

    timespec = "microseconds" if value.microsecond else "seconds"
    return value.isoformat(sep=" ", timespec=timespec)


def duration_seconds(value: str) -> float:
    text = str(value).strip()
    if not text:
        return math.nan
    days = 0.0
    if "day" in text:
        try:
            day_text, text = text.split(",", 1)
            days = float(day_text.split()[0])
            text = text.strip()
        except (IndexError, ValueError):
            return math.nan
    parts = text.split(":")
    if len(parts) != 3:
        return math.nan
    try:
        hour, minute, second = (float(part) for part in parts)
    except ValueError:
        return math.nan
    return days * 86400.0 + hour * 3600.0 + minute * 60.0 + second


def finite_delta(values: Iterable[float]) -> float:
    values_array = np.asarray(list(values), dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size < 2:
        return math.nan
    return float(np.max(values_array) - np.min(values_array))


def _source_directory(input_root: Path, config: DomainConfig) -> Path:
    nested = input_root / config.source_top_level
    if nested.is_dir():
        return nested
    if input_root.name.upper() == config.source_top_level.upper() and input_root.is_dir():
        return input_root
    raise FileNotFoundError(
        f"{config.domain_id}: expected {config.source_top_level!r} below {input_root}"
    )


def make_identity(path: Path, input_root: Path, config: DomainConfig) -> SourceIdentity:
    part_match = PART_RE.match(path.stem)
    if part_match is None:
        raise ValueError(f"Source filename lacks numeric chunk suffix: {path.name}")
    source_series = part_match.group("series")
    series_match = SERIES_RE.match(source_series)
    if series_match is None:
        raise ValueError(f"Cannot parse serial/C-rate/DOD from source file {path.name}")
    source_serial = series_match.group("serial")
    condition = f"{series_match.group('rate')}-{series_match.group('dod')}"
    dod_match = DOD_RE.search(condition)
    if dod_match is None:
        raise ValueError(f"Cannot parse DOD from condition {condition!r}")
    stat = path.stat()
    return SourceIdentity(
        path=path,
        relative_path=path.relative_to(input_root).as_posix(),
        config=config,
        source_series=source_series,
        source_serial=source_serial,
        condition=condition,
        dod_percent=int(dod_match.group("dod")),
        chunk_id=int(part_match.group("part")),
        logical_sequence_id=(
            f"{config.domain_id}__{slug(source_serial)}__{slug(condition)}"
        ),
        file_size_bytes=int(stat.st_size),
        file_mtime_ns=int(stat.st_mtime_ns),
    )


def list_identities(
    input_root: Path, config: DomainConfig, max_source_files: int | None
) -> list[SourceIdentity]:
    source_directory = _source_directory(input_root, config)
    identities = [
        make_identity(path, input_root, config)
        for path in source_directory.rglob("*.csv")
        if path.is_file()
    ]
    identities.sort(
        key=lambda item: (
            item.condition,
            item.logical_sequence_id,
            item.chunk_id,
            item.relative_path,
        )
    )
    if max_source_files is not None:
        if max_source_files <= 0:
            raise ValueError("--max-source-files must be positive")
        identities = identities[:max_source_files]
    if not identities:
        raise ValueError(f"No SmartHealth CSVs found for {config.domain_id} under {source_directory}")
    return identities


def events(points: Sequence[Point], target_step_type: str) -> list[list[Point]]:
    """Return contiguous same-step events of a source work-step type."""

    output: list[list[Point]] = []
    current: list[Point] = []
    previous: Point | None = None
    for point in points:
        if point.step_type != target_step_type:
            if current:
                output.append(current)
                current = []
            previous = None
            continue
        if previous is not None and (
            point.source_row_index != previous.source_row_index + 1
            or point.step_id != previous.step_id
        ):
            output.append(current)
            current = []
        current.append(point)
        previous = point
    if current:
        output.append(current)
    return output


def pick_event(
    candidates: Sequence[Sequence[Point]],
    capacity_field: str,
    *,
    prefer_capacity_span: bool,
) -> tuple[int, list[Point]] | None:
    if not candidates:
        return None

    def score(pair: tuple[int, Sequence[Point]]) -> tuple[float, float, int]:
        index, current = pair
        capacity_span = finite_delta(getattr(point, capacity_field) for point in current)
        finite_span = -1.0 if not math.isfinite(capacity_span) else capacity_span
        if prefer_capacity_span:
            return finite_span, float(len(current)), -index
        return float(len(current)), finite_span, -index

    index, selected = max(enumerate(candidates), key=score)
    return index, list(selected)


def min_cv_points_for_condition(
    domain_id: str, dod_percent: int, default_min_cv_points: int
) -> int:
    """Resolve the CV detection threshold without changing the data contract."""

    return CV_MIN_POINTS_BY_DOMAIN_DOD.get(domain_id, {}).get(
        int(dod_percent), default_min_cv_points
    )


def split_combined_charge(
    points: Sequence[Point],
    args: argparse.Namespace,
    *,
    min_cv_points: int | None = None,
) -> PhaseResult:
    """Infer CC/CV inside the source's combined ``恒流恒压充电`` step."""

    # ``args.min_cv_points`` remains the historical fallback.  Callers that
    # know the source domain/DoD pass the resolved condition-specific value.
    min_cv_points = args.min_cv_points if min_cv_points is None else min_cv_points

    charge_events = events(points, CHARGE_STEP)
    discharge_events = events(points, DISCHARGE_STEP)
    if not charge_events:
        return PhaseResult(
            status="invalid",
            reason="no_combined_charge_event",
            discharge_event_count=len(discharge_events),
        )
    selected = pick_event(
        charge_events, "charge_capacity_ah", prefer_capacity_span=False
    )
    assert selected is not None
    event_index, event = selected
    result = PhaseResult(
        status="invalid",
        reason="",
        charge_event_count=len(charge_events),
        discharge_event_count=len(discharge_events),
        charge_event_index=event_index,
        charge_step_id=event[0].step_id,
        charge_points=len(event),
    )
    if len(event) < args.min_cc_points + min_cv_points:
        result.reason = "combined_charge_too_short"
        return result

    current = np.asarray([abs(point.current_a) for point in event], dtype=float)
    voltage = np.asarray([point.voltage_v for point in event], dtype=float)
    times = np.asarray([duration_seconds(point.time_text) for point in event], dtype=float)
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(voltage)):
        result.reason = "nonfinite_charge_current_or_voltage"
        return result
    if (
        not np.all(np.isfinite(times))
        or times.size < 2
        or np.any(np.diff(times) < 0)
        or float(times[-1] - times[0]) <= 0
    ):
        result.reason = "invalid_charge_time"
        return result
    if not np.all(current > 0):
        result.reason = "nonpositive_charge_current"
        return result

    early_count = min(
        len(event),
        max(
            int(args.cc_reference_min_points),
            int(math.ceil(args.cc_reference_fraction * len(event))),
        ),
    )
    cc_reference = float(np.quantile(current[:early_count], args.cc_reference_quantile))
    if not math.isfinite(cc_reference) or cc_reference <= 0:
        result.reason = "invalid_cc_current_reference"
        return result

    taper = current <= cc_reference * (1.0 - args.cv_taper_fraction)
    voltage_max = float(np.max(voltage))
    boundary: int | None = None
    for index in range(args.min_cc_points, len(event) - min_cv_points + 1):
        if index + args.cv_persistence_points > len(event):
            break
        if not np.all(taper[index : index + args.cv_persistence_points]):
            continue
        if voltage[index] < voltage_max - args.cv_voltage_tolerance_v:
            continue
        boundary = index
        break

    result.cc_current_reference_a = cc_reference
    result.charge_voltage_max_v = voltage_max
    if boundary is None:
        result.reason = "no_persistent_taper_near_charge_voltage_max"
        return result

    result.cc = event[:boundary]
    result.cv = event[boundary:]
    result.inferred_cc_points = len(result.cc)
    result.inferred_cv_points = len(result.cv)
    result.cv_start_source_row_index = result.cv[0].source_row_index
    result.cv_start_voltage_v = float(voltage[boundary])
    result.cv_start_current_a = float(current[boundary])
    if result.inferred_cc_points < args.min_cc_points or result.inferred_cv_points < min_cv_points:
        result.reason = "phase_point_count_below_minimum"
        return result
    result.status = "ok"
    result.reason = "persistent_current_taper_near_charge_voltage_max"
    return result


def select_model_windows(
    phase: PhaseResult, config: DomainConfig, args: argparse.Namespace
) -> None:
    """Select only already-inferred CC and CV points for the model contract."""

    if phase.status != "ok":
        return
    phase.selected_cc = [
        point
        for point in phase.cc
        if config.cc_voltage_low_v <= point.voltage_v <= config.cc_voltage_high_v
    ]
    cv_lower = config.cv_c_rate_low - config.cv_selection_tolerance_c
    cv_upper = config.cv_c_rate_high + config.cv_selection_tolerance_c
    phase.selected_cv = [
        point
        for point in phase.cv
        if math.isfinite(point.current_a)
        and cv_lower <= abs(point.current_a) / config.nominal_capacity_ah <= cv_upper
    ]
    phase.selected_cc_points = len(phase.selected_cc)
    phase.selected_cv_points = len(phase.selected_cv)
    if phase.selected_cc:
        cc_voltage = np.asarray([point.voltage_v for point in phase.selected_cc], dtype=float)
        phase.selected_cc_voltage_min_v = float(np.min(cc_voltage))
        phase.selected_cc_voltage_max_v = float(np.max(cc_voltage))
        phase.cc_window_lower_covered = bool(
            phase.selected_cc_voltage_min_v
            <= config.cc_voltage_low_v + config.cc_coverage_tolerance_v
        )
        phase.cc_window_upper_covered = bool(
            phase.selected_cc_voltage_max_v
            >= config.cc_voltage_high_v - config.cc_coverage_tolerance_v
        )
    if phase.selected_cv:
        cv_rates = np.asarray(
            [abs(point.current_a) / config.nominal_capacity_ah for point in phase.selected_cv],
            dtype=float,
        )
        phase.selected_cv_c_rate_min = float(np.min(cv_rates))
        phase.selected_cv_c_rate_max = float(np.max(cv_rates))
    all_cv_rates = np.asarray(
        [abs(point.current_a) / config.nominal_capacity_ah for point in phase.cv], dtype=float
    )
    all_cv_rates = all_cv_rates[np.isfinite(all_cv_rates)]
    if all_cv_rates.size:
        phase.cv_window_high_covered = bool(
            float(np.max(all_cv_rates)) >= config.cv_high_coverage_c_rate
        )
        phase.cv_window_low_covered = bool(
            float(np.min(all_cv_rates)) <= config.cv_low_coverage_c_rate
        )
    phase.cc_window_complete = bool(
        phase.selected_cc_points >= args.min_selected_cc_points
        and phase.cc_window_lower_covered
        and phase.cc_window_upper_covered
    )
    phase.cv_window_complete = bool(
        phase.selected_cv_points >= args.min_selected_cv_points
        and phase.cv_window_high_covered
        and phase.cv_window_low_covered
    )
    selected_points = [*phase.selected_cc, *phase.selected_cv]
    phase.temperature_complete = bool(
        selected_points and all(math.isfinite(point.temperature_c) for point in selected_points)
    )


def model_cc_window_accepted(
    phase: PhaseResult, dod_percent: int, args: argparse.Namespace
) -> bool:
    """Return whether the selected CC trace is usable as a model input.

    A full-DOD charge is expected to cover the common 3.45--3.58 V interval.
    In a partial-DOD protocol, however, the source charge legitimately starts
    at a higher state of charge, so the lower endpoint is not observable. We
    still require the configured minimum number of points and upper-endpoint
    coverage; this does not admit empty or short CC fragments.
    """

    if phase.status != "ok":
        return False
    if phase.cc_window_complete:
        return True
    return bool(
        dod_percent < 100
        and phase.selected_cc_points >= args.min_selected_cc_points
        and phase.cc_window_upper_covered
    )


def principal_discharge_capacity(points: Sequence[Point]) -> float:
    """Use the largest discharge-capacity span, not charge throughput, as Q."""

    selected = pick_event(
        events(points, DISCHARGE_STEP),
        "discharge_capacity_ah",
        prefer_capacity_span=True,
    )
    if selected is None:
        return math.nan
    return finite_delta(point.discharge_capacity_ah for point in selected[1])


def source_cycle_duration_hours(start: datetime, end: datetime) -> float:
    """Return the source wall-clock span of one candidate cycle in hours."""

    return (end - start).total_seconds() / 3600.0


def candidate_from_points(
    identity: SourceIdentity,
    source_cycle_id: int,
    points: Sequence[Point],
    source_temperature_column_present: bool,
    args: argparse.Namespace,
) -> CycleCandidate:
    phase = split_combined_charge(
        points,
        args,
        min_cv_points=min_cv_points_for_condition(
            identity.config.domain_id, identity.dod_percent, args.min_cv_points
        ),
    )
    select_model_windows(phase, identity.config, args)
    phase.cc_window_accepted = model_cc_window_accepted(
        phase, identity.dod_percent, args
    )
    discharge_capacity = principal_discharge_capacity(points)
    source_times = [point.absolute_time for point in points]
    if not source_times:
        raise ValueError(
            f"No source timestamps for {identity.relative_path}, source cycle {source_cycle_id}"
        )
    source_start = min(source_times)
    source_end = max(source_times)
    reasons: list[str] = []
    if (
        source_cycle_duration_hours(source_start, source_end)
        > args.max_source_cycle_duration_hours
    ):
        # The empirical source distribution has a p99 below 13 h. A cycle
        # spanning a day is a chunking/export artefact, not one usable raw
        # CC/CV observation. Keep it in provenance, but never export it.
        reasons.append("source_cycle_duration_exceeds_limit")
    if phase.status != "ok":
        reasons.append(phase.reason)
    else:
        if not phase.cc_window_accepted:
            reasons.append("incomplete_selected_cc_voltage_window")
        if not phase.cv_window_complete:
            reasons.append("incomplete_selected_cv_c_rate_window")
        if not phase.temperature_complete:
            reasons.append("missing_or_nonfinite_temperature_in_selected_cccv")
    if not math.isfinite(discharge_capacity) or discharge_capacity <= 0:
        reasons.append("invalid_principal_discharge_capacity")
    candidate = CycleCandidate(
        identity=identity,
        source_cycle=source_cycle_id,
        source_absolute_start_time=source_start,
        source_absolute_end_time=source_end,
        source_rows=len(points),
        source_temperature_column_present=source_temperature_column_present,
        phase=phase,
        cycle_discharge_capacity_ah=discharge_capacity,
        candidate_eligible=not reasons,
        candidate_eligibility_reason=(
            ";".join(reasons)
            if reasons
            else (
                "phase_window_temperature_and_discharge_valid_partial_dod_cc_lower_coverage_not_required"
                if not phase.cc_window_complete
                else "phase_window_temperature_and_discharge_valid"
            )
        ),
    )
    # The first streaming pass retains only audit summaries.  Pass two re-reads
    # the selected source candidate and verifies these decisions before export.
    phase.cc = []
    phase.cv = []
    phase.selected_cc = []
    phase.selected_cv = []
    return candidate


class NulSanitizingLineIterator:
    """Make a malformed CSV parseable while preserving an audit count.

    A small number of SmartHealth exports contain literal NUL characters in
    otherwise GB18030 CSV text.  Python's :mod:`csv` reader rejects such a
    line before field-level validation can run.  NUL has no semantic meaning
    in this tabular format, so remove it at the text-line boundary and record
    the exact number in the source-file audit rather than silently dropping a
    row or a whole logical sequence.
    """

    def __init__(self, handle) -> None:
        self.handle = handle
        self.nul_characters_removed = 0

    def __iter__(self) -> "NulSanitizingLineIterator":
        return self

    def __next__(self) -> str:
        line = next(self.handle)
        count = line.count("\x00")
        if count:
            self.nul_characters_removed += count
            line = line.replace("\x00", "")
        return line


def visit_cycles(
    identity: SourceIdentity, callback: Callable[[int, list[Point], bool], None]
) -> dict[str, object]:
    """Stream one GB18030 source CSV, yielding contiguous source cycles.

    Literal NUL characters are removed before CSV tokenization and their count
    is returned to the source-file audit. A non-empty but truncated source row
    with no complete point-level record is skipped, while its reason and row
    index remain in that audit. It is never repaired or assigned a fabricated
    timestamp.
    """

    with identity.path.open("r", encoding=SOURCE_ENCODING, newline="") as handle:
        sanitized_lines = NulSanitizingLineIterator(handle)
        reader = csv.reader(sanitized_lines)
        try:
            header = [item.lstrip("\ufeff").strip() for item in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"Empty source CSV: {identity.path}") from exc
        index = {name: position for position, name in enumerate(header) if name}
        missing = sorted(SOURCE_REQUIRED_COLUMNS - set(index))
        if missing:
            raise ValueError(f"{identity.path} missing required columns: {missing}")
        has_temperature = "temp1_1" in index

        def value(row: Sequence[str], name: str) -> str:
            position = index.get(name)
            return "" if position is None or position >= len(row) else row[position].strip()

        current_cycle: int | None = None
        points: list[Point] = []
        seen: set[int] = set()
        source_row_count = 0
        source_cycle_count = 0
        malformed_rows_skipped = 0
        malformed_row_reason_counts: Counter[str] = Counter()
        malformed_row_examples: list[str] = []
        for source_row_index, row in enumerate(reader):
            if not row or not any(field.strip() for field in row):
                continue
            source_row_count += 1
            point_values = {
                name: value(row, name) for name in SOURCE_POINT_REQUIRED_COLUMNS
            }
            missing_values = [
                name for name, source_value in point_values.items() if not source_value
            ]
            if missing_values:
                malformed_rows_skipped += 1
                reason = "missing_required_point_values:" + "|".join(missing_values)
                malformed_row_reason_counts[reason] += 1
                if len(malformed_row_examples) < 10:
                    malformed_row_examples.append(f"{source_row_index}:{reason}")
                continue
            try:
                cycle = source_cycle(
                    point_values["循环号"], identity.path, source_row_index
                )
                absolute_time = source_absolute_time(
                    point_values["绝对时间"], identity.path, source_row_index
                )
            except ValueError:
                malformed_rows_skipped += 1
                reason = "invalid_cycle_or_absolute_time"
                malformed_row_reason_counts[reason] += 1
                if len(malformed_row_examples) < 10:
                    malformed_row_examples.append(f"{source_row_index}:{reason}")
                continue
            if current_cycle is None:
                current_cycle = cycle
            elif cycle != current_cycle:
                if current_cycle in seen:
                    raise ValueError(
                        f"Non-contiguous repeated source cycle {current_cycle} in {identity.path}"
                    )
                callback(current_cycle, points, has_temperature)
                seen.add(current_cycle)
                source_cycle_count += 1
                points = []
                current_cycle = cycle
            points.append(
                Point(
                    source_row_index=source_row_index,
                    cycle=cycle,
                    step_id=point_values["工步号"],
                    step_type=point_values["工步类型"],
                    time_text=point_values["时间"],
                    absolute_time=absolute_time,
                    current_a=to_float(point_values["电流(A)"]),
                    voltage_v=to_float(point_values["电压(V)"]),
                    charge_capacity_ah=to_float(point_values["充电容量(Ah)"]),
                    discharge_capacity_ah=to_float(point_values["放电容量(Ah)"]),
                    temperature_c=(
                        to_float(value(row, "temp1_1")) if has_temperature else math.nan
                    ),
                )
            )
        if current_cycle is not None:
            if current_cycle in seen:
                raise ValueError(
                    f"Non-contiguous repeated source cycle {current_cycle} in {identity.path}"
                )
            callback(current_cycle, points, has_temperature)
            source_cycle_count += 1
    return {
        "header": header,
        "temperature_column_present": has_temperature,
        "source_cycle_count": source_cycle_count,
        "source_row_count": source_row_count,
        "malformed_rows_skipped": malformed_rows_skipped,
        "malformed_row_reason_counts": json.dumps(
            dict(sorted(malformed_row_reason_counts.items())), ensure_ascii=False
        ),
        "malformed_row_examples": "|".join(malformed_row_examples),
        "nul_characters_removed": sanitized_lines.nul_characters_removed,
    }


def scan_one_source(
    identity: SourceIdentity, args: argparse.Namespace
) -> SourceScanResult:
    """Scan one source CSV without touching shared state or output files."""

    source_candidates: list[CycleCandidate] = []
    boundary_counts: Counter[str] = Counter()
    usable_cycles = 0

    def callback(cycle: int, points: list[Point], has_temperature: bool) -> None:
        nonlocal usable_cycles
        candidate = candidate_from_points(identity, cycle, points, has_temperature, args)
        source_candidates.append(candidate)
        boundary_counts[candidate.phase.reason or candidate.phase.status] += 1
        usable_cycles += int(candidate.candidate_eligible)

    info = visit_cycles(identity, callback)
    return SourceScanResult(
        identity=identity,
        candidates=source_candidates,
        audit_row={
            "dataset": "smarthealth",
            "domain_id": identity.config.domain_id,
            "manufacturer": identity.config.manufacturer,
            "source_serial": identity.source_serial,
            "logical_sequence_id": identity.logical_sequence_id,
            "condition": identity.condition,
            "source_file": identity.relative_path,
            "chunk_id": identity.chunk_id,
            "source_file_part": identity.chunk_id,
            "source_file_size_bytes": identity.file_size_bytes,
            "source_file_mtime_ns": identity.file_mtime_ns,
            "header_columns": "|".join(info["header"]),
            "temperature_column_present": info["temperature_column_present"],
            "source_cycle_count": info["source_cycle_count"],
            "source_row_count": info["source_row_count"],
            "malformed_rows_skipped": info["malformed_rows_skipped"],
            "malformed_row_reason_counts": info["malformed_row_reason_counts"],
            "malformed_row_examples": info["malformed_row_examples"],
            "nul_characters_removed": info["nul_characters_removed"],
            "boundary_status_counts": json.dumps(
                dict(sorted(boundary_counts.items())), ensure_ascii=False
            ),
            "candidate_eligible_cycles": usable_cycles,
        },
    )


def scan_sources(
    identities: Sequence[SourceIdentity], args: argparse.Namespace
) -> tuple[
    dict[SourceEventKey, list[CycleCandidate]],
    dict[str, CellSummary],
    list[dict[str, object]],
]:
    """Process-independent scan, merged in inventory order for reproducibility."""

    # A source ``循环号`` is local to a numbered CSV chunk and can reset or
    # overlap after a continuation.  The complete source time interval is the
    # event identity; exact interval duplicates are reconciled later.
    candidates: dict[SourceEventKey, list[CycleCandidate]] = defaultdict(list)
    cells: dict[str, CellSummary] = {}
    file_audit: list[dict[str, object]] = []

    def merge(source_index: int, result: SourceScanResult) -> None:
        identity = result.identity
        cell = cells.setdefault(
            identity.logical_sequence_id, CellSummary(identity=identity)
        )
        cell.source_chunk_count += 1
        cell.source_cycle_candidates += len(result.candidates)
        for candidate in result.candidates:
            candidates[
                (
                    identity.logical_sequence_id,
                    candidate.source_absolute_start_time,
                    candidate.source_absolute_end_time,
                )
            ].append(candidate)
        file_audit.append(result.audit_row)
        nul_count = int(result.audit_row["nul_characters_removed"])
        malformed_count = int(result.audit_row["malformed_rows_skipped"])
        print(
            f"[scan {source_index}/{len(identities)}] {identity.relative_path}: "
            f"{result.audit_row['source_cycle_count']} cycles, "
            f"{result.audit_row['candidate_eligible_cycles']} phase/window eligible"
            + (f"; sanitized {nul_count} NUL characters" if nul_count else "")
            + (f"; skipped {malformed_count} malformed row(s)" if malformed_count else ""),
            flush=True,
        )

    if args.workers == 1 or len(identities) == 1:
        for source_index, identity in enumerate(identities, start=1):
            merge(source_index, scan_one_source(identity, args))
        return candidates, cells, file_audit

    pending: dict[int, SourceScanResult] = {}
    next_index = 1
    worker_count = min(args.workers, len(identities))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(scan_one_source, identity, args): source_index
            for source_index, identity in enumerate(identities, start=1)
        }
        for future in as_completed(futures):
            source_index = futures[future]
            identity = identities[source_index - 1]
            try:
                pending[source_index] = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Scan failed for {identity.relative_path}"
                ) from exc
            while next_index in pending:
                merge(next_index, pending.pop(next_index))
                next_index += 1
    return candidates, cells, file_audit


def candidate_quality_key(candidate: CycleCandidate) -> tuple[object, ...]:
    """Strict duplicate ranking in the protocol's documented priority order."""

    phase = candidate.phase
    return (
        -int(phase.status == "ok"),
        -int(phase.temperature_complete),
        -int(phase.cc_window_complete and phase.cv_window_complete),
        -int(candidate.candidate_eligible),
        -(phase.selected_cc_points + phase.selected_cv_points),
        -phase.charge_points,
        -candidate.source_rows,
        candidate.identity.chunk_id,
        candidate.identity.relative_path,
    )


def chronological_candidate_key(candidate: CycleCandidate) -> tuple[object, ...]:
    """Deterministic physical ordering for one logical SmartHealth sequence."""

    return (
        candidate.source_absolute_start_time,
        candidate.source_absolute_end_time,
        candidate.identity.relative_path,
        candidate.source_cycle,
    )


def resolve_duplicate_candidates(
    candidates: dict[SourceEventKey, list[CycleCandidate]],
    cells: Mapping[str, CellSummary],
) -> dict[SourceEventKey, CycleCandidate]:
    """Resolve exact and overlapping chunk duplicates without merging resets.

    Source-local cycle numbers legitimately reset in later numbered chunks, so
    equal numbers at disjoint timestamps remain distinct. Conversely, a chunk
    boundary can copy one physical source cycle into both files with a slightly
    different end timestamp. Such candidates have the same logical sequence,
    the same local source cycle, and overlapping wall-clock intervals; they
    are one event and are quality-ranked together.

    ``candidates`` is rewritten in place to use one key per resolved physical
    event. Its values retain *all* candidates so cycle provenance continues to
    document discarded duplicates.
    """

    # Preserve historical exact-interval reconciliation as the first level.
    components_by_source_cycle: dict[
        tuple[str, int], list[tuple[SourceEventKey, list[CycleCandidate], CycleCandidate]]
    ] = defaultdict(list)
    exact_duplicate_counts: Counter[str] = Counter()
    for key, same_interval in sorted(candidates.items()):
        ordered = sorted(same_interval, key=candidate_quality_key)
        winner = ordered[0]
        if len(ordered) > 1:
            exact_duplicate_counts[winner.identity.logical_sequence_id] += 1
        components_by_source_cycle[
            (winner.identity.logical_sequence_id, winner.source_cycle)
        ].append((key, ordered, winner))

    selected: dict[SourceEventKey, CycleCandidate] = {}
    resolved_groups: dict[SourceEventKey, list[CycleCandidate]] = {}
    by_cell: dict[str, list[CycleCandidate]] = defaultdict(list)
    overlap_duplicate_counts: Counter[str] = Counter()

    for (logical_sequence_id, _), components in sorted(components_by_source_cycle.items()):
        components.sort(key=lambda item: chronological_candidate_key(item[2]))
        clusters: list[list[tuple[SourceEventKey, list[CycleCandidate], CycleCandidate]]] = []
        cluster: list[tuple[SourceEventKey, list[CycleCandidate], CycleCandidate]] = []
        latest_end: datetime | None = None
        for component in components:
            candidate = component[2]
            # Do not merge adjacent physical cycles whose timestamps only
            # touch at a chunk boundary: overlap must have positive duration.
            if cluster and latest_end is not None and candidate.source_absolute_start_time >= latest_end:
                clusters.append(cluster)
                cluster = []
                latest_end = None
            cluster.append(component)
            if latest_end is None or candidate.source_absolute_end_time > latest_end:
                latest_end = candidate.source_absolute_end_time
        if cluster:
            clusters.append(cluster)

        for cluster in clusters:
            same_event = [
                candidate
                for _, component_candidates, _ in cluster
                for candidate in component_candidates
            ]
            ordered = sorted(same_event, key=candidate_quality_key)
            winner = ordered[0]
            event_key = (
                logical_sequence_id,
                winner.source_absolute_start_time,
                winner.source_absolute_end_time,
            )
            if event_key in resolved_groups:
                raise RuntimeError(f"Duplicate resolved SmartHealth event identity: {event_key}")
            resolved_groups[event_key] = ordered
            selected[event_key] = winner
            by_cell[logical_sequence_id].append(winner)

            has_overlapping_chunks = len(cluster) > 1
            if has_overlapping_chunks:
                overlap_duplicate_counts[logical_sequence_id] += len(cluster) - 1
            selection_reason = (
                "unique_source_time_interval_event"
                if len(ordered) == 1
                else (
                    "overlapping_source_cycle_quality_rank:boundary,temperature,selected_cccv_points,label_eligibility,charge_points,source_rows,earlier_chunk"
                    if has_overlapping_chunks
                    else "duplicate_source_time_interval_quality_rank:boundary,temperature,selected_cccv_points,label_eligibility,charge_points,source_rows,earlier_chunk"
                )
            )
            for index, candidate in enumerate(ordered):
                candidate.candidate_count_for_event = len(ordered)
                candidate.selected_candidate = index == 0
                if index == 0:
                    candidate.selection_reason = selection_reason
                    continue
                candidate.selection_reason = (
                    "overlapping_source_cycle_not_selected"
                    if has_overlapping_chunks
                    else "duplicate_source_time_interval_not_selected"
                )
                candidate.output_status = "not_selected"
                candidate.output_reason = "lower_duplicate_candidate_quality"

    candidates.clear()
    candidates.update(resolved_groups)
    for logical_sequence_id, cell in cells.items():
        series = sorted(by_cell.get(logical_sequence_id, []), key=chronological_candidate_key)
        cell.unique_source_events = len(series)
        cell.duplicate_timestamp_candidates = exact_duplicate_counts[logical_sequence_id]
        cell.overlapping_source_cycle_candidates = overlap_duplicate_counts[logical_sequence_id]
        cell.phase_eligible_cycles = sum(candidate.candidate_eligible for candidate in series)
    return selected


def assign_chronological_cycle_ids(
    selected_events: Mapping[SourceEventKey, CycleCandidate],
    candidates: Mapping[SourceEventKey, Sequence[CycleCandidate]],
) -> dict[tuple[str, int], CycleCandidate]:
    """Assign one-based canonical cycle IDs after source-time ordering.

    ``source_cycle`` remains the untouched local ID from a GB18030 CSV.  The
    exported ``cycle`` is instead a unique chronological index within the
    logical sequence.  Every exact-time-interval duplicate (including a
    rejected one) receives the winner's canonical ID in provenance.
    """

    by_cell: dict[str, list[tuple[SourceEventKey, CycleCandidate]]] = defaultdict(list)
    for event_key, candidate in selected_events.items():
        by_cell[candidate.identity.logical_sequence_id].append((event_key, candidate))

    selected: dict[tuple[str, int], CycleCandidate] = {}
    for logical_sequence_id, events_for_cell in by_cell.items():
        ordered = sorted(events_for_cell, key=lambda item: chronological_candidate_key(item[1]))
        for canonical_cycle, (event_key, winner) in enumerate(ordered, start=1):
            for candidate in candidates[event_key]:
                candidate.canonical_cycle = canonical_cycle
            key = (logical_sequence_id, canonical_cycle)
            if key in selected:
                raise RuntimeError(f"Duplicate canonical SmartHealth cycle identity: {key}")
            selected[key] = winner
    return selected


def canonical_cycle_id(candidate: CycleCandidate) -> int:
    if candidate.canonical_cycle is None:
        raise RuntimeError(
            "SmartHealth candidate has no canonical cycle ID; assign chronological IDs first"
        )
    return int(candidate.canonical_cycle)


def _calibration_reason(
    candidate: CycleCandidate,
    max_source_cycle_duration_hours: float = DEFAULT_MAX_SOURCE_CYCLE_DURATION_HOURS,
) -> str:
    """Recognize source-supported full-capacity discharge calibrations.

    Calibration eligibility is intentionally independent of model CC/CV
    eligibility. A source cycle may contain a valid full-capacity discharge
    after a partial-DOD/high-rate charge whose CC trace does not cover 3.45 V.
    """

    if (
        source_cycle_duration_hours(
            candidate.source_absolute_start_time,
            candidate.source_absolute_end_time,
        )
        > max_source_cycle_duration_hours
    ):
        return ""
    if (
        not math.isfinite(candidate.cycle_discharge_capacity_ah)
        or candidate.cycle_discharge_capacity_ah <= 0
    ):
        return ""
    identity = candidate.identity
    if identity.dod_percent == 100:
        return "full_dod_principal_discharge"
    threshold = identity.config.nominal_capacity_ah * identity.config.calibration_min_nominal_fraction
    if candidate.cycle_discharge_capacity_ah >= threshold:
        return "periodic_full_capacity_discharge_threshold"
    return ""


def label_calibration_soh(
    selected: Mapping[tuple[str, int], CycleCandidate],
    cells: Mapping[str, CellSummary],
    max_source_cycle_duration_hours: float = DEFAULT_MAX_SOURCE_CYCLE_DURATION_HOURS,
) -> None:
    """Attach calibration capacities and fixed-nominal SOH without extrapolation.

    ``reference_calibration_capacity_ah`` remains auditable provenance for the
    early reliable calibration sequence.  It is not the Paper target
    denominator: all SmartHealth families use their fixed nominal capacity,
    matching the XJTU and MIT target convention.
    """

    by_cell: dict[str, list[CycleCandidate]] = defaultdict(list)
    for candidate in selected.values():
        by_cell[candidate.identity.logical_sequence_id].append(candidate)

    for logical_sequence_id, cell in cells.items():
        series = sorted(by_cell.get(logical_sequence_id, []), key=chronological_candidate_key)
        direct: list[CycleCandidate] = []
        for candidate in series:
            reason = _calibration_reason(candidate, max_source_cycle_duration_hours)
            candidate.calibration_reason = reason
            candidate.calibration_direct_candidate = bool(reason)
            if reason:
                direct.append(candidate)
        cell.direct_calibration_cycles = len(direct)
        if len(direct) < 3:
            for candidate in series:
                if not candidate.candidate_eligible:
                    candidate.output_status = "excluded"
                    candidate.output_reason = candidate.candidate_eligibility_reason
                else:
                    candidate.output_status = "excluded"
                    candidate.output_reason = "fewer_than_three_reliable_calibration_cycles"
            cell.excluded_selected_cycles = len(series)
            continue

        reference = float(median(item.cycle_discharge_capacity_ah for item in direct[:3]))
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError(f"Invalid calibration reference for {logical_sequence_id}")
        cell.reference_calibration_capacity_ah = reference
        cell.reference_calibration_cycle_count = 3
        cell.first_calibration_cycle = canonical_cycle_id(direct[0])
        cell.last_calibration_cycle = canonical_cycle_id(direct[-1])
        direct_by_cycle = {canonical_cycle_id(candidate): candidate for candidate in direct}

        for candidate in series:
            candidate.reference_calibration_capacity_ah = reference
            if not candidate.candidate_eligible:
                candidate.output_status = "excluded"
                candidate.output_reason = candidate.candidate_eligibility_reason
                cell.excluded_selected_cycles += 1
                continue
            candidate_cycle = canonical_cycle_id(candidate)
            if candidate_cycle in direct_by_cycle:
                label_capacity = candidate.cycle_discharge_capacity_ah
                label_source = "calibration_direct"
            else:
                left = [item for item in direct if canonical_cycle_id(item) < candidate_cycle]
                right = [item for item in direct if canonical_cycle_id(item) > candidate_cycle]
                if not left:
                    candidate.output_status = "excluded"
                    candidate.output_reason = "no_calibration_before_cycle_no_extrapolation"
                    cell.excluded_selected_cycles += 1
                    continue
                if not right:
                    candidate.output_status = "excluded"
                    candidate.output_reason = "no_calibration_after_cycle_no_extrapolation"
                    cell.excluded_selected_cycles += 1
                    continue
                lower, upper = left[-1], right[0]
                lower_cycle = canonical_cycle_id(lower)
                upper_cycle = canonical_cycle_id(upper)
                fraction = (candidate_cycle - lower_cycle) / (upper_cycle - lower_cycle)
                label_capacity = lower.cycle_discharge_capacity_ah + fraction * (
                    upper.cycle_discharge_capacity_ah - lower.cycle_discharge_capacity_ah
                )
                label_source = "calibration_interpolated"
            soh = label_capacity / candidate.identity.config.nominal_capacity_ah
            if not math.isfinite(soh) or soh <= 0:
                candidate.output_status = "excluded"
                candidate.output_reason = "invalid_soh_after_calibration_label"
                cell.excluded_selected_cycles += 1
                continue
            candidate.label_capacity_ah = float(label_capacity)
            candidate.soh = float(soh)
            candidate.label_source = label_source
            candidate.output_status = "selected_pending_export"
            candidate.output_reason = "phase_window_and_calibration_label_valid"
            cell.selected_output_cycles += 1
            if label_source == "calibration_interpolated":
                cell.interpolated_label_cycles += 1

        # Freeze the Paper-v2 denominator here, while the complete selected
        # source chronology is still available.  A direct calibration remains
        # a reference candidate even when its charge trace failed the model
        # window/temperature gate and will not be exported as a training row.
        if cell.selected_output_cycles:
            source_reference_rows = []
            for candidate in series:
                capacity = (
                    candidate.cycle_discharge_capacity_ah
                    if candidate.calibration_direct_candidate
                    else candidate.label_capacity_ah
                )
                source_reference_rows.append(
                    {
                        "cycle_id": canonical_cycle_id(candidate),
                        "capacity_Ah": capacity,
                        "calibration_direct": candidate.calibration_direct_candidate,
                        "model_eligible": candidate.candidate_eligible,
                        "label_source": candidate.label_source,
                    }
                )
            bol_reference = build_frozen_smarthealth_bol_reference(
                source_reference_rows,
                domain_id=cell.identity.config.domain_id,
                cell_id=logical_sequence_id,
            )
            cell.bol_reference = bol_reference
            for candidate in series:
                candidate.bol_q_ref_ah = float(bol_reference["Q_ref"])
                candidate.bol_q_ref_rule = str(bol_reference["rule_version"])
                candidate.bol_q_ref_source = str(bol_reference["reference_source"])


def inventory_signature(identities: Sequence[SourceIdentity]) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        digest.update(
            (
                f"{identity.relative_path}\t{identity.file_size_bytes}\t"
                f"{identity.file_mtime_ns}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def split_rank(config: DomainConfig, condition: str, logical_sequence_id: str) -> str:
    value = (
        f"{SPLIT_STRATEGY_VERSION}:condition_cell_split:"
        f"{config.domain_id}:{condition}:{logical_sequence_id}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assign_split_roles(
    config: DomainConfig,
    cells: Mapping[str, CellSummary],
    source_signature: str,
) -> dict[str, object]:
    """Assign two development and one held-out sequence per valid condition.

    A condition with anything other than exactly three usable logical sequences
    remains visible in the output, but it receives no silent fallback split.
    The resulting JSON is deliberately marked for manual confirmation and the
    shared split loader refuses to train from it.
    """

    by_condition: dict[str, list[CellSummary]] = defaultdict(list)
    for cell in cells.values():
        by_condition[cell.identity.condition].append(cell)
    conditions_payload: dict[str, dict[str, object]] = {}
    development_by_condition: dict[str, list[str]] = {}
    test_by_condition: dict[str, list[str]] = {}
    manual_confirmation_conditions: dict[str, str] = {}
    for condition, condition_cells in sorted(by_condition.items()):
        ranked = sorted(
            condition_cells,
            key=lambda cell: split_rank(
                config, condition, cell.identity.logical_sequence_id
            ),
        )
        for cell in ranked:
            cell.split_rank_sha256 = split_rank(
                config, condition, cell.identity.logical_sequence_id
            )
        unavailable = sorted(
            cell.identity.logical_sequence_id
            for cell in ranked
            if cell.selected_output_cycles == 0
        )
        issue_parts: list[str] = []
        if len(ranked) != 3:
            issue_parts.append(
                f"expected_exactly_3_logical_sequences_found_{len(ranked)}"
            )
        if len(ranked) - len(unavailable) != 3:
            issue_parts.append(
                "expected_exactly_3_eligible_logical_sequences_found_"
                f"{len(ranked) - len(unavailable)}"
            )
        if unavailable:
            issue_parts.append("no_exportable_cycles=" + ",".join(unavailable))
        issue = "; ".join(issue_parts)
        status = "complete" if not issue else "manual_confirmation_required"
        development_cells: list[str] = []
        test_cell: str | None = None
        if status == "complete":
            development_cells = [
                ranked[0].identity.logical_sequence_id,
                ranked[1].identity.logical_sequence_id,
            ]
            test_cell = ranked[2].identity.logical_sequence_id
            development_by_condition[condition] = list(development_cells)
            test_by_condition[condition] = [test_cell]
            for cell in ranked:
                cell.split_status = status
                cell.split_issue = ""
                cell.split_role = (
                    "development"
                    if cell.identity.logical_sequence_id in development_cells
                    else "test"
                )
        else:
            manual_confirmation_conditions[condition] = issue
            for cell in ranked:
                cell.split_status = status
                cell.split_issue = issue
                cell.split_role = "unassigned_manual_confirmation"
        conditions_payload[condition] = {
            "status": status,
            "required_logical_sequences": 3,
            "discovered_logical_sequences": len(ranked),
            "eligible_logical_sequences": len(ranked) - len(unavailable),
            "development_cells": development_cells,
            "test_cell": test_cell,
            "manual_confirmation_issue": issue or None,
            "ranking": {
                "algorithm": "SHA256",
                "input": (
                    "{split_strategy_version}:condition_cell_split:"
                    "{domain_id}:{condition}:{logical_sequence_id}"
                ),
                "sort_order": "ascending hexadecimal digest",
                "development_ranks": [0, 1],
                "held_out_test_rank": 2,
            },
            "logical_sequences": [
                {
                    "source_serial": cell.identity.source_serial,
                    "condition": cell.identity.condition,
                    "logical_sequence_id": cell.identity.logical_sequence_id,
                    "source_chunks": cell.source_chunk_count,
                    "eligible_cycles": cell.selected_output_cycles,
                    "selection": cell.split_role,
                    "split_rank_sha256": cell.split_rank_sha256,
                }
                for cell in sorted(ranked, key=lambda item: item.identity.logical_sequence_id)
            ],
        }
    split_status = (
        "manual_confirmation_required"
        if manual_confirmation_conditions
        else "complete"
    )
    return {
        "schema_version": 3,
        "name": f"{config.domain_id}_condition_stratified_2development_1test_v3",
        "dataset": "smarthealth",
        "domain_id": config.domain_id,
        "strategy_version": POLICY_VERSION,
        "preprocessing_strategy_version": POLICY_VERSION,
        "split_strategy_version": SPLIT_STRATEGY_VERSION,
        "split_status": split_status,
        "source_manifest_signature_sha256": source_signature,
        "logical_sequence_rule": "domain + source serial + C-rate + DOD; no cross-condition merge",
        "selection_rule": (
            "For each condition with exactly three exportable logical sequences, "
            "ascending deterministic SHA256 ranking selects ranks 0 and 1 as "
            "development and rank 2 as the held-out test. Other inventories are "
            "recorded for manual confirmation without a fallback split."
        ),
        "development_batteries_by_condition": development_by_condition,
        "test_batteries_by_condition": test_by_condition,
        "development_split": {
            "mode": "mixed_cycle",
            "scope": "all development logical sequences pooled within the battery family",
            "train_ratio": 0.8,
            "val_ratio": 0.2,
            "random_state": 420,
            "split_unit": "cycle",
            "train_val_battery_overlap_expected": True,
        },
        "manual_confirmation_conditions": manual_confirmation_conditions,
        "conditions": conditions_payload,
    }


def apply_split_roles_to_candidates(
    candidates: Iterable[CycleCandidate], cells: Mapping[str, CellSummary]
) -> None:
    """Propagate the deterministic cell decision to every auditable candidate.

    Canonical RAW rows are emitted only from the de-duplicated winner, but the
    cycle provenance also contains rejected chunk candidates.  Giving those
    rows the same cell-level decision makes an incomplete condition visible in
    the audit trail instead of leaving its split status implicit.
    """
    for candidate in candidates:
        cell = cells[candidate.identity.logical_sequence_id]
        candidate.split_role = cell.split_role
        candidate.split_status = cell.split_status
        candidate.split_issue = cell.split_issue


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _owned_domain_files(directory: Path, config: DomainConfig) -> list[Path]:
    return sorted(
        path
        for path in directory.glob(f"{config.domain_id}__*.csv")
        if path.is_file()
    )


def prepare_domain_output(
    directory: Path, config: DomainConfig, overwrite: bool, *, pointer_names: Sequence[str] = ()
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    existing = [*_owned_domain_files(directory, config)]
    existing.extend(path for name in pointer_names if (path := directory / name).is_file())
    if existing and not overwrite:
        raise FileExistsError(
            f"{directory} already contains {len(existing)} {config.domain_id} products; "
            "pass --overwrite to replace this domain only"
        )
    if overwrite:
        for path in existing:
            path.unlink()


def _selected_phase_for_export(
    points: Sequence[Point],
    config: DomainConfig,
    dod_percent: int,
    args: argparse.Namespace,
) -> PhaseResult:
    phase = split_combined_charge(
        points,
        args,
        min_cv_points=min_cv_points_for_condition(
            config.domain_id, dod_percent, args.min_cv_points
        ),
    )
    select_model_windows(phase, config, args)
    phase.cc_window_accepted = model_cc_window_accepted(phase, dod_percent, args)
    return phase


def _same_phase_summary(left: PhaseResult, right: PhaseResult) -> bool:
    return (
        left.status == right.status
        and left.reason == right.reason
        and left.inferred_cc_points == right.inferred_cc_points
        and left.inferred_cv_points == right.inferred_cv_points
        and left.selected_cc_points == right.selected_cc_points
        and left.selected_cv_points == right.selected_cv_points
        and left.cv_start_source_row_index == right.cv_start_source_row_index
        and left.cc_window_complete == right.cc_window_complete
        and left.cc_window_accepted == right.cc_window_accepted
        and left.cv_window_complete == right.cv_window_complete
        and left.temperature_complete == right.temperature_complete
    )


def make_raw_row(
    candidate: CycleCandidate,
    phase: PhaseResult,
    point: Point,
    segment: str,
    cycle_point_index: int,
    segment_point_index: int,
    time_zero: float,
) -> dict[str, object]:
    identity = candidate.identity
    relative_minutes = (duration_seconds(point.time_text) - time_zero) / 60.0
    return {
        "dataset": "smarthealth",
        "dataset_id": "smarthealth",
        "domain_id": identity.config.domain_id,
        "manufacturer": identity.config.manufacturer,
        "cell": identity.logical_sequence_id,
        "battery_id": identity.logical_sequence_id,
        "source_serial": identity.source_serial,
        "logical_sequence_id": identity.logical_sequence_id,
        "source_series": identity.source_series,
        "condition": identity.condition,
        "cycle": canonical_cycle_id(candidate),
        "SOH": candidate.soh,
        "label_source": candidate.label_source,
        "cycle_discharge_capacity_Ah": candidate.cycle_discharge_capacity_ah,
        "label_capacity_Ah": candidate.label_capacity_ah,
        "reference_calibration_capacity_Ah": candidate.reference_calibration_capacity_ah,
        "bol_q_ref_Ah": candidate.bol_q_ref_ah,
        "bol_q_ref_rule": candidate.bol_q_ref_rule,
        "bol_q_ref_source": candidate.bol_q_ref_source,
        "split_role": candidate.split_role,
        "split_status": candidate.split_status,
        "split_issue": candidate.split_issue,
        "split_strategy_version": SPLIT_STRATEGY_VERSION,
        "segment": segment,
        "cycle_point_index": cycle_point_index,
        "segment_point_index": segment_point_index,
        "source_row_index": point.source_row_index,
        "relative_time": relative_minutes,
        "relative_time_min": relative_minutes,
        "relative_time_unit": "min",
        "voltage_V": point.voltage_v,
        "current_A": point.current_a,
        "c_rate": abs(point.current_a) / identity.config.nominal_capacity_ah,
        "charge_capacity_Ah": point.charge_capacity_ah,
        "discharge_capacity_Ah": point.discharge_capacity_ah,
        "temperature_C": point.temperature_c,
        "power_W": point.voltage_v * point.current_a,
        "source_file": identity.relative_path,
        "chunk_id": identity.chunk_id,
        "source_file_part": identity.chunk_id,
        "source_cycle": candidate.source_cycle,
        "source_absolute_start_time": format_source_absolute_time(
            candidate.source_absolute_start_time
        ),
        "source_absolute_end_time": format_source_absolute_time(
            candidate.source_absolute_end_time
        ),
        "source_step_id": point.step_id,
        "strategy_version": POLICY_VERSION,
        "phase_policy_version": POLICY_VERSION,
        "cc_current_reference_A": phase.cc_current_reference_a,
        "cv_start_source_row_index": phase.cv_start_source_row_index,
        "cv_start_voltage_V": phase.cv_start_voltage_v,
        "cv_start_current_A": phase.cv_start_current_a,
        "cc_voltage_low_V": identity.config.cc_voltage_low_v,
        "cc_voltage_high_V": identity.config.cc_voltage_high_v,
        "cv_c_rate_low": identity.config.cv_c_rate_low,
        "cv_c_rate_high": identity.config.cv_c_rate_high,
    }


def export_one_logical_sequence(
    logical_sequence_id: str,
    identities: Sequence[SourceIdentity],
    selected_candidates: Sequence[CycleCandidate],
    output_path: Path,
    args: argparse.Namespace,
) -> LogicalSequenceExportResult:
    """Export one logical sequence in an isolated worker-owned CSV file."""

    candidates_by_source_cycle: dict[tuple[str, int], CycleCandidate] = {}
    wanted_by_source: dict[str, set[int]] = defaultdict(set)
    for candidate in selected_candidates:
        key = (candidate.identity.relative_path, candidate.source_cycle)
        if key in candidates_by_source_cycle:
            raise RuntimeError(
                f"Duplicate export task identity for {logical_sequence_id}: {key}"
            )
        candidates_by_source_cycle[key] = candidate
        wanted_by_source[candidate.identity.relative_path].add(candidate.source_cycle)
    if not candidates_by_source_cycle:
        raise RuntimeError(f"No selected cycles to export for {logical_sequence_id}")

    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}"
    )
    raw_rows_by_cycle: dict[int, int] = {}
    source_files_with_selected_cycles = 0
    try:
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for identity in identities:
                wanted = wanted_by_source.get(identity.relative_path)
                if not wanted:
                    continue
                source_files_with_selected_cycles += 1

                def callback(source_cycle_id: int, points: list[Point], has_temperature: bool) -> None:
                    del has_temperature
                    if source_cycle_id not in wanted:
                        return
                    candidate = candidates_by_source_cycle.get(
                        (identity.relative_path, source_cycle_id)
                    )
                    if candidate is None:
                        return
                    observed_start = min(point.absolute_time for point in points)
                    observed_end = max(point.absolute_time for point in points)
                    if (
                        observed_start != candidate.source_absolute_start_time
                        or observed_end != candidate.source_absolute_end_time
                    ):
                        raise RuntimeError(
                            "Non-reproducible source time-interval identity in "
                            f"{identity.relative_path}, source cycle {source_cycle_id}"
                        )
                    phase = _selected_phase_for_export(
                        points, identity.config, candidate.identity.dod_percent, args
                    )
                    if not _same_phase_summary(candidate.phase, phase):
                        raise RuntimeError(
                            "Non-reproducible CC/CV decision in "
                            f"{identity.relative_path}, source cycle {source_cycle_id}"
                        )
                    if not phase.selected_cc or not phase.selected_cv:
                        raise RuntimeError(
                            "Selected CC/CV window became empty in "
                            f"{identity.relative_path}, source cycle {source_cycle_id}"
                        )
                    time_zero = duration_seconds(phase.cc[0].time_text)
                    rows = 0
                    for segment, phase_points, offset in (
                        ("CC", phase.selected_cc, 0),
                        ("CV", phase.selected_cv, len(phase.selected_cc)),
                    ):
                        for point_index, point in enumerate(phase_points):
                            writer.writerow(
                                make_raw_row(
                                    candidate,
                                    phase,
                                    point,
                                    segment,
                                    offset + point_index,
                                    point_index,
                                    time_zero,
                                )
                            )
                            rows += 1
                    canonical_cycle = canonical_cycle_id(candidate)
                    if canonical_cycle in raw_rows_by_cycle:
                        raise RuntimeError(
                            f"Duplicate canonical cycle during export for {logical_sequence_id}: "
                            f"{canonical_cycle}"
                        )
                    raw_rows_by_cycle[canonical_cycle] = rows

                visit_cycles(identity, callback)
        missing = sorted(
            canonical_cycle_id(candidate)
            for candidate in candidates_by_source_cycle.values()
            if canonical_cycle_id(candidate) not in raw_rows_by_cycle
        )
        if missing:
            raise RuntimeError(
                f"Raw export did not reach selected cycles for {logical_sequence_id}: "
                f"{missing[:10]}"
            )
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return LogicalSequenceExportResult(
        logical_sequence_id=logical_sequence_id,
        raw_rows_by_cycle=raw_rows_by_cycle,
        source_files_with_selected_cycles=source_files_with_selected_cycles,
    )


def export_raw_products(
    identities: Sequence[SourceIdentity],
    selected: Mapping[tuple[str, int], CycleCandidate],
    cells: Mapping[str, CellSummary],
    raw_domain_directory: Path,
    args: argparse.Namespace,
) -> None:
    """Parallel second pass, with one worker-owned output file per cell."""

    identities_by_cell: dict[str, list[SourceIdentity]] = defaultdict(list)
    for identity in identities:
        identities_by_cell[identity.logical_sequence_id].append(identity)
    pending_by_cell: dict[str, list[CycleCandidate]] = defaultdict(list)
    for candidate in selected.values():
        if candidate.output_status == "selected_pending_export":
            pending_by_cell[candidate.identity.logical_sequence_id].append(candidate)
    tasks = [
        (
            logical_sequence_id,
            identities_by_cell[logical_sequence_id],
            sorted(candidates, key=canonical_cycle_id),
            raw_domain_directory / f"{logical_sequence_id}.csv",
        )
        for logical_sequence_id, candidates in sorted(pending_by_cell.items())
    ]

    exported: set[tuple[str, int]] = set()

    def merge(task_index: int, result: LogicalSequenceExportResult) -> None:
        logical_sequence_id = result.logical_sequence_id
        written_rows = 0
        for cycle, rows in sorted(result.raw_rows_by_cycle.items()):
            candidate = selected[(logical_sequence_id, cycle)]
            if candidate.output_status != "selected_pending_export":
                raise RuntimeError(
                    f"Unexpected export state for {logical_sequence_id} cycle {cycle}: "
                    f"{candidate.output_status}"
                )
            candidate.raw_rows_written = rows
            candidate.output_status = "exported"
            candidate.output_reason = "phase_window_and_calibration_label_exported"
            cells[logical_sequence_id].raw_rows_written += rows
            exported.add((logical_sequence_id, cycle))
            written_rows += rows
        print(
            f"[export {task_index}/{len(tasks)}] {logical_sequence_id}: "
            f"selected {len(result.raw_rows_by_cycle)} cycles from "
            f"{result.source_files_with_selected_cycles} source file(s), {written_rows} rows",
            flush=True,
        )

    if args.workers == 1 or len(tasks) == 1:
        for task_index, task in enumerate(tasks, start=1):
            merge(task_index, export_one_logical_sequence(*task, args))
    elif tasks:
        pending: dict[int, LogicalSequenceExportResult] = {}
        next_index = 1
        worker_count = min(args.workers, len(tasks))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(export_one_logical_sequence, *task, args): task_index
                for task_index, task in enumerate(tasks, start=1)
            }
            for future in as_completed(futures):
                task_index = futures[future]
                logical_sequence_id = tasks[task_index - 1][0]
                try:
                    pending[task_index] = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Export failed for {logical_sequence_id}"
                    ) from exc
                while next_index in pending:
                    merge(next_index, pending.pop(next_index))
                    next_index += 1
    missing = [
        (candidate.identity.logical_sequence_id, canonical_cycle_id(candidate))
        for candidate in selected.values()
        if candidate.output_status == "selected_pending_export"
    ]
    if missing:
        raise RuntimeError(f"Raw export did not reach selected cycles: {missing[:10]}")
    if not exported:
        raise RuntimeError("No SmartHealth raw cycles were exported")


def candidate_provenance(candidate: CycleCandidate) -> dict[str, object]:
    identity = candidate.identity
    phase = candidate.phase
    return {
        "dataset": "smarthealth",
        "domain_id": identity.config.domain_id,
        "manufacturer": identity.config.manufacturer,
        "source_serial": identity.source_serial,
        "logical_sequence_id": identity.logical_sequence_id,
        "source_series": identity.source_series,
        "condition": identity.condition,
        "cycle": canonical_cycle_id(candidate),
        "source_cycle": candidate.source_cycle,
        "source_file": identity.relative_path,
        "chunk_id": identity.chunk_id,
        "source_file_part": identity.chunk_id,
        "source_file_size_bytes": identity.file_size_bytes,
        "source_file_mtime_ns": identity.file_mtime_ns,
        "source_absolute_start_time": format_source_absolute_time(
            candidate.source_absolute_start_time
        ),
        "source_absolute_end_time": format_source_absolute_time(
            candidate.source_absolute_end_time
        ),
        "source_cycle_duration_hours": source_cycle_duration_hours(
            candidate.source_absolute_start_time, candidate.source_absolute_end_time
        ),
        "source_rows": candidate.source_rows,
        "source_charge_event_count": phase.charge_event_count,
        "source_discharge_event_count": phase.discharge_event_count,
        "source_temperature_column_present": candidate.source_temperature_column_present,
        "selected_charge_event_index": phase.charge_event_index,
        "selected_charge_step_id": phase.charge_step_id,
        "selected_charge_points": phase.charge_points,
        "inferred_cc_points": phase.inferred_cc_points,
        "inferred_cv_points": phase.inferred_cv_points,
        "selected_cc_points": phase.selected_cc_points,
        "selected_cv_points": phase.selected_cv_points,
        "cc_current_reference_A": phase.cc_current_reference_a,
        "cv_start_source_row_index": phase.cv_start_source_row_index,
        "cv_start_voltage_V": phase.cv_start_voltage_v,
        "cv_start_current_A": phase.cv_start_current_a,
        "charge_voltage_max_V": phase.charge_voltage_max_v,
        "selected_cc_voltage_min_V": phase.selected_cc_voltage_min_v,
        "selected_cc_voltage_max_V": phase.selected_cc_voltage_max_v,
        "selected_cv_c_rate_min": phase.selected_cv_c_rate_min,
        "selected_cv_c_rate_max": phase.selected_cv_c_rate_max,
        "cc_window_lower_covered": phase.cc_window_lower_covered,
        "cc_window_upper_covered": phase.cc_window_upper_covered,
        "cc_window_complete": phase.cc_window_complete,
        "cc_window_accepted": phase.cc_window_accepted,
        "cv_window_high_covered": phase.cv_window_high_covered,
        "cv_window_low_covered": phase.cv_window_low_covered,
        "cv_window_complete": phase.cv_window_complete,
        "temperature_complete": phase.temperature_complete,
        "boundary_detection_status": phase.status,
        "boundary_detection_reason": phase.reason,
        "candidate_eligible": candidate.candidate_eligible,
        "candidate_eligibility_reason": candidate.candidate_eligibility_reason,
        "candidate_count_for_event": candidate.candidate_count_for_event,
        "selected_candidate": candidate.selected_candidate,
        "selection_reason": candidate.selection_reason,
        "cycle_discharge_capacity_Ah": candidate.cycle_discharge_capacity_ah,
        "calibration_direct_candidate": candidate.calibration_direct_candidate,
        "calibration_reason": candidate.calibration_reason,
        "reference_calibration_capacity_Ah": candidate.reference_calibration_capacity_ah,
        "bol_q_ref_Ah": candidate.bol_q_ref_ah,
        "bol_q_ref_rule": candidate.bol_q_ref_rule,
        "bol_q_ref_source": candidate.bol_q_ref_source,
        "label_capacity_Ah": candidate.label_capacity_ah,
        "SOH": candidate.soh,
        "label_source": candidate.label_source,
        "split_role": candidate.split_role,
        "split_status": candidate.split_status,
        "split_issue": candidate.split_issue,
        "split_strategy_version": SPLIT_STRATEGY_VERSION,
        "output_status": candidate.output_status,
        "output_reason": candidate.output_reason,
        "raw_rows_written": candidate.raw_rows_written,
        "strategy_version": POLICY_VERSION,
    }


def cell_provenance(cell: CellSummary) -> dict[str, object]:
    identity = cell.identity
    bol = cell.bol_reference
    return {
        "dataset": "smarthealth",
        "domain_id": identity.config.domain_id,
        "manufacturer": identity.config.manufacturer,
        "paper_alias": identity.config.paper_alias,
        "source_serial": identity.source_serial,
        "logical_sequence_id": identity.logical_sequence_id,
        "source_series": identity.source_series,
        "condition": identity.condition,
        "dod_percent": identity.dod_percent,
        "nominal_capacity_Ah": identity.config.nominal_capacity_ah,
        "source_chunk_count": cell.source_chunk_count,
        "source_cycle_candidates": cell.source_cycle_candidates,
        "unique_source_events": cell.unique_source_events,
        "duplicate_timestamp_candidates": cell.duplicate_timestamp_candidates,
        "overlapping_source_cycle_candidates": cell.overlapping_source_cycle_candidates,
        "phase_eligible_cycles": cell.phase_eligible_cycles,
        "direct_calibration_cycles": cell.direct_calibration_cycles,
        "interpolated_label_cycles": cell.interpolated_label_cycles,
        "selected_output_cycles": cell.selected_output_cycles,
        "excluded_selected_cycles": cell.excluded_selected_cycles,
        "reference_calibration_capacity_Ah": cell.reference_calibration_capacity_ah,
        "reference_calibration_cycle_count": cell.reference_calibration_cycle_count,
        "first_calibration_cycle": cell.first_calibration_cycle,
        "last_calibration_cycle": cell.last_calibration_cycle,
        "bol_q_ref_Ah": bol.get("Q_ref", math.nan),
        "bol_q_ref_rule": bol.get("rule_version", ""),
        "bol_q_ref_source": bol.get("reference_source", ""),
        "bol_reference_candidate_count": bol.get("candidate_count", 0),
        "bol_reference_valid_candidate_count_after_outlier_filter": bol.get(
            "valid_candidate_count_after_outlier_filter", 0
        ),
        "bol_reference_window_start_cycle": bol.get("reference_window_start_cycle", ""),
        "bol_reference_window_end_cycle": bol.get("reference_window_end_cycle", ""),
        "bol_reference_window_observation_count": bol.get(
            "reference_window_observation_count", 0
        ),
        "bol_reference_window_initial_size": bol.get("reference_window_initial_size", 0),
        "bol_reference_window_expanded_after_mad": bol.get(
            "reference_window_expanded_after_mad", False
        ),
        "bol_reference_source_observation_count": bol.get("source_observation_count", 0),
        "bol_reference_source_direct_calibration_count": bol.get(
            "source_direct_calibration_count", 0
        ),
        "bol_reference_source_model_ineligible_direct_calibration_count": bol.get(
            "source_model_ineligible_direct_calibration_count", 0
        ),
        "bol_reference_selected_cycle_ids_json": json.dumps(
            bol.get("selected_cycle_ids", []), ensure_ascii=False, separators=(",", ":")
        ),
        "bol_reference_selected_capacities_Ah_json": json.dumps(
            bol.get("selected_top5_capacity_values_Ah", []),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "bol_reference_rejected_outliers_json": json.dumps(
            bol.get("rejected_outliers", []), ensure_ascii=False, separators=(",", ":")
        ),
        "raw_rows_written": cell.raw_rows_written,
        "split_role": cell.split_role,
        "split_status": cell.split_status,
        "split_issue": cell.split_issue,
        "split_rank_sha256": cell.split_rank_sha256,
        "split_file": cell.split_file,
        "split_strategy_version": SPLIT_STRATEGY_VERSION,
        "strategy_version": POLICY_VERSION,
    }


def build_raw_report(
    config: DomainConfig,
    input_root: Path,
    identities: Sequence[SourceIdentity],
    candidates: Mapping[SourceEventKey, Sequence[CycleCandidate]],
    cells: Mapping[str, CellSummary],
    split_path: Path,
    split_payload: Mapping[str, object],
    workers: int,
) -> dict[str, object]:
    all_candidates = [
        candidate
        for same_cycle in candidates.values()
        for candidate in same_cycle
    ]
    selected = [candidate for candidate in all_candidates if candidate.selected_candidate]
    by_condition: dict[str, dict[str, object]] = {}
    split_conditions = dict(split_payload.get("conditions", {}))
    for condition in sorted({identity.condition for identity in identities}):
        source_identity = [item for item in identities if item.condition == condition]
        condition_cells = [cell for cell in cells.values() if cell.identity.condition == condition]
        condition_selected = [candidate for candidate in selected if candidate.identity.condition == condition]
        condition_split = dict(split_conditions.get(condition, {}))
        by_condition[condition] = {
            "source_files": len(source_identity),
            "logical_sequences": len(condition_cells),
            "duplicate_timestamp_candidates": sum(
                cell.duplicate_timestamp_candidates for cell in condition_cells
            ),
            "overlapping_source_cycle_candidates": sum(
                cell.overlapping_source_cycle_candidates for cell in condition_cells
            ),
            "boundary_success": sum(candidate.phase.status == "ok" for candidate in condition_selected),
            "boundary_failure": sum(candidate.phase.status != "ok" for candidate in condition_selected),
            "cc_window_coverage": sum(candidate.phase.cc_window_complete for candidate in condition_selected),
            "cc_window_accepted_coverage": sum(candidate.phase.cc_window_accepted for candidate in condition_selected),
            "cv_window_coverage": sum(candidate.phase.cv_window_complete for candidate in condition_selected),
            "temperature_exclusions": sum(
                candidate.selected_candidate
                and "temperature" in candidate.candidate_eligibility_reason
                for candidate in all_candidates
                if candidate.identity.condition == condition
            ),
            "calibration_direct": sum(
                candidate.label_source == "calibration_direct" for candidate in condition_selected
            ),
            "calibration_interpolated": sum(
                candidate.label_source == "calibration_interpolated" for candidate in condition_selected
            ),
            "final_eligible_cycles": sum(candidate.output_status == "exported" for candidate in condition_selected),
            "exclusion_reasons": dict(
                sorted(
                    Counter(
                        candidate.output_reason
                        for candidate in condition_selected
                        if candidate.output_status != "exported"
                    ).items()
                )
            ),
            "split_status": condition_split.get("status", "unknown"),
            "development_cells": condition_split.get("development_cells", []),
            "test_cell": condition_split.get("test_cell"),
            "manual_confirmation_issue": condition_split.get(
                "manual_confirmation_issue"
            ),
        }
    return {
        "schema_version": 6,
        "strategy_version": POLICY_VERSION,
        "preprocessing_strategy_version": POLICY_VERSION,
        "split_strategy_version": SPLIT_STRATEGY_VERSION,
        "split_status": split_payload.get("split_status", "unknown"),
        "manual_confirmation_conditions": split_payload.get(
            "manual_confirmation_conditions", {}
        ),
        "dataset": "smarthealth",
        "domain_id": config.domain_id,
        "input_root": str(input_root.resolve()),
        "execution": {
            "workers": int(workers),
            "scan_parallelization": "independent source CSV process workers",
            "export_parallelization": "one worker-owned CSV per logical sequence",
            "merge_order": "source inventory/logical sequence order; canonical data and split are worker-count invariant",
            "canonical_cycle_identity": "logical_sequence_id + source absolute-time interval order; source_cycle remains provenance only",
        },
        "source_manifest_signature_sha256": inventory_signature(identities),
        "source_files": len(identities),
        "logical_sequences": len(cells),
        "unique_source_events": len(candidates),
        "duplicate_timestamp_candidates": sum(
            cell.duplicate_timestamp_candidates for cell in cells.values()
        ),
        "overlapping_source_cycle_candidates": sum(
            cell.overlapping_source_cycle_candidates for cell in cells.values()
        ),
        "boundary_detection_success": sum(candidate.phase.status == "ok" for candidate in selected),
        "boundary_detection_failure": sum(candidate.phase.status != "ok" for candidate in selected),
        "cc_window_coverage": sum(candidate.phase.cc_window_complete for candidate in selected),
        "cc_window_accepted_coverage": sum(candidate.phase.cc_window_accepted for candidate in selected),
        "cv_window_coverage": sum(candidate.phase.cv_window_complete for candidate in selected),
        "temperature_exclusions": sum(
            candidate.selected_candidate and "temperature" in candidate.candidate_eligibility_reason
            for candidate in all_candidates
        ),
        "calibration_direct_labels": sum(
            candidate.label_source == "calibration_direct" for candidate in selected
        ),
        "calibration_interpolated_labels": sum(
            candidate.label_source == "calibration_interpolated" for candidate in selected
        ),
        "final_eligible_cycles": sum(candidate.output_status == "exported" for candidate in selected),
        "final_raw_rows": sum(candidate.raw_rows_written for candidate in selected),
        "exclusion_reasons": dict(
            sorted(
                Counter(
                    candidate.output_reason
                    for candidate in selected
                    if candidate.output_status != "exported"
                ).items()
            )
        ),
        "conditions": by_condition,
        "cc_upper_bound_audit": CC_UPPER_BOUND_AUDIT,
        "phase_policy": {
            "source_charge_step": CHARGE_STEP,
            "boundary": "first persistent current taper near source charge-voltage maximum",
            "cc_window_v": [config.cc_voltage_low_v, config.cc_voltage_high_v],
            "cc_window_acceptance": (
                "100%DOD requires lower and upper CC coverage; partial-DOD accepts "
                "the observable CC interval when it has the minimum selected-point "
                "count and reaches the upper endpoint"
            ),
            "cv_window_c_rate": [config.cv_c_rate_low, config.cv_c_rate_high],
            "current_normalization": "abs(current_A) / nominal_capacity_Ah only",
            "temperature": "all selected CC/CV points must have finite source temperature; no imputation",
        },
        "soh_policy": {
            "target": "calibration-derived label capacity / fixed nominal family capacity",
            "reference": "median of first three reliable calibration discharge capacities; provenance only, not the target denominator",
            "direct": "calibration_direct",
            "interpolation": "linear only between direct calibration cycles; no leading/trailing extrapolation",
            "excluded": "partial-DOD discharge capacity / nominal capacity is never an SOH label",
            "rul_eol": "not generated in v5",
        },
        "bol_reference_contract": {
            "contract_version": BOL_REFERENCE_CONTRACT_VERSION,
            "rule_version": BOL_RULE_VERSION,
            "reference_source": BOL_REFERENCE_SOURCE,
            "cell_references": {
                key: cells[key].bol_reference
                for key in sorted(cells)
                if cells[key].bol_reference
            },
        },
        "split_file": str(split_path),
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_source_root() -> Path | None:
    value = os.environ.get("SMARTHEALTH_SOURCE_ROOT")
    return Path(value).expanduser() if value else None


def _default_raw_root() -> Path:
    return _repository_root() / "datasets/SmartHealth_raw"


def _default_feature_root() -> Path:
    return _repository_root() / "datasets/SmartHealth_features"


def _default_split_root() -> Path:
    return _repository_root() / "splits/smarthealth"


def build_raw_parser(config: DomainConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Build canonical RAW SmartHealth products for {config.domain_id}. "
            "This entry point cannot process another family."
        )
    )
    parser.add_argument("--input-root", type=Path, default=_default_source_root())
    parser.add_argument("--raw-output-root", type=Path, default=_default_raw_root())
    parser.add_argument("--splits-output-root", type=Path, default=_default_split_root())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-source-files", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help=(
            "process workers for independent source scans and logical-sequence "
            "exports (default: min(8, available CPUs); use 1 for serial debugging)"
        ),
    )
    parser.add_argument("--min-cc-points", type=int, default=60)
    parser.add_argument("--min-cv-points", type=int, default=60)
    parser.add_argument("--min-selected-cc-points", type=int, default=10)
    parser.add_argument("--min-selected-cv-points", type=int, default=10)
    parser.add_argument("--cc-reference-fraction", type=float, default=0.20)
    parser.add_argument("--cc-reference-min-points", type=int, default=120)
    parser.add_argument("--cc-reference-quantile", type=float, default=0.90)
    parser.add_argument("--cv-taper-fraction", type=float, default=0.01)
    parser.add_argument("--cv-persistence-points", type=int, default=30)
    parser.add_argument("--cv-voltage-tolerance-v", type=float, default=0.02)
    parser.add_argument(
        "--max-source-cycle-duration-hours",
        type=float,
        default=DEFAULT_MAX_SOURCE_CYCLE_DURATION_HOURS,
        help=(
            "reject a source cycle spanning longer than this wall-clock duration "
            "(default: 24 h)"
        ),
    )
    return parser


def parse_raw_args(config: DomainConfig, argv: Sequence[str] | None) -> argparse.Namespace:
    parser = build_raw_parser(config)
    args = parser.parse_args(argv)
    if args.input_root is None:
        parser.error("--input-root is required (or set SMARTHEALTH_SOURCE_ROOT)")
    if min(
        args.min_cc_points,
        args.min_cv_points,
        args.min_selected_cc_points,
        args.min_selected_cv_points,
    ) < 2:
        parser.error("all minimum point counts must be at least 2")
    if not 0 < args.cc_reference_fraction <= 1:
        parser.error("--cc-reference-fraction must be in (0, 1]")
    if not 0 < args.cc_reference_quantile <= 1:
        parser.error("--cc-reference-quantile must be in (0, 1]")
    if not 0 < args.cv_taper_fraction < 1:
        parser.error("--cv-taper-fraction must be in (0, 1)")
    if args.cv_persistence_points < 2 or args.cv_voltage_tolerance_v < 0:
        parser.error("invalid CV persistence or voltage tolerance")
    if args.max_source_cycle_duration_hours <= 0:
        parser.error("--max-source-cycle-duration-hours must be positive")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def raw_audit_paths(audit_root: Path, config: DomainConfig) -> dict[str, Path]:
    stem = config.domain_id.upper()
    return {
        "source_file_audit": audit_root / f"{stem}_SOURCE_FILE_AUDIT.csv",
        "cycle_provenance": audit_root / f"{stem}_CYCLE_PROVENANCE.csv",
        "cell_provenance": audit_root / f"{stem}_CELL_PROVENANCE.csv",
        "report": audit_root / f"{stem}_PREPROCESSING_REPORT.json",
    }


def run_raw_preprocessing(config: DomainConfig, argv: Sequence[str] | None = None) -> int:
    """Run one family-specific source-to-canonical RAW pipeline."""

    args = parse_raw_args(config, argv)
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    identities = list_identities(input_root, config, args.max_source_files)
    signature = inventory_signature(identities)
    print(
        f"{config.domain_id}: scanning {len(identities)} source chunks "
        f"with {args.workers} worker(s) (manifest={signature[:12]}...)",
        flush=True,
    )
    candidates, cells, source_audit = scan_sources(identities, args)
    selected_events = resolve_duplicate_candidates(candidates, cells)
    selected = assign_chronological_cycle_ids(selected_events, candidates)
    label_calibration_soh(
        selected,
        cells,
        max_source_cycle_duration_hours=args.max_source_cycle_duration_hours,
    )
    split_payload = assign_split_roles(config, cells, signature)
    apply_split_roles_to_candidates(
        (candidate for per_cycle in candidates.values() for candidate in per_cycle),
        cells,
    )

    raw_domain_directory = args.raw_output_root / config.domain_id
    audit_root = args.raw_output_root / "audit"
    audit_paths = raw_audit_paths(audit_root, config)
    raw_manifest_name = "SMARTHEALTH_CANONICAL_MANIFEST.json"
    prepare_domain_output(
        raw_domain_directory,
        config,
        args.overwrite,
        pointer_names=(raw_manifest_name,),
    )
    audit_root.mkdir(parents=True, exist_ok=True)
    audit_existing = [path for path in audit_paths.values() if path.is_file()]
    split_path = args.splits_output_root / f"{config.domain_id}_cell_split.json"
    if (audit_existing or split_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Existing audit/split products for {config.domain_id}; pass --overwrite to replace them"
        )
    if args.overwrite:
        for path in audit_existing:
            path.unlink()
        if split_path.is_file():
            split_path.unlink()

    export_raw_products(identities, selected, cells, raw_domain_directory, args)
    args.splits_output_root.mkdir(parents=True, exist_ok=True)
    for cell in cells.values():
        cell.split_file = str(split_path)
    write_json(split_path, split_payload)
    write_csv(
        audit_paths["source_file_audit"], SOURCE_FILE_AUDIT_COLUMNS, source_audit
    )
    all_candidates = [candidate for same_cycle in candidates.values() for candidate in same_cycle]
    write_csv(
        audit_paths["cycle_provenance"],
        CYCLE_PROVENANCE_COLUMNS,
        (candidate_provenance(candidate) for candidate in all_candidates),
    )
    write_csv(
        audit_paths["cell_provenance"],
        CELL_PROVENANCE_COLUMNS,
        (cell_provenance(cells[key]) for key in sorted(cells)),
    )
    report = build_raw_report(
        config,
        input_root,
        identities,
        candidates,
        cells,
        split_path,
        split_payload,
        args.workers,
    )
    write_json(audit_paths["report"], report)
    write_json(
        raw_domain_directory / raw_manifest_name,
        {
            "schema_version": 6,
            "strategy_version": POLICY_VERSION,
            "preprocessing_strategy_version": POLICY_VERSION,
            "split_strategy_version": SPLIT_STRATEGY_VERSION,
            "split_status": split_payload["split_status"],
            "manual_confirmation_conditions": split_payload[
                "manual_confirmation_conditions"
            ],
            "dataset": "smarthealth",
            "domain_id": config.domain_id,
            "source_manifest_signature_sha256": signature,
            "execution": {
                "workers": int(args.workers),
                "scan_parallelization": "independent source CSV process workers",
                "export_parallelization": "one worker-owned CSV per logical sequence",
                "worker_count_invariant_data_contract": True,
            },
            "cycle_identity": {
                "canonical_cycle": "one-based chronological index within logical_sequence_id, ordered by source_absolute_start_time then source_absolute_end_time",
                "source_cycle": "untouched local source 循环号; provenance only and not globally unique across chunks",
                "deduplication": "exact-time duplicates and same-source-cycle overlapping chunk intervals are quality-ranked; disjoint local-cycle resets remain distinct",
                "max_source_cycle_duration_hours": args.max_source_cycle_duration_hours,
            },
            "raw_schema": RAW_COLUMNS,
            "bol_reference_contract": {
                "contract_version": BOL_REFERENCE_CONTRACT_VERSION,
                "rule_version": BOL_RULE_VERSION,
                "reference_source": BOL_REFERENCE_SOURCE,
                "cell_references": {
                    key: cells[key].bol_reference
                    for key in sorted(cells)
                    if cells[key].bol_reference
                },
            },
            "audit_files": {name: str(path) for name, path in audit_paths.items()},
            "split_file": str(split_path),
            "feature_input_contract": "feature extraction reads this canonical RAW directory and its cycle provenance only; never source CSV",
        },
    )
    print(
        json.dumps(
            {
                "domain_id": config.domain_id,
                "raw_output": str(raw_domain_directory),
                "audit_output": str(audit_root),
                "split_output": str(split_path),
                "workers": args.workers,
                "exported_cycles": sum(candidate.output_status == "exported" for candidate in selected.values()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0)) if values.size else math.nan


def safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else math.nan


def safe_kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    std = safe_std(values)
    if not math.isfinite(std) or math.isclose(std, 0.0):
        return 0.0
    z = (values - safe_mean(values)) / std
    return float(np.mean(z ** 4) - 3.0)


def safe_skewness(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    std = safe_std(values)
    if not math.isfinite(std) or math.isclose(std, 0.0):
        return 0.0
    z = (values - safe_mean(values)) / std
    return float(np.mean(z ** 3))


def safe_entropy(values: np.ndarray, bins: int) -> float:
    finite = values[np.isfinite(values)]
    if finite.size <= 1 or math.isclose(float(np.min(finite)), float(np.max(finite))):
        return 0.0
    histogram, _ = np.histogram(finite, bins=min(int(bins), int(finite.size)))
    probability = histogram[histogram > 0].astype(float)
    probability /= probability.sum()
    return float(-np.sum(probability * np.log(probability)))


def safe_slope(times: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(times) & np.isfinite(values)
    x, y = times[finite], values[finite]
    if x.size < 2:
        return 0.0
    x = x - float(np.mean(x))
    denominator = float(np.dot(x, x))
    if math.isclose(denominator, 0.0):
        return 0.0
    return float(np.dot(x, y - float(np.mean(y))) / denominator)


def _raw_required_fields() -> set[str]:
    return {
        "dataset",
        "domain_id",
        "cell",
        "logical_sequence_id",
        "condition",
        "cycle",
        "segment",
        "relative_time",
        "voltage_V",
        "current_A",
        "charge_capacity_Ah",
        "temperature_C",
        "SOH",
        "label_source",
        "cycle_discharge_capacity_Ah",
        "label_capacity_Ah",
        "reference_calibration_capacity_Ah",
        "bol_q_ref_Ah",
        "bol_q_ref_rule",
        "bol_q_ref_source",
        "split_role",
        "split_status",
        "split_issue",
        "split_strategy_version",
        "source_file",
        "chunk_id",
        "source_cycle",
        "source_absolute_start_time",
        "source_absolute_end_time",
        "strategy_version",
    }


def _as_finite(row: Mapping[str, str], name: str, path: Path) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid {name}={row.get(name)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}: non-finite {name}={row.get(name)!r}")
    return value


def _feature_row_from_raw_cycle(
    rows: Sequence[Mapping[str, str]], path: Path, entropy_bins: int
) -> dict[str, object]:
    if not rows:
        raise ValueError(f"Empty raw cycle in {path}")
    identities = {
        (
            row["domain_id"],
            row["cell"],
            row["logical_sequence_id"],
            row["condition"],
            row["cycle"],
            row["SOH"],
            row["label_source"],
            row["bol_q_ref_Ah"],
            row["bol_q_ref_rule"],
            row["bol_q_ref_source"],
            row["source_file"],
            row["chunk_id"],
            row["source_cycle"],
            row["source_absolute_start_time"],
            row["source_absolute_end_time"],
            row["strategy_version"],
            row["split_role"],
            row["split_status"],
            row["split_issue"],
            row["split_strategy_version"],
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError(f"{path}: canonical metadata varies within one raw cycle")
    segments = [str(row["segment"]).strip().upper() for row in rows]
    if not segments or segments[0] != "CC" or "CV" not in segments:
        raise ValueError(f"{path}: raw cycle does not contain contiguous CC then CV points")
    first_cv = segments.index("CV")
    if any(segment != "CC" for segment in segments[:first_cv]) or any(
        segment != "CV" for segment in segments[first_cv:]
    ):
        raise ValueError(f"{path}: raw segments are not a contiguous CC/CV pair")
    cc_rows = rows[:first_cv]
    cv_rows = rows[first_cv:]
    if not cc_rows or not cv_rows:
        raise ValueError(f"{path}: empty selected CC or CV segment")

    def arrays(current_rows: Sequence[Mapping[str, str]]) -> tuple[np.ndarray, ...]:
        times = np.asarray([_as_finite(row, "relative_time", path) for row in current_rows])
        voltage = np.asarray([_as_finite(row, "voltage_V", path) for row in current_rows])
        current = np.asarray([abs(_as_finite(row, "current_A", path)) for row in current_rows])
        capacity = np.asarray([_as_finite(row, "charge_capacity_Ah", path) for row in current_rows])
        temperature = np.asarray([_as_finite(row, "temperature_C", path) for row in current_rows])
        return times, voltage, current, capacity, temperature

    cc_t, cc_v, _, cc_q, cc_temp = arrays(cc_rows)
    cv_t, _, cv_i, cv_q, cv_temp = arrays(cv_rows)
    first = rows[0]
    soh = _as_finite(first, "SOH", path)
    label_capacity = _as_finite(first, "label_capacity_Ah", path)
    bol_q_ref = _as_finite(first, "bol_q_ref_Ah", path)
    if bol_q_ref <= 0.0:
        raise ValueError(f"{path}: frozen bol_q_ref_Ah must be positive")
    if str(first["bol_q_ref_rule"]) != BOL_RULE_VERSION:
        raise ValueError(f"{path}: frozen BOL rule version mismatch")
    if str(first["bol_q_ref_source"]) != BOL_REFERENCE_SOURCE:
        raise ValueError(f"{path}: frozen BOL reference source mismatch")
    row: dict[str, object] = {
        "dataset": str(first["dataset"]),
        "dataset_id": str(first.get("dataset_id", first["dataset"])),
        "domain_id": str(first["domain_id"]),
        "manufacturer": str(first.get("manufacturer", "")),
        "cell": str(first["cell"]),
        "battery_id": str(first.get("battery_id", first["cell"])),
        "source_serial": str(first.get("source_serial", "")),
        "logical_sequence_id": str(first["logical_sequence_id"]),
        "source_series": str(first.get("source_series", "")),
        "condition": str(first["condition"]),
        "cycle": int(float(first["cycle"])),
        "SOH": soh,
        "label_source": str(first["label_source"]),
        "cycle_discharge_capacity_Ah": _as_finite(first, "cycle_discharge_capacity_Ah", path),
        "label_capacity_Ah": label_capacity,
        "reference_calibration_capacity_Ah": _as_finite(
            first, "reference_calibration_capacity_Ah", path
        ),
        "bol_q_ref_Ah": bol_q_ref,
        "bol_q_ref_rule": str(first["bol_q_ref_rule"]),
        "bol_q_ref_source": str(first["bol_q_ref_source"]),
        "split_role": str(first.get("split_role", "")),
        "split_status": str(first.get("split_status", "")),
        "split_issue": str(first.get("split_issue", "")),
        "split_strategy_version": str(first.get("split_strategy_version", "")),
        "source_file": str(first["source_file"]),
        "chunk_id": int(float(first["chunk_id"])),
        "source_file_part": str(first.get("source_file_part", first["chunk_id"])),
        "source_cycle": int(float(first["source_cycle"])),
        "source_absolute_start_time": str(first["source_absolute_start_time"]),
        "source_absolute_end_time": str(first["source_absolute_end_time"]),
        "strategy_version": str(first["strategy_version"]),
        "phase_policy_version": str(first.get("phase_policy_version", first["strategy_version"])),
        "voltage mean": safe_mean(cc_v),
        "voltage std": safe_std(cc_v),
        "voltage kurtosis": safe_kurtosis(cc_v),
        "voltage skewness": safe_skewness(cc_v),
        "CC Q": finite_delta(cc_q),
        "CC charge time": finite_delta(cc_t),
        "voltage slope": safe_slope(cc_t, cc_v),
        "voltage entropy": safe_entropy(cc_v, entropy_bins),
        "current mean": safe_mean(cv_i),
        "current std": safe_std(cv_i),
        "current kurtosis": safe_kurtosis(cv_i),
        "current skewness": safe_skewness(cv_i),
        "CV Q": finite_delta(cv_q),
        "CV charge time": finite_delta(cv_t),
        "current slope": safe_slope(cv_t, cv_i),
        "current entropy": safe_entropy(cv_i, entropy_bins),
        "T_CC_mean": safe_mean(cc_temp),
        "T_CC_max": safe_max(cc_temp),
        "T_CC_delta": finite_delta(cc_temp),
        "T_CC_slope": safe_slope(cc_t, cc_temp),
        "T_CV_mean": safe_mean(cv_temp),
        "T_CV_max": safe_max(cv_temp),
        "T_CV_delta": finite_delta(cv_temp),
        "T_CV_slope": safe_slope(cv_t, cv_temp),
        # ``capacity`` follows the calibration label, never a partial-DOD
        # source discharge.  The observed source discharge is retained above.
        "capacity": label_capacity,
    }
    for column in FEATURE_COLUMNS:
        value = float(row[column])
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite feature {column!r}")
    return row


def _load_exported_cycle_provenance(
    config: DomainConfig, raw_domain_directory: Path
) -> tuple[dict[tuple[str, int], dict[str, str]], dict[str, object]]:
    manifest_path = raw_domain_directory / "SMARTHEALTH_CANONICAL_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{config.domain_id}: canonical RAW manifest is missing: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("domain_id") != config.domain_id or manifest.get("strategy_version") != POLICY_VERSION:
        raise ValueError(f"{manifest_path}: incompatible canonical RAW manifest")
    if int(manifest.get("schema_version", 0)) < 6:
        raise ValueError(
            f"{manifest_path}: canonical RAW schema predates frozen BOL references"
        )
    if manifest.get("split_strategy_version") != SPLIT_STRATEGY_VERSION:
        raise ValueError(f"{manifest_path}: incompatible canonical split strategy")
    bol_contract = manifest.get("bol_reference_contract", {})
    if (
        bol_contract.get("contract_version") != BOL_REFERENCE_CONTRACT_VERSION
        or bol_contract.get("rule_version") != BOL_RULE_VERSION
        or bol_contract.get("reference_source") != BOL_REFERENCE_SOURCE
    ):
        raise ValueError(f"{manifest_path}: incompatible frozen BOL reference contract")
    cycle_path = Path(manifest["audit_files"]["cycle_provenance"])
    if not cycle_path.is_file():
        raise FileNotFoundError(f"Missing canonical cycle provenance: {cycle_path}")
    required = {
        "domain_id",
        "logical_sequence_id",
        "cycle",
        "source_cycle",
        "source_absolute_start_time",
        "source_absolute_end_time",
        "selected_candidate",
        "output_status",
        "raw_rows_written",
        "SOH",
        "label_source",
        "bol_q_ref_Ah",
        "bol_q_ref_rule",
        "bol_q_ref_source",
        "split_role",
        "split_status",
        "split_strategy_version",
        "strategy_version",
    }
    expected: dict[tuple[str, int], dict[str, str]] = {}
    with cycle_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{cycle_path}: missing provenance columns {missing}")
        for row in reader:
            if row["domain_id"] != config.domain_id:
                raise ValueError(f"{cycle_path}: unexpected domain row")
            if row["strategy_version"] != POLICY_VERSION:
                raise ValueError(f"{cycle_path}: unexpected strategy version")
            if row["split_strategy_version"] != SPLIT_STRATEGY_VERSION:
                raise ValueError(f"{cycle_path}: unexpected split strategy version")
            if str(row["selected_candidate"]).lower() not in {"true", "1"}:
                continue
            if row["output_status"] != "exported":
                continue
            key = (str(row["logical_sequence_id"]), int(float(row["cycle"])))
            if key in expected:
                raise ValueError(f"{cycle_path}: duplicate exported cycle provenance {key}")
            expected[key] = row
    if not expected:
        raise ValueError(f"{cycle_path}: no exported canonical cycles")
    return expected, manifest


def build_feature_parser(config: DomainConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Extract baseline features from canonical RAW only for {config.domain_id}. "
            "It never opens SmartHealth source CSV files."
        )
    )
    parser.add_argument("--raw-input-root", type=Path, default=_default_raw_root())
    parser.add_argument("--feature-output-root", type=Path, default=_default_feature_root())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--entropy-bins", type=int, default=128)
    return parser


def run_feature_extraction(config: DomainConfig, argv: Sequence[str] | None = None) -> int:
    """Read the canonical RAW/provenance contract and derive one feature row/cycle."""

    parser = build_feature_parser(config)
    args = parser.parse_args(argv)
    if args.entropy_bins < 2:
        parser.error("--entropy-bins must be at least 2")
    raw_domain_directory = args.raw_input_root / config.domain_id
    expected, manifest = _load_exported_cycle_provenance(config, raw_domain_directory)
    raw_paths = _owned_domain_files(raw_domain_directory, config)
    if not raw_paths:
        raise FileNotFoundError(f"No canonical RAW CSV files under {raw_domain_directory}")
    feature_domain_directory = args.feature_output_root / config.domain_id
    pointer_name = "SMARTHEALTH_FEATURE_PROVENANCE_POINTER.json"
    report_name = f"{config.domain_id.upper()}_FEATURE_REPORT.json"
    prepare_domain_output(
        feature_domain_directory,
        config,
        args.overwrite,
        pointer_names=(pointer_name, report_name),
    )
    found: set[tuple[str, int]] = set()
    feature_rows = 0
    for raw_path in raw_paths:
        output_path = feature_domain_directory / raw_path.name
        with raw_path.open("r", encoding="utf-8-sig", newline="") as source, output_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as destination:
            reader = csv.DictReader(source)
            missing = sorted(_raw_required_fields() - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{raw_path}: missing required canonical raw columns {missing}")
            writer = csv.DictWriter(
                destination,
                fieldnames=[*FEATURE_PREFIX_COLUMNS, *FEATURE_COLUMNS],
                extrasaction="ignore",
            )
            writer.writeheader()
            active_key: tuple[str, int] | None = None
            active_rows: list[dict[str, str]] = []
            seen_local: set[tuple[str, int]] = set()

            def flush() -> None:
                nonlocal feature_rows, active_key, active_rows
                if active_key is None:
                    return
                if active_key in seen_local:
                    raise ValueError(f"{raw_path}: non-contiguous repeated raw cycle {active_key}")
                if active_key in found:
                    raise ValueError(f"Duplicate canonical raw cycle across files: {active_key}")
                provenance = expected.get(active_key)
                if provenance is None:
                    raise ValueError(f"{raw_path}: raw cycle has no exported provenance: {active_key}")
                feature = _feature_row_from_raw_cycle(active_rows, raw_path, args.entropy_bins)
                if not math.isclose(
                    float(provenance["SOH"]), float(feature["SOH"]), rel_tol=1e-7, abs_tol=1e-8
                ) or provenance["label_source"] != feature["label_source"]:
                    raise ValueError(f"{raw_path}: raw/provenance label mismatch for {active_key}")
                writer.writerow(feature)
                seen_local.add(active_key)
                found.add(active_key)
                feature_rows += 1
                active_key = None
                active_rows = []

            for row in reader:
                try:
                    key = (str(row["logical_sequence_id"]), int(float(row["cycle"])))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{raw_path}: invalid cycle identity row {row}") from exc
                if row["domain_id"] != config.domain_id:
                    raise ValueError(f"{raw_path}: contains another domain")
                if row["strategy_version"] != POLICY_VERSION:
                    raise ValueError(f"{raw_path}: strategy version mismatch")
                if row["split_strategy_version"] != SPLIT_STRATEGY_VERSION:
                    raise ValueError(f"{raw_path}: split strategy version mismatch")
                if active_key is None:
                    active_key = key
                elif key != active_key:
                    flush()
                    active_key = key
                active_rows.append(row)
            flush()
    if found != set(expected):
        raise ValueError(
            f"RAW/provenance mismatch for {config.domain_id}: "
            f"missing={sorted(set(expected) - found)[:8]}, unexpected={sorted(found - set(expected))[:8]}"
        )
    write_json(
        feature_domain_directory / pointer_name,
        {
            "schema_version": 6,
            "strategy_version": POLICY_VERSION,
            "preprocessing_strategy_version": POLICY_VERSION,
            "split_strategy_version": SPLIT_STRATEGY_VERSION,
            "split_status": manifest["split_status"],
            "manual_confirmation_conditions": manifest[
                "manual_confirmation_conditions"
            ],
            "dataset": "smarthealth",
            "domain_id": config.domain_id,
            "raw_domain_directory": str(raw_domain_directory.resolve()),
            "raw_manifest": str((raw_domain_directory / "SMARTHEALTH_CANONICAL_MANIFEST.json").resolve()),
            "split_file": manifest["split_file"],
            "raw_source_manifest_signature_sha256": manifest["source_manifest_signature_sha256"],
            "feature_schema": [*FEATURE_PREFIX_COLUMNS, *FEATURE_COLUMNS],
            "bol_reference_contract": manifest["bol_reference_contract"],
            "feature_contract": "one feature row for each exported canonical RAW cycle; no source CSV parsing",
        },
    )
    write_json(
        feature_domain_directory / report_name,
        {
            "schema_version": 6,
            "strategy_version": POLICY_VERSION,
            "preprocessing_strategy_version": POLICY_VERSION,
            "split_strategy_version": SPLIT_STRATEGY_VERSION,
            "split_status": manifest["split_status"],
            "manual_confirmation_conditions": manifest[
                "manual_confirmation_conditions"
            ],
            "domain_id": config.domain_id,
            "raw_cycles": len(expected),
            "feature_rows": feature_rows,
            "electrical_features": len(FEATURE_ELECTRICAL_COLUMNS),
            "temperature_features": len(FEATURE_TEMPERATURE_COLUMNS),
            "capacity": "calibration-labelled capacity, never partial-DOD source capacity",
            "bol_reference_contract_version": BOL_REFERENCE_CONTRACT_VERSION,
        },
    )
    print(
        json.dumps(
            {
                "domain_id": config.domain_id,
                "feature_output": str(feature_domain_directory),
                "feature_rows": feature_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def cli_main_raw(config: DomainConfig) -> int:
    try:
        return run_raw_preprocessing(config)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


def cli_main_features(config: DomainConfig) -> int:
    try:
        return run_feature_extraction(config)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
