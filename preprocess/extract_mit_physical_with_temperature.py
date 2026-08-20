#!/usr/bin/env python3
"""Extract canonical MIT/A123 physical-cell raw and feature products.

The historical MIT batch archive has 140 cell records but not 140 independent
physical cells.  This extractor applies the original ``LoadData.m``
continuation mapping before extracting data, then (by default) applies the
author's documented curation to produce the paper's 124 physical cells.

The legacy ``MIT_raw_t_v1`` and ``MIT_t_v1`` products are intentionally never
modified.  This script writes a new, provenance-rich pair of directories:

* point-level CC/CV records with a global physical ``cycle``;
* statistical features with the same physical ID/cycle and source lineage.

The proposed raw product has its own phase-aware CC/CV policy.  The paired
handcrafted feature table deliberately keeps the original feature extractor's
window definition and is not redefined by the proposed raw policy.

The first physical cycle is skipped by default once per physical cell.  In
particular, the first local cycle in a continuation segment is retained: it
is not a new physical cell and must not be dropped merely because that source
file's local MATLAB index restarts at one.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import multiprocessing as mp
import os
import re
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from extract_mit_cycle_raw_with_temperature import (
    MIT_CV_HIGH_COVERAGE_C_RATE,
    MIT_CV_LOW_COVERAGE_C_RATE,
    MIT_CV_SELECTION_TOLERANCE_C,
    MIT_PROPOSED_CC_VOLTAGE_RANGE,
    MIT_PROPOSED_CV_C_RATE_RANGE,
    MIT_PROPOSED_RAW_POLICY_VERSION,
    RAW_COLUMNS,
    MITRawBatch,
    extract_cycle_rows,
    find_all_batch_files,
    parse_float_pair,
    row_has_nonfinite as raw_row_has_nonfinite,
)
from extract_mit_features_with_temperature import (
    FEATURE_COLUMNS,
    MITBatch,
    extract_cycle_features,
    row_has_nonfinite as feature_row_has_nonfinite,
)
from mit_physical_provenance import (
    BATCH_DATES,
    KNOWN_INVALID_SOURCE_CYCLES,
    LOAD_DATA_URL,
    PAPER_URL,
    PhysicalCell,
    SourceSegment,
    build_physical_cells,
    physical_test_batteries,
    source_cycle_is_known_invalid,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "MIT_raw"
DEFAULT_FEATURE_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "MIT_features"


def default_input_root() -> Path | None:
    value = os.environ.get("MIT_SOURCE_ROOT")
    return Path(value).expanduser() if value else None

DATE_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")

PHYSICAL_RAW_COLUMNS = [
    "physical_cell_id",
    "paper_batch",
    "primary_batch_date",
    "cycle",
    "source_batch_date",
    "source_cell",
    "source_cycle",
    *[column for column in RAW_COLUMNS if column != "cycle"],
]
PHYSICAL_FEATURE_COLUMNS = [
    "physical_cell_id",
    "paper_batch",
    "primary_batch_date",
    "cycle",
    "source_batch_date",
    "source_cell",
    "source_cycle",
    *FEATURE_COLUMNS,
]
REPORT_COLUMNS = [
    "physical_cell_id",
    "physical_index",
    "paper_batch",
    "primary_batch_date",
    "source_segments",
    "source_capacity_cycle_count",
    "source_raw_cycle_count",
    "source_cycle_count_mismatch_segments",
    "start_cycle",
    "max_cycles",
    "written_raw_cycles",
    "written_raw_rows",
    "written_feature_rows",
    "empty_cc_cycles",
    "empty_cv_cycles",
    "phase_window_rejected_cycles",
    "skipped_nonfinite_raw_rows",
    "skipped_nonfinite_feature_rows",
    "length_mismatch_cycles",
    "failed_raw_cycles",
    "failed_feature_cycles",
    "dropped_known_invalid_cycles",
    "eol_threshold_Ah",
    "eol_cycle",
    "reported_cycle_life",
    "termination_status",
    "last_observed_cycle",
    "last_observed_capacity_Ah",
    "raw_file",
    "feature_file",
]

# HDF5 objects must never be inherited from the parent process.  Each spawned
# worker opens its own read-only MIT raw/feature handles once, then processes
# multiple physical cells with those handles.  This avoids HDF5 fork hazards
# and eliminates the much larger overhead of reopening all six source files
# for every physical-cell task.
_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_RAW_BATCHES: Dict[str, MITRawBatch] | None = None
_WORKER_FEATURE_BATCHES: Dict[str, MITBatch] | None = None


def _close_worker_batches() -> None:
    """Close worker-local HDF5 handles before a process exits."""

    global _WORKER_RAW_BATCHES, _WORKER_FEATURE_BATCHES
    for batches in (_WORKER_RAW_BATCHES, _WORKER_FEATURE_BATCHES):
        if batches is None:
            continue
        for batch in batches.values():
            batch.close()
    _WORKER_RAW_BATCHES = None
    _WORKER_FEATURE_BATCHES = None


def _initialise_physical_worker(
    batch_paths: Mapping[str, Path], args: argparse.Namespace
) -> None:
    """Initialise an isolated read-only HDF5 cache for one worker process."""

    global _WORKER_ARGS, _WORKER_RAW_BATCHES, _WORKER_FEATURE_BATCHES
    _close_worker_batches()
    _WORKER_ARGS = args
    _WORKER_RAW_BATCHES = {
        date: MITRawBatch(path) for date, path in batch_paths.items()
    }
    _WORKER_FEATURE_BATCHES = {
        date: MITBatch(path) for date, path in batch_paths.items()
    }
    atexit.register(_close_worker_batches)


def _process_physical_worker(
    physical: PhysicalCell,
) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]]]:
    """Process one physical cell with this worker's local HDF5 handles."""

    if (
        _WORKER_ARGS is None
        or _WORKER_RAW_BATCHES is None
        or _WORKER_FEATURE_BATCHES is None
    ):
        raise RuntimeError("MIT physical worker was not initialised")
    return _process_physical_cell(
        _WORKER_ARGS,
        physical,
        _WORKER_RAW_BATCHES,
        _WORKER_FEATURE_BATCHES,
    )


def _batch_paths_by_date(input_root: Path) -> Dict[str, Path]:
    """Resolve the three official source batch files by their date prefix."""

    resolved: Dict[str, Path] = {}
    for path in find_all_batch_files(input_root):
        match = DATE_RE.search(path.name)
        if match is None:
            continue
        batch_date = match.group("date")
        if batch_date not in BATCH_DATES:
            continue
        if batch_date in resolved:
            raise ValueError(
                f"multiple MIT batch files found for {batch_date}: "
                f"{resolved[batch_date]} and {path}"
            )
        resolved[batch_date] = path
    missing = [date for date in BATCH_DATES if date not in resolved]
    if missing:
        raise FileNotFoundError(
            "missing official MIT batch files for " + ", ".join(missing)
        )
    return resolved


def _canonical_csv_name(physical: PhysicalCell) -> str:
    return physical.output_stem + ".csv"


def _existing_canonical_files(root: Path) -> List[Path]:
    return sorted(root.glob("MIT_*_physical-*.csv"))


def _prepare_output_dir(root: Path, overwrite: bool) -> None:
    """Protect existing canonical data unless a scoped overwrite was requested."""

    root.mkdir(parents=True, exist_ok=True)
    existing = _existing_canonical_files(root)
    if existing and not overwrite:
        raise FileExistsError(
            f"{root} already contains {len(existing)} canonical MIT physical CSVs. "
            "Choose a new output directory or pass --overwrite explicitly."
        )
    if overwrite:
        # Only touch files made by this canonical extractor, never legacy CSVs
        # or other user files in the same parent directory.
        for path in existing:
            path.unlink()


def _segment_inventory(
    physical: PhysicalCell, raw_batches: Mapping[str, MITRawBatch], strict: bool
) -> Tuple[List[Dict[str, object]], np.ndarray, int, int, int]:
    """Build global cycle offsets and capacity labels for one physical cell."""

    segments: List[Dict[str, object]] = []
    capacities: List[np.ndarray] = []
    global_offset = 0
    raw_total = 0
    mismatch_segments = 0
    for source in physical.source_segments:
        batch = raw_batches[source.batch_date]
        cell_index = source.cell - 1
        capacity_series = np.asarray(batch.capacity_series(cell_index), dtype=float)
        capacity_count, raw_cycle_count = batch.cycle_counts(cell_index)
        usable_count = min(capacity_count, raw_cycle_count)
        mismatch = capacity_count != raw_cycle_count
        if mismatch:
            mismatch_segments += 1
            if strict:
                raise ValueError(
                    f"cycle-count mismatch for {source.source_file_identity}: "
                    f"capacity={capacity_count}, raw={raw_cycle_count}"
                )
        if usable_count <= 0:
            raise ValueError(f"no usable cycles for {source.source_file_identity}")
        if capacity_series.size < usable_count:
            raise ValueError(
                f"capacity series shorter than usable cycle count for "
                f"{source.source_file_identity}"
            )
        segments.append(
            {
                "batch_date": source.batch_date,
                "cell": int(source.cell),
                "source_file_identity": source.source_file_identity,
                "capacity_cycle_count": int(capacity_count),
                "raw_cycle_count": int(raw_cycle_count),
                "used_cycle_count": int(usable_count),
                "cycle_count_mismatch": bool(mismatch),
                "physical_cycle_start": int(global_offset + 1),
                "physical_cycle_end": int(global_offset + usable_count),
            }
        )
        capacities.append(capacity_series[:usable_count])
        global_offset += usable_count
        raw_total += raw_cycle_count
    return (
        segments,
        np.concatenate(capacities, axis=0),
        int(global_offset),
        int(raw_total),
        int(mismatch_segments),
    )


def _lifetime_metadata(capacity_series: np.ndarray, nominal_capacity: float) -> Dict[str, object]:
    """Mirror the paper's LoadData.m cycle-life label semantics.

    ``eol_cycle`` is the first physical source cycle with QDischarge < 0.88 Ah
    (80% of 1.1 Ah).  If no observed capacity crosses that threshold, the
    original code records ``n_cycles + 1``; this is right-censoring, not proof
    that the physical cell survived indefinitely.
    """

    values = np.asarray(capacity_series, dtype=float).reshape(-1)
    threshold = 0.8 * float(nominal_capacity)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        raise ValueError("a physical cell has no finite QDischarge capacities")
    below = np.flatnonzero(np.isfinite(values) & (values < threshold))
    last_index = int(finite_indices[-1])
    if below.size:
        eol_cycle = int(below[0] + 1)
        reported_cycle_life = eol_cycle
        termination_status = "capacity_below_80pct_nominal"
    else:
        eol_cycle = None
        # Exact LoadData.m fallback for an uncrossed record.
        reported_cycle_life = int(values.size + 1)
        termination_status = "right_censored_no_observed_80pct_crossing"
    return {
        "eol_threshold_Ah": float(threshold),
        "eol_cycle": eol_cycle,
        "reported_cycle_life": int(reported_cycle_life),
        "termination_status": termination_status,
        "last_observed_cycle": int(last_index + 1),
        "last_observed_capacity_Ah": float(values[last_index]),
    }


def _physical_row_prefix(
    physical: PhysicalCell, global_cycle: int, source: SourceSegment, source_cycle: int
) -> Dict[str, object]:
    return {
        "physical_cell_id": physical.physical_cell_id,
        "paper_batch": physical.paper_batch,
        "primary_batch_date": physical.primary_batch_date,
        "cycle": int(global_cycle),
        "source_batch_date": source.batch_date,
        "source_cell": int(source.cell),
        "source_cycle": int(source_cycle),
    }


def _raw_output_row(prefix: Mapping[str, object], row: Mapping[str, object]) -> Dict[str, object]:
    output = dict(prefix)
    output.update({key: value for key, value in row.items() if key != "cycle"})
    return output


def _feature_output_row(prefix: Mapping[str, object], row: Mapping[str, float]) -> Dict[str, object]:
    output = dict(prefix)
    output.update(row)
    return output


def _write_feature_row(writer: csv.DictWriter, row: Mapping[str, object]) -> None:
    """Use legacy blank representation only when non-finite rows are retained."""

    formatted: MutableMapping[str, object] = dict(row)
    for column in FEATURE_COLUMNS:
        value = formatted.get(column)
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            formatted[column] = ""
    writer.writerow(formatted)


def _cycle_selected(args: argparse.Namespace, global_cycle: int) -> bool:
    if global_cycle < args.start_cycle:
        return False
    if args.max_cycles is None:
        return True
    return global_cycle <= args.start_cycle + args.max_cycles - 1


def _process_physical_cell(
    args: argparse.Namespace,
    physical: PhysicalCell,
    raw_batches: Mapping[str, MITRawBatch],
    feature_batches: Mapping[str, MITBatch],
) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]]]:
    """Write raw/features for one physical cell and return audit metadata."""

    (
        segment_info,
        combined_capacities,
        source_capacity_cycle_count,
        source_raw_cycle_count,
        mismatch_segments,
    ) = _segment_inventory(physical, raw_batches, strict=args.strict)
    lifetime = _lifetime_metadata(combined_capacities, args.nominal_capacity)
    raw_path = args.raw_output_dir / _canonical_csv_name(physical)
    feature_path = args.feature_output_dir / _canonical_csv_name(physical)

    metrics: Dict[str, int] = {
        "written_raw_cycles": 0,
        "written_raw_rows": 0,
        "written_feature_rows": 0,
        "empty_cc_cycles": 0,
        "empty_cv_cycles": 0,
        "phase_window_rejected_cycles": 0,
        "skipped_nonfinite_raw_rows": 0,
        "skipped_nonfinite_feature_rows": 0,
        "length_mismatch_cycles": 0,
        "failed_raw_cycles": 0,
        "failed_feature_cycles": 0,
        "dropped_known_invalid_cycles": 0,
    }
    invalid_cycles: List[Dict[str, object]] = []

    with raw_path.open("w", newline="", encoding="utf-8-sig") as raw_handle, feature_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as feature_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=PHYSICAL_RAW_COLUMNS)
        feature_writer = csv.DictWriter(feature_handle, fieldnames=PHYSICAL_FEATURE_COLUMNS)
        raw_writer.writeheader()
        feature_writer.writeheader()

        for info, source in zip(segment_info, physical.source_segments):
            raw_batch = raw_batches[source.batch_date]
            feature_batch = feature_batches[source.batch_date]
            cell_index = source.cell - 1
            local_capacities = np.asarray(raw_batch.capacity_series(cell_index), dtype=float)
            source_cycle_count = int(info["used_cycle_count"])
            global_start = int(info["physical_cycle_start"])

            for source_cycle in range(1, source_cycle_count + 1):
                global_cycle = global_start + source_cycle - 1
                invalid_reason = source_cycle_is_known_invalid(
                    source.batch_date, source.cell, source_cycle
                )
                if invalid_reason is not None and args.drop_known_invalid_cycles:
                    metrics["dropped_known_invalid_cycles"] += 1
                    invalid_cycles.append(
                        {
                            "battery_id": physical.physical_cell_id,
                            "cycle_id": int(global_cycle),
                            "source_batch_date": source.batch_date,
                            "source_cell": int(source.cell),
                            "source_cycle": int(source_cycle),
                            "reason": invalid_reason,
                        }
                    )
                    continue
                if not _cycle_selected(args, global_cycle):
                    continue

                capacity_ah = float(local_capacities[source_cycle - 1])
                soh = (
                    capacity_ah / args.nominal_capacity
                    if args.label_mode == "capacity_to_soh"
                    else capacity_ah
                )
                prefix = _physical_row_prefix(physical, global_cycle, source, source_cycle)

                raw_cycle_written = False
                try:
                    (
                        rows,
                        empty_cc,
                        empty_cv,
                        length_mismatch,
                        phase_window_rejected,
                    ) = extract_cycle_rows(
                        batch=raw_batch,
                        cell_index=cell_index,
                        cycle=source_cycle,
                        soh=soh,
                        capacity_ah=capacity_ah,
                        nominal_capacity_ah=args.nominal_capacity,
                    )
                    metrics["empty_cc_cycles"] += int(empty_cc)
                    metrics["empty_cv_cycles"] += int(empty_cv)
                    metrics["phase_window_rejected_cycles"] += int(phase_window_rejected)
                    metrics["length_mismatch_cycles"] += int(length_mismatch)
                    if length_mismatch and args.strict:
                        raise ValueError("raw channel lengths do not match")
                    output_rows = []
                    for row in rows:
                        if args.drop_nonfinite_rows and raw_row_has_nonfinite(row):
                            metrics["skipped_nonfinite_raw_rows"] += 1
                            continue
                        output_rows.append(_raw_output_row(prefix, row))
                    if output_rows:
                        raw_writer.writerows(output_rows)
                        raw_cycle_written = True
                        metrics["written_raw_cycles"] += 1
                        metrics["written_raw_rows"] += len(output_rows)
                except Exception as exc:
                    metrics["failed_raw_cycles"] += 1
                    if args.strict:
                        raise RuntimeError(
                            f"failed raw extraction for {physical.physical_cell_id} "
                            f"{source.source_file_identity} source cycle {source_cycle}"
                        ) from exc

                # A feature without an accessible raw terminal signal cannot be
                # evaluated against the raw path.  Keeping the common support
                # makes physical-cycle matched evaluation exact rather than
                # capacity-order inferred.
                if not raw_cycle_written:
                    continue
                try:
                    feature = extract_cycle_features(
                        batch=feature_batch,
                        cell_index=cell_index,
                        cycle=source_cycle,
                        capacity_series=local_capacities,
                        cc_voltage_range=args.feature_cc_voltage_range,
                        cv_current_range=args.feature_cv_current_range,
                        cv_slope_current_range=args.cv_slope_current_range,
                    )
                    if args.drop_nonfinite_rows and feature_row_has_nonfinite(feature):
                        metrics["skipped_nonfinite_feature_rows"] += 1
                        continue
                    _write_feature_row(feature_writer, _feature_output_row(prefix, feature))
                    metrics["written_feature_rows"] += 1
                except Exception as exc:
                    metrics["failed_feature_cycles"] += 1
                    if args.strict:
                        raise RuntimeError(
                            f"failed feature extraction for {physical.physical_cell_id} "
                            f"{source.source_file_identity} source cycle {source_cycle}"
                        ) from exc

    report: Dict[str, object] = {
        "physical_cell_id": physical.physical_cell_id,
        "physical_index": int(physical.physical_index),
        "paper_batch": physical.paper_batch,
        "primary_batch_date": physical.primary_batch_date,
        "source_segments": json.dumps(segment_info, ensure_ascii=False, sort_keys=True),
        "source_capacity_cycle_count": source_capacity_cycle_count,
        "source_raw_cycle_count": source_raw_cycle_count,
        "source_cycle_count_mismatch_segments": mismatch_segments,
        "start_cycle": int(args.start_cycle),
        "max_cycles": "" if args.max_cycles is None else int(args.max_cycles),
        **metrics,
        **lifetime,
        "raw_file": raw_path.name,
        "feature_file": feature_path.name,
    }
    provenance = physical.as_dict()
    provenance.update(
        {
            "source_segments": segment_info,
            "lifetime": lifetime,
            "extraction": {
                "start_cycle": int(args.start_cycle),
                "max_cycles": args.max_cycles,
                "drop_known_invalid_cycles": bool(args.drop_known_invalid_cycles),
                "drop_nonfinite_rows": bool(args.drop_nonfinite_rows),
                "label_mode": args.label_mode,
                "nominal_capacity_Ah": float(args.nominal_capacity),
                "workers": int(args.workers),
                "proposed_raw_phase_policy": {
                    "version": MIT_PROPOSED_RAW_POLICY_VERSION,
                    "cc_voltage_range_V": list(MIT_PROPOSED_CC_VOLTAGE_RANGE),
                    "cv_c_rate_range": list(MIT_PROPOSED_CV_C_RATE_RANGE),
                    "cv_sampling_tolerance_C": MIT_CV_SELECTION_TOLERANCE_C,
                    "cv_endpoint_coverage_C": [
                        MIT_CV_LOW_COVERAGE_C_RATE,
                        MIT_CV_HIGH_COVERAGE_C_RATE,
                    ],
                    "current_normalization": "abs(current_A) / nominal_capacity_Ah",
                },
                "feature_baseline_windows": {
                    "cc_voltage_range_V": list(args.feature_cc_voltage_range),
                    "cv_current_range_A": list(args.feature_cv_current_range),
                    "cv_slope_current_range_A": list(args.cv_slope_current_range),
                },
                "metrics": metrics,
            },
            "raw_file": raw_path.name,
            "feature_file": feature_path.name,
        }
    )
    return report, provenance, invalid_cycles


def _process_cells_serial(
    args: argparse.Namespace,
    physical_cells: Sequence[PhysicalCell],
    batch_paths: Mapping[str, Path],
) -> Iterable[
    Tuple[PhysicalCell, Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]]]]
]:
    """Yield physical-cell extraction results using one shared HDF5 reader set."""

    with ExitStack() as stack:
        raw_batches = {
            date: stack.enter_context(MITRawBatch(path))
            for date, path in batch_paths.items()
        }
        feature_batches = {
            date: stack.enter_context(MITBatch(path))
            for date, path in batch_paths.items()
        }
        for physical in physical_cells:
            yield physical, _process_physical_cell(
                args, physical, raw_batches, feature_batches
            )


def _process_cells_parallel(
    args: argparse.Namespace,
    physical_cells: Sequence[PhysicalCell],
    batch_paths: Mapping[str, Path],
) -> Iterable[
    Tuple[PhysicalCell, Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]]]]
]:
    """Yield independently written physical-cell results from spawned workers.

    A physical cell is the parallelism boundary: it owns one unique raw CSV
    and one unique feature CSV, while raw/feature pairing and all per-cell
    cycle ordering remain unchanged.  ``spawn`` is deliberate because HDF5
    file objects are not safe to inherit through ``fork``.
    """

    output_names = [_canonical_csv_name(physical) for physical in physical_cells]
    if len(output_names) != len(set(output_names)):
        raise ValueError("physical-cell mapping produced duplicate output filenames")

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        mp_context=context,
        initializer=_initialise_physical_worker,
        initargs=(dict(batch_paths), args),
    ) as executor:
        futures: Dict[Future, PhysicalCell] = {
            executor.submit(_process_physical_worker, physical): physical
            for physical in physical_cells
        }
        for future in as_completed(futures):
            physical = futures[future]
            try:
                yield physical, future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f"MIT physical worker failed for {physical.physical_cell_id}"
                ) from exc


def _write_csv(rows: Iterable[Mapping[str, object]], columns: Sequence[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _manifest_csv_rows(provenance_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in provenance_rows:
        lifetime = dict(item["lifetime"])
        metrics = dict(item["extraction"]["metrics"])
        rows.append(
            {
                "physical_cell_id": item["physical_cell_id"],
                "physical_index": item["physical_index"],
                "paper_batch": item["paper_batch"],
                "primary_batch_date": item["primary_batch_date"],
                "source_segments": json.dumps(item["source_segments"], sort_keys=True),
                "curation_note": item["curation_note"],
                **lifetime,
                "written_raw_cycles": metrics["written_raw_cycles"],
                "written_raw_rows": metrics["written_raw_rows"],
                "written_feature_rows": metrics["written_feature_rows"],
                "dropped_known_invalid_cycles": metrics["dropped_known_invalid_cycles"],
                "raw_file": item["raw_file"],
                "feature_file": item["feature_file"],
            }
        )
    return rows


MANIFEST_CSV_COLUMNS = [
    "physical_cell_id",
    "physical_index",
    "paper_batch",
    "primary_batch_date",
    "source_segments",
    "curation_note",
    "eol_threshold_Ah",
    "eol_cycle",
    "reported_cycle_life",
    "termination_status",
    "last_observed_cycle",
    "last_observed_capacity_Ah",
    "written_raw_cycles",
    "written_raw_rows",
    "written_feature_rows",
    "dropped_known_invalid_cycles",
    "raw_file",
    "feature_file",
]


def _split_spec(physical_cells: Sequence[PhysicalCell], cohort: str, invalid_cycles: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    test_batteries = physical_test_batteries(physical_cells, modulus=5, remainder=0)
    return {
        "name": f"mit_{cohort}_mixed_physical_index_mod5_test",
        "dataset_id": "mit",
        "status": "recommended",
        "source": "MIT public batch source; continuation/cell curation from LoadData.m",
        "cohort": cohort,
        "physical_cell_count": len(physical_cells),
        "test_rule": {
            "type": "physical_id_modulo",
            "modulus": 5,
            "remainder": 0,
            "identity": "physical_cell_id",
        },
        "test_batteries": test_batteries,
        "development_split": {
            "mode": "mixed_cycle",
            "scope": "single_domain_pool",
            "val_ratio": 0.2,
            "random_state": 420,
            "train_val_battery_overlap_expected": True,
        },
        "invalid_cycles": list(invalid_cycles),
        "notes": [
            "The test rule applies to canonical physical IDs (mit_p###), never source file IDs.",
            "All segments of a continuation cell have one physical_cell_id and therefore one split role.",
            "The raw and feature products use identical physical IDs, physical cycles, and known-label-QC exclusions.",
            "Mixed-cycle development keeps train/validation cycle overlap by design; only the test cells are physically held out.",
        ],
    }


def _write_metadata(
    args: argparse.Namespace,
    cohort: str,
    batch_paths: Mapping[str, Path],
    all_physical_cells: Sequence[PhysicalCell],
    provenance_rows: Sequence[Mapping[str, object]],
    reports: Sequence[Mapping[str, object]],
    invalid_cycles: Sequence[Mapping[str, object]],
) -> None:
    manifest = {
        "schema_version": "mit_physical_provenance_v3",
        "cohort": cohort,
        "physical_cell_count": len(provenance_rows),
        "full_cohort_physical_cell_count": len(all_physical_cells),
        "subset_export": len(provenance_rows) != len(all_physical_cells),
        "source_batch_files": {date: str(path) for date, path in batch_paths.items()},
        "identity_definition": {
            "battery_id": "physical_cell_id (mit_p###), not an extracted file ID",
            "cycle": "one-based global physical cycle; continuation local cycles are offset after their batch-1 segment",
            "source_provenance_columns": [
                "source_batch_date",
                "source_cell",
                "source_cycle",
            ],
        },
        "continuation_mapping": [
            {
                "physical_parent": (
                    f"{physical['source_segments'][0]['batch_date']}_battery-"
                    f"{physical['source_segments'][0]['cell']}"
                ),
                "continuation": (
                    f"{physical['source_segments'][1]['batch_date']}_battery-"
                    f"{physical['source_segments'][1]['cell']}"
                ),
                "physical_cell_id": physical["physical_cell_id"],
            }
            for physical in provenance_rows
            if len(physical["source_segments"]) == 2
        ],
        "paper_124_curation": {
            "reference": LOAD_DATA_URL,
            "description": "five continuation appends; five documented batch-1 removals; six documented batch-3 removals",
        },
        "cycle_life_definition": {
            "reference": LOAD_DATA_URL,
            "nominal_capacity_Ah": float(args.nominal_capacity),
            "eol_threshold_Ah": 0.8 * float(args.nominal_capacity),
            "rule": "first global physical cycle with QDischarge < 0.8 * nominal_capacity",
            "right_censor_rule": "if no observed crossing, reported_cycle_life = observed_capacity_cycle_count + 1",
            "paper_reference": PAPER_URL,
        },
        "known_invalid_source_cycles": [
            {
                "source_batch_date": key[0],
                "source_cell": key[1],
                "source_cycle": key[2],
                "reason": reason,
            }
            for key, reason in KNOWN_INVALID_SOURCE_CYCLES.items()
        ],
        "invalid_cycles_applied": list(invalid_cycles),
        "extraction_arguments": {
            "start_cycle": int(args.start_cycle),
            "max_cycles": args.max_cycles,
            "drop_known_invalid_cycles": bool(args.drop_known_invalid_cycles),
            "drop_nonfinite_rows": bool(args.drop_nonfinite_rows),
            "label_mode": args.label_mode,
            "nominal_capacity_Ah": float(args.nominal_capacity),
            "workers": int(args.workers),
            "proposed_raw_phase_policy": {
                "version": MIT_PROPOSED_RAW_POLICY_VERSION,
                "cc_voltage_range_V": list(MIT_PROPOSED_CC_VOLTAGE_RANGE),
                "cv_c_rate_range": list(MIT_PROPOSED_CV_C_RATE_RANGE),
                "cv_sampling_tolerance_C": MIT_CV_SELECTION_TOLERANCE_C,
                "cv_endpoint_coverage_C": [
                    MIT_CV_LOW_COVERAGE_C_RATE,
                    MIT_CV_HIGH_COVERAGE_C_RATE,
                ],
                "current_normalization": "abs(current_A) / nominal_capacity_Ah",
            },
            "feature_baseline_windows": {
                "cc_voltage_range_V": list(args.feature_cc_voltage_range),
                "cv_current_range_A": list(args.feature_cv_current_range),
                "cv_slope_current_range_A": list(args.cv_slope_current_range),
            },
        },
        "physical_cells": list(provenance_rows),
    }
    manifest_path = args.raw_output_dir / "MIT_PHYSICAL_PROVENANCE.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(
        _manifest_csv_rows(provenance_rows),
        MANIFEST_CSV_COLUMNS,
        args.raw_output_dir / "MIT_PHYSICAL_PROVENANCE.csv",
    )
    _write_csv(reports, REPORT_COLUMNS, args.raw_output_dir / "mit_physical_extraction_report.csv")
    split_path = args.split_json or args.raw_output_dir / "mit_physical_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(_split_spec(provenance_rows_to_cells(provenance_rows), cohort, invalid_cycles), indent=2)
        + "\n",
        encoding="utf-8",
    )
    # Feature users should not have to infer which raw directory owns its
    # provenance.  Store a small pointer, not a duplicated giant manifest.
    feature_pointer = {
        "schema_version": "mit_physical_feature_pointer_v3",
        "raw_provenance": str(manifest_path),
        "raw_output_dir": str(args.raw_output_dir),
        "feature_output_dir": str(args.feature_output_dir),
        "cohort": cohort,
        "physical_cell_count": len(provenance_rows),
    }
    (args.feature_output_dir / "MIT_PHYSICAL_PROVENANCE_POINTER.json").write_text(
        json.dumps(feature_pointer, indent=2) + "\n", encoding="utf-8"
    )


def provenance_rows_to_cells(rows: Sequence[Mapping[str, object]]) -> List[PhysicalCell]:
    """Recover the small identity objects needed for split JSON creation."""

    cells = []
    for row in rows:
        sources = tuple(
            SourceSegment(str(item["batch_date"]), int(item["cell"]))
            for item in row["source_segments"]
        )
        cells.append(
            PhysicalCell(
                physical_cell_id=str(row["physical_cell_id"]),
                physical_index=int(row["physical_index"]),
                paper_batch=str(row["paper_batch"]),
                primary_batch_date=str(row["primary_batch_date"]),
                source_segments=sources,
                cohort=str(row["cohort"]),
                curation_note=str(row["curation_note"]),
            )
        )
    return cells


def _print_summary(reports: Sequence[Mapping[str, object]]) -> None:
    total = {column: 0 for column in REPORT_COLUMNS if column.startswith(("written_", "empty_", "skipped_", "failed_", "dropped_"))}
    for report in reports:
        for key in total:
            total[key] += int(report[key])
    print("physical_cells=" + str(len(reports)))
    print(" ".join(f"{key}={value}" for key, value in total.items()))
    print(
        "termination_counts="
        + json.dumps(
            {
                status: sum(1 for row in reports if row["termination_status"] == status)
                for status in sorted({str(row["termination_status"]) for row in reports})
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MIT raw/features by canonical physical cell identity."
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--raw-output-dir", type=Path, default=DEFAULT_RAW_OUTPUT_DIR)
    parser.add_argument("--feature-output-dir", type=Path, default=DEFAULT_FEATURE_OUTPUT_DIR)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--cohort", choices=("paper124", "source135"), default="paper124")
    parser.add_argument("--start-cycle", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--max-physical-cells", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
        help=(
            "physical-cell worker processes (default: min(4, CPU count)); "
            "use 1 for serial extraction"
        ),
    )
    parser.add_argument(
        "--drop-known-invalid-cycles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exclude known source label spikes from both raw and features",
    )
    parser.add_argument(
        "--drop-nonfinite-rows", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--feature-cc-voltage-range",
        "--cc-voltage-range",
        dest="feature_cc_voltage_range",
        type=parse_float_pair,
        default=[3.4, 3.595],
        help=(
            "legacy handcrafted-feature CC window; the proposed raw CC window "
            "is fixed at 3.45,3.60"
        ),
    )
    parser.add_argument(
        "--feature-cv-current-range",
        "--cv-current-range",
        dest="feature_cv_current_range",
        type=parse_float_pair,
        default=[0.5, 0.1],
        help=(
            "legacy handcrafted-feature CV-current window; the proposed raw CV "
            "window is fixed at nominal 0.25C,0.05C"
        ),
    )
    parser.add_argument(
        "--cv-slope-current-range", type=parse_float_pair, default=[0.5, 0.1]
    )
    parser.add_argument("--nominal-capacity", type=float, default=1.1)
    parser.add_argument(
        "--label-mode", choices=("capacity_to_soh", "capacity"), default="capacity_to_soh"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.input_root is None:
        parser.error("--input-root is required (or set MIT_SOURCE_ROOT)")
    args.input_root = args.input_root.expanduser().resolve()
    args.raw_output_dir = args.raw_output_dir.expanduser().resolve()
    args.feature_output_dir = args.feature_output_dir.expanduser().resolve()
    if args.split_json is not None:
        args.split_json = args.split_json.expanduser().resolve()
    if args.start_cycle < 1:
        parser.error("--start-cycle must be at least 1")
    if args.max_cycles is not None and args.max_cycles < 1:
        parser.error("--max-cycles must be at least 1")
    if args.max_physical_cells is not None and args.max_physical_cells < 1:
        parser.error("--max-physical-cells must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.nominal_capacity <= 0:
        parser.error("--nominal-capacity must be positive")
    if args.raw_output_dir == args.feature_output_dir:
        parser.error("raw and feature output directories must be different")
    return args


def main() -> None:
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {args.input_root}")
    all_physical_cells = build_physical_cells(args.cohort)
    physical_cells = list(all_physical_cells)
    if args.max_physical_cells is not None:
        physical_cells = physical_cells[: args.max_physical_cells]
    args.workers = min(int(args.workers), len(physical_cells))
    _prepare_output_dir(args.raw_output_dir, args.overwrite)
    _prepare_output_dir(args.feature_output_dir, args.overwrite)
    batch_paths = _batch_paths_by_date(args.input_root)

    print(
        f"cohort={args.cohort}; exporting {len(physical_cells)}/"
        f"{len(all_physical_cells)} physical cells"
    )
    execution_mode = "parallel" if args.workers > 1 else "serial"
    print(f"execution={execution_mode}; workers={args.workers}; hdf5_start_method=spawn")
    result_iterator = (
        _process_cells_parallel(args, physical_cells, batch_paths)
        if args.workers > 1
        else _process_cells_serial(args, physical_cells, batch_paths)
    )
    completed: List[
        Tuple[PhysicalCell, Dict[str, object], Dict[str, object], List[Dict[str, object]]]
    ] = []
    for completed_count, (physical, result) in enumerate(result_iterator, start=1):
        report, provenance, invalid = result
        completed.append((physical, report, provenance, invalid))
        print(
            f"OK [{completed_count}/{len(physical_cells)}] "
            f"{physical.physical_cell_id} ({physical.output_stem}): "
            f"raw_cycles={report['written_raw_cycles']} "
            f"raw_rows={report['written_raw_rows']} "
            f"features={report['written_feature_rows']} "
            f"eol={report['eol_cycle'] or 'censored'}"
        )

    # Completion order is intentionally nondeterministic under parallelism;
    # emitted CSV rows remain cell-local and deterministic.  Restore canonical
    # physical-index ordering before serialising the aggregate audit artifacts.
    completed.sort(key=lambda item: int(item[0].physical_index))
    reports = [report for _, report, _, _ in completed]
    provenance_rows = [provenance for _, _, provenance, _ in completed]
    invalid_cycles = [
        item
        for _, _, _, cell_invalid_cycles in completed
        for item in cell_invalid_cycles
    ]
    invalid_cycles.sort(
        key=lambda item: (
            str(item["battery_id"]),
            int(item["cycle_id"]),
            str(item["source_batch_date"]),
            int(item["source_cell"]),
            int(item["source_cycle"]),
        )
    )

    _write_metadata(
        args,
        args.cohort,
        batch_paths,
        all_physical_cells,
        provenance_rows,
        reports,
        invalid_cycles,
    )
    _print_summary(reports)
    print(f"raw output: {args.raw_output_dir}")
    print(f"feature output: {args.feature_output_dir}")
    print(f"provenance: {args.raw_output_dir / 'MIT_PHYSICAL_PROVENANCE.json'}")
    print(f"split: {args.split_json or args.raw_output_dir / 'mit_physical_split.json'}")


if __name__ == "__main__":
    main()
