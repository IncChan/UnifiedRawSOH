#!/usr/bin/env python3
"""Extract point-level MIT/A123 end-of-charge CC/CV records.

The MIT batch files are MATLAB v7.3/HDF5 files.  This extractor follows the
stage definitions in ``Battery-dataset-preprocessing-code-library/
MITBatteryClass.py`` while reading the raw temperature channel from
``cycles['T']`` as well.  It writes one long CSV per source cell and does not
use the legacy ``MIT data`` or ``MIT_t_v1_aligned`` files for alignment.

The default source contains 140 cells across the three MIT batch files:
46 + 48 + 46.  The default output directory is a new ``MIT_raw_t_v1``
directory so that point-level records cannot be confused with the existing
cycle-level feature tables in ``MIT_t_v1``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np


RAW_COLUMNS = [
    "cycle",
    "SOH",
    "capacity_Ah",
    "segment",
    "cycle_point_index",
    "segment_point_index",
    "source_point_index",
    "relative_time_min",
    "voltage_V",
    "current_A",
    "c_rate",
    "charge_capacity_Ah",
    "temperature_C",
    "power_W",
    "phase_policy_version",
    "phase_detection_status",
    "phase_detection_reason",
    "cc_current_reference_A",
    "cv_start_source_point_index",
    "cv_start_voltage_V",
    "cv_start_current_A",
    "cc_voltage_low_V",
    "cc_voltage_high_V",
    "cv_c_rate_low",
    "cv_c_rate_high",
]

REPORT_COLUMNS = [
    "batch_file",
    "cell",
    "policy",
    "capacity_cycle_count",
    "raw_cycle_count",
    "processed_cycle_count",
    "start_cycle",
    "written_cycles",
    "written_rows",
    "empty_cc_cycles",
    "empty_cv_cycles",
    "phase_window_rejected_cycles",
    "skipped_nonfinite_rows",
    "length_mismatch_cycles",
    "failed_cycles",
    "cycle_count_mismatch",
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "MIT_raw"


def default_input_root() -> Path | None:
    value = os.environ.get("MIT_SOURCE_ROOT")
    return Path(value).expanduser() if value else None

# Proposed/canonical raw policy. These are intentionally separate from the
# historical handcrafted-feature extractor, whose author-defined 3.4--3.595 V
# and 0.5--0.1 A windows remain unchanged.
MIT_PROPOSED_RAW_POLICY_VERSION = "mit_proposed_phase_aware_cccv_v3"
MIT_PROPOSED_CC_VOLTAGE_RANGE = (3.45, 3.60)
MIT_PROPOSED_CV_C_RATE_RANGE = (0.05, 0.25)
MIT_CC_COVERAGE_TOLERANCE_V = 0.01
MIT_CV_SELECTION_TOLERANCE_C = 0.002
# A sampled point need only land within the selection tolerance of each
# nominal endpoint. Requiring >= 0.252C while selection itself caps at 0.252C
# wrongly rejects a normal 0.2596C -> 0.2490C sample transition.
MIT_CV_HIGH_COVERAGE_C_RATE = (
    MIT_PROPOSED_CV_C_RATE_RANGE[1] - MIT_CV_SELECTION_TOLERANCE_C
)
MIT_CV_LOW_COVERAGE_C_RATE = (
    MIT_PROPOSED_CV_C_RATE_RANGE[0] + MIT_CV_SELECTION_TOLERANCE_C
)

# Keep the same evidence pattern as SmartHealth v2: a CC/CV boundary is a
# persistent current taper near the charge-voltage maximum, not a point that
# merely happens to fall inside a voltage/current window. MIT stores fewer
# points per end-of-charge trace, hence the smaller point-count minima.
MIT_PHASE_MIN_CC_POINTS = 8
MIT_PHASE_MIN_CV_POINTS = 8
MIT_CC_REFERENCE_FRACTION = 0.20
MIT_CC_REFERENCE_MIN_POINTS = 8
MIT_CC_REFERENCE_QUANTILE = 0.90
MIT_CV_TAPER_FRACTION = 0.01
MIT_CV_PERSISTENCE_POINTS = 5
MIT_CV_VOLTAGE_TOLERANCE_V = 0.02


def parse_float_pair(text: str) -> List[float]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected two comma-separated floats, got: {text}"
        )
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float pair: {text}") from exc


def parse_int_list(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            try:
                values.append(int(item))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid integer list: {text}") from exc
    return values


def decode_h5_string(dataset: h5py.Dataset) -> str:
    """Decode the UTF-16-like MATLAB string representation used in the files."""

    raw = dataset[()]
    if isinstance(raw, bytes):
        return raw.decode(errors="ignore").strip("\x00")

    array = np.asarray(raw)
    if array.dtype.kind in {"S", "U"}:
        return "".join(str(item) for item in array.reshape(-1)).strip("\x00")
    if hasattr(raw, "tobytes"):
        return raw.tobytes()[::2].decode(errors="ignore").strip("\x00")
    return str(raw)


def select_proposed_cccv_windows(
    data: Dict[str, np.ndarray],
    cccv_indices: np.ndarray,
    nominal_capacity_ah: float,
) -> Dict[str, object]:
    """Infer phase first, then select the fixed proposed MIT raw windows.

    The historical MIT archive has no separate CC/CV step label at this
    point.  Therefore a voltage or current predicate alone cannot decide a
    phase.  This follows the SmartHealth-v2-style persistent-taper decision
    inside the end-of-charge trace and only then applies the CC voltage and
    nominal-C-rate CV windows.
    """

    if nominal_capacity_ah <= 0:
        raise ValueError("nominal_capacity_ah must be positive")

    empty = np.empty(0, dtype=int)
    metadata: Dict[str, object] = {
        "phase_policy_version": MIT_PROPOSED_RAW_POLICY_VERSION,
        "phase_detection_status": "invalid",
        "phase_detection_reason": "",
        "selection_reason": "",
        "selection_eligible": False,
        "cc_current_reference_A": math.nan,
        "cv_start_source_point_index": None,
        "cv_start_voltage_V": math.nan,
        "cv_start_current_A": math.nan,
        "cc_window_lower_covered": False,
        "cc_window_upper_covered": False,
        "cc_window_complete": False,
        "cv_window_high_covered": False,
        "cv_window_low_covered": False,
        "cv_window_complete": False,
    }
    result: Dict[str, object] = {"CC": empty, "CV": empty, "metadata": metadata}

    indices = np.asarray(cccv_indices, dtype=int).reshape(-1)
    if indices.size == 0:
        metadata["phase_detection_reason"] = "no_end_of_charge_trace"
        return result
    source_length = int(np.asarray(data["current_A"]).size)
    if np.any(indices < 0) or np.any(indices >= source_length):
        metadata["phase_detection_reason"] = "invalid_end_of_charge_indices"
        return result
    if indices.size < MIT_PHASE_MIN_CC_POINTS + MIT_PHASE_MIN_CV_POINTS:
        metadata["phase_detection_reason"] = "end_of_charge_trace_too_short"
        return result

    current = np.abs(np.asarray(data["current_A"], dtype=float)[indices])
    voltage = np.asarray(data["voltage_V"], dtype=float)[indices]
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(voltage)):
        metadata["phase_detection_reason"] = "nonfinite_charge_current_or_voltage"
        return result
    if not np.all(current > 0):
        metadata["phase_detection_reason"] = "nonpositive_charge_current"
        return result

    reference_count = min(
        indices.size,
        max(
            MIT_CC_REFERENCE_MIN_POINTS,
            int(math.ceil(MIT_CC_REFERENCE_FRACTION * indices.size)),
        ),
    )
    if reference_count < MIT_PHASE_MIN_CC_POINTS:
        metadata["phase_detection_reason"] = "insufficient_cc_reference_points"
        return result
    cc_reference = float(
        np.quantile(current[:reference_count], MIT_CC_REFERENCE_QUANTILE)
    )
    if not math.isfinite(cc_reference) or cc_reference <= 0:
        metadata["phase_detection_reason"] = "invalid_cc_current_reference"
        return result

    taper = current <= cc_reference * (1.0 - MIT_CV_TAPER_FRACTION)
    voltage_max = float(np.max(voltage))
    boundary: Optional[int] = None
    for index in range(
        MIT_PHASE_MIN_CC_POINTS,
        indices.size - MIT_PHASE_MIN_CV_POINTS + 1,
    ):
        if index + MIT_CV_PERSISTENCE_POINTS > indices.size:
            break
        if not np.all(taper[index : index + MIT_CV_PERSISTENCE_POINTS]):
            continue
        if voltage[index] < voltage_max - MIT_CV_VOLTAGE_TOLERANCE_V:
            continue
        boundary = index
        break

    metadata["cc_current_reference_A"] = cc_reference
    if boundary is None:
        metadata["phase_detection_reason"] = "no_persistent_taper_near_charge_voltage_max"
        return result

    inferred_cc = indices[:boundary]
    inferred_cv = indices[boundary:]
    if (
        inferred_cc.size < MIT_PHASE_MIN_CC_POINTS
        or inferred_cv.size < MIT_PHASE_MIN_CV_POINTS
    ):
        metadata["phase_detection_reason"] = "phase_point_count_below_minimum"
        return result

    metadata.update(
        {
            "phase_detection_status": "ok",
            "phase_detection_reason": "persistent_current_taper_near_charge_voltage_max",
            "cv_start_source_point_index": int(indices[boundary]),
            "cv_start_voltage_V": float(voltage[boundary]),
            "cv_start_current_A": float(current[boundary]),
        }
    )

    cc_low, cc_high = MIT_PROPOSED_CC_VOLTAGE_RANGE
    selected_cc = inferred_cc[
        (data["voltage_V"][inferred_cc] >= cc_low)
        & (data["voltage_V"][inferred_cc] <= cc_high)
    ]
    cv_low, cv_high = MIT_PROPOSED_CV_C_RATE_RANGE
    cv_rates = np.abs(data["current_A"][inferred_cv]) / float(nominal_capacity_ah)
    selected_cv = inferred_cv[
        (cv_rates >= cv_low - MIT_CV_SELECTION_TOLERANCE_C)
        & (cv_rates <= cv_high + MIT_CV_SELECTION_TOLERANCE_C)
    ]

    if selected_cc.size:
        selected_cc_voltage = np.asarray(data["voltage_V"][selected_cc], dtype=float)
        metadata["cc_window_lower_covered"] = bool(
            float(np.min(selected_cc_voltage)) <= cc_low + MIT_CC_COVERAGE_TOLERANCE_V
        )
        metadata["cc_window_upper_covered"] = bool(
            float(np.max(selected_cc_voltage)) >= cc_high - MIT_CC_COVERAGE_TOLERANCE_V
        )
    metadata["cc_window_complete"] = bool(
        metadata["cc_window_lower_covered"] and metadata["cc_window_upper_covered"]
    )
    if selected_cv.size:
        selected_cv_rates = np.abs(data["current_A"][selected_cv]) / float(
            nominal_capacity_ah
        )
        metadata["cv_window_high_covered"] = bool(
            float(np.max(selected_cv_rates)) >= MIT_CV_HIGH_COVERAGE_C_RATE
        )
        metadata["cv_window_low_covered"] = bool(
            float(np.min(selected_cv_rates)) <= MIT_CV_LOW_COVERAGE_C_RATE
        )
    metadata["cv_window_complete"] = bool(
        metadata["cv_window_high_covered"] and metadata["cv_window_low_covered"]
    )
    if not metadata["cc_window_complete"]:
        metadata["selection_reason"] = "incomplete_selected_cc_voltage_window"
        return result
    if not metadata["cv_window_complete"]:
        metadata["selection_reason"] = "incomplete_selected_cv_c_rate_window"
        return result

    metadata["selection_eligible"] = True
    metadata["selection_reason"] = "phase_aware_cccv_windows_complete"
    result["CC"] = selected_cc
    result["CV"] = selected_cv
    return result


class MITRawBatch:
    """Small HDF5 reader containing the MITBatteryClass stage semantics."""

    RAW_FIELDS = {
        "current_A": "I",
        "voltage_V": "V",
        "charge_capacity_Ah": "Qc",
        "discharge_capacity_Ah": "Qd",
        "time_min": "t",
        "temperature_C": "T",
    }

    def __init__(self, path: Path):
        self.path = path
        self.file = h5py.File(path, "r")
        if "batch" not in self.file:
            raise ValueError(f"MIT batch group is missing: {path}")
        self.batch = self.file["batch"]
        self.num_cells = int(self.batch["summary"].shape[0])

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "MITRawBatch":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _ref_array(self, ref: object) -> np.ndarray:
        return np.asarray(self.file[ref][()]).reshape(-1).astype(float, copy=False)

    def _ref_scalar(self, ref: object) -> float:
        values = self._ref_array(ref)
        if values.size == 0:
            return np.nan
        return float(values[0])

    def _summary_group(self, cell_index: int):
        return self.file[self.batch["summary"][cell_index, 0]]

    def _cycles_group(self, cell_index: int):
        return self.file[self.batch["cycles"][cell_index, 0]]

    def policy(self, cell_index: int) -> str:
        return decode_h5_string(self.file[self.batch["policy_readable"][cell_index, 0]])

    def capacity_series(self, cell_index: int) -> np.ndarray:
        """Return QDischarge for real cycles, excluding MATLAB's dummy row."""

        summary = self._summary_group(cell_index)
        values = np.asarray(summary["QDischarge"][()]).reshape(-1)
        if values.dtype.kind == "O":
            values = np.asarray(
                [self._ref_scalar(ref) for ref in values[1:]], dtype=float
            )
        else:
            values = values[1:].astype(float, copy=False)
        return values

    def cycle_counts(self, cell_index: int) -> Tuple[int, int]:
        """Return ``(capacity_count, raw_cycle_count)``.

        ``batch/cycle_life`` is not used because it is inconsistent with the
        actual cycle groups in some of the supplied MIT batches.
        """

        capacity_count = int(self.capacity_series(cell_index).size)
        cycles = self._cycles_group(cell_index)
        raw_cycle_count = int(cycles["I"].shape[0] - 1)
        return capacity_count, raw_cycle_count

    def cycle_data(
        self, cell_index: int, cycle: int
    ) -> Tuple[Dict[str, np.ndarray], bool]:
        """Read one real cycle and return data plus a length-mismatch flag."""

        cycles = self._cycles_group(cell_index)
        if cycle < 1 or cycle >= cycles["I"].shape[0]:
            raise ValueError(
                f"cycle should be in [1,{cycles['I'].shape[0] - 1}], got {cycle}"
            )

        data: Dict[str, np.ndarray] = {}
        for output_name, source_name in self.RAW_FIELDS.items():
            if source_name not in cycles:
                raise ValueError(
                    f"{self.path.name} cell {cell_index + 1} has no cycles['{source_name}']"
                )
            data[output_name] = self._ref_array(cycles[source_name][cycle, 0])

        lengths = {values.size for values in data.values()}
        length_mismatch = len(lengths) > 1
        if length_mismatch:
            n = min(lengths)
            data = {key: values[:n] for key, values in data.items()}
        return data, length_mismatch

    @staticmethod
    def _first_contiguous_block(indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return indices
        breaks = np.flatnonzero(np.diff(indices) != 1)
        if breaks.size == 0:
            return indices
        return indices[: breaks[0] + 1]

    def charge_indices(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """MITBatteryClass: current > -0.1, then first contiguous block."""

        indices = np.flatnonzero(data["current_A"] > -1e-1)
        return self._first_contiguous_block(indices)

    def cccv_indices(
        self, data: Dict[str, np.ndarray], charge_indices: np.ndarray
    ) -> np.ndarray:
        """MITBatteryClass: Qc >= 79% max and current > 0.01."""

        if charge_indices.size == 0:
            return charge_indices

        q = data["charge_capacity_Ah"][charge_indices]
        current = data["current_A"][charge_indices]
        indices = np.flatnonzero(
            (q >= 0.79 * np.max(q)) & (current > 0.01)
        )
        if indices.size == 0:
            return indices

        # This follows get_one_battery_one_cycle_CCCV_stage exactly: if the
        # candidate indices contain a gap, keep the part after the first gap.
        breaks = np.flatnonzero(np.diff(indices) != 1)
        if breaks.size > 0:
            indices = indices[breaks[0] + 1 :]
        return charge_indices[indices]

    def end_of_charge_indices(
        self,
        data: Dict[str, np.ndarray],
        nominal_capacity_ah: float,
    ) -> Dict[str, object]:
        charge = self.charge_indices(data)
        cccv = self.cccv_indices(data, charge)
        return select_proposed_cccv_windows(data, cccv, nominal_capacity_ah)


def row_has_nonfinite(row: Dict[str, object]) -> bool:
    numeric_columns = [
        "SOH",
        "capacity_Ah",
        "relative_time_min",
        "voltage_V",
        "current_A",
        "c_rate",
        "charge_capacity_Ah",
        "temperature_C",
        "power_W",
    ]
    return any(not np.isfinite(float(row[column])) for column in numeric_columns)


def find_all_batch_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*batchdata*_struct*.mat"))


def output_name(batch_path: Path, cell_index: int) -> str:
    return f"MIT_{batch_path.stem}_cell-{cell_index + 1:03d}.csv"


def selected_cells(args: argparse.Namespace, batch: MITRawBatch) -> List[int]:
    if args.cells is None:
        return list(range(batch.num_cells))

    if not args.cells:
        raise ValueError("--cells must contain at least one 1-based cell index")
    if len(set(args.cells)) != len(args.cells):
        raise ValueError(f"--cells contains duplicates: {args.cells}")

    selected = []
    for cell_number in args.cells:
        if cell_number < 1 or cell_number > batch.num_cells:
            raise ValueError(
                f"cell {cell_number} is outside [1,{batch.num_cells}] for {batch.path.name}"
            )
        selected.append(cell_number - 1)
    return selected


def make_row(
    cycle: int,
    soh: float,
    capacity_ah: float,
    segment: str,
    cycle_point_index: int,
    segment_point_index: int,
    source_point_index: int,
    data: Dict[str, np.ndarray],
    nominal_capacity_ah: float,
    phase_metadata: Mapping[str, object],
) -> Dict[str, object]:
    voltage = float(data["voltage_V"][source_point_index])
    current = float(data["current_A"][source_point_index])
    return {
        "cycle": cycle,
        "SOH": soh,
        "capacity_Ah": capacity_ah,
        "segment": segment,
        "cycle_point_index": cycle_point_index,
        "segment_point_index": segment_point_index,
        "source_point_index": source_point_index,
        "relative_time_min": float(data["time_min"][source_point_index]),
        "voltage_V": voltage,
        "current_A": current,
        "c_rate": abs(current) / float(nominal_capacity_ah),
        "charge_capacity_Ah": float(
            data["charge_capacity_Ah"][source_point_index]
        ),
        "temperature_C": float(data["temperature_C"][source_point_index]),
        "power_W": voltage * current,
        "phase_policy_version": phase_metadata["phase_policy_version"],
        "phase_detection_status": phase_metadata["phase_detection_status"],
        "phase_detection_reason": phase_metadata["phase_detection_reason"],
        "cc_current_reference_A": phase_metadata["cc_current_reference_A"],
        "cv_start_source_point_index": phase_metadata[
            "cv_start_source_point_index"
        ],
        "cv_start_voltage_V": phase_metadata["cv_start_voltage_V"],
        "cv_start_current_A": phase_metadata["cv_start_current_A"],
        "cc_voltage_low_V": MIT_PROPOSED_CC_VOLTAGE_RANGE[0],
        "cc_voltage_high_V": MIT_PROPOSED_CC_VOLTAGE_RANGE[1],
        "cv_c_rate_low": MIT_PROPOSED_CV_C_RATE_RANGE[0],
        "cv_c_rate_high": MIT_PROPOSED_CV_C_RATE_RANGE[1],
    }


def extract_cycle_rows(
    batch: MITRawBatch,
    cell_index: int,
    cycle: int,
    soh: float,
    capacity_ah: float,
    nominal_capacity_ah: float,
) -> Tuple[List[Dict[str, object]], bool, bool, bool, bool]:
    data, length_mismatch = batch.cycle_data(cell_index, cycle)
    stages = batch.end_of_charge_indices(
        data,
        nominal_capacity_ah=nominal_capacity_ah,
    )
    cc_indices = np.asarray(stages["CC"], dtype=int)
    cv_indices = np.asarray(stages["CV"], dtype=int)
    phase_metadata = dict(stages["metadata"])
    phase_window_rejected = not bool(phase_metadata["selection_eligible"])
    if phase_window_rejected:
        return [], cc_indices.size == 0, cv_indices.size == 0, length_mismatch, True

    rows: List[Dict[str, object]] = []
    cycle_point_index = 0
    for segment in ("CC", "CV"):
        indices = cc_indices if segment == "CC" else cv_indices
        for segment_point_index, source_point_index in enumerate(indices.tolist()):
            rows.append(
                make_row(
                    cycle=cycle,
                    soh=soh,
                    capacity_ah=capacity_ah,
                    segment=segment,
                    cycle_point_index=cycle_point_index,
                    segment_point_index=segment_point_index,
                    source_point_index=source_point_index,
                    data=data,
                    nominal_capacity_ah=nominal_capacity_ah,
                    phase_metadata=phase_metadata,
                )
            )
            cycle_point_index += 1

    return rows, cc_indices.size == 0, cv_indices.size == 0, length_mismatch, False


def write_csv_header(writer: csv.DictWriter) -> None:
    writer.writeheader()


def write_report(rows: List[Dict[str, object]], report_csv: Path) -> None:
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def process_one_cell(
    args: argparse.Namespace,
    batch_path: Path,
    batch: MITRawBatch,
    cell_index: int,
) -> Dict[str, object]:
    capacity_series = batch.capacity_series(cell_index)
    capacity_count, raw_cycle_count = batch.cycle_counts(cell_index)
    cycle_count_mismatch = capacity_count != raw_cycle_count
    if cycle_count_mismatch and args.strict:
        raise ValueError(
            f"cycle count mismatch in {batch_path.name} cell {cell_index + 1}: "
            f"capacity={capacity_count}, raw={raw_cycle_count}"
        )

    process_cycle_count = min(capacity_count, raw_cycle_count)
    end_cycle = process_cycle_count
    if args.max_cycles is not None:
        end_cycle = min(end_cycle, args.start_cycle + args.max_cycles - 1)

    output_csv = args.output_dir / output_name(batch_path, cell_index)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    capacity_ah_for_label = capacity_series
    written_cycles = 0
    written_rows = 0
    empty_cc_cycles = 0
    empty_cv_cycles = 0
    phase_window_rejected_cycles = 0
    skipped_nonfinite_rows = 0
    length_mismatch_cycles = 0
    failed_cycles = 0

    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        write_csv_header(writer)

        for cycle in range(args.start_cycle, end_cycle + 1):
            capacity_ah = float(capacity_ah_for_label[cycle - 1])
            if args.label_mode == "capacity_to_soh":
                soh = capacity_ah / args.nominal_capacity
            else:
                soh = capacity_ah

            try:
                (
                    cycle_rows,
                    empty_cc,
                    empty_cv,
                    length_mismatch,
                    phase_window_rejected,
                ) = extract_cycle_rows(
                    batch=batch,
                    cell_index=cell_index,
                    cycle=cycle,
                    soh=soh,
                    capacity_ah=capacity_ah,
                    nominal_capacity_ah=args.nominal_capacity,
                )
                if length_mismatch:
                    length_mismatch_cycles += 1
                    if args.strict:
                        raise ValueError("raw channel lengths do not match")
            except Exception as exc:
                failed_cycles += 1
                if args.strict:
                    raise RuntimeError(
                        f"failed extracting {batch_path.name} cell {cell_index + 1} cycle {cycle}"
                    ) from exc
                continue

            empty_cc_cycles += int(empty_cc)
            empty_cv_cycles += int(empty_cv)
            phase_window_rejected_cycles += int(phase_window_rejected)

            finite_rows = []
            for row in cycle_rows:
                if args.drop_nonfinite_rows and row_has_nonfinite(row):
                    skipped_nonfinite_rows += 1
                else:
                    finite_rows.append(row)

            if finite_rows:
                writer.writerows(finite_rows)
                written_cycles += 1
                written_rows += len(finite_rows)

    return {
        "batch_file": batch_path.name,
        "cell": cell_index + 1,
        "policy": batch.policy(cell_index),
        "capacity_cycle_count": capacity_count,
        "raw_cycle_count": raw_cycle_count,
        "processed_cycle_count": process_cycle_count,
        "start_cycle": args.start_cycle,
        "written_cycles": written_cycles,
        "written_rows": written_rows,
        "empty_cc_cycles": empty_cc_cycles,
        "empty_cv_cycles": empty_cv_cycles,
        "phase_window_rejected_cycles": phase_window_rejected_cycles,
        "skipped_nonfinite_rows": skipped_nonfinite_rows,
        "length_mismatch_cycles": length_mismatch_cycles,
        "failed_cycles": failed_cycles,
        "cycle_count_mismatch": int(cycle_count_mismatch),
    }


def print_summary(report_rows: List[Dict[str, object]]) -> None:
    grouped = defaultdict(Counter)
    for row in report_rows:
        group = grouped[str(row["batch_file"])]
        group["cells"] += 1
        for key in REPORT_COLUMNS:
            if key in {"batch_file", "policy"}:
                continue
            group[key] += int(row[key])

    print(
        "batch_file,cells,written_cycles,written_rows,empty_cc_cycles,"
        "empty_cv_cycles,phase_window_rejected_cycles,skipped_nonfinite_rows,length_mismatch_cycles,"
        "failed_cycles,cycle_count_mismatch"
    )
    for batch_file in sorted(grouped):
        group = grouped[batch_file]
        print(
            f"{batch_file},{group['cells']},{group['written_cycles']},"
            f"{group['written_rows']},{group['empty_cc_cycles']},"
            f"{group['empty_cv_cycles']},{group['phase_window_rejected_cycles']},"
            f"{group['skipped_nonfinite_rows']},"
            f"{group['length_mismatch_cycles']},{group['failed_cycles']},"
            f"{group['cycle_count_mismatch']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract phase-aware proposed MIT/A123 CC/CV raw records: inferred "
            "CC 3.45--3.60 V and inferred-CV nominal 0.25C--0.05C."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--start-cycle", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument(
        "--cells",
        type=parse_int_list,
        default=None,
        help="optional 1-based cell list per batch, e.g. 1,2,8; default is all cells",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--drop-nonfinite-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--nominal-capacity", type=float, default=1.1)
    parser.add_argument(
        "--label-mode",
        choices=("capacity_to_soh", "capacity"),
        default="capacity_to_soh",
        help="write SOH as QDischarge/nominal capacity or raw QDischarge",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.input_root is None:
        parser.error("--input-root is required (or set MIT_SOURCE_ROOT)")
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.report_csv is None:
        args.report_csv = args.output_dir / "mit_raw_extraction_report.csv"
    else:
        args.report_csv = args.report_csv.expanduser().resolve()

    if args.start_cycle < 1:
        parser.error("--start-cycle must be at least 1")
    if args.max_cycles is not None and args.max_cycles < 1:
        parser.error("--max-cycles must be at least 1")
    if args.nominal_capacity <= 0:
        parser.error("--nominal-capacity must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {args.input_root}")

    batch_files = find_all_batch_files(args.input_root)
    if args.max_batches is not None:
        batch_files = batch_files[: args.max_batches]
    if not batch_files:
        raise FileNotFoundError(f"no MIT batch .mat files found under: {args.input_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_rows: List[Dict[str, object]] = []
    print(f"Found {len(batch_files)} MIT batch files under {args.input_root}")

    for batch_path in batch_files:
        with MITRawBatch(batch_path) as batch:
            cells = selected_cells(args, batch)
            print(f"{batch_path.name}: {batch.num_cells} cells; processing {len(cells)}")
            for cell_index in cells:
                report = process_one_cell(args, batch_path, batch, cell_index)
                report_rows.append(report)
                print(
                    f"OK: {batch_path.name} cell {report['cell']:03d} -> "
                    f"{args.output_dir / output_name(batch_path, cell_index)} "
                    f"(cycles={report['written_cycles']}, rows={report['written_rows']}, "
                    f"failed={report['failed_cycles']})"
                )

    write_report(report_rows, args.report_csv)
    print_summary(report_rows)
    print(f"wrote raw CSVs: {args.output_dir}")
    print(f"wrote report: {args.report_csv}")
    print(f"battery files written: {len(report_rows)}")


if __name__ == "__main__":
    main()
