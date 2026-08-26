"""Adapters for the MIT feature and point-level raw sources.

The feature adapter remains intentionally separate from the raw adapter.  The
raw adapter reads the point-level CSVs produced by the MIT extractors and
emits the same cycle contract as the XJTU adapter; it never fabricates a
sequence or falls back to an aligned feature table.  Legacy v1 source-file
CSVs remain readable, while the canonical physical-cell product uses
physical-cell IDs and global physical cycles.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np

from .base import RawTerminalSignalUnavailable, UNIFIED_SAMPLE_KEYS
from .splits import resolve_test_batteries


MIT_REQUIRED_INTERMEDIATE_COLUMNS = {
    "CC Q",
    "CC charge time",
    "CV Q",
    "CV charge time",
    "capacity",
    "T_CC_mean",
    "T_CC_max",
    "T_CC_delta",
    "T_CC_slope",
    "T_CV_mean",
    "T_CV_max",
    "T_CV_delta",
    "T_CV_slope",
}
MIT_DATE_RE = re.compile(r"MIT_(?P<date>\d{4}-\d{2}-\d{2})_.*cell-(?P<cell>\d+)\.csv$")
MIT_PHYSICAL_RE = re.compile(
    r"MIT_(?P<date>\d{4}-\d{2}-\d{2})_physical-(?P<physical>\d+)\.csv$"
)
MIT_PHYSICAL_ID_RE = re.compile(r"^mit_p\d{3}$")
MIT_RAW_REQUIRED_COLUMNS = {
    "cycle",
    "SOH",
    "capacity_Ah",
    "segment",
    "relative_time_min",
    "voltage_V",
    "current_A",
    "temperature_C",
}
MIT_PHYSICAL_RAW_PROVENANCE_COLUMNS = {
    "physical_cell_id",
    "primary_batch_date",
    "source_batch_date",
    "source_cell",
    "source_cycle",
}
MIT_PROPOSED_RAW_PHASE_POLICY_VERSION = "mit_proposed_phase_aware_cccv_v3"
MIT_PROPOSED_RAW_REQUIRED_COLUMNS = {
    "c_rate",
    "phase_policy_version",
    "phase_detection_status",
    "phase_detection_reason",
    "cc_voltage_low_V",
    "cc_voltage_high_V",
    "cv_c_rate_low",
    "cv_c_rate_high",
}
MIT_PROPOSED_CC_VOLTAGE_RANGE = (3.45, 3.60)
MIT_PROPOSED_CV_C_RATE_RANGE = (0.05, 0.25)
MIT_CC_COVERAGE_TOLERANCE_V = 0.01
MIT_CV_SELECTION_TOLERANCE_C = 0.002
MIT_CV_HIGH_COVERAGE_C_RATE = (
    MIT_PROPOSED_CV_C_RATE_RANGE[1] - MIT_CV_SELECTION_TOLERANCE_C
)
MIT_CV_LOW_COVERAGE_C_RATE = (
    MIT_PROPOSED_CV_C_RATE_RANGE[0] + MIT_CV_SELECTION_TOLERANCE_C
)


def parse_mit_file_identity(path):
    """Return ``(condition, battery_id, is_physical)`` from a MIT CSV name.

    The physical filename keeps the primary batch date only for convenient
    batch filtering.  Its actual identity is ``mit_p###`` and is also checked
    against the CSV provenance column by the raw adapter.
    """

    path = Path(path)
    physical_match = MIT_PHYSICAL_RE.match(path.name)
    if physical_match is not None:
        return (
            physical_match.group("date"),
            f"mit_p{int(physical_match.group('physical')):03d}",
            True,
        )
    source_match = MIT_DATE_RE.match(path.name)
    if source_match is None:
        raise ValueError(f"Cannot parse MIT CSV filename: {path.name}")
    return (
        source_match.group("date"),
        f"{source_match.group('date')}_battery-{int(source_match.group('cell'))}",
        False,
    )


def _is_mit_battery_csv(path):
    """Whether ``path`` is a data-bearing MIT battery CSV, not audit metadata."""

    name = Path(path).name
    return MIT_PHYSICAL_RE.match(name) is not None or MIT_DATE_RE.match(name) is not None


def list_mit_raw_files(data_root, batch=None):
    root = Path(data_root)
    if not root.is_dir():
        raise ValueError(f"MIT raw data root does not exist: {root}")
    files = []
    for path in sorted(root.glob("*.csv")):
        # The physical124 export intentionally stores provenance/report CSVs
        # beside battery files.  Never pass those metadata tables to the raw
        # adapter merely because they have a .csv suffix.
        if not _is_mit_battery_csv(path):
            continue
        if batch is not None and not path.name.startswith(f"MIT_{batch}_"):
            continue
        files.append(path)
    if not files:
        raise ValueError(f"No MIT raw CSV files found under {root} for batch={batch!r}")
    return files


def inspect_mit_raw_inventory(data_root, batch=None):
    """Return a cheap, no-training readiness audit for canonical MIT raw CSVs.

    A header-only physical export is not a valid raw source.  Detect it before
    the launcher creates one failed process per seed and never substitute the
    historical v2/raw-aligned products as a fallback.
    """

    files = list_mit_raw_files(data_root, batch=batch)
    header_only = []
    missing_headers = {}
    nonempty = []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            _, _, is_physical_file = parse_mit_file_identity(path)
            required = set(MIT_RAW_REQUIRED_COLUMNS)
            if is_physical_file:
                required.update(MIT_PHYSICAL_RAW_PROVENANCE_COLUMNS)
                required.update(MIT_PROPOSED_RAW_REQUIRED_COLUMNS)
            missing = sorted(required - fields)
            has_data = next(reader, None) is not None
            if not has_data:
                header_only.append(path.name)
            if missing:
                missing_headers[path.name] = missing
                continue
            if has_data:
                nonempty.append(path.name)
    return {
        "files": len(files),
        "nonempty_files": nonempty,
        "header_only_files": header_only,
        "missing_required_headers": missing_headers,
    }


def validate_mit_physical_cohort(
    observed_battery_ids,
    split_spec,
    *,
    require_full_physical_cohort=False,
):
    """Validate the physical-cell cohort required by an MIT experiment JSON.

    This is deliberately independent of filesystem layout so both the raw
    loader and the statistical-feature loader enforce the same split cohort.
    The official Paper E1 configs request all 124 canonical physical cells;
    a separately declared debugging subset can opt out and supply its own
    split JSON without changing the official protocol.
    """

    observed = {str(item).strip() for item in observed_battery_ids if str(item).strip()}
    if not observed:
        raise ValueError("MIT cohort validation received no observed physical cells")
    invalid = sorted(item for item in observed if MIT_PHYSICAL_ID_RE.match(item) is None)
    if invalid:
        raise ValueError(
            "MIT cohort contains non-canonical physical IDs: " + ", ".join(invalid[:5])
        )
    if require_full_physical_cohort:
        declared_count = split_spec.get("physical_cell_count")
        if declared_count is None:
            raise ValueError(
                "MIT config requires a full physical cohort but its split JSON has no "
                "physical_cell_count"
            )
        declared_count = int(declared_count)
        expected = {f"mit_p{index:03d}" for index in range(1, declared_count + 1)}
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                f"MIT physical cohort does not match declared Paper-{declared_count}: "
                f"found={len(observed)}, missing={missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
                + f", unexpected={unexpected[:5]}"
                + (" ..." if len(unexpected) > 5 else "")
            )
    test_batteries = set(
        resolve_test_batteries(split_spec, observed_battery_ids=observed)
    )
    development_batteries = observed - test_batteries
    if not development_batteries:
        raise ValueError("MIT split leaves no physical cells for train/validation")
    return {
        "physical_ids": observed,
        "test_ids": test_batteries,
        "development_ids": development_batteries,
    }


def _normalize_mit_soh(raw_soh, nominal_capacity, label_scale_mode):
    mode = str(label_scale_mode or "auto_capacity_to_soh")
    nominal_capacity = float(nominal_capacity)
    if nominal_capacity <= 0:
        raise ValueError("nominal_capacity must be positive")
    if mode == "none":
        return float(raw_soh), 1.0, mode
    if mode == "capacity_to_soh":
        return float(raw_soh) / nominal_capacity, nominal_capacity, mode
    if mode == "auto_capacity_to_soh":
        if abs(float(raw_soh)) > 1.2:
            return float(raw_soh) / nominal_capacity, nominal_capacity, "auto_capacity_to_soh_applied"
        return float(raw_soh), 1.0, "auto_capacity_to_soh_noop"
    raise ValueError(f"Unsupported label_scale_mode: {mode}")


def read_mit_raw_file(path, nominal_capacity=1.1, label_scale_mode="none"):
    """Read one extracted MIT point-level CSV into raw cycle records."""

    path = Path(path)
    nominal_capacity = float(nominal_capacity)
    if nominal_capacity <= 0:
        raise ValueError("nominal_capacity must be positive")
    filename_condition, filename_battery_id, is_physical_file = parse_mit_file_identity(path)
    grouped = {}
    cycle_order = []
    cycle_provenance = {}
    physical_ids = set()
    primary_batch_dates = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(MIT_RAW_REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} is missing MIT raw columns: {missing}")
        if is_physical_file:
            missing = sorted(MIT_PHYSICAL_RAW_PROVENANCE_COLUMNS - set(reader.fieldnames))
            if missing:
                raise ValueError(
                    f"Canonical physical MIT raw file {path} is missing provenance columns: {missing}"
                )
            missing = sorted(MIT_PROPOSED_RAW_REQUIRED_COLUMNS - set(reader.fieldnames))
            if missing:
                raise ValueError(
                    f"Canonical proposed MIT raw file {path} is missing phase-policy columns: {missing}"
                )
        for row in reader:
            try:
                cycle_id = int(float(row["cycle"]))
                segment = str(row["segment"]).strip().upper()
                current = float(row["current_A"])
                recorded_c_rate = (
                    float(row["c_rate"])
                    if is_physical_file
                    else abs(current) / nominal_capacity
                )
                item = (
                    segment,
                    float(row["relative_time_min"]),
                    float(row["voltage_V"]),
                    current,
                    float(row["temperature_C"]),
                    float(row["SOH"]),
                    recorded_c_rate,
                    float(row["capacity_Ah"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric row in {path}: {row}") from exc
            if segment not in {"CC", "CV"}:
                continue
            if is_physical_file:
                if row["phase_policy_version"].strip() != MIT_PROPOSED_RAW_PHASE_POLICY_VERSION:
                    raise ValueError(
                        f"MIT phase-policy version mismatch in {path}: "
                        f"{row['phase_policy_version']!r}"
                    )
                if row["phase_detection_status"].strip() != "ok":
                    raise ValueError(
                        f"MIT canonical row lacks a successful phase decision in {path}: {row}"
                    )
                if not math.isclose(
                    recorded_c_rate,
                    abs(current) / nominal_capacity,
                    rel_tol=1e-7,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"MIT c_rate is not nominal-capacity normalized in {path}: {row}"
                    )
                if not math.isclose(
                    float(row["cc_voltage_low_V"]),
                    MIT_PROPOSED_CC_VOLTAGE_RANGE[0],
                    abs_tol=1e-9,
                ) or not math.isclose(
                    float(row["cc_voltage_high_V"]),
                    MIT_PROPOSED_CC_VOLTAGE_RANGE[1],
                    abs_tol=1e-9,
                ) or not math.isclose(
                    float(row["cv_c_rate_low"]),
                    MIT_PROPOSED_CV_C_RATE_RANGE[0],
                    abs_tol=1e-9,
                ) or not math.isclose(
                    float(row["cv_c_rate_high"]),
                    MIT_PROPOSED_CV_C_RATE_RANGE[1],
                    abs_tol=1e-9,
                ):
                    raise ValueError(f"MIT canonical window metadata mismatch in {path}: {row}")
                if segment == "CC" and not (
                    MIT_PROPOSED_CC_VOLTAGE_RANGE[0] - 1e-9
                    <= item[2]
                    <= MIT_PROPOSED_CC_VOLTAGE_RANGE[1] + 1e-9
                ):
                    raise ValueError(f"MIT CC point outside proposed window in {path}: {row}")
                if segment == "CV" and not (
                    MIT_PROPOSED_CV_C_RATE_RANGE[0] - MIT_CV_SELECTION_TOLERANCE_C - 1e-9
                    <= recorded_c_rate
                    <= MIT_PROPOSED_CV_C_RATE_RANGE[1] + MIT_CV_SELECTION_TOLERANCE_C + 1e-9
                ):
                    raise ValueError(f"MIT CV point outside proposed C-rate window in {path}: {row}")
            if cycle_id not in grouped:
                grouped[cycle_id] = []
                cycle_order.append(cycle_id)
            grouped[cycle_id].append(item)
            if is_physical_file:
                physical_id = str(row["physical_cell_id"]).strip()
                primary_batch_date = str(row["primary_batch_date"]).strip()
                if not physical_id or not primary_batch_date:
                    raise ValueError(f"Missing physical identity metadata in {path}: {row}")
                physical_ids.add(physical_id)
                primary_batch_dates.add(primary_batch_date)
                provenance = {
                    "physical_cell_id": physical_id,
                    "source_batch_date": str(row["source_batch_date"]).strip(),
                    "source_cell": int(float(row["source_cell"])),
                    "source_cycle": int(float(row["source_cycle"])),
                }
                previous = cycle_provenance.setdefault(cycle_id, provenance)
                if previous != provenance:
                    raise ValueError(
                        f"Physical provenance is not constant within cycle {cycle_id} in {path}"
                    )

    condition = filename_condition
    battery_id = filename_battery_id
    if is_physical_file:
        if len(physical_ids) != 1:
            raise ValueError(f"Expected one physical_cell_id in {path}, got {sorted(physical_ids)}")
        if len(primary_batch_dates) != 1:
            raise ValueError(
                f"Expected one primary_batch_date in {path}, got {sorted(primary_batch_dates)}"
            )
        battery_id = next(iter(physical_ids))
        condition = next(iter(primary_batch_dates))
        if battery_id != filename_battery_id:
            raise ValueError(
                f"Physical CSV filename/id mismatch in {path}: "
                f"filename={filename_battery_id}, column={battery_id}"
            )
        if condition != filename_condition:
            raise ValueError(
                f"Physical CSV filename/primary batch mismatch in {path}: "
                f"filename={filename_condition}, column={condition}"
            )
    records = []
    for raw_order_index, cycle_id in enumerate(cycle_order):
        rows = sorted(grouped[cycle_id], key=lambda item: item[1])
        segments = np.asarray([item[0] for item in rows], dtype=object)
        if is_physical_file:
            if segments.size == 0 or segments[0] != "CC" or "CV" not in segments:
                raise ValueError(f"MIT canonical cycle lacks a contiguous CC/CV pair: {path}, {cycle_id}")
            first_cv = int(np.flatnonzero(segments == "CV")[0])
            if np.any(segments[:first_cv] != "CC") or np.any(segments[first_cv:] != "CV"):
                raise ValueError(
                    f"MIT canonical cycle has non-contiguous phases: {path}, {cycle_id}"
                )
            cc_voltage = np.asarray([item[2] for item in rows[:first_cv]], dtype=float)
            cv_c_rate = np.asarray([item[6] for item in rows[first_cv:]], dtype=float)
            if (
                cc_voltage.size == 0
                or float(np.min(cc_voltage))
                > MIT_PROPOSED_CC_VOLTAGE_RANGE[0] + MIT_CC_COVERAGE_TOLERANCE_V
                or float(np.max(cc_voltage))
                < MIT_PROPOSED_CC_VOLTAGE_RANGE[1] - MIT_CC_COVERAGE_TOLERANCE_V
            ):
                raise ValueError(
                    f"MIT canonical cycle lacks full proposed CC coverage: {path}, {cycle_id}"
                )
            if (
                cv_c_rate.size == 0
                or float(np.max(cv_c_rate)) < MIT_CV_HIGH_COVERAGE_C_RATE
                or float(np.min(cv_c_rate)) > MIT_CV_LOW_COVERAGE_C_RATE
            ):
                raise ValueError(
                    f"MIT canonical cycle lacks full proposed CV coverage: {path}, {cycle_id}"
                )
        times = np.asarray([item[1] for item in rows], dtype=np.float32)
        voltage = np.asarray([item[2] for item in rows], dtype=np.float32)
        current = np.asarray([item[3] for item in rows], dtype=np.float32)
        temperature = np.asarray([item[4] for item in rows], dtype=np.float32)
        soh_values = np.asarray([item[5] for item in rows], dtype=np.float32)
        if not np.allclose(soh_values, soh_values[0], rtol=1e-5, atol=1e-6):
            raise ValueError(f"SOH is not constant within cycle {cycle_id} in {path}")
        capacity_values = np.asarray([item[7] for item in rows], dtype=np.float32)
        if not np.all(np.isfinite(capacity_values)) or not np.all(capacity_values > 0.0):
            raise ValueError(f"capacity_Ah is not finite and positive within cycle {cycle_id} in {path}")
        if not np.allclose(capacity_values, capacity_values[0], rtol=1e-5, atol=1e-6):
            raise ValueError(f"capacity_Ah is not constant within cycle {cycle_id} in {path}")
        raw_soh = float(soh_values[0])
        capacity_ah = float(capacity_values[0])
        soh, scale_factor, resolved_mode = _normalize_mit_soh(
            raw_soh, nominal_capacity, label_scale_mode
        )
        records.append(
            {
                "dataset_id": "mit",
                "condition": condition,
                "battery_id": battery_id,
                "cycle_id": int(cycle_id),
                "raw_cycle_order_index": int(raw_order_index),
                "segment": segments,
                "time": times,
                "voltage": voltage,
                "current": current,
                "temperature": temperature,
                "soh": float(soh),
                "soh_raw": raw_soh,
                # MIT's formal Paper-v2 source capacity is capacity_Ah;
                # retaining it does not alter the historical SOH field.
                "capacity_Ah": capacity_ah,
                "capacity": capacity_ah,
                "source_capacity_field": "capacity_Ah",
                "soh_scale_factor": float(scale_factor),
                "soh_scale_mode": resolved_mode,
                "nominal_capacity": nominal_capacity,
                "source_file": str(path),
                "physical_cell_id": battery_id if is_physical_file else None,
                "phase_policy_version": (
                    MIT_PROPOSED_RAW_PHASE_POLICY_VERSION if is_physical_file else None
                ),
                **cycle_provenance.get(cycle_id, {}),
            }
        )
    if not records:
        raise ValueError(f"No CC/CV cycles found in MIT raw file: {path}")
    return records


def load_mit_raw_records(data_root, batch=None, nominal_capacity=1.1, label_scale_mode="none"):
    records = []
    for path in list_mit_raw_files(data_root, batch=batch):
        records.extend(read_mit_raw_file(path, nominal_capacity, label_scale_mode))
    if not records:
        raise ValueError(f"No MIT raw records loaded from {data_root}")
    return records


class MITRawAdapter:
    """MIT point-level adapter for the common raw-cycle contract."""

    dataset_id = "mit"
    def __init__(self, data_root, nominal_capacity=1.1, label_scale_mode="none"):
        self.data_root = Path(data_root)
        self.nominal_capacity = float(nominal_capacity)
        self.label_scale_mode = str(label_scale_mode)
        try:
            self.inventory = inspect_mit_raw_inventory(self.data_root)
        except ValueError as exc:
            self.inventory = None
            self.raw_terminal_signals = False
            self.readiness_error = str(exc)
        else:
            header_only = self.inventory["header_only_files"]
            invalid_headers = self.inventory["missing_required_headers"]
            self.raw_terminal_signals = bool(self.inventory["nonempty_files"]) and not (
                header_only or invalid_headers
            )
            if self.raw_terminal_signals:
                self.readiness_error = ""
            else:
                self.readiness_error = (
                    "MIT canonical raw export is not runnable: "
                    f"files={self.inventory['files']}, "
                    f"nonempty={len(self.inventory['nonempty_files'])}, "
                    f"header_only={len(header_only)}, "
                    f"missing_headers={len(invalid_headers)}. "
                    "Regenerate/copy the phase-aware physical124 export into "
                    "UnifiedRawSOH/datasets/MIT_raw; no historical raw or aligned fallback is used."
                )

    def load_records(self, batch=None):
        if not self.raw_terminal_signals:
            raise RawTerminalSignalUnavailable(self.readiness_error)
        return load_mit_raw_records(
            self.data_root,
            batch=batch,
            nominal_capacity=self.nominal_capacity,
            label_scale_mode=self.label_scale_mode,
        )


def list_mit_feature_files(data_root):
    root = Path(data_root)
    if not root.is_dir():
        raise ValueError(f"MIT data root does not exist: {root}")
    files = sorted(path for path in root.glob("*.csv") if _is_mit_battery_csv(path))
    if not files:
        raise ValueError(f"No MIT feature CSV files found under {root}")
    return files


class MITFeatureAdapter:
    dataset_id = "mit"
    raw_terminal_signals = False
    unified_sample_keys = UNIFIED_SAMPLE_KEYS

    def __init__(self, data_root):
        self.data_root = Path(data_root)

    def inspect(self):
        files = list_mit_feature_files(self.data_root)
        row_count = 0
        capacity_count = 0
        finite_capacity_count = 0
        headers = set()
        for path in files:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or ())
                headers.update(fields)
                missing = sorted(MIT_REQUIRED_INTERMEDIATE_COLUMNS - fields)
                if missing:
                    raise ValueError(f"MIT feature file {path} is missing columns: {missing}")
                for row in reader:
                    row_count += 1
                    value = row.get("capacity", "")
                    if value != "":
                        capacity_count += 1
                        try:
                            finite_capacity_count += int(math.isfinite(float(value)))
                        except ValueError:
                            pass
        return {
            "dataset_id": self.dataset_id,
            "files": len(files),
            "rows": row_count,
            "header_count": len(headers),
            "capacity_values": capacity_count,
            "finite_capacity_values": finite_capacity_count,
            "raw_terminal_signals": False,
            "available": sorted(headers),
            "missing_for_unified_raw_contract": [
                "point-level voltage",
                "point-level current",
                "point-level time",
                "point-level temperature",
                "validated CC/CV segment boundaries",
            ],
        }

    def load_records(self):
        """Return auditable intermediate rows, never model-ready raw samples."""

        records = []
        for path in list_mit_feature_files(self.data_root):
            condition, filename_battery_id, is_physical_file = parse_mit_file_identity(path)
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                physical_ids = set()
                for cycle_index, row in enumerate(reader):
                    battery_id = filename_battery_id
                    cycle_id = cycle_index
                    if is_physical_file:
                        physical_id = str(row.get("physical_cell_id", "")).strip()
                        if not physical_id:
                            raise ValueError(f"Physical MIT feature row has no physical_cell_id: {path}")
                        physical_ids.add(physical_id)
                        battery_id = physical_id
                        try:
                            cycle_id = int(float(row["cycle"]))
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ValueError(
                                f"Physical MIT feature row has invalid global cycle: {path}"
                            ) from exc
                    records.append(
                        {
                            "dataset_id": self.dataset_id,
                            "battery_id": battery_id,
                            "condition": condition,
                            "cycle_id": cycle_id,
                            "capacity": float(row["capacity"]),
                            "source_file": str(path),
                            "raw_terminal_signals": False,
                        }
                    )
                if is_physical_file and physical_ids != {filename_battery_id}:
                    raise ValueError(
                        f"Physical MIT feature filename/id mismatch in {path}: "
                        f"filename={filename_battery_id}, rows={sorted(physical_ids)}"
                    )
        return records

    def to_unified_samples(self, *args, **kwargs):
        raise RawTerminalSignalUnavailable(
            "MIT_features contains cycle-level summary features only. Use the "
            "separate MIT_raw source and its MITRawAdapter for point-level "
            "voltage/current/time/temperature; no synthetic CC/CV sequence or "
            "aligned fallback is permitted."
        )
