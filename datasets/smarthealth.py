"""SmartHealth source audit and v2 canonical raw-cycle adapter.

The GB18030 source has a combined ``恒流恒压充电`` step and is audit-only when
supplied directly.  The three family-specific v2 RAW preprocessors write the
validated boundary, selected CC/CV windows, calibration-derived SOH, and
cell/cycle provenance that this adapter consumes.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .base import RawTerminalSignalUnavailable


SMARTHEALTH_ENCODING = "gb18030"
SMARTHEALTH_REQUIRED_COLUMNS = {
    "循环号",
    "工步号",
    "工步类型",
    "电流(A)",
    "电压(V)",
    "充电容量(Ah)",
    "放电容量(Ah)",
    "temp1_1",
}
SMARTHEALTH_MANUFACTURERS = {
    "smarthealth_lishen40": "LISHEN",
    "smarthealth_catl280": "CATL",
    "smarthealth_eve280": "EVE",
}
SMARTHEALTH_NOMINAL_CAPACITY_AH = {
    "smarthealth_lishen40": 40.0,
    "smarthealth_catl280": 280.0,
    "smarthealth_eve280": 280.0,
}
SMARTHEALTH_CANONICAL_POLICY_VERSION = "smarthealth_cccv_calibration_v2"
SMARTHEALTH_CC_VOLTAGE_RANGE = (3.45, 3.58)
SMARTHEALTH_CV_C_RATE_RANGE = (0.05, 0.25)
SMARTHEALTH_CV_SELECTION_TOLERANCE_C = 0.002
_PART_RE = re.compile(r"^(?P<series>.+)-(?P<part>\d+)$")
_PROCESSED_FILE_RE = re.compile(
    r"^(?P<domain>smarthealth_(?:lishen40|catl280|eve280))__.+[.]csv$"
)
SMARTHEALTH_PROCESSED_REQUIRED_COLUMNS = {
    "dataset",
    "dataset_id",
    "domain_id",
    "condition",
    "cell",
    "battery_id",
    "source_serial",
    "logical_sequence_id",
    "cycle",
    "SOH",
    "label_source",
    "split_role",
    "split_status",
    "split_issue",
    "split_strategy_version",
    "segment",
    "cycle_point_index",
    "segment_point_index",
    "relative_time",
    "voltage_V",
    "current_A",
    "c_rate",
    "temperature_C",
    "source_file",
    "chunk_id",
    "source_cycle",
    "strategy_version",
    "phase_policy_version",
    "cc_voltage_low_V",
    "cc_voltage_high_V",
    "cv_c_rate_low",
    "cv_c_rate_high",
}


def list_smarthealth_csv_files(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    if not root.is_dir():
        raise ValueError(f"SmartHealth data root does not exist: {root}")
    files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    if not files:
        raise ValueError(f"No SmartHealth CSV files found under {root}")
    return files


def _header(path: Path) -> list[str]:
    with path.open("r", encoding=SMARTHEALTH_ENCODING, newline="") as handle:
        reader = csv.reader(handle)
        try:
            return [str(item).strip() for item in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"SmartHealth CSV is empty: {path}") from exc


def _series_id(path: Path) -> str:
    match = _PART_RE.match(path.stem)
    return match.group("series") if match is not None else path.stem


def audit_smarthealth_source(data_root: str | Path, domain_id: str | None = None) -> dict:
    """Inspect real SmartHealth headers and file organization without parsing cycles."""

    files = list_smarthealth_csv_files(data_root)
    invalid_headers: list[dict] = []
    header_sets = set()
    conditions = Counter()
    series = set()
    for path in files:
        header = _header(path)
        header_set = frozenset(item for item in header if item)
        header_sets.add(header_set)
        missing = sorted(SMARTHEALTH_REQUIRED_COLUMNS - set(header_set))
        if missing:
            invalid_headers.append({"path": str(path), "missing": missing})
        conditions[path.parent.name] += 1
        series.add(_series_id(path))
    return {
        "domain_id": domain_id,
        "data_root": str(Path(data_root).resolve()),
        "files": len(files),
        "logical_series": len(series),
        "condition_directories": dict(sorted(conditions.items())),
        "header_variants": len(header_sets),
        "required_columns": sorted(SMARTHEALTH_REQUIRED_COLUMNS),
        "invalid_header_files": len(invalid_headers),
        "invalid_headers": invalid_headers[:10],
        "raw_signal_columns_confirmed": not invalid_headers,
        "encoding": SMARTHEALTH_ENCODING,
        "charge_step_label": "恒流恒压充电",
        "raw_contract_status": "source_audit_only_canonical_export_required",
        "blockers": [
            "The direct GB18030 source is not a model input; run the matching process_smarthealth_<family>_raw.py entry point to produce the versioned canonical export.",
            "Source chunks without the temperature channel remain excluded from the temperature-aware canonical export, never imputed.",
        ],
    }


def list_smarthealth_raw_files(
    data_root: str | Path,
    domain_id: str | None = None,
) -> list[Path]:
    """List canonical processed files, never GB18030 source chunks."""

    root = Path(data_root)
    if not root.is_dir():
        raise ValueError(f"SmartHealth data root does not exist: {root}")
    search_root = root
    if domain_id is not None and (root / str(domain_id)).is_dir():
        search_root = root / str(domain_id)
    files = []
    for path in sorted(search_root.rglob("smarthealth_*.csv")):
        match = _PROCESSED_FILE_RE.match(path.name)
        if match is None:
            continue
        if domain_id is not None and match.group("domain") != str(domain_id):
            continue
        files.append(path)
    return files


def _processed_soh(raw_soh: float, label_scale_mode: str) -> tuple[float, float, str]:
    """Canonical exports already contain a dimensionless SOH label."""

    mode = str(label_scale_mode or "none")
    if mode not in {"none", "auto_capacity_to_soh", "exported_soh_direct"}:
        raise ValueError(
            "SmartHealth canonical SOH is already dimensionless; use "
            "label_scale_mode='none' or 'auto_capacity_to_soh'."
        )
    return float(raw_soh), 1.0, "exported_soh_direct"


def _validate_canonical_phase_row(
    row: dict[str, str],
    *,
    segment: str,
    voltage: float,
    current: float,
    domain_id: str,
    path: Path,
) -> None:
    """Reject a stale or differently-windowed SmartHealth canonical export.

    The raw adapter deliberately validates the export contract rather than
    applying a second window.  Re-windowing here could hide an upstream phase
    selection error and would make the raw and Only-F products incomparable.
    """

    if row["strategy_version"].strip() != SMARTHEALTH_CANONICAL_POLICY_VERSION:
        raise ValueError(
            f"SmartHealth strategy mismatch in {path}: {row['strategy_version']!r}"
        )
    if row["phase_policy_version"].strip() != SMARTHEALTH_CANONICAL_POLICY_VERSION:
        raise ValueError(
            f"SmartHealth phase-policy mismatch in {path}: {row['phase_policy_version']!r}"
        )
    expected_metadata = {
        "cc_voltage_low_V": SMARTHEALTH_CC_VOLTAGE_RANGE[0],
        "cc_voltage_high_V": SMARTHEALTH_CC_VOLTAGE_RANGE[1],
        "cv_c_rate_low": SMARTHEALTH_CV_C_RATE_RANGE[0],
        "cv_c_rate_high": SMARTHEALTH_CV_C_RATE_RANGE[1],
    }
    for column, expected in expected_metadata.items():
        try:
            actual = float(row[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {column} in canonical SmartHealth row: {path}") from exc
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"SmartHealth window metadata mismatch in {path}: {column}={actual}, "
                f"expected={expected}"
            )
    nominal_capacity = SMARTHEALTH_NOMINAL_CAPACITY_AH.get(domain_id)
    if nominal_capacity is None:
        raise ValueError(f"Unknown SmartHealth domain in canonical row: {domain_id!r}")
    try:
        recorded_c_rate = float(row["c_rate"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid c_rate in canonical SmartHealth row: {path}") from exc
    expected_c_rate = abs(float(current)) / nominal_capacity
    if not math.isclose(recorded_c_rate, expected_c_rate, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError(
            f"SmartHealth c_rate is not nominal-capacity normalized in {path}: "
            f"recorded={recorded_c_rate}, expected={expected_c_rate}"
        )
    if segment == "CC" and not (
        SMARTHEALTH_CC_VOLTAGE_RANGE[0] - 1e-9
        <= voltage
        <= SMARTHEALTH_CC_VOLTAGE_RANGE[1] + 1e-9
    ):
        raise ValueError(f"SmartHealth CC point outside 3.45--3.58 V window in {path}")
    if segment == "CV" and not (
        SMARTHEALTH_CV_C_RATE_RANGE[0] - SMARTHEALTH_CV_SELECTION_TOLERANCE_C - 1e-9
        <= recorded_c_rate
        <= SMARTHEALTH_CV_C_RATE_RANGE[1] + SMARTHEALTH_CV_SELECTION_TOLERANCE_C + 1e-9
    ):
        raise ValueError(f"SmartHealth CV point outside 0.25C--0.05C window in {path}")


def read_smarthealth_raw_file(
    path: str | Path,
    domain_id: str | None = None,
    label_scale_mode: str = "none",
) -> list[dict]:
    """Read one canonical per-logical-cell SmartHealth raw CSV."""

    path = Path(path)
    grouped: dict[int, list[tuple[str, float, float, float, float, float]]] = {}
    cycle_provenance: dict[int, dict] = {}
    battery_ids: set[str] = set()
    domains: set[str] = set()
    conditions: set[str] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(SMARTHEALTH_PROCESSED_REQUIRED_COLUMNS - fields)
        if missing:
            raise ValueError(f"Canonical SmartHealth raw file {path} is missing: {missing}")
        for row in reader:
            try:
                cycle = int(float(row["cycle"]))
                segment = str(row["segment"]).strip().upper()
                item = (
                    segment,
                    float(row["relative_time"]),
                    float(row["voltage_V"]),
                    float(row["current_A"]),
                    float(row["temperature_C"]),
                    float(row["SOH"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid canonical SmartHealth row in {path}: {row}") from exc
            if segment not in {"CC", "CV"}:
                raise ValueError(f"Unexpected segment {segment!r} in {path}")
            if not all(math.isfinite(value) for value in item[1:]):
                raise ValueError(
                    f"Canonical SmartHealth export has a non-finite model value in {path}, cycle {cycle}"
                )
            row_domain = str(row["domain_id"]).strip()
            battery_id = str(row["battery_id"]).strip()
            condition = str(row["condition"]).strip()
            if not row_domain or not battery_id or not condition:
                raise ValueError(f"Canonical SmartHealth row lacks identity metadata in {path}")
            if domain_id is not None and row_domain != str(domain_id):
                raise ValueError(
                    f"SmartHealth file/domain mismatch: requested={domain_id}, row={row_domain}, path={path}"
                )
            _validate_canonical_phase_row(
                row,
                segment=segment,
                voltage=item[2],
                current=item[3],
                domain_id=row_domain,
                path=path,
            )
            battery_ids.add(battery_id)
            domains.add(row_domain)
            conditions.add(condition)
            grouped.setdefault(cycle, []).append(item)
            provenance = {
                "cell": str(row.get("cell", battery_id)).strip(),
                "logical_sequence_id": str(row.get("logical_sequence_id", battery_id)).strip(),
                "source_serial": str(row.get("source_serial", "")).strip(),
                "source_series": str(row.get("source_series", "")).strip(),
                "source_file": str(row.get("source_file", "")).strip(),
                "chunk_id": str(row.get("chunk_id", "")).strip(),
                "source_cycle": int(float(row.get("source_cycle", cycle))),
                "strategy_version": str(row["strategy_version"]).strip(),
                "split_role": str(row["split_role"]).strip(),
                "split_status": str(row["split_status"]).strip(),
                "split_issue": str(row["split_issue"]).strip(),
                "split_strategy_version": str(row["split_strategy_version"]).strip(),
                "label_source": str(row["label_source"]).strip(),
            }
            previous = cycle_provenance.setdefault(cycle, provenance)
            if previous != provenance:
                raise ValueError(
                    f"SmartHealth provenance is not constant within {path} cycle {cycle}"
                )
    if not battery_ids:
        raise ValueError(f"Canonical SmartHealth raw file has no data rows: {path}")
    if len(battery_ids) != 1 or len(domains) != 1 or len(conditions) != 1:
        raise ValueError(f"Canonical SmartHealth file contains mixed identities: {path}")
    battery_id = next(iter(battery_ids))
    current_domain = next(iter(domains))
    condition = next(iter(conditions))
    records = []
    for order, cycle in enumerate(sorted(grouped)):
        rows = sorted(grouped[cycle], key=lambda item: item[1])
        segment = np.asarray([item[0] for item in rows], dtype=object)
        cv_positions = np.flatnonzero(segment == "CV")
        first_cv = int(cv_positions[0]) if cv_positions.size else -1
        if (
            segment.size == 0
            or first_cv <= 0
            or np.any(segment[:first_cv] != "CC")
            or np.any(segment[first_cv:] != "CV")
        ):
            raise ValueError(f"Canonical SmartHealth cycle has no CC/CV pair: {path}, cycle {cycle}")
        values = np.asarray([item[5] for item in rows], dtype=np.float32)
        if not np.allclose(values, values[0], rtol=1e-5, atol=1e-6):
            raise ValueError(f"SOH is not constant within {path} cycle {cycle}")
        soh, factor, mode = _processed_soh(float(values[0]), label_scale_mode)
        records.append(
            {
                "dataset_id": "smarthealth",
                "domain_id": current_domain,
                "condition": condition,
                "battery_id": battery_id,
                "cycle_id": int(cycle),
                "raw_cycle_order_index": int(order),
                "segment": segment,
                "time": np.asarray([item[1] for item in rows], dtype=np.float32),
                "voltage": np.asarray([item[2] for item in rows], dtype=np.float32),
                "current": np.asarray([item[3] for item in rows], dtype=np.float32),
                "temperature": np.asarray([item[4] for item in rows], dtype=np.float32),
                "soh": soh,
                "soh_raw": float(values[0]),
                "soh_scale_factor": factor,
                "soh_scale_mode": mode,
                "label_source": str(cycle_provenance[cycle]["label_source"]),
                "source_file": str(path),
                **cycle_provenance[cycle],
            }
        )
    if not records:
        raise ValueError(f"No canonical SmartHealth records in {path}")
    return records


def load_smarthealth_raw_records(
    data_root: str | Path,
    domain_id: str | None = None,
    batch: str | None = None,
    label_scale_mode: str = "none",
) -> list[dict]:
    """Load canonical records, optionally filtering by one source condition."""

    files = list_smarthealth_raw_files(data_root, domain_id=domain_id)
    if not files:
        raise RawTerminalSignalUnavailable(
            "No canonical SmartHealth raw export was found. Run "
            "the matching sqj_soh/preprocess/process_smarthealth_<family>_raw.py first; "
            "the direct GB18030 source root is audit-only."
        )
    records: list[dict] = []
    for path in files:
        current = read_smarthealth_raw_file(
            path, domain_id=domain_id, label_scale_mode=label_scale_mode
        )
        if batch is not None:
            current = [record for record in current if record["condition"] == str(batch)]
        records.extend(current)
    if not records:
        raise ValueError(
            f"No canonical SmartHealth records for domain={domain_id!r}, batch={batch!r}"
        )
    return records


class SmartHealthRawAdapter:
    """Use canonical processed products while retaining source-root audit safety."""

    dataset_id = "smarthealth"

    def __init__(self, data_root, domain_id=None, label_scale_mode="none", **_kwargs):
        self.data_root = Path(data_root)
        self.domain_id = None if domain_id is None else str(domain_id)
        self.label_scale_mode = str(label_scale_mode)
        self.raw_terminal_signals = bool(
            list_smarthealth_raw_files(self.data_root, domain_id=self.domain_id)
        ) if self.data_root.is_dir() else False

    def inspect(self):
        processed = list_smarthealth_raw_files(self.data_root, domain_id=self.domain_id)
        if processed:
            return {
                "dataset_id": self.dataset_id,
                "domain_id": self.domain_id,
                "data_root": str(self.data_root.resolve()),
                "files": len(processed),
                "raw_terminal_signals": True,
                "raw_contract_status": "canonical_cccv_calibration_v2_available",
                "required_columns": sorted(SMARTHEALTH_PROCESSED_REQUIRED_COLUMNS),
            }
        return audit_smarthealth_source(self.data_root, domain_id=self.domain_id)

    def load_records(self, batch=None):
        return load_smarthealth_raw_records(
            self.data_root,
            domain_id=self.domain_id,
            batch=batch,
            label_scale_mode=self.label_scale_mode,
        )
