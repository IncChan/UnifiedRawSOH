"""Shared streaming and protocol logic for the normalized SMVIC cell CSVs.

The normalized source is already point-level and can be several GiB.  This
module therefore keeps at most one physical cycle in memory.  It deliberately
contains no train/validation fitting: protocol windows and SOH denominators
are fixed physical metadata from the SMVIC specification.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


SMVIC_SCHEMA = "smvic_model_ready_v1"
SOURCE_SCHEMA = "smvic_normalized_cell_csv_v2"


@dataclass(frozen=True)
class FamilySpec:
    group: str
    domain_id: str
    nominal_capacity_ah: float
    charge_cutoff_v: float
    discharge_cutoff_v: float
    cc_c_rate_range: tuple[float, float]
    discharge_c_rate_range: tuple[float, float]
    terminal_cc_delta_v: float
    cv_c_rate_range: tuple[float, float] = (0.04, 0.30)
    charge_temperature_range_c: tuple[float, float] | None = None
    minimum_cycle: int | None = None

    def normalization(self) -> dict[str, Any]:
        cv_low, cv_high = self.cv_c_rate_range
        return {
            "voltage_low": self.charge_cutoff_v - self.terminal_cc_delta_v,
            "voltage_high": self.charge_cutoff_v,
            "current_scale": self.nominal_capacity_ah,
            "cc_voltage_low": self.charge_cutoff_v - self.terminal_cc_delta_v,
            "cc_voltage_high": self.charge_cutoff_v,
            "cv_current_low": cv_low * self.nominal_capacity_ah,
            "cv_current_high": cv_high * self.nominal_capacity_ah,
            "temp_room": 25.0,
            "temp_abs_scale": 20.0,
            "temp_delta_scale": 10.0,
            "time_scale_min": 10.0,
            "schema_version": 2,
            "current_mode": "nominal_c_rate",
            "nominal_capacity_ah": self.nominal_capacity_ah,
        }


FAMILY_SPECS: dict[str, FamilySpec] = {
    "Battery01": FamilySpec(
        "Battery01", "smvic_e72_69ah", 69.4, 4.20, 2.80,
        (0.25, 0.42), (0.80, 1.20), 0.20,
    ),
    "Battery02": FamilySpec(
        "Battery02", "smvic_s5e891_51ah", 51.0, 4.18, 2.80,
        (0.32, 0.52), (0.80, 1.20), 0.20,
    ),
    "Battery03": FamilySpec(
        "Battery03", "smvic_type1_18ah", 18.0, 4.40, 2.50,
        (0.75, 1.15), (0.75, 1.15), 0.20,
    ),
    "Battery04": FamilySpec(
        "Battery04", "smvic_type2_150ah_t40", 150.0, 4.85, 3.50,
        (0.75, 1.20), (0.75, 1.20), 0.25,
        charge_temperature_range_c=(35.0, 50.0),
    ),
    "Battery05": FamilySpec(
        "Battery05", "smvic_type3_108ah", 108.0, 4.78, 3.50,
        (0.80, 1.20), (0.80, 1.20), 0.25,
        minimum_cycle=1,
    ),
    "Battery06": FamilySpec(
        "Battery06", "smvic_type4_11ah", 11.4, 4.20, 3.00,
        (0.75, 1.15), (0.75, 1.15), 0.20,
    ),
}


REQUIRED_COLUMNS = {
    "battery_group", "battery_id", "source_battery_id", "condition",
    "nominal_capacity_Ah", "charge_cutoff_V", "discharge_cutoff_V",
    "cycle", "cycle_charge_capacity_Ah", "cycle_discharge_capacity_Ah",
    "SOH_nominal", "is_complete_cycle", "quality_flags", "segment",
    "relative_time_s", "absolute_time", "voltage_V", "current_A",
    "temperature_C", "source_file",
}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def list_cell_csvs(source_root: str | Path, groups: Iterable[str] | None = None) -> list[Path]:
    root = Path(source_root).expanduser().resolve()
    selected = set(FAMILY_SPECS if groups is None else groups)
    unknown = sorted(selected - set(FAMILY_SPECS))
    if unknown:
        raise ValueError(f"Unknown SMVIC groups: {unknown}")
    files = [path for group in sorted(selected) for path in sorted((root / group).glob("Cell*.csv"))]
    if not files:
        raise FileNotFoundError(f"No SMVIC Cell CSVs found under {root}")
    return files


def _cycle_from_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    first = rows[0]
    group = str(first["battery_group"]).strip()
    if group not in FAMILY_SPECS:
        raise ValueError(f"Unexpected battery_group={group!r} in {path}")
    cycle_id = _int(first["cycle"])
    for row in rows:
        if str(row["battery_group"]).strip() != group or _int(row["cycle"]) != cycle_id:
            raise ValueError(f"Non-constant cycle identity in {path}: {group}/{cycle_id}")
    return {
        "path": str(path),
        "group": group,
        "battery_id": str(first["battery_id"]).strip(),
        "source_battery_id": str(first["source_battery_id"]).strip(),
        "condition": str(first["condition"]).strip() or "unknown",
        "cycle_id": cycle_id,
        "nominal_capacity_ah": _float(first["nominal_capacity_Ah"]),
        "charge_cutoff_v": _float(first["charge_cutoff_V"]),
        "discharge_cutoff_v": _float(first["discharge_cutoff_V"]),
        "charge_capacity_ah": _float(first["cycle_charge_capacity_Ah"]),
        "discharge_capacity_ah": _float(first["cycle_discharge_capacity_Ah"]),
        "soh_source": _float(first["SOH_nominal"]),
        "is_complete_cycle": _int(first["is_complete_cycle"], 0) == 1,
        "quality_flags": str(first["quality_flags"]).strip(),
        "segment": np.asarray([str(row["segment"]).strip().upper() for row in rows], dtype=object),
        "time_s": np.asarray([_float(row["relative_time_s"]) for row in rows], dtype=np.float64),
        "voltage": np.asarray([_float(row["voltage_V"]) for row in rows], dtype=np.float64),
        "current": np.asarray([_float(row["current_A"]) for row in rows], dtype=np.float64),
        "temperature": np.asarray([_float(row["temperature_C"]) for row in rows], dtype=np.float64),
        "absolute_time": np.asarray([str(row["absolute_time"]).strip() for row in rows], dtype=object),
        "source_file": str(first["source_file"]).strip(),
        "source_rows": len(rows),
    }


def iter_cell_cycles(path: str | Path, max_cycles: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield contiguous cycles from one normalized cell CSV."""

    path = Path(path).resolve()
    yielded = 0
    seen: set[int] = set()
    current_cycle: int | None = None
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - fields)
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            cycle_id = _int(row["cycle"])
            if current_cycle is None:
                current_cycle = cycle_id
            if cycle_id != current_cycle:
                if cycle_id in seen:
                    raise ValueError(f"Cycle {cycle_id} is non-contiguous in {path}")
                seen.add(current_cycle)
                yield _cycle_from_rows(path, rows)
                yielded += 1
                if max_cycles is not None and yielded >= int(max_cycles):
                    return
                rows = []
                current_cycle = cycle_id
            # Do not retain the remaining provenance/device columns for the
            # whole cycle; a dense source cycle can contain many point rows.
            rows.append({key: row[key] for key in REQUIRED_COLUMNS})
    if rows and (max_cycles is None or yielded < int(max_cycles)):
        yield _cycle_from_rows(path, rows)


def _charge_event_indices(cycle: Mapping[str, Any]) -> list[np.ndarray]:
    segments = np.asarray(cycle["segment"], dtype=object)
    current = np.asarray(cycle["current"], dtype=float)
    events: list[list[int]] = []
    active: list[int] = []
    saw_cc = False
    saw_cv = False
    for index, (segment, amps) in enumerate(zip(segments, current)):
        is_charge = segment in {"CC", "CV"} and math.isfinite(float(amps)) and float(amps) > 0.0
        if not is_charge:
            if active and saw_cc and saw_cv:
                events.append(active)
            active, saw_cc, saw_cv = [], False, False
            continue
        if segment == "CC" and saw_cv:
            if active and saw_cc and saw_cv:
                events.append(active)
            active, saw_cc, saw_cv = [], False, False
        active.append(index)
        saw_cc = saw_cc or segment == "CC"
        saw_cv = saw_cv or segment == "CV"
    if active and saw_cc and saw_cv:
        events.append(active)
    return [np.asarray(item, dtype=np.int64) for item in events]


def select_principal_terminal_event(
    cycle: Mapping[str, Any], spec: FamilySpec, min_phase_points: int = 4
) -> tuple[dict[str, np.ndarray] | None, str | None, dict[str, float]]:
    events = _charge_event_indices(cycle)
    if not events:
        return None, "no_contiguous_cc_cv_event", {}
    segment = np.asarray(cycle["segment"], dtype=object)
    voltage = np.asarray(cycle["voltage"], dtype=float)
    current = np.asarray(cycle["current"], dtype=float)
    temperature = np.asarray(cycle["temperature"], dtype=float)
    time_s = np.asarray(cycle["time_s"], dtype=float)
    candidates = []
    for event in events:
        cc_all = event[segment[event] == "CC"]
        cv_all = event[segment[event] == "CV"]
        cc = cc_all[
            np.isfinite(voltage[cc_all])
            & (voltage[cc_all] >= spec.charge_cutoff_v - spec.terminal_cc_delta_v)
            & (voltage[cc_all] <= spec.charge_cutoff_v + 0.03)
        ]
        c_rate = np.abs(current[cv_all]) / spec.nominal_capacity_ah
        cv = cv_all[
            np.isfinite(c_rate)
            & np.isfinite(voltage[cv_all])
            & (c_rate >= spec.cv_c_rate_range[0])
            & (c_rate <= spec.cv_c_rate_range[1])
            & (voltage[cv_all] >= spec.charge_cutoff_v - 0.08)
            & (voltage[cv_all] <= spec.charge_cutoff_v + 0.03)
        ]
        if len(cc) < min_phase_points or len(cv) < min_phase_points:
            continue
        score = float(len(cc) + len(cv))
        if np.all(np.isfinite(time_s[event])):
            score += max(0.0, float(np.max(time_s[event]) - np.min(time_s[event]))) / 60.0
        candidates.append((score, cc, cv, cc_all, cv_all))
    if not candidates:
        return None, "terminal_window_too_short", {"charge_event_count": float(len(events))}
    _, cc, cv, cc_all, cv_all = max(candidates, key=lambda item: item[0])
    selected = np.concatenate((cc, cv))
    if not (
        np.all(np.isfinite(time_s[selected]))
        and np.all(np.isfinite(voltage[selected]))
        and np.all(np.isfinite(current[selected]))
    ):
        return None, "nonfinite_electrical_terminal", {}
    if not np.all(np.isfinite(temperature[selected])):
        return None, "nonfinite_temperature_terminal", {}
    cc_rate = float(np.median(np.abs(current[cc_all])) / spec.nominal_capacity_ah)
    cv_rate_start = float(np.max(np.abs(current[cv])) / spec.nominal_capacity_ah)
    cv_rate_end = float(np.min(np.abs(current[cv])) / spec.nominal_capacity_ah)
    return {
        "cc": cc,
        "cv": cv,
    }, None, {
        "charge_event_count": float(len(events)),
        "cc_c_rate": cc_rate,
        "cv_c_rate_start": cv_rate_start,
        "cv_c_rate_end": cv_rate_end,
        "charge_temperature_median_c": float(np.median(temperature[np.concatenate((cc_all, cv_all))])),
    }


def classify_cycle(
    cycle: Mapping[str, Any], spec: FamilySpec, min_phase_points: int = 4
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Classify one cycle and return a canonical terminal record plus audit."""

    audit: dict[str, Any] = {
        "domain_id": spec.domain_id,
        "battery_group": cycle["group"],
        "battery_id": cycle["battery_id"],
        "source_battery_id": cycle["source_battery_id"],
        "cycle_id": int(cycle["cycle_id"]),
        "source_rows": int(cycle["source_rows"]),
        "quality_flags": cycle["quality_flags"],
    }

    def reject(reason: str) -> tuple[None, dict[str, Any]]:
        audit.update({"eligible": 0, "reason": reason})
        return None, audit

    if not cycle["is_complete_cycle"]:
        return reject("incomplete_cycle")
    if spec.minimum_cycle is not None and int(cycle["cycle_id"]) < spec.minimum_cycle:
        return reject("preconditioning_cycle")
    for key, expected in (
        ("nominal_capacity_ah", spec.nominal_capacity_ah),
        ("charge_cutoff_v", spec.charge_cutoff_v),
        ("discharge_cutoff_v", spec.discharge_cutoff_v),
    ):
        if not math.isclose(float(cycle[key]), expected, rel_tol=0.0, abs_tol=1e-6):
            return reject(f"metadata_mismatch:{key}")
    capacity = float(cycle["discharge_capacity_ah"])
    source_soh = float(cycle["soh_source"])
    if not math.isfinite(capacity) or capacity <= 0.0:
        return reject("invalid_discharge_capacity")
    soh = capacity / spec.nominal_capacity_ah
    if not math.isfinite(source_soh) or not math.isclose(soh, source_soh, rel_tol=1e-8, abs_tol=1e-8):
        return reject("soh_nominal_mismatch")
    if not 0.0 < soh < 2.0:
        return reject("implausible_soh")

    event, event_reason, event_stats = select_principal_terminal_event(
        cycle, spec, min_phase_points=min_phase_points
    )
    audit.update(event_stats)
    if event is None:
        return reject(str(event_reason))
    cc_c_rate = float(event_stats["cc_c_rate"])
    if not spec.cc_c_rate_range[0] <= cc_c_rate <= spec.cc_c_rate_range[1]:
        return reject("non_aging_charge_c_rate")

    segment = np.asarray(cycle["segment"], dtype=object)
    current = np.asarray(cycle["current"], dtype=float)
    discharge = np.abs(current[(segment == "DISCHARGE") & np.isfinite(current) & (current < 0.0)])
    discharge = discharge[discharge / spec.nominal_capacity_ah >= 0.02]
    if discharge.size == 0:
        return reject("no_discharge_current")
    discharge_c_rate = float(np.median(discharge) / spec.nominal_capacity_ah)
    audit["discharge_c_rate"] = discharge_c_rate
    if not spec.discharge_c_rate_range[0] <= discharge_c_rate <= spec.discharge_c_rate_range[1]:
        return reject("non_aging_discharge_c_rate")
    if spec.charge_temperature_range_c is not None:
        median_temp = float(event_stats["charge_temperature_median_c"])
        if not spec.charge_temperature_range_c[0] <= median_temp <= spec.charge_temperature_range_c[1]:
            return reject("non_aging_charge_temperature")

    cc, cv = event["cc"], event["cv"]
    indices = np.concatenate((cc, cv))
    record = {
        "dataset_id": spec.domain_id,
        "domain_id": spec.domain_id,
        "condition": str(cycle["condition"]),
        "strategy_id": str(cycle["condition"]),
        "battery_id": str(cycle["battery_id"]),
        "cycle_id": int(cycle["cycle_id"]),
        "segment": np.concatenate((np.full(len(cc), "CC", dtype=object), np.full(len(cv), "CV", dtype=object))),
        "time": np.asarray(cycle["time_s"], dtype=np.float64)[indices] / 60.0,
        "voltage": np.asarray(cycle["voltage"], dtype=np.float64)[indices],
        "current": np.asarray(cycle["current"], dtype=np.float64)[indices],
        "temperature": np.asarray(cycle["temperature"], dtype=np.float64)[indices],
        "soh": float(soh),
        "soh_raw": float(capacity),
        "nominal_capacity": spec.nominal_capacity_ah,
        "source_file": str(cycle["path"]),
        "source_cycle": int(cycle["cycle_id"]),
        "source_absolute_start_time": str(np.asarray(cycle["absolute_time"], dtype=object)[indices[0]]),
        "source_absolute_end_time": str(np.asarray(cycle["absolute_time"], dtype=object)[indices[-1]]),
    }
    audit.update({
        "eligible": 1,
        "reason": "eligible",
        "soh": float(soh),
        "discharge_capacity_ah": capacity,
        "cc_terminal_points": int(len(cc)),
        "cv_terminal_points": int(len(cv)),
    })
    return record, audit


def iter_classified_cycles(
    source_root: str | Path,
    groups: Iterable[str] | None = None,
    *,
    max_cycles_per_cell: int | None = None,
    min_phase_points: int = 4,
) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any]]]:
    for path in list_cell_csvs(source_root, groups):
        for cycle in iter_cell_cycles(path, max_cycles=max_cycles_per_cell):
            spec = FAMILY_SPECS[str(cycle["group"])]
            yield classify_cycle(cycle, spec, min_phase_points=min_phase_points)


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value
