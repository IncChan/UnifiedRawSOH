"""Full-charge source support for Paper-Backup E2.

The regular repository adapters intentionally expose only canonical terminal
products.  This module accepts a separately configured full source and fails
closed when it is absent or cannot be linked to the canonical terminal cycle.
It never promotes a terminal record to a full record.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class FullSourceUnavailable(RuntimeError):
    """Raised when a real, provenance-linkable full source is unavailable."""


FULL_REQUIRED_RECORD_KEYS = (
    "battery_id",
    "cycle_id",
    "condition",
    "segment",
    "time",
    "voltage",
    "current",
    "temperature",
    "soh",
)


def _as_array(record: dict[str, Any], key: str) -> np.ndarray:
    value = record.get(key)
    if value is None:
        raise FullSourceUnavailable(f"Full record is missing {key!r}")
    value = np.asarray(value)
    if value.ndim != 1:
        value = value.reshape(-1)
    return value


def validate_full_record(record: dict[str, Any]) -> None:
    """Validate the non-negotiable full-record marker and point contract."""

    if not bool(record.get("is_full", False)) or str(record.get("source_view", "")) != "full_cccv":
        raise FullSourceUnavailable(
            "A terminal-only record cannot be used as full_cccv; "
            "the source record must carry is_full=true and source_view='full_cccv'."
        )
    missing = [key for key in FULL_REQUIRED_RECORD_KEYS if key not in record]
    if missing:
        raise FullSourceUnavailable(f"Full record is missing required keys: {missing}")
    arrays = {key: _as_array(record, key) for key in ("segment", "time", "voltage", "current", "temperature")}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 4:
        raise FullSourceUnavailable("Full record has inconsistent or insufficient point arrays")
    segments = np.asarray([str(value).upper() for value in arrays["segment"]], dtype=object)
    if "CC" not in segments or "CV" not in segments:
        raise FullSourceUnavailable("Full record must contain both CC and CV points")
    if not all(np.all(np.isfinite(value.astype(float, copy=False))) for key, value in arrays.items() if key != "segment"):
        raise FullSourceUnavailable("Full record contains non-finite point values")


def _normalise_label(value: float, nominal_capacity: float, mode: str) -> float:
    mode = str(mode or "none")
    if mode == "none":
        return float(value)
    if mode in {"capacity_to_soh", "auto_capacity_to_soh"}:
        if mode == "capacity_to_soh" or abs(float(value)) > 1.2:
            return float(value) / float(nominal_capacity)
        return float(value)
    raise ValueError(f"Unsupported full-source label_scale_mode: {mode}")


def _read_csv_full_records(root: Path, domain_id: str, data_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Read an explicit normalized full CSV export.

    This format is useful for a regenerated full product without coupling the
    training namespace to a vendor-specific source parser. Identity must be
    present in columns; it is never inferred from the filename.
    """

    files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    if not files:
        raise FullSourceUnavailable(f"No full-source CSV files found under {root}")
    nominal = float(data_config.get("nominal_capacity", 1.0))
    label_mode = data_config.get("full_label_scale_mode", "none")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    required = {"battery_id", "cycle_id", "condition", "segment", "voltage_V", "current_A", "temperature_C"}
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            time_key = "relative_time_min" if "relative_time_min" in fields else "relative_time"
            label_key = "soh" if "soh" in fields else ("SOH" if "SOH" in fields else None)
            missing = sorted(required - fields)
            if time_key not in fields:
                missing.append("relative_time_min or relative_time")
            if label_key is None:
                missing.append("soh or SOH")
            if missing:
                if path.name.lower().startswith(("manifest", "report", "audit")):
                    continue
                raise FullSourceUnavailable(f"Full CSV {path} is missing columns: {sorted(set(missing))}")
            for row in reader:
                try:
                    battery_id = str(row["battery_id"]).strip()
                    cycle_id = int(float(row["cycle_id"]))
                    condition = str(row["condition"]).strip()
                    segment = str(row["segment"]).strip().upper()
                    time = float(row[time_key])
                    voltage = float(row["voltage_V"])
                    current = float(row["current_A"])
                    temperature = float(row["temperature_C"])
                    label = _normalise_label(float(row[label_key]), nominal, label_mode)
                except (TypeError, ValueError) as exc:
                    raise FullSourceUnavailable(f"Invalid full-source row in {path}: {row}") from exc
                if not battery_id or not condition or segment not in {"CC", "CV"}:
                    raise FullSourceUnavailable(f"Full-source row has invalid identity/segment in {path}: {row}")
                grouped[(battery_id, cycle_id)].append(
                    {
                        "segment": segment,
                        "time": time,
                        "voltage": voltage,
                        "current": current,
                        "temperature": temperature,
                        "soh": label,
                        "condition": condition,
                        "battery_id": battery_id,
                        "cycle_id": cycle_id,
                        "source_file": str(path),
                    }
                )
    records = []
    for (battery_id, cycle_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item["time"])
        first = rows[0]
        record = {
            "dataset_id": str(domain_id),
            "domain_id": str(domain_id),
            "condition": first["condition"],
            "battery_id": battery_id,
            "cycle_id": cycle_id,
            "segment": np.asarray([row["segment"] for row in rows], dtype=object),
            "time": np.asarray([row["time"] for row in rows], dtype=np.float32),
            "voltage": np.asarray([row["voltage"] for row in rows], dtype=np.float32),
            "current": np.asarray([row["current"] for row in rows], dtype=np.float32),
            "temperature": np.asarray([row["temperature"] for row in rows], dtype=np.float32),
            "soh": float(first["soh"]),
            "soh_raw": float(first["soh"]),
            "source_file": first["source_file"],
            "source_view": "full_cccv",
            "is_full": True,
        }
        validate_full_record(record)
        records.append(record)
    if not records:
        raise FullSourceUnavailable(f"Full-source CSV root contains no usable records: {root}")
    return records


def _xjtu_battery_files(source_root: Path) -> list[Path]:
    return sorted(path for path in source_root.rglob("*.mat") if path.name != "Temperature_Compensation_Data.mat")


def _load_xjtu_full_records(
    source_root: Path,
    terminal_records: Iterable[dict[str, Any]],
    data_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Materialize XJTU full charging stages from the source MATLAB files."""

    files = _xjtu_battery_files(source_root)
    if not files:
        raise FullSourceUnavailable(f"No XJTU MATLAB files found under {source_root}")
    wanted = {(str(item["battery_id"]), int(item["cycle_id"])) for item in terminal_records}
    if not wanted:
        raise FullSourceUnavailable("No terminal cycle keys were supplied for full matching")
    try:
        from ..xjtu import parse_file_identity
        from ...preprocess.XJTUBatteryClass import Battery
    except Exception as exc:  # pragma: no cover - depends on local scipy/matplotlib
        raise FullSourceUnavailable("XJTU full source parser is unavailable") from exc

    labels = {(str(item["battery_id"]), int(item["cycle_id"])): item for item in terminal_records}
    records: list[dict[str, Any]] = []
    for path in files:
        condition, battery_id = parse_file_identity(path)
        local_wanted = sorted(cycle for battery, cycle in wanted if battery == battery_id)
        if not local_wanted:
            continue
        try:
            battery = Battery(str(path))
        except Exception as exc:
            raise FullSourceUnavailable(f"Cannot open XJTU full source {path}") from exc
        for cycle_id in local_wanted:
            try:
                if "test capacity" in str(battery.get_one_cycle_description(cycle_id)).lower():
                    continue
                values = {
                    key: np.asarray(battery.get_partial_value(cycle_id, key, stage=1), dtype=np.float32).reshape(-1)
                    for key in ("relative_time_min", "voltage_V", "current_A", "temperature_C")
                }
            except Exception as exc:
                raise FullSourceUnavailable(f"Cannot read XJTU full cycle {battery_id}/{cycle_id} from {path}") from exc
            lengths = {len(value) for value in values.values()}
            if len(lengths) != 1 or not lengths:
                raise FullSourceUnavailable(f"XJTU full cycle has inconsistent arrays: {path}, cycle {cycle_id}")
            order = np.argsort(values["relative_time_min"], kind="stable")
            time = values["relative_time_min"][order]
            voltage = values["voltage_V"][order]
            current = values["current_A"][order]
            temperature = values["temperature_C"][order]
            boundary_candidates = np.flatnonzero(voltage >= 4.199)
            if boundary_candidates.size == 0:
                raise FullSourceUnavailable(f"XJTU full cycle has no CC/CV boundary: {path}, cycle {cycle_id}")
            boundary = int(boundary_candidates[0])
            if boundary < 2 or boundary >= len(time) - 2:
                raise FullSourceUnavailable(f"XJTU full cycle has insufficient CC or CV points: {path}, cycle {cycle_id}")
            terminal = labels[(battery_id, cycle_id)]
            record = {
                "dataset_id": "xjtu",
                "domain_id": str(terminal.get("domain_id", "xjtu")),
                "condition": condition,
                "battery_id": battery_id,
                "cycle_id": cycle_id,
                "segment": np.asarray(["CC"] * boundary + ["CV"] * (len(time) - boundary), dtype=object),
                "time": time,
                "voltage": voltage,
                "current": current,
                "temperature": temperature,
                "soh": float(terminal["soh"]),
                "soh_raw": float(terminal.get("soh_raw", terminal["soh"])),
                "source_file": str(path),
                "source_view": "full_cccv",
                "is_full": True,
                "full_source_kind": "xjtu_mat_charging_stage",
            }
            validate_full_record(record)
            records.append(record)
    if not records:
        raise FullSourceUnavailable(
            "No XJTU full cycles matched canonical terminal identities; "
            "check source root and cycle provenance."
        )
    return records


def _parse_datetime(value: str, path: Path) -> datetime:
    text = str(value).strip().replace("/", "-").replace("T", " ")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
        for layout in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                parsed = datetime.strptime(text, layout)
                break
            except ValueError:
                continue
        if parsed is None:
            raise FullSourceUnavailable(f"Invalid source timestamp {value!r} in {path}")
    return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


def _smarthealth_boundary(current: np.ndarray, voltage: np.ndarray) -> int | None:
    if len(current) < 16:
        return None
    current = np.abs(current.astype(float))
    voltage = voltage.astype(float)
    reference_count = min(len(current), max(8, int(math.ceil(0.2 * len(current)))))
    reference = float(np.quantile(current[:reference_count], 0.90))
    if not np.isfinite(reference) or reference <= 0:
        return None
    taper = current <= reference * 0.99
    voltage_max = float(np.max(voltage))
    for index in range(8, len(current) - 8 + 1):
        if index + 5 > len(current) or not np.all(taper[index : index + 5]):
            continue
        if voltage[index] >= voltage_max - 0.02:
            return index
    return None


def _load_smarthealth_full_records(
    source_root: Path,
    terminal_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read full SmartHealth charge events through canonical source linkage."""

    records: list[dict[str, Any]] = []
    for terminal in terminal_records:
        source_name = str(terminal.get("source_file", "")).strip()
        source_cycle = terminal.get("source_cycle")
        start_value = terminal.get("source_absolute_start_time")
        end_value = terminal.get("source_absolute_end_time")
        if not source_name or source_cycle is None or start_value is None or end_value is None:
            raise FullSourceUnavailable(
                f"SmartHealth terminal record lacks source linkage for "
                f"{terminal.get('battery_id')}/{terminal.get('cycle_id')}"
            )
        source_path = Path(source_name)
        if not source_path.is_absolute():
            candidates = [source_root / source_path]
            if source_root.parent != source_root:
                candidates.append(source_root.parent / source_path)
            source_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if not source_path.is_file():
            raise FullSourceUnavailable(f"Linked SmartHealth full source is missing: {source_path}")
        start = _parse_datetime(str(start_value), source_path)
        end = _parse_datetime(str(end_value), source_path)
        rows = []
        with source_path.open("r", encoding="gb18030", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"循环号", "工步类型", "绝对时间", "电压(V)", "电流(A)", "temp1_1"}
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise FullSourceUnavailable(f"SmartHealth full source {source_path} is missing {missing}")
            for row_index, row in enumerate(reader):
                try:
                    if int(float(row["循环号"])) != int(source_cycle):
                        continue
                    if str(row["工步类型"]).strip() != "恒流恒压充电":
                        continue
                    timestamp = _parse_datetime(row["绝对时间"], source_path)
                    if timestamp < start or timestamp > end:
                        continue
                    rows.append(
                        (
                            timestamp,
                            int(row_index),
                            float(row["电压(V)"]),
                            float(row["电流(A)"]),
                            float(row["temp1_1"]),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise FullSourceUnavailable(f"Invalid SmartHealth full row in {source_path}: {row}") from exc
        if len(rows) < 16:
            raise FullSourceUnavailable(
                f"Linked SmartHealth event has too few full charge points: "
                f"{terminal.get('battery_id')}/{terminal.get('cycle_id')}"
            )
        rows.sort(key=lambda item: item[0])
        times = np.asarray([(item[0] - rows[0][0]).total_seconds() / 60.0 for item in rows], dtype=np.float32)
        voltage = np.asarray([item[2] for item in rows], dtype=np.float32)
        current = np.asarray([item[3] for item in rows], dtype=np.float32)
        temperature = np.asarray([item[4] for item in rows], dtype=np.float32)
        boundary = _smarthealth_boundary(current, voltage)
        if boundary is None:
            raise FullSourceUnavailable(
                f"Could not infer a persistent CC/CV boundary for linked SmartHealth event: "
                f"{terminal.get('battery_id')}/{terminal.get('cycle_id')}"
            )
        record = {
            "dataset_id": str(terminal.get("dataset_id", terminal.get("domain_id"))),
            "domain_id": str(terminal.get("domain_id")),
            "condition": str(terminal.get("condition")),
            "battery_id": str(terminal.get("battery_id")),
            "cycle_id": int(terminal.get("cycle_id")),
            "segment": np.asarray(["CC"] * boundary + ["CV"] * (len(times) - boundary), dtype=object),
            "time": times,
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "soh": float(terminal["soh"]),
            "soh_raw": float(terminal.get("soh_raw", terminal["soh"])),
            "source_file": str(source_path),
            "source_cycle": int(source_cycle),
            "source_view": "full_cccv",
            "is_full": True,
            "full_source_kind": "smarthealth_gb18030_charge_event",
        }
        validate_full_record(record)
        records.append(record)
    return records


def materialize_full_records(
    terminal_records: Iterable[dict[str, Any]],
    *,
    domain_id: str,
    data_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load full records only through an explicitly configured source."""

    source_value = data_config.get("full_data_root")
    if not source_value:
        raise FullSourceUnavailable(
            f"No full_data_root configured for {domain_id}; canonical terminal data cannot be used as full data."
        )
    root = Path(source_value).expanduser().resolve()
    if not root.is_dir():
        raise FullSourceUnavailable(f"Configured full_data_root does not exist: {root}")
    terminal_records = list(terminal_records)
    source_format = str(data_config.get("full_source_format", "auto"))
    if source_format == "csv" or (source_format == "auto" and any(root.rglob("*.csv"))):
        # SmartHealth needs canonical source linkage; a normalized full CSV
        # export is explicitly selected with full_source_format=csv.
        if source_format == "csv":
            return _read_csv_full_records(root, domain_id, data_config)
    if domain_id == "xjtu":
        return _load_xjtu_full_records(root, terminal_records, data_config)
    if domain_id.startswith("smarthealth_"):
        return _load_smarthealth_full_records(root, terminal_records)
    return _read_csv_full_records(root, domain_id, data_config)


def match_full_terminal_records(
    terminal_records: Iterable[dict[str, Any]],
    full_records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair views by physical battery/cycle identity and report exclusions."""

    terminal = list(terminal_records)
    full = list(full_records)
    full_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_full = []
    for record in full:
        validate_full_record(record)
        key = (str(record["battery_id"]), int(record["cycle_id"]))
        if key in full_by_key:
            duplicate_full.append(key)
        full_by_key[key] = record
    if duplicate_full:
        raise FullSourceUnavailable(f"Full source has duplicate physical cycle identities: {duplicate_full[:5]}")
    matched = []
    missing = []
    label_mismatch = []
    for record in terminal:
        key = (str(record["battery_id"]), int(record["cycle_id"]))
        candidate = full_by_key.get(key)
        if candidate is None:
            missing.append({"battery_id": key[0], "cycle_id": key[1], "reason": "full_cycle_missing"})
            continue
        if not math.isclose(float(record["soh"]), float(candidate["soh"]), rel_tol=1e-5, abs_tol=1e-6):
            label_mismatch.append({"battery_id": key[0], "cycle_id": key[1], "reason": "label_mismatch"})
            continue
        if str(record.get("condition")) != str(candidate.get("condition")):
            raise FullSourceUnavailable(f"Full/terminal strategy mismatch for {key}: {record.get('condition')} vs {candidate.get('condition')}")
        candidate = dict(candidate)
        candidate["split"] = record.get("split")
        candidate["terminal_source_file"] = record.get("source_file")
        matched.append(candidate)
    audit = {
        "terminal_records": len(terminal),
        "full_records": len(full),
        "matched_records": len(matched),
        "missing_records": missing,
        "label_mismatch_records": label_mismatch,
        "matched_batteries": sorted({str(item["battery_id"]) for item in matched}),
        "matched_cycles_by_battery": {
            battery: sum(str(item["battery_id"]) == battery for item in matched)
            for battery in sorted({str(item["battery_id"]) for item in matched})
        },
        "pair_key": "(physical battery_id, cycle_id)",
    }
    if not matched:
        raise FullSourceUnavailable(f"No full/terminal physical cycles matched; audit={audit}")
    return matched, audit


__all__ = [
    "FullSourceUnavailable",
    "FULL_REQUIRED_RECORD_KEYS",
    "match_full_terminal_records",
    "materialize_full_records",
    "validate_full_record",
]
