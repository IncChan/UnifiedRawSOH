"""XJTU point-level raw adapter and the common raw-cycle sample dataset."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from UnifiedRawSOH.data.normalization import PhysicalWindowNormalizer


REQUIRED_COLUMNS = {
    "cycle",
    "SOH",
    "segment",
    "relative_time_min",
    "voltage_V",
    "current_A",
    "temperature_C",
}


def parse_file_identity(path):
    stem = Path(path).stem
    if "_battery-" not in stem:
        return stem, stem
    condition, battery_suffix = stem.split("_battery-", 1)
    if condition == "Sim_satellite":
        condition = "satellite"
    return condition, f"{condition}_battery-{battery_suffix}"


def list_xjtu_csv_files(data_root, batch=None):
    root = Path(data_root)
    if not root.is_dir():
        raise ValueError(f"XJTU data root does not exist: {root}")
    files = []
    for path in sorted(root.glob("*.csv")):
        if path.name.endswith("_report.csv"):
            continue
        if batch is not None:
            matches = path.name.startswith("Sim_satellite") if batch == "satellite" else path.name.startswith(f"{batch}_")
            if not matches:
                continue
        files.append(path)
    if not files:
        suffix = f" for batch {batch!r}" if batch is not None else ""
        raise ValueError(f"No XJTU battery CSV files found under {root}{suffix}")
    return files


def read_xjtu_file(path, nominal_capacity=2.0, label_scale_mode="auto_capacity_to_soh"):
    """Read one point-level XJTU file into cycle records.

    This follows the v2 C5B reader semantics: rows are grouped by physical
    cycle, sorted by relative time, and SOH is auto-scaled from capacity-like
    values to a unit SOH using the fixed nominal capacity.
    """

    condition, battery_id = parse_file_identity(path)
    grouped = {}
    cycle_order = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            try:
                cycle_id = int(float(row["cycle"]))
                item = (
                    str(row["segment"]).strip().upper(),
                    float(row["relative_time_min"]),
                    float(row["voltage_V"]),
                    float(row["current_A"]),
                    float(row["temperature_C"]),
                    float(row["SOH"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric row in {path}: {row}") from exc
            if cycle_id not in grouped:
                grouped[cycle_id] = []
                cycle_order.append(cycle_id)
            grouped[cycle_id].append(item)

    records = []
    for raw_order_index, cycle_id in enumerate(cycle_order):
        rows = sorted(grouped[cycle_id], key=lambda item: item[1])
        segments = np.asarray([item[0] for item in rows], dtype=object)
        times = np.asarray([item[1] for item in rows], dtype=np.float32)
        voltage = np.asarray([item[2] for item in rows], dtype=np.float32)
        current = np.asarray([item[3] for item in rows], dtype=np.float32)
        temperature = np.asarray([item[4] for item in rows], dtype=np.float32)
        soh_values = np.asarray([item[5] for item in rows], dtype=np.float32)
        if not np.allclose(soh_values, soh_values[0], rtol=1e-5, atol=1e-6):
            raise ValueError(f"SOH is not constant within cycle {cycle_id} in {path}")
        raw_soh = float(soh_values[0])
        soh, scale_factor, resolved_mode = _normalize_soh(
            raw_soh, nominal_capacity, label_scale_mode
        )
        records.append(
            {
                "dataset_id": "xjtu",
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
                "soh_scale_factor": float(scale_factor),
                "soh_scale_mode": resolved_mode,
                "nominal_capacity": float(nominal_capacity),
                "source_file": str(path),
            }
        )
    return records


def _normalize_soh(raw_soh, nominal_capacity, label_scale_mode):
    mode = str(label_scale_mode or "auto_capacity_to_soh")
    nominal_capacity = float(nominal_capacity)
    if nominal_capacity <= 0:
        raise ValueError("nominal_capacity must be positive.")
    if mode == "none":
        return float(raw_soh), 1.0, mode
    if mode == "capacity_to_soh":
        return float(raw_soh) / nominal_capacity, nominal_capacity, mode
    if mode == "auto_capacity_to_soh":
        if abs(float(raw_soh)) > 1.2:
            return float(raw_soh) / nominal_capacity, nominal_capacity, "auto_capacity_to_soh_applied"
        return float(raw_soh), 1.0, "auto_capacity_to_soh_noop"
    raise ValueError(f"Unsupported label_scale_mode: {mode}")


def load_xjtu_records(data_root, batch=None, nominal_capacity=2.0, label_scale_mode="auto_capacity_to_soh"):
    records = []
    for path in list_xjtu_csv_files(data_root, batch=batch):
        records.extend(read_xjtu_file(path, nominal_capacity, label_scale_mode))
    if not records:
        raise ValueError(f"No XJTU records loaded from {data_root}")
    return records


class XJTURawAdapter:
    """Dataset adapter whose output is the common raw-cycle record contract."""

    dataset_id = "xjtu"
    raw_terminal_signals = True

    def __init__(self, data_root, nominal_capacity=2.0, label_scale_mode="auto_capacity_to_soh"):
        self.data_root = Path(data_root)
        self.nominal_capacity = float(nominal_capacity)
        self.label_scale_mode = str(label_scale_mode)

    def load_records(self, batch=None):
        return load_xjtu_records(
            self.data_root,
            batch=batch,
            nominal_capacity=self.nominal_capacity,
            label_scale_mode=self.label_scale_mode,
        )


def build_full_life_cycle_metadata(records):
    by_battery = {}
    for record in records:
        by_battery.setdefault(str(record["battery_id"]), []).append(int(record["raw_cycle_order_index"]))
    return {
        battery_id: {
            "full_life_min_cycle": min(indices),
            "full_life_max_cycle": max(indices),
            "full_life_cycle_count": len(indices),
        }
        for battery_id, indices in by_battery.items()
    }


class UnifiedCCCVSampleDataset(Dataset):
    """Leakage-free XJTU samples under the paper-wide sample interface.

    Each item includes both the stable public names (``cc``, ``cv``, ``t0``)
    and explicit model names (``cc_signal``, ``cv_signal``,
    ``t0_temperature_norm``) so adapters can be tested independently of the
    model implementation.
    """

    def __init__(self, records, data_config, normalization_config, split_name, seed=0, cycle_metadata=None):
        self.records = list(records)
        self.data_config = dict(data_config)
        self.split_name = str(split_name)
        self.seed = int(seed)
        self.normalizer = PhysicalWindowNormalizer(normalization_config)
        self.raw_len_cc = int(self.data_config.get("raw_len_cc", 128))
        self.raw_len_cv = int(self.data_config.get("raw_len_cv", 256))
        self.min_cc_points = int(self.data_config.get("min_cc_points", 4))
        self.min_cv_points = int(self.data_config.get("min_cv_points", 4))
        self.use_time = bool(self.data_config.get("use_real_time", True))
        self.use_temperature = bool(self.data_config.get("use_temperature", True))
        self.use_t0 = bool(self.data_config.get("use_t0_temperature_meta", True))
        self.temperature_reference_c = float(self.data_config.get("temperature_reference_c", 25.0))
        self.temperature_scale_c = float(self.data_config.get("temperature_scale_c", 20.0))
        self.temperature_delta_scale_c = float(self.data_config.get("temperature_delta_scale_c", 10.0))
        self.t0_reference_c = float(self.data_config.get("t0_temperature_reference_c", 25.0))
        self.t0_scale_c = float(self.data_config.get("t0_temperature_scale_c", 20.0))
        if self.use_t0 and not self.use_temperature:
            raise ValueError("use_t0_temperature_meta requires use_temperature=true.")
        self.cycle_metadata = cycle_metadata or build_full_life_cycle_metadata(self.records)
        self.skipped = Counter()
        self.samples = []
        self.valid_records = []
        for record in self.records:
            sample = self._make_sample(record)
            if sample is not None:
                self.samples.append(sample)
                self.valid_records.append(record)
        if not self.samples:
            raise ValueError(f"No usable XJTU samples for split {split_name!r}: {dict(self.skipped)}")

    def _make_sample(self, record):
        if not np.isfinite(record["soh"]) or not np.isfinite(record["soh_raw"]):
            self.skipped["nonfinite_label"] += 1
            return None
        segments = np.asarray(record["segment"], dtype=object)
        cc_mask = np.asarray([str(item).upper() == "CC" for item in segments], dtype=bool)
        cv_mask = np.asarray([str(item).upper() == "CV" for item in segments], dtype=bool)
        if int(cc_mask.sum()) < self.min_cc_points:
            self.skipped["too_few_cc_points"] += 1
            return None
        if int(cv_mask.sum()) < self.min_cv_points:
            self.skipped["too_few_cv_points"] += 1
            return None
        charge_mask = cc_mask | cv_mask
        charge_time = np.asarray(record["time"][charge_mask], dtype=np.float32)
        charge_temperature = (
            np.asarray(record["temperature"][charge_mask], dtype=np.float32)
            if self.use_temperature
            else None
        )
        if self.use_time and (not np.all(np.isfinite(charge_time)) or charge_time.size == 0):
            self.skipped["invalid_charge_time"] += 1
            return None
        if self.use_temperature and (
            not np.all(np.isfinite(charge_temperature)) or charge_temperature.size == 0
        ):
            self.skipped["invalid_charge_temperature"] += 1
            return None
        time_zero = float(np.min(charge_time)) if self.use_time else 0.0
        temperature_zero = (
            float(charge_temperature[int(np.argmin(charge_time))])
            if self.use_temperature
            else 0.0
        )
        cc = self._phase(record, cc_mask, self.raw_len_cc, "CC", time_zero, temperature_zero)
        cv = self._phase(record, cv_mask, self.raw_len_cv, "CV", time_zero, temperature_zero)
        if cc is None or cv is None:
            return None

        battery_id = str(record["battery_id"])
        metadata = self.cycle_metadata.get(battery_id)
        if metadata is None:
            raise ValueError(f"Missing complete lifetime metadata for {battery_id!r}.")
        denominator = int(metadata["full_life_max_cycle"]) - int(metadata["full_life_min_cycle"])
        cycle_norm = 0.0 if denominator <= 0 else (
            2.0 * (int(record["raw_cycle_order_index"]) - int(metadata["full_life_min_cycle"])) / denominator - 1.0
        )
        if not -1.000001 <= cycle_norm <= 1.000001:
            raise ValueError(f"Cycle target is outside [-1, 1] for {battery_id!r}.")
        t0 = (
            np.asarray([(temperature_zero - self.t0_reference_c) / self.t0_scale_c], dtype=np.float32)
            if self.use_t0
            else np.zeros(1, dtype=np.float32)
        )
        sample = {
            "cc_signal": cc["signal"],
            "cv_signal": cv["signal"],
            "cc_mask": np.ones(self.raw_len_cc, dtype=np.float32),
            "cv_mask": np.ones(self.raw_len_cv, dtype=np.float32),
            "cc_time": cc["time"],
            "cv_time": cv["time"],
            "cc_temperature": cc["temperature"],
            "cv_temperature": cv["temperature"],
            "t0_temperature_norm": t0,
            "soh": np.asarray([record["soh"]], dtype=np.float32),
            "soh_raw": float(record["soh_raw"]),
            "cycle_life_norm_target": np.asarray([cycle_norm], dtype=np.float32),
            "battery_id": battery_id,
            "dataset_id": str(record.get("dataset_id", "xjtu")),
            "domain_id": str(record.get("domain_id", record.get("dataset_id", "xjtu"))),
            "condition": str(record["condition"]),
            "batch_name": str(record["condition"]),
            "cycle_id": int(record["cycle_id"]),
            "split": self.split_name,
        }
        return sample

    def _phase(self, record, mask, target_len, phase, time_zero, temperature_zero):
        time = np.asarray(record["time"][mask], dtype=np.float32)
        order = np.argsort(time, kind="stable")
        time = time[order]
        if time.size < 2 or not np.all(np.isfinite(time)):
            self.skipped[f"invalid_{phase.lower()}_time"] += 1
            return None
        span = float(time[-1] - time[0])
        if not np.isfinite(span) or span <= 0:
            self.skipped[f"invalid_{phase.lower()}_time_span"] += 1
            return None
        if phase == "CC":
            physical = np.asarray(record["voltage"][mask], dtype=np.float32)[order]
        else:
            physical = np.asarray(record["current"][mask], dtype=np.float32)[order]
        if not np.all(np.isfinite(physical)):
            self.skipped[f"nonfinite_{phase.lower()}_signal"] += 1
            return None
        temperature = None
        if self.use_temperature:
            temperature = np.asarray(record["temperature"][mask], dtype=np.float32)[order]
            if not np.all(np.isfinite(temperature)):
                self.skipped[f"nonfinite_{phase.lower()}_temperature"] += 1
                return None
        sample_time = np.linspace(float(time[0]), float(time[-1]), int(target_len), dtype=np.float32)
        physical = np.interp(sample_time, time, physical).astype(np.float32)
        if phase == "CC":
            physical = self.normalizer.normalize_cc_voltage(physical)
        else:
            physical = self.normalizer.normalize_cv_current(physical)
        tau = ((sample_time - float(time[0])) / span).astype(np.float32)
        signal = np.stack([physical, 2.0 * tau - 1.0], axis=-1).astype(np.float32)
        phase_time = (sample_time - time_zero).astype(np.float32)
        if self.use_temperature:
            temperature = np.interp(sample_time, time, temperature).astype(np.float32)
            temp_abs = ((temperature - self.temperature_reference_c) / self.temperature_scale_c).astype(np.float32)
            temp_delta = ((temperature - temperature_zero) / self.temperature_delta_scale_c).astype(np.float32)
            temp = np.stack([temp_abs, temp_delta], axis=-1).astype(np.float32)
        else:
            temp = np.zeros((int(target_len), 2), dtype=np.float32)
        if not (np.all(np.isfinite(signal)) and np.all(np.isfinite(phase_time)) and np.all(np.isfinite(temp))):
            self.skipped[f"nonfinite_{phase.lower()}_interpolation"] += 1
            return None
        return {"signal": signal, "time": phase_time, "temperature": temp}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        output = {
            "cc": torch.from_numpy(sample["cc_signal"]),
            "cv": torch.from_numpy(sample["cv_signal"]),
            "cc_signal": torch.from_numpy(sample["cc_signal"]),
            "cv_signal": torch.from_numpy(sample["cv_signal"]),
            "cc_mask": torch.from_numpy(sample["cc_mask"]),
            "cv_mask": torch.from_numpy(sample["cv_mask"]),
            "cc_time": torch.from_numpy(sample["cc_time"]),
            "cv_time": torch.from_numpy(sample["cv_time"]),
            "cc_temperature": torch.from_numpy(sample["cc_temperature"]),
            "cv_temperature": torch.from_numpy(sample["cv_temperature"]),
            "t0": torch.from_numpy(sample["t0_temperature_norm"]),
            "t0_temperature_norm": torch.from_numpy(sample["t0_temperature_norm"]),
            "soh": torch.from_numpy(sample["soh"]),
            "cycle_life_norm_target": torch.from_numpy(sample["cycle_life_norm_target"]),
            "soh_raw": sample["soh_raw"],
            "battery_id": sample["battery_id"],
            "dataset_id": sample["dataset_id"],
            "domain_id": sample["domain_id"],
            "condition": sample["condition"],
            "batch_name": sample["batch_name"],
            "cycle_id": sample["cycle_id"],
            "split": sample["split"],
        }
        return output
