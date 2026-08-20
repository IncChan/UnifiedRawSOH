"""Protocol-level sample filters shared by raw and statistical experiments.

The Paper-v1 raw model does not consume handcrafted features.  The optional
PINN F-only filter only reproduces the paired experiment's retained cycle
membership (3-sigma cleaning plus adjacent-x1 sampling) so that E1 compares
the two models on the same samples.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def filter_records_by_invalid_cycles(records, invalid_cycles):
    """Remove source-documented invalid cycles by battery and cycle ID."""

    invalid_keys = {
        (str(item["battery_id"]), int(item["cycle_id"]))
        for item in invalid_cycles
    }
    if not invalid_keys:
        return list(records), {
            "enabled": False,
            "removed_records": 0,
            "invalid_cycles": [],
        }
    kept = []
    removed = []
    for record in records:
        key = (str(record["battery_id"]), int(record["cycle_id"]))
        if key in invalid_keys:
            removed.append(key)
        else:
            kept.append(record)
    return kept, {
        "enabled": True,
        "removed_records": len(removed),
        "invalid_cycles": [
            {"battery_id": battery_id, "cycle_id": cycle_id}
            for battery_id, cycle_id in sorted(invalid_keys)
        ],
        "matched_records": [
            {"battery_id": battery_id, "cycle_id": cycle_id}
            for battery_id, cycle_id in sorted(set(removed))
        ],
    }


def _retained_reference_rows(reference_path, cycle_column="cycle index"):
    import pandas as pd

    frame = pd.read_csv(reference_path)
    if "capacity" not in frame.columns:
        raise ValueError(f"PINN F-only reference is missing capacity: {reference_path}")
    if cycle_column in frame.columns:
        raise ValueError(f"Reserved cycle column already exists in {reference_path}")
    frame.insert(frame.shape[1] - 1, cycle_column, np.arange(len(frame), dtype=np.int64))
    values = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    outlier_indices = set()
    for column in values.columns:
        series = values[column]
        std = series.std()
        if not math.isfinite(float(std)) or float(std) == 0.0:
            continue
        rule = (series.mean() - 3 * std > series) | (series.mean() + 3 * std < series)
        outlier_indices.update(np.flatnonzero(rule.to_numpy()).tolist())
    if outlier_indices:
        values = values.drop(sorted(outlier_indices), axis=0)
    return {
        "reference_rows": int(len(frame)),
        "retained_row_indices": values[cycle_column].to_numpy(dtype=np.int64),
        "capacity": frame["capacity"].to_numpy(dtype=np.float64),
    }


def filter_raw_records_to_pinn_fonly_samples(
    records,
    reference_root,
    match_atol=1e-6,
    drop_adjacent_x1_last=True,
):
    """Retain raw cycles corresponding to the validated PINN F-only samples."""

    reference_root = Path(reference_root)
    if not reference_root.is_dir():
        raise ValueError(f"PINN F-only reference root is not a directory: {reference_root}")
    match_atol = float(match_atol)
    if match_atol <= 0.0:
        raise ValueError("match_atol must be positive")

    grouped = {}
    for record in records:
        grouped.setdefault(str(record["battery_id"]), []).append(record)

    filtered = []
    per_battery = {}
    for battery_id, battery_records in grouped.items():
        source_names = {Path(record["source_file"]).name for record in battery_records}
        if len(source_names) != 1:
            raise ValueError(f"Expected one raw source file for {battery_id}, got {source_names}")
        source_name = next(iter(source_names))
        reference_path = reference_root / source_name
        if not reference_path.is_file():
            raise ValueError(f"Missing PINN F-only reference file: {reference_path}")

        reference = _retained_reference_rows(reference_path)
        retained = reference["retained_row_indices"]
        sample_rows = retained[:-1] if drop_adjacent_x1_last else retained
        sample_row_set = {int(index) for index in sample_rows}
        capacities = reference["capacity"]
        cursor = 0
        mapped = []
        max_match_error = 0.0
        for record in battery_records:
            target = float(record["soh_raw"])
            matched_index = None
            matched_error = None
            while cursor < len(capacities):
                error = abs(float(capacities[cursor]) - target)
                if error <= match_atol:
                    matched_index = int(cursor)
                    matched_error = float(error)
                    cursor += 1
                    break
                cursor += 1
            if matched_index is None:
                raise ValueError(
                    f"Could not map raw cycle {record['cycle_id']} ({target:.12g}) for {battery_id} "
                    f"to {reference_path} within atol={match_atol}."
                )
            item = dict(record)
            item["pinn_fonly_reference_row_index"] = matched_index
            mapped.append(item)
            max_match_error = max(max_match_error, float(matched_error))

        battery_filtered = [
            record
            for record in mapped
            if int(record["pinn_fonly_reference_row_index"]) in sample_row_set
        ]
        filtered.extend(battery_filtered)
        per_battery[battery_id] = {
            "reference_file": str(reference_path),
            "reference_rows": int(reference["reference_rows"]),
            "reference_rows_after_3sigma": int(len(retained)),
            "raw_records_before": int(len(mapped)),
            "retained_raw_records": int(len(battery_filtered)),
            "removed_raw_records": int(len(mapped) - len(battery_filtered)),
            "max_capacity_match_abs_error": float(max_match_error),
        }

    if not filtered:
        raise ValueError("PINN F-only sample filter removed every raw cycle")
    return filtered, {
        "enabled": True,
        "mode": "pinn_fonly_3sigma_adjacent_x1",
        "reference_root": str(reference_root),
        "filter_scope": "reference statistical rows; never a model input",
        "drop_adjacent_x1_last": bool(drop_adjacent_x1_last),
        "match_key": "monotonic_capacity_to_raw_SOH",
        "match_atol": float(match_atol),
        "raw_records_before": int(len(records)),
        "raw_records_after": int(len(filtered)),
        "removed_raw_records": int(len(records) - len(filtered)),
        "per_battery": per_battery,
    }
