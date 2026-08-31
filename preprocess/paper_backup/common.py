"""Shared numerical contract for Paper-Backup offline materialization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PAPER_BACKUP_PREPROCESS_POLICY = "paper_backup_preprocessed_v1"
PAPER_BACKUP_PREPROCESS_SCHEMA = 1
PAPER_BACKUP_PREPROCESS_POLICIES = {
    1: "paper_backup_preprocessed_v1",
    2: "paper_backup_preprocessed_crate_v2",
}
RICH_CHANNEL_NAMES = (
    "voltage_norm",
    "current_norm",
    "relative_time_norm",
    "temperature_abs_norm",
    "temperature_delta_norm",
    "phase_signal_norm",
    "phase_tau",
)
RICH_CHANNEL_NAMES_BY_SCHEMA = {
    1: RICH_CHANNEL_NAMES,
    2: (
        "voltage_norm",
        "current_c_rate",
        "relative_time_norm",
        "temperature_abs_norm",
        "temperature_delta_norm",
        "legacy_phase_signal_norm",
        "phase_tau",
    ),
}
FEATURE_NAMES = (
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
    "T_CC_mean",
    "T_CC_max",
    "T_CC_delta",
    "T_CC_slope",
    "T_CV_mean",
    "T_CV_max",
    "T_CV_delta",
    "T_CV_slope",
)


def finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain at least two finite values")
    return array


def phase_arrays(record: Mapping[str, Any], phase: str) -> dict[str, np.ndarray]:
    segments = np.asarray([str(item).upper() for item in record["segment"]], dtype=object)
    mask = segments == str(phase).upper()
    if int(mask.sum()) < 2:
        raise ValueError(f"Cycle {record.get('cycle_id')} lacks two {phase} points")
    time = finite_array(np.asarray(record["time"])[mask], f"{phase}.time")
    order = np.argsort(time, kind="stable")
    time = time[order]
    unique_time, unique_index = np.unique(time, return_index=True)
    if unique_time.size < 2 or float(unique_time[-1] - unique_time[0]) <= 0:
        raise ValueError(f"Cycle {record.get('cycle_id')} has invalid {phase} time span")
    output = {"time": unique_time}
    for key in ("voltage", "current", "temperature"):
        values = finite_array(np.asarray(record[key])[mask], f"{phase}.{key}")
        output[key] = values[order][unique_index]
    return output


def resample_phase(source: Mapping[str, np.ndarray], target_len: int) -> dict[str, np.ndarray]:
    target_len = int(target_len)
    if target_len < 2:
        raise ValueError("Target phase length must be at least two")
    sample_time = np.linspace(
        float(source["time"][0]),
        float(source["time"][-1]),
        target_len,
        dtype=np.float64,
    )
    return {
        "time": sample_time,
        **{
            key: np.interp(sample_time, source["time"], source[key])
            for key in ("voltage", "current", "temperature")
        },
    }


def preprocessing_policy(schema_version: int) -> str:
    try:
        return PAPER_BACKUP_PREPROCESS_POLICIES[int(schema_version)]
    except KeyError as exc:
        raise ValueError(f"Unsupported Paper-Backup preprocessing schema: {schema_version}") from exc


def rich_channel_names(schema_version: int) -> tuple[str, ...]:
    try:
        return RICH_CHANNEL_NAMES_BY_SCHEMA[int(schema_version)]
    except KeyError as exc:
        raise ValueError(f"Unsupported Paper-Backup preprocessing schema: {schema_version}") from exc


def normalization_contract(
    config: Mapping[str, Any], schema_version: int | None = None
) -> dict[str, Any]:
    norm = dict(config.get("normalization", {}))
    data = dict(config.get("data", {}))
    schema_version = int(
        schema_version
        if schema_version is not None
        else data.get("preprocessed_schema_version", PAPER_BACKUP_PREPROCESS_SCHEMA)
    )
    values = {
        "voltage_low": float(norm.get("raw_voltage_low", norm["cc_voltage_low"])),
        "voltage_high": float(norm.get("raw_voltage_high", norm["cv_voltage_ref"])),
        "current_scale": float(norm.get("raw_current_scale", norm["cc_current_ref"])),
        "cc_voltage_low": float(norm["cc_voltage_low"]),
        "cc_voltage_high": float(norm["cc_voltage_high"]),
        "cv_current_low": float(norm["cv_current_low"]),
        "cv_current_high": float(norm["cv_current_high"]),
        "temp_room": float(norm.get("temp_room", 25.0)),
        "temp_abs_scale": float(norm.get("temp_abs_scale", 20.0)),
        "temp_delta_scale": float(norm.get("temp_delta_scale", 10.0)),
        "time_scale_min": float(data.get("time_scale_min", 10.0)),
        "schema_version": schema_version,
        "current_mode": (
            "nominal_c_rate" if schema_version == 2 else "legacy_affine_current"
        ),
    }
    if schema_version == 2:
        values["nominal_capacity_ah"] = float(data.get("nominal_capacity", 0.0))
    positive = (
        "current_scale",
        "temp_abs_scale",
        "temp_delta_scale",
        "time_scale_min",
    )
    if any(values[key] <= 0 for key in positive):
        raise ValueError(f"Invalid Paper-Backup normalization scales: {values}")
    if schema_version == 2 and values["nominal_capacity_ah"] <= 0:
        raise ValueError("C-rate preprocessing requires a positive fixed nominal capacity")
    if values["voltage_high"] <= values["voltage_low"]:
        raise ValueError("Invalid global voltage normalization range")
    if values["cc_voltage_high"] <= values["cc_voltage_low"]:
        raise ValueError("Invalid terminal CC voltage normalization range")
    if values["cv_current_high"] <= values["cv_current_low"]:
        raise ValueError("Invalid terminal CV current normalization range")
    return values


def rich_phase_tensor(
    phase: Mapping[str, np.ndarray],
    *,
    phase_name: str,
    time_zero: float,
    temperature_zero: float,
    normalization: Mapping[str, float],
) -> np.ndarray:
    voltage = np.asarray(phase["voltage"], dtype=np.float64)
    current = np.abs(np.asarray(phase["current"], dtype=np.float64))
    time = np.asarray(phase["time"], dtype=np.float64)
    temperature = np.asarray(phase["temperature"], dtype=np.float64)
    voltage_norm = (
        2.0
        * (voltage - normalization["voltage_low"])
        / (normalization["voltage_high"] - normalization["voltage_low"])
        - 1.0
    )
    if normalization.get("current_mode") == "nominal_c_rate":
        current_norm = current / normalization["nominal_capacity_ah"]
    else:
        current_norm = 2.0 * current / normalization["current_scale"] - 1.0
    relative_time_norm = (time - float(time_zero)) / normalization["time_scale_min"]
    temperature_abs_norm = (
        temperature - normalization["temp_room"]
    ) / normalization["temp_abs_scale"]
    temperature_delta_norm = (
        temperature - float(temperature_zero)
    ) / normalization["temp_delta_scale"]
    if str(phase_name).upper() == "CC":
        signal_norm = (
            2.0
            * (voltage - normalization["cc_voltage_low"])
            / (normalization["cc_voltage_high"] - normalization["cc_voltage_low"])
            - 1.0
        )
    elif str(phase_name).upper() == "CV":
        signal_norm = (
            2.0
            * (current - normalization["cv_current_low"])
            / (normalization["cv_current_high"] - normalization["cv_current_low"])
            - 1.0
        )
    else:
        raise ValueError(f"Unknown phase {phase_name!r}")
    tau = np.linspace(-1.0, 1.0, len(time), dtype=np.float64)
    output = np.stack(
        (
            voltage_norm,
            current_norm,
            relative_time_norm,
            temperature_abs_norm,
            temperature_delta_norm,
            signal_norm,
            tau,
        ),
        axis=-1,
    ).astype(np.float32)
    channel_names = rich_channel_names(int(normalization.get("schema_version", 1)))
    if output.shape != (len(time), len(channel_names)) or not np.all(np.isfinite(output)):
        raise ValueError("Offline normalized phase contains invalid values")
    return output


def materialize_record_tensors(
    record: Mapping[str, Any],
    *,
    cc_len: int,
    cv_len: int,
    normalization: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    cc_source = phase_arrays(record, "CC")
    cv_source = phase_arrays(record, "CV")
    cc = resample_phase(cc_source, cc_len)
    cv = resample_phase(cv_source, cv_len)
    time_zero = min(float(cc_source["time"][0]), float(cv_source["time"][0]))
    temperature_zero = float(cc_source["temperature"][0])
    cc_tensor = rich_phase_tensor(
        cc,
        phase_name="CC",
        time_zero=time_zero,
        temperature_zero=temperature_zero,
        normalization=normalization,
    )
    cv_tensor = rich_phase_tensor(
        cv,
        phase_name="CV",
        time_zero=time_zero,
        temperature_zero=temperature_zero,
        normalization=normalization,
    )
    all_time = np.concatenate((cc_source["time"], cv_source["time"]))
    stats = {
        "cc_raw_points": float(len(cc_source["time"])),
        "cv_raw_points": float(len(cv_source["time"])),
        "raw_point_count": float(len(cc_source["time"]) + len(cv_source["time"])),
        "duration_min": float(np.max(all_time) - np.min(all_time)),
        "cc_duration_min": float(cc_source["time"][-1] - cc_source["time"][0]),
        "cv_duration_min": float(cv_source["time"][-1] - cv_source["time"][0]),
    }
    return cc_tensor, cv_tensor, stats


def materialize_full_joint_tensor(
    record: Mapping[str, Any],
    *,
    joint_len: int,
    normalization: Mapping[str, float],
) -> tuple[np.ndarray, int]:
    """Resample one complete CC+CV charge event on one physical-time grid.

    Unlike :func:`materialize_record_tensors`, this function does not assign a
    fixed budget to each phase.  The requested ``joint_len`` points cover the
    entire principal charge event, so the effective CC/CV allocation follows
    their physical durations.  The returned boundary is the insertion index
    for the first CV point in the resampled sequence.
    """

    joint_len = int(joint_len)
    if joint_len < 4:
        raise ValueError("Full joint charging sequence requires at least four points")
    cc_source = phase_arrays(record, "CC")
    cv_source = phase_arrays(record, "CV")
    time = np.concatenate((cc_source["time"], cv_source["time"])).astype(np.float64)
    voltage = np.concatenate((cc_source["voltage"], cv_source["voltage"])).astype(np.float64)
    current = np.concatenate((cc_source["current"], cv_source["current"])).astype(np.float64)
    temperature = np.concatenate((cc_source["temperature"], cv_source["temperature"])).astype(np.float64)
    phase = np.concatenate(
        (
            np.zeros(len(cc_source["time"]), dtype=np.int8),
            np.ones(len(cv_source["time"]), dtype=np.int8),
        )
    )
    order = np.argsort(time, kind="stable")
    time, voltage, current, temperature, phase = (
        value[order] for value in (time, voltage, current, temperature, phase)
    )
    unique_time, unique_index = np.unique(time, return_index=True)
    if unique_time.size < 4 or float(unique_time[-1] - unique_time[0]) <= 0:
        raise ValueError("Full charging event has an invalid physical-time span")
    voltage = voltage[unique_index]
    current = current[unique_index]
    temperature = temperature[unique_index]
    phase = phase[unique_index]
    cv_times = unique_time[phase == 1]
    if cv_times.size < 2:
        raise ValueError("Full charging event has no persistent CV phase")
    boundary_time = float(cv_times[0])
    sample_time = np.linspace(unique_time[0], unique_time[-1], joint_len, dtype=np.float64)
    boundary_index = int(np.searchsorted(sample_time, boundary_time, side="left"))
    boundary_index = min(max(boundary_index, 1), joint_len - 1)

    voltage_sample = np.interp(sample_time, unique_time, voltage)
    current_sample = np.interp(sample_time, unique_time, current)
    temperature_sample = np.interp(sample_time, unique_time, temperature)
    voltage_norm = (
        2.0
        * (voltage_sample - normalization["voltage_low"])
        / (normalization["voltage_high"] - normalization["voltage_low"])
        - 1.0
    )
    current_abs = np.abs(current_sample)
    if normalization.get("current_mode") == "nominal_c_rate":
        current_norm = current_abs / normalization["nominal_capacity_ah"]
    else:
        current_norm = 2.0 * current_abs / normalization["current_scale"] - 1.0
    relative_time_norm = (sample_time - float(sample_time[0])) / normalization["time_scale_min"]
    temperature_zero = float(temperature_sample[0])
    temperature_abs_norm = (
        temperature_sample - normalization["temp_room"]
    ) / normalization["temp_abs_scale"]
    temperature_delta_norm = (
        temperature_sample - temperature_zero
    ) / normalization["temp_delta_scale"]

    legacy_signal = np.empty(joint_len, dtype=np.float64)
    legacy_signal[:boundary_index] = (
        2.0
        * (voltage_sample[:boundary_index] - normalization["cc_voltage_low"])
        / (normalization["cc_voltage_high"] - normalization["cc_voltage_low"])
        - 1.0
    )
    legacy_signal[boundary_index:] = (
        2.0
        * (current_abs[boundary_index:] - normalization["cv_current_low"])
        / (normalization["cv_current_high"] - normalization["cv_current_low"])
        - 1.0
    )
    tau = np.empty(joint_len, dtype=np.float64)
    tau[:boundary_index] = np.linspace(-1.0, 1.0, boundary_index)
    tau[boundary_index:] = np.linspace(-1.0, 1.0, joint_len - boundary_index)
    output = np.stack(
        (
            voltage_norm,
            current_norm,
            relative_time_norm,
            temperature_abs_norm,
            temperature_delta_norm,
            legacy_signal,
            tau,
        ),
        axis=-1,
    ).astype(np.float32)
    if output.shape != (joint_len, len(rich_channel_names(int(normalization.get("schema_version", 1))))):
        raise ValueError("Full joint tensor has an invalid shape")
    if not np.all(np.isfinite(output)):
        raise ValueError("Full joint tensor contains non-finite values")
    return output, boundary_index


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0))


def _slope(time: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(time, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    x = x - float(np.mean(x))
    denominator = float(np.dot(x, x))
    return 0.0 if math.isclose(denominator, 0.0) else float(np.dot(x, y - np.mean(y)) / denominator)


def _skew(values: np.ndarray) -> float:
    std = _std(values)
    if len(values) < 3 or math.isclose(std, 0.0):
        return 0.0
    return float(np.mean(((values - np.mean(values)) / std) ** 3))


def _kurtosis(values: np.ndarray) -> float:
    std = _std(values)
    if len(values) < 4 or math.isclose(std, 0.0):
        return 0.0
    return float(np.mean(((values - np.mean(values)) / std) ** 4) - 3.0)


def _entropy(values: np.ndarray, bins: int = 128) -> float:
    if len(values) <= 1 or math.isclose(float(np.min(values)), float(np.max(values))):
        return 0.0
    counts, _ = np.histogram(values, bins=min(int(bins), len(values)))
    probability = counts[counts > 0].astype(np.float64)
    probability /= probability.sum()
    return float(-np.sum(probability * np.log(probability)))


def _charge_ah(time_min: np.ndarray, current_a: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(current_a), time_min) / 60.0)


def feature_vector(record: Mapping[str, Any]) -> np.ndarray:
    """Extract the 16 electrical + 8 thermal features before resampling."""

    cc = phase_arrays(record, "CC")
    cv = phase_arrays(record, "CV")
    cc_v = cc["voltage"]
    cv_i = np.abs(cv["current"])
    cc_t = cc["time"]
    cv_t = cv["time"]
    cc_temp = cc["temperature"]
    cv_temp = cv["temperature"]
    values: Sequence[float] = (
        _mean(cc_v),
        _std(cc_v),
        _kurtosis(cc_v),
        _skew(cc_v),
        _charge_ah(cc_t, cc["current"]),
        float(cc_t[-1] - cc_t[0]),
        _slope(cc_t, cc_v),
        _entropy(cc_v),
        _mean(cv_i),
        _std(cv_i),
        _kurtosis(cv_i),
        _skew(cv_i),
        _charge_ah(cv_t, cv["current"]),
        float(cv_t[-1] - cv_t[0]),
        _slope(cv_t, cv_i),
        _entropy(cv_i),
        _mean(cc_temp),
        float(np.max(cc_temp)),
        float(cc_temp[-1] - cc_temp[0]),
        _slope(cc_t, cc_temp),
        _mean(cv_temp),
        float(np.max(cv_temp)),
        float(cv_temp[-1] - cv_temp[0]),
        _slope(cv_t, cv_temp),
    )
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(output)):
        raise ValueError("Offline feature vector is invalid")
    return output
