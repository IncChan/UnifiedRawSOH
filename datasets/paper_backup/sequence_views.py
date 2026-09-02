"""Canonical sequence views for Paper-Backup.

This adapter consumes the existing canonical terminal records but owns the
Paper-Backup sample schema.  Metadata is kept beside the tensors for auditing
and evaluation and is never passed to a model.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..base import RawTerminalSignalUnavailable
from ..domains import build_default_domain_registry, canonical_domain_id
from ..filters import filter_records_by_invalid_cycles
from ..registry import build_default_registry
from ..splits import load_invalid_cycles, load_split_spec, split_records_from_spec
from ..xjtu import build_full_life_cycle_metadata
from .full_cccv import FullSourceUnavailable, match_full_terminal_records, materialize_full_records
from .preprocessed import paper_backup_dataloader_kwargs


TERMINAL_VIEW_IDS = ("terminal_joint", "terminal_cc", "terminal_cv", "terminal_phase")
FULL_VIEW_IDS = ("full_cccv", "full_joint")
ALL_VIEW_IDS = (*TERMINAL_VIEW_IDS, *FULL_VIEW_IDS)
SEQUENCE_CHANNEL_NAMES = (
    "voltage_norm",
    "current_norm",
    "relative_time_norm",
    "temperature_abs_norm",
    "temperature_delta_norm",
)


def _is_preprocessed_mode(value: Any) -> bool:
    return str(value) in {"preprocessed_v1", "preprocessed_v2"}


def _resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(repo_root) / path).resolve()


def _domain_id(config: Mapping[str, Any]) -> str:
    experiment = config.get("experiment", {})
    data = config.get("data", {})
    return canonical_domain_id(
        experiment.get(
            "domain_id",
            experiment.get("family_id", data.get("domain_id", data.get("dataset", "xjtu"))),
        )
    )


def _split_path(config: Mapping[str, Any], repo_root: Path) -> Path:
    value = config.get("data", {}).get("split_file") or config.get("experiment", {}).get("split_file")
    if not value:
        raise ValueError("Paper-Backup sequence config must provide data.split_file or experiment.split_file")
    path = _resolve_path(repo_root, value)
    if not path.is_file():
        raise ValueError(f"Configured split file does not exist: {path}")
    return path


def load_terminal_records(config: Mapping[str, Any], repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load canonical terminal records without deriving a lifetime target."""

    if _is_preprocessed_mode(config.get("data", {}).get("source_mode", "legacy_runtime")):
        from .preprocessed import load_preprocessed_records

        return load_preprocessed_records(config, repo_root, source_view="terminal")

    domain_id = _domain_id(config)
    data = dict(config.get("data", {}))
    domain = build_default_domain_registry().get(domain_id)
    data_root_value = data.get("terminal_data_root", data.get("data_root", domain.data_root))
    if not data_root_value:
        raise ValueError(f"No terminal_data_root is configured for {domain_id}")
    data_root = _resolve_path(repo_root, data_root_value)
    adapter_id = str(data.get("adapter_id", domain.adapter_id))
    nominal = float(data.get("nominal_capacity", domain.nominal_capacity_ah or 2.0))
    label_mode = str(data.get("label_scale_mode", "auto_capacity_to_soh"))
    adapter = build_default_registry().create(
        adapter_id,
        data_root=data_root,
        nominal_capacity=nominal,
        label_scale_mode=label_mode,
        domain_id=domain_id,
    )
    if not getattr(adapter, "raw_terminal_signals", False):
        message = str(getattr(adapter, "readiness_error", ""))
        raise RawTerminalSignalUnavailable(
            f"Terminal source for {domain_id!r} is not available through the canonical adapter. {message}"
        )
    configured_batches = list(config.get("experiment", {}).get("batches", []))
    if configured_batches:
        records: list[dict[str, Any]] = []
        for batch in configured_batches:
            records.extend(adapter.load_records(batch=batch))
    else:
        records = list(adapter.load_records())
    for record in records:
        record["domain_id"] = domain_id
        record["source_view"] = "terminal"
        record["is_full"] = False
    split_path = _split_path(config, repo_root)
    invalid = load_invalid_cycles(split_path)
    records, invalid_audit = filter_records_by_invalid_cycles(records, invalid)
    return records, {
        "domain_id": domain_id,
        "data_root": str(data_root),
        "adapter_id": adapter_id,
        "nominal_capacity_ah": nominal,
        "terminal_product": "canonical terminal RAW; no full fallback",
        "invalid_cycle_filter": invalid_audit,
        "split_file": str(split_path),
    }


def split_terminal_records(
    records: Iterable[dict[str, Any]],
    config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    split_path = _split_path(config, repo_root)
    split_spec = load_split_spec(split_path)
    split_records, split_info = split_records_from_spec(list(records), split_spec, split_file=split_path)
    overlaps = {
        "train_val": sorted({str(x["battery_id"]) for x in split_records["train"]} & {str(x["battery_id"]) for x in split_records["val"]}),
        "train_test": sorted({str(x["battery_id"]) for x in split_records["train"]} & {str(x["battery_id"]) for x in split_records["test"]}),
        "val_test": sorted({str(x["battery_id"]) for x in split_records["val"]} & {str(x["battery_id"]) for x in split_records["test"]}),
    }
    if overlaps["train_test"] or overlaps["val_test"]:
        raise ValueError(f"Paper-Backup split leaks test batteries: {overlaps}")
    split_info = dict(split_info)
    split_info["battery_overlap"] = overlaps
    split_info["development_batteries"] = sorted({str(x["battery_id"]) for name in ("train", "val") for x in split_records[name]})
    return split_records, split_info


def attach_cycle_order_auxiliary_targets(
    split_records: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Attach a no-future-lifetime cycle-order target to every split record.

    Ranks are chronological positions within a battery.  The nonlinear scale
    is fitted only on training records; neither test statistics nor a battery's
    final cycle/EOL is used as a per-sample denominator.
    """

    cycle_ids_by_battery: dict[str, set[int]] = {}
    for records in split_records.values():
        for record in records:
            battery_id = str(record["battery_id"])
            cycle_ids_by_battery.setdefault(battery_id, set()).add(
                int(record["cycle_id"])
            )
    rank_by_key: dict[tuple[str, int], int] = {}
    for battery_id, cycle_ids in cycle_ids_by_battery.items():
        for rank, cycle_id in enumerate(sorted(cycle_ids)):
            rank_by_key[(battery_id, cycle_id)] = int(rank)

    train_ranks = [
        rank_by_key[(str(record["battery_id"]), int(record["cycle_id"]))]
        for record in split_records["train"]
    ]
    if not train_ranks:
        raise ValueError("Cycle auxiliary target requires non-empty training data")
    train_max_rank = max(train_ranks)
    log_scale = math.log1p(train_max_rank)
    if not math.isfinite(log_scale) or log_scale <= 0.0:
        log_scale = 1.0

    target_ranges: dict[str, list[float]] = {}
    for split_name, records in split_records.items():
        targets = []
        for record in records:
            key = (str(record["battery_id"]), int(record["cycle_id"]))
            rank = rank_by_key[key]
            target = math.log1p(rank) / log_scale
            record["cycle_aux_rank"] = int(rank)
            record["cycle_aux_target"] = float(target)
            targets.append(float(target))
        target_ranges[str(split_name)] = [
            float(min(targets)) if targets else float("nan"),
            float(max(targets)) if targets else float("nan"),
        ]
    return {
        "enabled": True,
        "target": "log1p chronological rank / train log1p(max rank)",
        "uses_battery_final_cycle_as_sample_denominator": False,
        "uses_test_statistics": False,
        "train_max_rank": int(train_max_rank),
        "train_log_scale": float(log_scale),
        "target_ranges": target_ranges,
    }


def _safe_float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"Record field {name!r} is empty or non-finite")
    return array


def _phase_arrays(record: Mapping[str, Any], phase: str) -> dict[str, np.ndarray]:
    segments = np.asarray([str(item).upper() for item in record["segment"]], dtype=object)
    mask = segments == phase
    if int(mask.sum()) < 2:
        raise ValueError(f"Cycle {record.get('cycle_id')} lacks two {phase} points")
    time = _safe_float_array(np.asarray(record["time"])[mask], "time")
    order = np.argsort(time, kind="stable")
    time = time[order]
    unique_time, unique_indices = np.unique(time, return_index=True)
    if unique_time.size < 2 or float(unique_time[-1] - unique_time[0]) <= 0:
        raise ValueError(f"Cycle {record.get('cycle_id')} has invalid {phase} time span")
    source = {
        "time": unique_time,
        "voltage": _safe_float_array(np.asarray(record["voltage"])[mask], "voltage")[order][unique_indices],
        "current": _safe_float_array(np.asarray(record["current"])[mask], "current")[order][unique_indices],
        "temperature": _safe_float_array(np.asarray(record["temperature"])[mask], "temperature")[order][unique_indices],
    }
    return source


def _resample_phase(record: Mapping[str, Any], phase: str, target_len: int) -> dict[str, np.ndarray]:
    source = _phase_arrays(record, phase)
    target_len = int(target_len)
    if target_len < 2:
        raise ValueError("raw phase target length must be at least 2")
    sample_time = np.linspace(float(source["time"][0]), float(source["time"][-1]), target_len, dtype=np.float32)
    return {
        key: np.interp(sample_time, source["time"], source[key]).astype(np.float32)
        for key in ("voltage", "current", "temperature")
    } | {"time": sample_time}


def _normalization_values(config: Mapping[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    norm = config.get("normalization", {})
    voltage_low = float(norm.get("raw_voltage_low", norm.get("cc_voltage_low", 0.0)))
    voltage_high = float(norm.get("raw_voltage_high", norm.get("cv_voltage_ref", voltage_low + 1.0)))
    current_scale = float(norm.get("raw_current_scale", norm.get("cc_current_ref", 1.0)))
    temp_room = float(norm.get("temp_room", 25.0))
    temp_abs_scale = float(norm.get("temp_abs_scale", 20.0))
    temp_delta_scale = float(norm.get("temp_delta_scale", 10.0))
    time_scale = float(config.get("data", {}).get("time_scale_min", 10.0))
    if voltage_high <= voltage_low or current_scale <= 0 or temp_abs_scale <= 0 or temp_delta_scale <= 0 or time_scale <= 0:
        raise ValueError("Paper-Backup physical normalization has invalid range or scale")
    return voltage_low, voltage_high, current_scale, temp_room, temp_abs_scale, temp_delta_scale, time_scale


def _sequence_from_phases(
    phases: list[dict[str, np.ndarray]],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    voltage_low, voltage_high, current_scale, temp_room, temp_abs_scale, temp_delta_scale, time_scale = _normalization_values(config)
    all_time = np.concatenate([phase["time"] for phase in phases])
    time_zero = float(np.min(all_time))
    temp_zero = float(phases[0]["temperature"][0])
    rows = []
    point_count = 0
    for phase in phases:
        voltage = phase["voltage"]
        current = np.abs(phase["current"])
        temp = phase["temperature"]
        rows.append(
            np.stack(
                [
                    2.0 * (voltage - voltage_low) / (voltage_high - voltage_low) - 1.0,
                    2.0 * current / current_scale - 1.0,
                    (phase["time"] - time_zero) / time_scale,
                    (temp - temp_room) / temp_abs_scale,
                    (temp - temp_zero) / temp_delta_scale,
                ],
                axis=-1,
            ).astype(np.float32)
        )
        point_count += len(phase["time"])
    sequence = np.concatenate(rows, axis=0)
    if not np.all(np.isfinite(sequence)):
        raise ValueError("Normalized Paper-Backup sequence contains non-finite values")
    duration = float(np.max(all_time) - np.min(all_time))
    return sequence, {"raw_point_count": float(point_count), "duration_min": duration}


def _source_view_stats(record: Mapping[str, Any], view_id: str) -> dict[str, float]:
    """Report provenance statistics before fixed-length interpolation."""

    segments = np.asarray([str(item).upper() for item in record["segment"]], dtype=object)
    selected = (segments == "CC") | (segments == "CV")
    if view_id == "terminal_cc":
        selected = segments == "CC"
    elif view_id == "terminal_cv":
        selected = segments == "CV"
    time = _safe_float_array(np.asarray(record["time"])[selected], "time")
    return {
        "raw_point_count": float(int(selected.sum())),
        "duration_min": float(np.max(time) - np.min(time)),
    }


def _phase_sample(record: Mapping[str, Any], config: Mapping[str, Any], split_name: str, view_id: str) -> dict[str, Any]:
    data = config.get("data", {})
    cc = _resample_phase(record, "CC", int(data.get("raw_len_cc", 128)))
    cv = _resample_phase(record, "CV", int(data.get("raw_len_cv", 256)))
    voltage_low, voltage_high, current_scale, temp_room, temp_abs_scale, temp_delta_scale, time_scale = _normalization_values(config)
    del current_scale, temp_room, temp_abs_scale, temp_delta_scale, time_scale
    normal = config.get("normalization", {})
    cc_low = float(normal["cc_voltage_low"])
    cc_high = float(normal["cc_voltage_high"])
    cv_low = float(normal["cv_current_low"])
    cv_high = float(normal["cv_current_high"])
    cc_tau = np.linspace(-1.0, 1.0, len(cc["time"]), dtype=np.float32)
    cv_tau = np.linspace(-1.0, 1.0, len(cv["time"]), dtype=np.float32)
    cc_signal = np.stack([2.0 * (cc["voltage"] - cc_low) / (cc_high - cc_low) - 1.0, cc_tau], axis=-1).astype(np.float32)
    cv_signal = np.stack([2.0 * (np.abs(cv["current"]) - cv_low) / (cv_high - cv_low) - 1.0, cv_tau], axis=-1).astype(np.float32)
    temperature_zero = float(cc["temperature"][0])
    cc_temperature = np.stack(
        [(cc["temperature"] - float(normal.get("temp_room", 25.0))) / float(normal.get("temp_abs_scale", 20.0)),
         (cc["temperature"] - temperature_zero) / float(normal.get("temp_delta_scale", 10.0))], axis=-1
    ).astype(np.float32)
    cv_temperature = np.stack(
        [(cv["temperature"] - float(normal.get("temp_room", 25.0))) / float(normal.get("temp_abs_scale", 20.0)),
         (cv["temperature"] - temperature_zero) / float(normal.get("temp_delta_scale", 10.0))], axis=-1
    ).astype(np.float32)
    all_time = np.concatenate([cc["time"], cv["time"]])
    segments = np.asarray([str(item).upper() for item in record["segment"]], dtype=object)
    stats = {
        "raw_point_count": float(int(np.sum(segments == "CC") + np.sum(segments == "CV"))),
        "duration_min": float(np.max(all_time) - np.min(all_time)),
    }
    return {
        "cc_signal": cc_signal,
        "cv_signal": cv_signal,
        "cc_time": (cc["time"] - np.min(all_time)).astype(np.float32),
        "cv_time": (cv["time"] - np.min(all_time)).astype(np.float32),
        "cc_temperature": cc_temperature,
        "cv_temperature": cv_temperature,
        "t0_temperature_norm": np.asarray([(temperature_zero - float(normal.get("temp_room", 25.0))) / float(normal.get("temp_abs_scale", 20.0))], dtype=np.float32),
        "cc_mask": np.ones(len(cc["time"]), dtype=np.bool_),
        "cv_mask": np.ones(len(cv["time"]), dtype=np.bool_),
        "view_stats": stats,
        "view_id": view_id,
        "split": split_name,
    }


def _base_metadata(record: Mapping[str, Any], split_name: str, view_id: str) -> dict[str, Any]:
    if "soh" not in record:
        raise ValueError("Canonical record has no SOH label")
    metadata = {
        "soh": np.asarray([float(record["soh"])], dtype=np.float32),
        "battery_id": str(record["battery_id"]),
        "domain_id": str(record.get("domain_id", record.get("dataset_id", "unknown"))),
        "strategy_id": str(record.get("strategy_id", record.get("condition", "unknown"))),
        "condition": str(record.get("condition", record.get("strategy_id", "unknown"))),
        "cycle_id": int(record["cycle_id"]),
        "split": split_name,
        "view_id": view_id,
        "input_view_id": view_id,
    }
    if "cycle_aux_target" in record:
        metadata["cycle_aux_target"] = np.asarray(
            [float(record["cycle_aux_target"])], dtype=np.float32
        )
        metadata["cycle_aux_rank"] = int(record["cycle_aux_rank"])
    return metadata


class SequenceViewDataset(Dataset):
    """Fixed-length raw view dataset with an evaluation-only provenance schema."""

    def __init__(self, records: Iterable[dict[str, Any]], config: Mapping[str, Any], split_name: str, view_id: str):
        if view_id not in ALL_VIEW_IDS:
            raise ValueError(f"Unknown Paper-Backup view_id={view_id!r}; allowed={ALL_VIEW_IDS}")
        self.records = list(records)
        self.config = copy.deepcopy(dict(config))
        self.split_name = str(split_name)
        self.view_id = str(view_id)
        self.samples: list[dict[str, Any]] = []
        self.skipped = Counter()
        self.preprocessed = bool(self.records) and all(
            "_preprocessed_directory" in record for record in self.records
        )
        if self.preprocessed:
            # Keep only the compact cycle index here. Fixed arrays stay mmap'd
            # and one selected row is copied in __getitem__.
            return
        for record in self.records:
            try:
                if "_preprocessed_directory" in record:
                    from .preprocessed import sample_from_preprocessed_record

                    sample = sample_from_preprocessed_record(
                        record, self.view_id, self.split_name, self.config
                    )
                    self.samples.append(sample)
                    continue
                if self.view_id == "terminal_phase":
                    view = _phase_sample(record, self.config, self.split_name, self.view_id)
                elif self.view_id == "full_joint":
                    raise ValueError(
                        "full_joint requires the audited offline preprocessing product"
                    )
                else:
                    cc = _resample_phase(record, "CC", int(self.config.get("data", {}).get("raw_len_cc", 128)))
                    cv = _resample_phase(record, "CV", int(self.config.get("data", {}).get("raw_len_cv", 256)))
                    phases = {"terminal_joint": [cc, cv], "full_cccv": [cc, cv], "terminal_cc": [cc], "terminal_cv": [cv]}[self.view_id]
                    sequence, stats = _sequence_from_phases(phases, self.config)
                    stats.update(_source_view_stats(record, self.view_id))
                    view = {
                        "sequence": sequence,
                        "mask": np.ones(sequence.shape[0], dtype=np.bool_),
                        "view_stats": stats,
                        "split": self.split_name,
                        "view_id": self.view_id,
                    }
                metadata = _base_metadata(record, self.split_name, self.view_id)
                sample = {**view, **metadata}
                sample["raw_point_count"] = np.asarray([float(sample["view_stats"]["raw_point_count"])], dtype=np.float32)
                sample["duration_min"] = np.asarray([float(sample["view_stats"]["duration_min"])], dtype=np.float32)
                self.samples.append(sample)
            except (TypeError, ValueError, KeyError) as exc:
                self.skipped[type(exc).__name__ + ":" + str(exc)[:120]] += 1
        if not self.samples:
            raise ValueError(f"No usable Paper-Backup {view_id} samples for split {split_name}: {dict(self.skipped)}")

    def limit(self, count: int) -> None:
        if int(count) > 0:
            self.records = self.records[: int(count)]
            if not self.preprocessed:
                self.samples = self.samples[: int(count)]

    def __len__(self) -> int:
        return len(self.records) if self.preprocessed else len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.preprocessed:
            from .preprocessed import sample_from_preprocessed_record

            sample = sample_from_preprocessed_record(
                self.records[index], self.view_id, self.split_name, self.config
            )
        else:
            sample = self.samples[index]
        output: dict[str, Any] = {}
        for key, value in sample.items():
            if key == "view_stats":
                continue
            if isinstance(value, np.ndarray):
                output[key] = torch.from_numpy(value)
            else:
                output[key] = value
        return output


def _strategy_counts(dataset: Dataset) -> dict[str, int]:
    if getattr(dataset, "preprocessed", False):
        return dict(
            Counter(
                str(item.get("strategy_id", item.get("condition", "unknown")))
                for item in dataset.records
            )
        )
    return dict(Counter(str(dataset[index]["strategy_id"]) for index in range(len(dataset))))


def build_strategy_sampler(dataset: Dataset, seed: int = 0) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    """Equal strategy mass, then equal battery mass, then equal cycle mass."""

    if getattr(dataset, "preprocessed", False):
        metadata = [
            (
                str(item.get("strategy_id", item.get("condition", "unknown"))),
                str(item["battery_id"]),
            )
            for item in dataset.records
        ]
    else:
        metadata = []
        for index in range(len(dataset)):
            item = dataset[index]
            metadata.append((str(item["strategy_id"]), str(item["battery_id"])))
    if not metadata:
        raise ValueError("Cannot build strategy sampler for an empty dataset")
    strategies = sorted({strategy for strategy, _ in metadata})
    batteries_per_strategy = Counter(strategy for strategy, _ in set(metadata))
    cycles_per_battery = Counter(metadata)
    weights = [
        1.0 / (len(strategies) * batteries_per_strategy[strategy] * cycles_per_battery[(strategy, battery)])
        for strategy, battery in metadata
    ]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    audit = {
        "policy": "strategy -> battery -> cycle hierarchical equal-mass",
        "strategies": strategies,
        "strategy_counts": dict(Counter(strategy for strategy, _ in metadata)),
        "battery_counts": {f"{strategy}::{battery}": count for (strategy, battery), count in sorted(cycles_per_battery.items())},
        "weights_sum": float(sum(weights)),
        "strategy_id_in_model_input": False,
    }
    return sampler, audit


def _make_loaders(datasets: dict[str, Dataset], config: Mapping[str, Any], seed: int, strategy_balanced: bool = False):
    train_sampler = None
    sampler_audit = {"enabled": False}
    if strategy_balanced:
        train_sampler, sampler_audit = build_strategy_sampler(datasets["train"], seed=seed)
    batch_size = int(config.get("train", {}).get("batch_size", 64))
    common = paper_backup_dataloader_kwargs(config, batch_size=batch_size)
    loaders = {
        "train": DataLoader(datasets["train"], sampler=train_sampler, shuffle=train_sampler is None, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return loaders, sampler_audit, dict(common)


def _full_data_config(config: Mapping[str, Any], repo_root: Path, root_value: Any) -> dict[str, Any]:
    data_config = dict(config.get("data", {}))
    if root_value:
        root = Path(str(root_value)).expanduser()
        if not root.is_absolute():
            root = _resolve_path(repo_root, root)
        data_config["full_data_root"] = str(root)
    data_config["nominal_capacity"] = config.get("data", {}).get("nominal_capacity", 1.0)
    return data_config


def build_sequence_loaders(
    config: Mapping[str, Any],
    repo_root: str | Path,
    seed: int = 42,
    *,
    view_id: str | None = None,
    split_records: dict[str, list[dict[str, Any]]] | None = None,
    strategy_balanced: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    """Build terminal or matched full sequence loaders."""

    repo_root = Path(repo_root).resolve()
    view_id = str(view_id or config.get("data", {}).get("input_view", config.get("experiment", {}).get("input_view", "terminal_joint")))
    if view_id not in ALL_VIEW_IDS:
        raise ValueError(f"Unknown Paper-Backup input view {view_id!r}")
    source_mode = str(config.get("data", {}).get("source_mode", "legacy_runtime"))
    terminal_records, source_info = load_terminal_records(config, repo_root)
    matching_audit = None
    matched_keys: set[tuple[str, int]] | None = None
    if view_id in FULL_VIEW_IDS:
        if _is_preprocessed_mode(source_mode):
            from .preprocessed import load_preprocessed_records

            terminal_records, full_source_info = load_preprocessed_records(
                config, repo_root, source_view="full_cccv"
            )
            matching_audit = {
                "policy": "offline full_matched cohort",
                "matched_records": len(terminal_records),
                "source": full_source_info,
            }
            source_info = full_source_info
        else:
            full_records = materialize_full_records(
                terminal_records,
                domain_id=_domain_id(config),
                data_config=_full_data_config(config, repo_root, config.get("data", {}).get("full_data_root")),
            )
            matched, matching_audit = match_full_terminal_records(terminal_records, full_records)
            terminal_by_key = {(str(item["battery_id"]), int(item["cycle_id"])): item for item in terminal_records}
            # Use the full record for input, but retain canonical terminal label and
            # metadata. match_full_terminal_records already checked the linkage.
            terminal_records = []
            matched_keys = set()
            for item in matched:
                key = (str(item["battery_id"]), int(item["cycle_id"]))
                matched_keys.add(key)
                merged = dict(item)
                merged["domain_id"] = terminal_by_key[key].get("domain_id", _domain_id(config))
                merged["condition"] = terminal_by_key[key]["condition"]
                merged["soh"] = terminal_by_key[key]["soh"]
                terminal_records.append(merged)
    else:
        matched_root = config.get("data", {}).get("matched_full_data_root")
        if matched_root and not _is_preprocessed_mode(source_mode):
            full_records = materialize_full_records(
                terminal_records,
                domain_id=_domain_id(config),
                data_config=_full_data_config(config, repo_root, matched_root),
            )
            matched, matching_audit = match_full_terminal_records(terminal_records, full_records)
            matched_keys = {
                (str(item["battery_id"]), int(item["cycle_id"]))
                for item in matched
            }
            terminal_records = [
                item for item in terminal_records
                if (str(item["battery_id"]), int(item["cycle_id"])) in matched_keys
            ]
            if not terminal_records:
                raise FullSourceUnavailable("Matched full/terminal cohort is empty for terminal E2 view")
    if split_records is None:
        split_records, split_info = split_terminal_records(terminal_records, config, repo_root)
    else:
        if matched_keys is not None:
            split_records = {
                name: [
                    item for item in values
                    if (str(item["battery_id"]), int(item["cycle_id"])) in matched_keys
                ]
                for name, values in split_records.items()
            }
        split_info = {"provided_by": "strategy_pooling", "battery_overlap": {}}
    cycle_aux_mode = str(
        config.get("data", {}).get("cycle_aux_target_mode", "disabled")
    )
    if cycle_aux_mode == "disabled":
        cycle_aux_audit = {"enabled": False}
    elif cycle_aux_mode == "log1p_rank_train_max":
        cycle_aux_audit = attach_cycle_order_auxiliary_targets(split_records)
    else:
        raise ValueError(f"Unknown cycle_aux_target_mode: {cycle_aux_mode!r}")
    datasets = {
        name: SequenceViewDataset(split_records[name], config, name, view_id)
        for name in ("train", "val", "test")
    }
    debug_n = int(config.get("debug", {}).get("debug_num_samples", 0) or 0)
    if debug_n > 0:
        for dataset in datasets.values():
            dataset.limit(debug_n)
    loaders, sampler_audit, loader_options = _make_loaders(
        datasets, config, int(seed), strategy_balanced=strategy_balanced
    )
    sequence_channel_names = list(SEQUENCE_CHANNEL_NAMES)
    if int(config.get("data", {}).get("preprocessed_schema_version", 1)) == 2:
        sequence_channel_names[1] = "current_c_rate"
    info = {
        "loader_type": "paper_backup_sequence",
        "domain_id": _domain_id(config),
        "view_id": view_id,
        "input_channel_names": (
            sequence_channel_names
            if view_id not in {"terminal_phase"}
            else [
                str(config.get("data", {}).get("phase_signal_mode", "legacy_local")),
                "relative_time",
                "temperature_abs",
                "temperature_delta",
            ]
        ),
        "source": source_info,
        "split": split_info,
        "sample_filter": {
            "mode": str(config.get("data", {}).get("sample_filter_mode", "none")),
            "filter_applied": False,
            "records_before": sum(len(values) for values in split_records.values()),
            "records_after": sum(len(values) for values in split_records.values()),
            "removed_records": 0,
        },
        "record_counts": {name: len(split_records[name]) for name in split_records},
        "sample_counts": {name: len(datasets[name]) for name in datasets},
        "battery_counts": {name: len({str(item["battery_id"]) for item in split_records[name]}) for name in split_records},
        "strategy_counts": {name: _strategy_counts(datasets[name]) for name in datasets},
        "sampler": sampler_audit,
        "dataloader": loader_options,
        "metadata_in_forward": False,
        "cycle_lifetime_auxiliary": False,
        "cycle_order_auxiliary": cycle_aux_audit,
        "matching": matching_audit,
    }
    return loaders, info


__all__ = [
    "ALL_VIEW_IDS",
    "FULL_VIEW_IDS",
    "SEQUENCE_CHANNEL_NAMES",
    "SequenceViewDataset",
    "TERMINAL_VIEW_IDS",
    "build_sequence_loaders",
    "build_strategy_sampler",
    "attach_cycle_order_auxiliary_targets",
    "load_terminal_records",
    "split_terminal_records",
]
