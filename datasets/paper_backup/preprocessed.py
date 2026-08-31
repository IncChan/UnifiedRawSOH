"""Memory-mapped reader for offline Paper-Backup preprocessing products."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..splits import load_split_spec, split_records_from_spec
from ...preprocess.paper_backup.common import (
    FEATURE_NAMES,
    PAPER_BACKUP_PREPROCESS_POLICY,
    PAPER_BACKUP_PREPROCESS_SCHEMA,
    RICH_CHANNEL_NAMES,
    normalization_contract,
    preprocessing_policy,
    rich_channel_names,
)


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _domain_id(config: Mapping[str, Any]) -> str:
    return str(
        config.get("data", {}).get(
            "domain_id", config.get("experiment", {}).get("domain_id", "")
        )
    )


def _root(config: Mapping[str, Any], repo_root: Path) -> Path:
    value = config.get("data", {}).get(
        "preprocessed_data_root", "datasets/PaperBackup_preprocessed"
    )
    return _resolve(repo_root, value)


def _index(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        output.append(
            {
                **row,
                "row": int(row["row"]),
                "cycle_id": int(row["cycle_id"]),
                "soh": float(row["soh"]),
                "soh_raw": float(row["soh_raw"]),
                "raw_point_count": float(row["raw_point_count"]),
                "duration_min": float(row["duration_min"]),
            }
        )
    if not output:
        raise ValueError(f"Empty Paper-Backup preprocessed index: {path}")
    return output


class PreprocessedStore:
    """One domain product with arrays opened lazily through numpy mmap."""

    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing Paper-Backup manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema_version = int(self.manifest.get("schema_version", 0))
        if self.manifest.get("policy_version") != preprocessing_policy(schema_version):
            raise ValueError(f"Incompatible Paper-Backup preprocessing policy: {manifest_path}")
        if self.manifest.get("rich_channel_names") != list(rich_channel_names(schema_version)):
            raise ValueError(f"Incompatible rich-channel schema: {manifest_path}")
        if self.manifest.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError(f"Incompatible feature schema: {manifest_path}")
        self._indices: dict[str, list[dict[str, Any]]] = {}
        self._arrays: dict[tuple[str, str], np.ndarray] = {}

    def has_view(self, source_view: str) -> bool:
        section = "terminal" if source_view == "terminal" else "full"
        return self.manifest.get(section) is not None

    def index(self, source_view: str) -> list[dict[str, Any]]:
        section = "terminal" if source_view == "terminal" else "full"
        if not self.has_view(source_view):
            raise ValueError(f"{self.directory.name} has no {source_view} product")
        if section not in self._indices:
            self._indices[section] = _index(
                self.directory / self.manifest[section]["index"]
            )
        return self._indices[section]

    def array(self, source_view: str, name: str) -> np.ndarray:
        section = "terminal" if source_view == "terminal" else "full"
        key = (section, name)
        if key not in self._arrays:
            contract = self.manifest[section]["arrays"][name]
            values = np.load(
                self.directory / contract["file"], mmap_mode="r", allow_pickle=False
            )
            if list(values.shape) != list(contract["shape"]):
                raise ValueError(f"Shape mismatch for {self.directory / contract['file']}")
            self._arrays[key] = values
        return self._arrays[key]


_STORE_CACHE: dict[str, PreprocessedStore] = {}


def _cached_store(directory: str | Path) -> PreprocessedStore:
    key = str(Path(directory).resolve())
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = PreprocessedStore(Path(key))
    return _STORE_CACHE[key]


def paper_backup_dataloader_kwargs(
    config: Mapping[str, Any], *, batch_size: int
) -> dict[str, Any]:
    """Resolve audited DataLoader performance settings.

    Worker-only options must not be passed when ``num_workers == 0`` because
    PyTorch rejects that combination. Keeping this logic in one place also
    makes feature and sequence loaders use the same runtime policy.
    """

    data = config.get("data", {})
    workers = int(data.get("num_workers", 1))
    if workers < 0:
        raise ValueError("Paper-Backup data.num_workers must be non-negative")
    options: dict[str, Any] = {
        "batch_size": int(batch_size),
        "num_workers": workers,
        "pin_memory": bool(data.get("pin_memory", True)),
    }
    if workers > 0:
        prefetch_factor = int(data.get("prefetch_factor", 2))
        if prefetch_factor < 1:
            raise ValueError("Paper-Backup data.prefetch_factor must be positive")
        options.update(
            {
                "persistent_workers": bool(data.get("persistent_workers", True)),
                "prefetch_factor": prefetch_factor,
            }
        )
    return options


def load_preprocessed_records(
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    source_view: str = "terminal",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    domain_id = _domain_id(config)
    directory = _root(config, repo_root) / domain_id
    store = PreprocessedStore(directory)
    if str(store.manifest.get("domain_id")) != domain_id:
        raise ValueError(f"Preprocessed domain mismatch: expected {domain_id}, got {store.manifest.get('domain_id')}")
    expected_schema = int(config.get("data", {}).get("preprocessed_schema_version", PAPER_BACKUP_PREPROCESS_SCHEMA))
    if expected_schema != int(store.manifest["schema_version"]):
        raise ValueError(f"Configured/preprocessed schema mismatch for {domain_id}")
    data = dict(config.get("data", {}))
    expected_cc = int(data.get("raw_len_cc", store.manifest["resampling"]["cc_length"]))
    expected_cv = int(data.get("raw_len_cv", store.manifest["resampling"]["cv_length"]))
    if expected_cc != int(store.manifest["resampling"]["cc_length"]) or expected_cv != int(store.manifest["resampling"]["cv_length"]):
        raise ValueError(
            f"Configured/preprocessed resampling mismatch for {domain_id}: "
            f"config=({expected_cc},{expected_cv}), product="
            f"({store.manifest['resampling']['cc_length']},{store.manifest['resampling']['cv_length']})"
        )
    if str(data.get("input_view", "")) == "full_joint":
        expected_joint = int(data.get("full_joint_len", 0))
        product_joint = int(store.manifest["resampling"].get("full_joint_length", 0) or 0)
        if expected_joint <= 0 or expected_joint != product_joint:
            raise ValueError(
                f"Configured/preprocessed FULL joint length mismatch for {domain_id}: "
                f"config={expected_joint}, product={product_joint}"
            )
        if "joint" not in store.manifest.get("full", {}).get("arrays", {}):
            raise ValueError(f"FULL joint arrays are absent for {domain_id}")
    if config.get("normalization"):
        expected_norm = normalization_contract(config, schema_version=expected_schema)
        product_norm = store.manifest["normalization"]
        mismatch = []
        for key, value in expected_norm.items():
            if key not in product_norm:
                mismatch.append(key)
                continue
            actual = product_norm[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                equal = math.isclose(float(value), float(actual), rel_tol=0.0, abs_tol=1e-12)
            else:
                equal = value == actual
            if not equal:
                mismatch.append(key)
        if mismatch:
            raise ValueError(f"Configured/preprocessed normalization mismatch for {domain_id}: {mismatch}")
    records = []
    for item in store.index(source_view):
        records.append(
            {
                **item,
                "domain_id": domain_id,
                "dataset_id": domain_id,
                "strategy_id": str(item["strategy_id"]),
                "condition": str(item["condition"]),
                "battery_id": str(item["battery_id"]),
                "source_view": source_view,
                "is_full": source_view == "full_cccv",
                "_preprocessed_directory": str(directory),
                "_preprocessed_source_view": source_view,
                "_preprocessed_row": int(item["row"]),
            }
        )
    cohort = str(config.get("data", {}).get("cohort", "all"))
    if str(config.get("output", {}).get("experiment_id", "")) in {
        "e2_charging_information",
        "e2_final_256budget",
        "e2_final_interaction_5seed",
    }:
        cohort = "full_matched"
    if source_view == "terminal" and cohort == "full_matched":
        if not store.has_view("full_cccv"):
            raise ValueError(f"full_matched cohort requested but FULL arrays are absent: {directory}")
        full_keys = {
            (str(item["battery_id"]), int(item["cycle_id"]))
            for item in store.index("full_cccv")
        }
        records = [
            item
            for item in records
            if (str(item["battery_id"]), int(item["cycle_id"])) in full_keys
        ]
    if not records:
        raise ValueError(f"No {source_view} records in {directory}")
    return records, {
        "source_mode": f"preprocessed_v{store.manifest['schema_version']}",
        "policy_version": store.manifest["policy_version"],
        "schema_version": int(store.manifest["schema_version"]),
        "domain_id": domain_id,
        "directory": str(directory),
        "source_view": source_view,
        "records": len(records),
        "cohort": cohort,
        "resampling": store.manifest["resampling"],
        "normalization": store.manifest["normalization"],
    }


def sample_from_preprocessed_record(
    record: Mapping[str, Any], view_id: str, split_name: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_view = "full_cccv" if view_id in {"full_cccv", "full_joint"} else "terminal"
    if str(record.get("_preprocessed_source_view")) != source_view:
        raise ValueError(f"Preprocessed record/view mismatch for {view_id}")
    store = _cached_store(str(record["_preprocessed_directory"]))
    row = int(record["_preprocessed_row"])
    # mmap arrays are intentionally read-only; materialize only the selected
    # sample so torch never receives a non-writable NumPy view.
    if view_id == "full_joint":
        sequence = np.array(
            store.array(source_view, "joint")[row, :, :5],
            dtype=np.float32,
            copy=True,
        )
        boundary_index = int(store.array(source_view, "boundary_index")[row])
        if not 0 < boundary_index < sequence.shape[0]:
            raise ValueError(f"Invalid FULL boundary index {boundary_index}")
        view = {
            "sequence": sequence,
            "mask": np.ones(sequence.shape[0], dtype=np.bool_),
            "boundary_index": np.asarray(boundary_index, dtype=np.int64),
        }
    else:
        cc = np.array(store.array(source_view, "cc")[row], dtype=np.float32, copy=True)
        cv = np.array(store.array(source_view, "cv")[row], dtype=np.float32, copy=True)
    if view_id == "terminal_phase":
        # Rich offline tensors store relative_time_norm = physical_minutes /
        # time_scale_min.  The phase Mamba input encoder owns that
        # normalization and divides physical minutes by the same scale.  Undo
        # the materialized normalization here so the offline path has exactly
        # the same time contract as the legacy runtime phase view.
        time_scale_min = float(store.manifest["normalization"]["time_scale_min"])
        if not math.isfinite(time_scale_min) or time_scale_min <= 0:
            raise ValueError(
                f"Invalid preprocessed time_scale_min={time_scale_min!r}: "
                f"{store.directory / 'manifest.json'}"
            )
        signal_mode = str((config or {}).get("data", {}).get("phase_signal_mode", "legacy_local"))
        if signal_mode == "legacy_local":
            cc_signal, cv_signal = cc[:, [5, 6]], cv[:, [5, 6]]
        elif signal_mode == "shared_dominant":
            cc_signal, cv_signal = cc[:, [0, 6]], cv[:, [1, 6]]
        elif signal_mode == "shared_full_vi":
            cc_signal, cv_signal = cc[:, [0, 1, 6]], cv[:, [0, 1, 6]]
        elif signal_mode == "shared_gated_full_vi":
            # Preserve the common rich-channel order.  The gated phase
            # encoder selects voltage as CC-dominant/current as secondary and
            # current as CV-dominant/voltage as secondary inside the model.
            cc_signal, cv_signal = cc[:, [0, 1, 6]], cv[:, [0, 1, 6]]
        else:
            raise ValueError(f"Unknown phase_signal_mode={signal_mode!r}")
        active_phase = str((config or {}).get("data", {}).get("active_phase", "both"))
        if active_phase not in {"both", "cc", "cv"}:
            raise ValueError(f"Unknown active_phase={active_phase!r}")
        cc_time = cc[:, 2] * time_scale_min
        cv_time = cv[:, 2] * time_scale_min
        cc_temperature = cc[:, [3, 4]]
        cv_temperature = cv[:, [3, 4]]
        cc_mask = np.ones(cc.shape[0], dtype=np.bool_)
        cv_mask = np.ones(cv.shape[0], dtype=np.bool_)
        t0_temperature = cc[0, 3]
        if active_phase == "cc":
            cv_signal = np.zeros_like(cv_signal)
            cv_time = np.zeros_like(cv_time)
            cv_temperature = np.zeros_like(cv_temperature)
        elif active_phase == "cv":
            cc_signal = np.zeros_like(cc_signal)
            cc_time = np.zeros_like(cc_time)
            cc_temperature = np.zeros_like(cc_temperature)
            t0_temperature = cv[0, 3]
        view = {
            "cc_signal": cc_signal,
            "cv_signal": cv_signal,
            "cc_time": cc_time,
            "cv_time": cv_time,
            "cc_temperature": cc_temperature,
            "cv_temperature": cv_temperature,
            "t0_temperature_norm": np.asarray([t0_temperature], dtype=np.float32),
            "cc_mask": cc_mask,
            "cv_mask": cv_mask,
        }
    elif view_id != "full_joint":
        phases = {
            "terminal_joint": (cc, cv),
            "full_cccv": (cc, cv),
            "terminal_cc": (cc,),
            "terminal_cv": (cv,),
        }[view_id]
        sequence = np.concatenate([phase[:, :5] for phase in phases], axis=0).astype(np.float32)
        view = {
            "sequence": sequence,
            "mask": np.ones(sequence.shape[0], dtype=np.bool_),
        }
        if view_id == "terminal_joint" and bool(
            (config or {}).get("model", {}).get("use_boundary_token", False)
        ):
            view["boundary_index"] = np.asarray(cc.shape[0], dtype=np.int64)
    active_phase = str((config or {}).get("data", {}).get("active_phase", "both"))
    raw_point_count = float(record["raw_point_count"])
    duration_min = float(record["duration_min"])
    if active_phase == "cc":
        raw_point_count = float(record.get("cc_raw_points", raw_point_count))
        duration_min = float(record.get("cc_duration_min", duration_min))
    elif active_phase == "cv":
        raw_point_count = float(record.get("cv_raw_points", raw_point_count))
        duration_min = float(record.get("cv_duration_min", duration_min))
    return {
        **view,
        "soh": np.asarray([float(record["soh"])], dtype=np.float32),
        "battery_id": str(record["battery_id"]),
        "domain_id": str(record["domain_id"]),
        "strategy_id": str(record.get("strategy_id", record.get("condition", "unknown"))),
        "condition": str(record.get("condition", record.get("strategy_id", "unknown"))),
        "cycle_id": int(record["cycle_id"]),
        "split": str(split_name),
        "view_id": str(view_id),
        "input_view_id": str(view_id),
        "raw_point_count": np.asarray([raw_point_count], dtype=np.float32),
        "duration_min": np.asarray([duration_min], dtype=np.float32),
    }


class PreprocessedFeatureDataset(Dataset):
    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        mean: np.ndarray,
        scale: np.ndarray,
        split_name: str,
    ):
        self.records = list(records)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.split_name = str(split_name)
        self._stores: dict[str, PreprocessedStore] = {}

    def limit(self, count: int) -> None:
        if int(count) > 0:
            self.records = self.records[: int(count)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        directory = str(record["_preprocessed_directory"])
        store = self._stores.setdefault(directory, _cached_store(directory))
        row = int(record["_preprocessed_row"])
        features = np.asarray(store.array("terminal", "features")[row], dtype=np.float32)
        features = ((features - self.mean) / self.scale).astype(np.float32)
        return {
            "features": torch.from_numpy(features),
            "soh": torch.as_tensor([float(record["soh"])], dtype=torch.float32),
            "battery_id": str(record["battery_id"]),
            "domain_id": str(record["domain_id"]),
            "strategy_id": str(record.get("strategy_id", record["condition"])),
            "condition": str(record["condition"]),
            "cycle_id": int(record["cycle_id"]),
            "view_id": "statistical_features",
            "raw_point_count": torch.as_tensor([float(record["raw_point_count"])], dtype=torch.float32),
            "duration_min": torch.as_tensor([float(record["duration_min"])], dtype=torch.float32),
        }


def _split_records(
    records: list[dict[str, Any]], config: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    split_value = config.get("data", {}).get("split_file") or config.get("experiment", {}).get("split_file")
    if not split_value:
        raise ValueError("Paper-Backup preprocessed loader requires a split file")
    split_path = _resolve(repo_root, split_value)
    split = load_split_spec(split_path)
    return split_records_from_spec(records, split, split_file=split_path)


def build_preprocessed_feature_loaders(
    config: Mapping[str, Any], repo_root: str | Path, seed: int = 42
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    del seed
    repo_root = Path(repo_root).resolve()
    records, source_info = load_preprocessed_records(config, repo_root, source_view="terminal")
    split_records, split_info = _split_records(records, config, repo_root)
    store = PreprocessedStore(Path(records[0]["_preprocessed_directory"]))
    feature_array = store.array("terminal", "features")
    train_rows = np.asarray([int(item["_preprocessed_row"]) for item in split_records["train"]], dtype=int)
    train_features = np.asarray(feature_array[train_rows], dtype=np.float64)
    mean = np.mean(train_features, axis=0).astype(np.float32)
    scale = np.std(train_features, axis=0, ddof=0).astype(np.float32)
    scale[~np.isfinite(scale) | np.isclose(scale, 0.0)] = 1.0
    datasets = {
        name: PreprocessedFeatureDataset(values, mean, scale, name)
        for name, values in split_records.items()
    }
    debug_n = int(config.get("debug", {}).get("debug_num_samples", 0) or 0)
    if debug_n > 0:
        for dataset in datasets.values():
            dataset.limit(debug_n)
    batch_size = int(config.get("train", {}).get("batch_size", 64))
    common = paper_backup_dataloader_kwargs(config, batch_size=batch_size)
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return loaders, {
        "loader_type": "paper_backup_preprocessed_features",
        "source": source_info,
        "split": split_info,
        "feature_names": list(FEATURE_NAMES),
        "standardization": "train_split_mean_std_only",
        "standardization_mean": mean.tolist(),
        "standardization_scale": scale.tolist(),
        "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "dataloader": dict(common),
        "metadata_in_forward": False,
    }


__all__ = [
    "PreprocessedFeatureDataset",
    "PreprocessedStore",
    "build_preprocessed_feature_loaders",
    "load_preprocessed_records",
    "paper_backup_dataloader_kwargs",
    "sample_from_preprocessed_record",
]
