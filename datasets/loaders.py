"""Shared single-domain and unified raw-cycle loader interfaces."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .base import RawTerminalSignalUnavailable, UNIFIED_SAMPLE_KEYS
from .domains import build_default_domain_registry, canonical_domain_id
from .filters import filter_raw_records_to_pinn_fonly_samples, filter_records_by_invalid_cycles
from .mit import validate_mit_physical_cohort
from .registry import build_default_registry
from .splits import (
    load_invalid_cycles,
    load_split_spec,
    split_records_from_spec,
)
from .xjtu import UnifiedCCCVSampleDataset, build_full_life_cycle_metadata
from .soh_labels import (
    BOL_LABEL_MODE,
    BOL_RULE_VERSION,
    apply_bol_relative_soh,
    build_bol_references,
    frozen_smarthealth_bol_references,
    is_bol_label_mode,
)


def _resolve_path(repo_root, value):
    value = Path(value)
    return value if value.is_absolute() else (Path(repo_root) / value).resolve()


def _canonical_domain_id(value):
    """Resolve a paper-level domain while accepting saved source identifiers."""

    return canonical_domain_id(value)


def _domain_id_from_config(config):
    experiment = config.get("experiment", {})
    data = config.get("data", {})
    return _canonical_domain_id(
        experiment.get(
            "domain_id",
            experiment.get("dataset_id", data.get("domain_id", data.get("dataset_id", data.get("dataset", "xjtu")))),
        )
    )


def _domain_mapping_value(mapping, domain_id, adapter_id, fallback):
    """Resolve per-domain overrides without forcing source-name keys."""

    mapping = mapping or {}
    for key in (domain_id, adapter_id):
        if key in mapping:
            return mapping[key]
    return fallback


def _make_balanced_sampler(dataset, balance_mode):
    if not balance_mode or balance_mode == "none":
        return None
    metadata = []
    for index in range(len(dataset)):
        item = dataset[index]
        metadata.append(
            (
                str(item.get("domain_id", item.get("dataset_id", "unknown"))),
                str(item.get("battery_id", "unknown")),
            )
        )
    if balance_mode in {"dataset", "domain"}:
        keys = [domain_id for domain_id, _ in metadata]
    elif balance_mode == "battery":
        keys = [battery_id for _, battery_id in metadata]
    elif balance_mode in {"dataset_battery", "domain_battery"}:
        keys = metadata
    elif balance_mode == "domain_battery_hierarchical":
        # Equal domain mass, then equal battery mass within each domain.
        group_counts = Counter(metadata)
        batteries_per_domain = Counter(domain_id for domain_id, _ in set(metadata))
        weights = torch.as_tensor(
            [1.0 / (batteries_per_domain[d] * group_counts[(d, b)]) for d, b in metadata],
            dtype=torch.double,
        )
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    else:
        raise ValueError(
            "balance_mode must be one of 'none', 'domain', 'battery', 'domain_battery', "
            "'domain_battery_hierarchical' "
            "(the legacy dataset aliases remain supported)"
        )
    counts = Counter(keys)
    weights = torch.as_tensor([1.0 / counts[key] for key in keys], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _build_loaders_from_datasets(datasets, config):
    train_cfg = config["train"]
    data_cfg = config["data"]
    batch_size = int(train_cfg.get("batch_size", 64))
    balance_mode = data_cfg.get("balance_mode", "none")
    train_sampler = _make_balanced_sampler(datasets["train"], balance_mode)
    common = {
        "batch_size": batch_size,
        "num_workers": int(data_cfg.get("num_workers", 0)),
    }
    return {
        "train": DataLoader(
            datasets["train"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            pin_memory=torch.cuda.is_available(),
            **common,
        ),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def _build_raw_domain(config, repo_root, seed, domain_id, data_root):
    data_cfg = config["data"]
    domain_registry = build_default_domain_registry()
    domain = domain_registry.get(domain_id)
    adapter_id = str(data_cfg.get("adapter_id", domain.adapter_id))
    adapter_registry = build_default_registry()
    split_file = (
        config.get("experiment", {}).get("split_file")
        or data_cfg.get("split_file")
        or domain.split_file
    )
    split_path = None
    if split_file:
        split_path = _resolve_path(repo_root, split_file)
    if split_path is None:
        raise ValueError(
            "Raw dataset configuration must provide experiment.split_file or data.split_file; "
            "dataset split policy belongs in JSON."
        )
    split_spec = load_split_spec(split_path)
    nominal_capacity = float(
        _domain_mapping_value(
            data_cfg.get("nominal_capacities", {}),
            domain_id,
            adapter_id,
            data_cfg.get("nominal_capacity", domain.nominal_capacity_ah or 2.0),
        )
    )
    label_scale_mode = _domain_mapping_value(
        data_cfg.get("label_scale_modes", {}),
        domain_id,
        adapter_id,
        data_cfg.get("label_scale_mode", "auto_capacity_to_soh"),
    )
    adapter = adapter_registry.create(
        adapter_id,
        data_root=data_root,
        nominal_capacity=nominal_capacity,
        label_scale_mode=label_scale_mode,
        domain_id=domain_id,
    )
    if not getattr(adapter, "raw_terminal_signals", False):
        readiness_error = str(getattr(adapter, "readiness_error", "")).strip()
        raise RawTerminalSignalUnavailable(
            f"Battery domain {domain_id!r} (adapter {adapter_id!r}) cannot emit the common "
            "raw-cycle contract."
            + (f" {readiness_error}" if readiness_error else "")
        )

    configured_batches = list(config.get("experiment", {}).get("batches", []))
    runtime_batch = config.get("runtime_batch")
    if runtime_batch:
        records_before_filter = adapter.load_records(batch=runtime_batch)
    elif data_cfg.get("data_mode", "single_domain") == "all_batch_pooled" or not configured_batches:
        records_before_filter = adapter.load_records()
    else:
        records_before_filter = []
        for batch in configured_batches:
            records_before_filter.extend(adapter.load_records(batch=batch))
    for record in records_before_filter:
        # Keep source ``dataset_id`` untouched for compatibility, while all
        # experiment composition/balancing/reporting uses the stable domain.
        record["domain_id"] = domain_id

    # Paper-v2 labels are constructed once from the complete canonical
    # trajectory, before invalid-cycle filtering, split assignment, debug
    # truncation, or raw-window normalization.  Q_ref is never placed in a
    # model input; only the derived soh_bol value is consumed downstream.
    label_mode = BOL_LABEL_MODE if is_bol_label_mode(config) else "rated_relative"
    label_provenance = {}
    if label_mode == BOL_LABEL_MODE:
        if str(domain_id).startswith("smarthealth_"):
            label_provenance = frozen_smarthealth_bol_references(
                records_before_filter, domain_id=domain_id
            )
        else:
            label_provenance = build_bol_references(
                records_before_filter, domain_id=domain_id
            )
        records_before_filter = apply_bol_relative_soh(
            records_before_filter,
            label_provenance,
            domain_id=domain_id,
        )
    cycle_metadata = build_full_life_cycle_metadata(records_before_filter)

    records = records_before_filter
    invalid_cycle_audit = {
        "enabled": False,
        "removed_records": 0,
        "invalid_cycles": [],
    }
    if split_path is not None:
        invalid_cycles = load_invalid_cycles(split_path)
        records, invalid_cycle_audit = filter_records_by_invalid_cycles(records, invalid_cycles)
    filter_mode = str(data_cfg.get("sample_filter_mode", "none"))
    filter_audit = {
        "enabled": False,
        "mode": "none",
        "raw_records_before": len(records),
        "raw_records_after": len(records),
        "removed_raw_records": 0,
    }
    if filter_mode == "pinn_fonly_3sigma_adjacent_x1":
        reference_root = _resolve_path(repo_root, data_cfg["pinn_fonly_reference_root"])
        records, filter_audit = filter_raw_records_to_pinn_fonly_samples(
            records,
            reference_root=reference_root,
            match_atol=float(data_cfg.get("pinn_fonly_match_atol", 1e-6)),
            drop_adjacent_x1_last=bool(data_cfg.get("pinn_fonly_drop_adjacent_x1_last", True)),
        )
    elif filter_mode != "none":
        raise ValueError("data.sample_filter_mode must be 'none' or 'pinn_fonly_3sigma_adjacent_x1'")

    if domain_id == "mit":
        # Keep the official continuation-aware Paper-124 cohort from being
        # bypassed when callers invoke main.py directly instead of its E1
        # launcher. The exact test rule remains owned by the split JSON.
        validate_mit_physical_cohort(
            (record["battery_id"] for record in records),
            split_spec,
            require_full_physical_cohort=bool(
                data_cfg.get("require_full_physical_cohort", False)
            ),
        )

    split_records, split_meta = split_records_from_spec(
        records,
        split_spec,
        split_file=split_path,
    )
    debug_n = int(config.get("debug", {}).get("debug_num_samples", 0) or 0)
    if debug_n > 0:
        split_records = {name: values[:debug_n] for name, values in split_records.items()}

    datasets = {
        name: UnifiedCCCVSampleDataset(
            split_records[name],
            data_cfg,
            config["normalization"],
            split_name=name,
            seed=int(seed) + index * 1000,
            cycle_metadata=cycle_metadata,
        )
        for index, name in enumerate(("train", "val", "test"))
    }
    split_info = {
        "domain_id": domain_id,
        "dataset_id": adapter_id,
        "domain_metadata": domain.metadata(),
        "data_root": str(data_root),
        "record_counts": {name: len(values) for name, values in split_records.items()},
        "sample_counts": {name: len(datasets[name]) for name in datasets},
        "battery_counts": {
            name: len({str(item["battery_id"]) for item in values})
            for name, values in split_records.items()
        },
        "split_overlap": {
            "train_val": sorted(
                {str(item["battery_id"]) for item in split_records["train"]}
                & {str(item["battery_id"]) for item in split_records["val"]}
            ),
            "train_test": sorted(
                {str(item["battery_id"]) for item in split_records["train"]}
                & {str(item["battery_id"]) for item in split_records["test"]}
            ),
            "val_test": sorted(
                {str(item["battery_id"]) for item in split_records["val"]}
                & {str(item["battery_id"]) for item in split_records["test"]}
            ),
        },
        "sample_contract": list(UNIFIED_SAMPLE_KEYS),
        "quality_skips": {name: dict(datasets[name].skipped) for name in datasets},
        "normalization": {
            "kind": "fixed_physical",
            "train_derived_statistics_used": False,
            "config": config["normalization"],
        },
        "full_life_metadata_built_before_filter_and_split": True,
        "invalid_cycle_filter": invalid_cycle_audit,
        "sample_filter": filter_audit,
        "label": {
            "label_mode": label_mode,
            "label_field": "soh_bol" if label_mode == BOL_LABEL_MODE else "soh",
            "reference_rule": BOL_RULE_VERSION if label_mode == BOL_LABEL_MODE else None,
            "reference_provenance": label_provenance,
            "q_ref_is_model_input": False,
            "q_ref_in_normalization": False,
        },
        **split_meta,
    }
    return datasets, split_info


def build_single_domain_loaders(config, repo_root, seed):
    domain_id = _domain_id_from_config(config)
    domain = build_default_domain_registry().get(domain_id)
    root_value = config["data"].get("data_root") or domain.data_root
    if not root_value:
        raise ValueError(f"Domain {domain_id!r} has no configured data root")
    data_root = _resolve_path(repo_root, root_value)
    datasets, split_info = _build_raw_domain(config, repo_root, seed, domain_id, data_root)
    return _build_loaders_from_datasets(datasets, config), split_info


def build_unified_loaders(config, repo_root, seed):
    """Build a shared-model loader with optional domain/battery balancing.

    Each domain can provide its own fixed physical normalization and split file
    while emitting the same common raw-cycle sample contract.
    """

    requested_domain_ids = config.get("experiment", {}).get(
        "domain_ids", config.get("experiment", {}).get("dataset_ids", [])
    )
    domain_ids = [_canonical_domain_id(item) for item in requested_domain_ids]
    if not domain_ids:
        raise ValueError("Unified loader requires experiment.domain_ids")
    if len(set(domain_ids)) != len(domain_ids):
        raise ValueError(f"Unified loader received duplicate domain IDs: {domain_ids}")
    data_cfg = config["data"]
    roots = data_cfg.get("data_roots", {})
    split_datasets = {"train": [], "val": [], "test": []}
    domain_info = {}
    domain_registry = build_default_domain_registry()
    for index, domain_id in enumerate(domain_ids):
        domain = domain_registry.get(domain_id)
        domain_config = dict(config)
        domain_config["experiment"] = dict(config.get("experiment", {}))
        domain_config["experiment"]["domain_id"] = domain_id
        domain_config["experiment"].pop("batches", None)
        batches_by_domain = data_cfg.get("batches_by_domain", data_cfg.get("batches_by_dataset", {}))
        if domain_id in batches_by_domain:
            domain_config["experiment"]["batches"] = list(batches_by_domain[domain_id])
        split_files = data_cfg.get("split_files", {})
        split_file = split_files.get(domain_id, domain.split_file)
        if not split_file:
            raise ValueError(
                f"Unified config must provide a split file for domain {domain_id!r}"
            )
        domain_config["experiment"]["split_file"] = split_file
        domain_config["data"] = dict(data_cfg)
        domain_config["data"]["data_mode"] = "single_domain"
        root_value = roots.get(domain_id) or data_cfg.get("data_root") or domain.data_root
        if not root_value:
            raise ValueError(f"Unified config has no data root for domain {domain_id!r}")
        domain_config["data"]["data_root"] = root_value
        domain_config["data"]["adapter_id"] = domain.adapter_id
        if "normalizations" in data_cfg and domain_id in data_cfg["normalizations"]:
            domain_config["normalization"] = data_cfg["normalizations"][domain_id]
        elif domain.normalization is not None:
            domain_config["normalization"] = dict(domain.normalization)
        datasets, info = _build_raw_domain(
            domain_config,
            repo_root,
            seed + index * 10_000,
            domain_id,
            _resolve_path(repo_root, domain_config["data"]["data_root"]),
        )
        for split_name in split_datasets:
            split_datasets[split_name].append(datasets[split_name])
        domain_info[domain_id] = info

    combined = {
        split_name: ConcatDataset(values)
        for split_name, values in split_datasets.items()
    }
    loaders = _build_loaders_from_datasets(combined, config)
    return loaders, {
        "loader_type": "unified_multi_dataset",
        "domain_ids": domain_ids,
        # Retained as a compatibility alias for older analysis utilities.
        "dataset_ids": domain_ids,
        "domain_info": domain_info,
        "balance_mode": data_cfg.get("balance_mode", "none"),
        "sample_contract": list(UNIFIED_SAMPLE_KEYS),
    }


def build_lodo_loaders(config, repo_root, seed):
    """Build source train/val loaders and a held-out-domain test loader.

    Every domain is read through its existing split JSON. Source test records
    and target train/validation records are deliberately excluded.
    """

    from UnifiedRawSOH.trainers.reusability import parse_reusability_protocol

    protocol = parse_reusability_protocol(config)
    if protocol["protocol"] != "leave_one_domain_out":
        raise ValueError(
            "LODO loader requires reusability.protocol=leave_one_domain_out"
        )
    source_domain_ids = protocol["source_domain_ids"]
    target_domain_ids = protocol["target_domain_ids"]
    if len(target_domain_ids) != 1:
        raise ValueError("LODO requires exactly one target domain")
    target_domain_id = target_domain_ids[0]
    requested = config.get("experiment", {}).get("domain_ids", [])
    configured_domain_ids = [_canonical_domain_id(value) for value in requested]
    expected_domain_ids = source_domain_ids + [target_domain_id]
    if set(configured_domain_ids) != set(expected_domain_ids):
        raise ValueError(
            "experiment.domain_ids must equal source_domain_ids plus target_domain_id; "
            f"configured={configured_domain_ids}, expected={expected_domain_ids}"
        )

    data_cfg = config["data"]
    roots = data_cfg.get("data_roots", {})
    domain_registry = build_default_domain_registry()
    datasets_by_domain = {}
    domain_info = {}
    for index, domain_id in enumerate(expected_domain_ids):
        domain = domain_registry.get(domain_id)
        domain_config = dict(config)
        domain_config["experiment"] = dict(config.get("experiment", {}))
        domain_config["experiment"]["domain_id"] = domain_id
        domain_config["experiment"].pop("batches", None)
        batches_by_domain = data_cfg.get(
            "batches_by_domain", data_cfg.get("batches_by_dataset", {})
        )
        if domain_id in batches_by_domain:
            domain_config["experiment"]["batches"] = list(
                batches_by_domain[domain_id]
            )
        split_file = data_cfg.get("split_files", {}).get(
            domain_id, domain.split_file
        )
        if not split_file:
            raise ValueError(f"LODO config requires a split file for {domain_id!r}")
        domain_config["experiment"]["split_file"] = split_file
        domain_config["data"] = dict(data_cfg)
        domain_config["data"]["data_mode"] = "single_domain"
        root_value = (
            roots.get(domain_id) or data_cfg.get("data_root") or domain.data_root
        )
        if not root_value:
            raise ValueError(f"LODO config has no data root for {domain_id!r}")
        domain_config["data"]["data_root"] = root_value
        domain_config["data"]["adapter_id"] = domain.adapter_id
        if domain_id in data_cfg.get("normalizations", {}):
            domain_config["normalization"] = data_cfg["normalizations"][domain_id]
        elif domain.normalization is not None:
            domain_config["normalization"] = dict(domain.normalization)
        datasets, info = _build_raw_domain(
            domain_config,
            repo_root,
            seed + index * 10_000,
            domain_id,
            _resolve_path(repo_root, root_value),
        )
        datasets_by_domain[domain_id] = datasets
        domain_info[domain_id] = info

    selected = {
        "train": ConcatDataset(
            [
                datasets_by_domain[domain_id]["train"]
                for domain_id in source_domain_ids
            ]
        ),
        "val": ConcatDataset(
            [
                datasets_by_domain[domain_id]["val"]
                for domain_id in source_domain_ids
            ]
        ),
        "test": datasets_by_domain[target_domain_id]["test"],
    }
    loaders = _build_loaders_from_datasets(selected, config)
    return loaders, {
        "loader_type": "leave_one_domain_out",
        "source_domain_ids": source_domain_ids,
        "target_domain_id": target_domain_id,
        "domain_ids": expected_domain_ids,
        "domain_info": domain_info,
        "split_usage": {
            "train": "source domains' original train split only",
            "val": "source domains' original val split only",
            "test": "left-out domain's original test split only",
            "excluded": [
                "source domains' test splits",
                "left-out domain's train and val splits",
            ],
        },
        "balance_mode": data_cfg.get("balance_mode", "none"),
        "sample_contract": list(UNIFIED_SAMPLE_KEYS),
    }
