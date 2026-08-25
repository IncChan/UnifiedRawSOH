"""One-reference-cell data construction for Paper-v1 E3 head-only adaptation."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset

from UnifiedRawSOH.datasets.domains import (
    build_default_domain_registry,
    canonical_domain_id,
)
from UnifiedRawSOH.datasets.loaders import _build_raw_domain, _resolve_path
from UnifiedRawSOH.datasets.splits import load_split_spec


def _stable_hex(*parts):
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _target_domain_config(config, repo_root):
    target = canonical_domain_id(config["one_cell"]["target_domain_id"])
    data_cfg = config["data"]
    domain = build_default_domain_registry().get(target)
    result = copy.deepcopy(config)
    result["experiment"]["domain_id"] = target
    result["experiment"]["split_file"] = data_cfg["split_files"][target]
    result["experiment"]["batches"] = list(
        data_cfg.get("batches_by_domain", {}).get(target, [])
    )
    result["data"]["data_mode"] = "single_domain"
    result["data"]["data_root"] = (
        data_cfg.get("data_roots", {}).get(target) or domain.data_root
    )
    result["data"]["adapter_id"] = domain.adapter_id
    if target in data_cfg.get("normalizations", {}):
        result["normalization"] = copy.deepcopy(
            data_cfg["normalizations"][target]
        )
    elif domain.normalization is not None:
        result["normalization"] = copy.deepcopy(domain.normalization)
    split_path = _resolve_path(repo_root, result["experiment"]["split_file"])
    return result, target, split_path


def _build_target_datasets(config, repo_root, seed):
    target_config, target, split_path = _target_domain_config(config, repo_root)
    data_root = _resolve_path(repo_root, target_config["data"]["data_root"])
    datasets, split_info = _build_raw_domain(
        target_config,
        repo_root,
        int(seed),
        target,
        data_root,
    )
    return datasets, split_info, load_split_spec(split_path), split_path


def _records(dataset):
    return list(getattr(dataset, "valid_records", []))


def discover_support_inventory(config, repo_root, seed=0, _prepared=None):
    """Return exact development candidates and fixed test cells by group."""

    if _prepared is None:
        datasets, split_info, split_spec, split_path = _build_target_datasets(
            config, repo_root, seed
        )
    else:
        datasets, split_info, split_spec, split_path = _prepared
    target = canonical_domain_id(config["one_cell"]["target_domain_id"])
    groups = [str(value) for value in config["one_cell"]["support_groups"]]
    development_records = _records(datasets["train"]) + _records(datasets["val"])
    test_records = _records(datasets["test"])

    candidates = {
        group: sorted(
            {
                str(record["battery_id"])
                for record in development_records
                if str(record["condition"]) == group
            }
        )
        for group in groups
    }
    test_by_group = {
        group: sorted(
            {
                str(record["battery_id"])
                for record in test_records
                if str(record["condition"]) == group
            }
        )
        for group in groups
    }
    missing_development = [group for group, values in candidates.items() if not values]
    missing_test = [group for group, values in test_by_group.items() if not values]
    if missing_development or missing_test:
        raise ValueError(
            "One-cell inventory has empty groups: "
            f"development={missing_development}, test={missing_test}"
        )

    explicit_development = split_spec.get(
        "development_batteries_by_condition"
    )
    if explicit_development is not None:
        ordered = {}
        for group in groups:
            expected = [str(value) for value in explicit_development[group]]
            if set(expected) != set(candidates[group]):
                raise ValueError(
                    f"Development cells differ from split JSON for {group}: "
                    f"observed={candidates[group]}, expected={expected}"
                )
            ordered[group] = expected
        candidates = ordered

    explicit_test = split_spec.get("test_batteries_by_condition")
    if explicit_test is not None:
        for group in groups:
            expected = sorted(str(value) for value in explicit_test[group])
            if expected != test_by_group[group]:
                raise ValueError(
                    f"Test cells differ from split JSON for {group}: "
                    f"observed={test_by_group[group]}, expected={expected}"
                )
    else:
        expected_all = sorted(str(value) for value in split_spec["test_batteries"])
        observed_all = sorted(
            {value for values in test_by_group.values() for value in values}
        )
        if expected_all != observed_all:
            raise ValueError(
                "Test cells differ from split JSON: "
                f"observed={observed_all}, expected={expected_all}"
            )

    development_all = {
        value for values in candidates.values() for value in values
    }
    test_all = {value for values in test_by_group.values() for value in values}
    overlap = sorted(development_all & test_all)
    if overlap:
        raise ValueError(f"Development/test physical-cell overlap: {overlap}")

    return {
        "target_domain_id": target,
        "split_file": str(Path(split_path).resolve()),
        "support_groups": groups,
        "development_cells_by_group": candidates,
        "test_cells_by_group": test_by_group,
        "all_test_cells": sorted(test_all),
        "all_test_sample_count": len(datasets["test"]),
        "split_info": split_info,
    }


def select_support_cell(config, inventory, support_group, support_choice):
    """Choose one development cell deterministically without Python hash()."""

    one_cell = config["one_cell"]
    target = canonical_domain_id(one_cell["target_domain_id"])
    group = str(support_group)
    if group not in inventory["development_cells_by_group"]:
        raise ValueError(f"Unknown support group {group!r}")
    candidates = list(inventory["development_cells_by_group"][group])
    if not candidates:
        raise ValueError(f"Support group {group!r} has no development cells")

    mode = str(one_cell["support_selection_mode"])
    if mode == "ordered_ab":
        choice = str(support_choice).upper()
        index = {"A": 0, "B": 1}.get(choice)
        if index is None:
            raise ValueError("SmartHealth support choice must be A or B")
        if len(candidates) < 2:
            raise ValueError(f"Support group {group!r} needs two A/B cells")
        selected = candidates[index]
    elif mode == "stable_seed_rotation":
        seed = int(support_choice)
        configured_seeds = [int(value) for value in one_cell["support_seeds"]]
        stable_candidates = sorted(
            candidates,
            key=lambda cell: _stable_hex(target, group, cell),
        )
        offset = int(_stable_hex(target, group)[:16], 16) % len(stable_candidates)
        if seed in configured_seeds:
            position = configured_seeds.index(seed)
        else:
            position = int(_stable_hex(target, group, seed)[:16], 16)
        selected = stable_candidates[(offset + position) % len(stable_candidates)]
    else:
        raise ValueError(f"Unknown support_selection_mode: {mode!r}")

    if selected in set(inventory["all_test_cells"]):
        raise ValueError(f"Selected support cell is a test cell: {selected}")
    return {
        "target_domain_id": target,
        "support_group": group,
        "support_choice": str(support_choice),
        "support_cell": selected,
        "selection_mode": mode,
        "candidate_cells": candidates,
        "stable_selection_sha256": _stable_hex(
            target, group, support_choice, selected
        ),
    }


def _sample_identity(dataset, index):
    item = dataset[index]
    return (
        str(item["battery_id"]),
        str(item["condition"]),
        int(item["cycle_id"]),
    )


def stratified_support_split(
    dataset,
    indices,
    validation_ratio,
    seed,
    bin_width=0.02,
):
    """Deterministically split one cell's cycles across SOH bins."""

    indices = [int(index) for index in indices]
    if len(indices) < 2:
        raise ValueError("One-cell support requires at least two valid cycles")
    bins = {}
    for index in indices:
        soh = float(dataset[index]["soh"].reshape(-1)[0])
        bin_id = int(np.floor(soh / float(bin_width)))
        bins.setdefault(bin_id, []).append(index)
    for bin_id, values in bins.items():
        values.sort(
            key=lambda index: _stable_hex(
                seed, bin_id, *_sample_identity(dataset, index)
            )
        )

    target_val = max(1, int(round(float(validation_ratio) * len(indices))))
    target_val = min(target_val, len(indices) - 1)
    allocations = {
        bin_id: min(
            len(values) - 1,
            int(np.floor(float(validation_ratio) * len(values))),
        )
        for bin_id, values in bins.items()
    }
    assigned = sum(allocations.values())
    ranked_bins = sorted(
        bins,
        key=lambda bin_id: (
            -(
                float(validation_ratio) * len(bins[bin_id])
                - np.floor(float(validation_ratio) * len(bins[bin_id]))
            ),
            _stable_hex(seed, bin_id),
        ),
    )
    while assigned < target_val:
        progressed = False
        for bin_id in ranked_bins:
            capacity = len(bins[bin_id]) - 1
            if allocations[bin_id] < capacity:
                allocations[bin_id] += 1
                assigned += 1
                progressed = True
                if assigned >= target_val:
                    break
        if not progressed:
            break
    if assigned == 0:
        # With sparse trajectories every SOH bin may contain one cycle. Keep
        # the global split non-empty and choose the validation singleton by a
        # stable hash instead of abandoning stratification determinism.
        selected_bin = min(
            bins,
            key=lambda bin_id: _stable_hex(seed, "singleton", bin_id),
        )
        allocations[selected_bin] = 1

    val_indices = []
    train_indices = []
    for bin_id, values in sorted(bins.items()):
        count = allocations[bin_id]
        val_indices.extend(values[:count])
        train_indices.extend(values[count:])
    if not train_indices or not val_indices:
        raise ValueError("Stratified support split produced an empty partition")
    return sorted(train_indices), sorted(val_indices)


def build_one_cell_loaders(
    config,
    repo_root,
    support_group,
    support_choice,
    selected_cell=None,
):
    """Build one-cell support train/val and the complete fixed target test."""

    one_cell = config["one_cell"]
    split_seed = int(one_cell.get("support_split_seed", 2025))
    choice_seed = (
        int(support_choice)
        if str(support_choice).lstrip("-").isdigit()
        else int(_stable_hex(support_choice)[:8], 16)
    )
    datasets, split_info, split_spec, split_path = _build_target_datasets(
        config, repo_root, split_seed
    )
    inventory = discover_support_inventory(
        config,
        repo_root,
        split_seed,
        _prepared=(datasets, split_info, split_spec, split_path),
    )
    selection = select_support_cell(
        config, inventory, support_group, support_choice
    )
    if selected_cell is not None and str(selected_cell) != selection["support_cell"]:
        raise ValueError(
            f"Planned support cell {selected_cell!r} does not match "
            f"deterministic selection {selection['support_cell']!r}"
        )

    development = ConcatDataset([datasets["train"], datasets["val"]])
    support_indices = [
        index
        for index in range(len(development))
        if (
            str(development[index]["battery_id"]) == selection["support_cell"]
            and str(development[index]["condition"]) == str(support_group)
        )
    ]
    train_indices, val_indices = stratified_support_split(
        development,
        support_indices,
        validation_ratio=float(one_cell["support_validation_ratio"]),
        seed=split_seed + choice_seed,
        bin_width=float(one_cell.get("support_soh_bin_width", 0.02)),
    )
    batch_size = int(one_cell.get("batch_size", config["train"]["batch_size"]))
    num_workers = int(config["data"].get("num_workers", 0))
    loaders = {
        "train": DataLoader(
            Subset(development, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            Subset(development, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    support_split = {
        "split_file": str(Path(split_path).resolve()),
        "support_cell": selection["support_cell"],
        "support_group": str(support_group),
        "train_sample_count": len(train_indices),
        "validation_sample_count": len(val_indices),
        "validation_ratio": float(one_cell["support_validation_ratio"]),
        "validation_mode": "soh_stratified",
        "train_samples": [
            {
                "battery_id": _sample_identity(development, index)[0],
                "condition": _sample_identity(development, index)[1],
                "cycle_id": _sample_identity(development, index)[2],
            }
            for index in train_indices
        ],
        "validation_samples": [
            {
                "battery_id": _sample_identity(development, index)[0],
                "condition": _sample_identity(development, index)[1],
                "cycle_id": _sample_identity(development, index)[2],
            }
            for index in val_indices
        ],
        "test_cells_by_group": inventory["test_cells_by_group"],
        "all_test_cells": inventory["all_test_cells"],
        "all_test_sample_count": inventory["all_test_sample_count"],
        "source_split_info": split_info,
    }
    return loaders, selection, support_split, inventory
