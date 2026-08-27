"""Static and runtime leakage audits for Paper-v2 zero-cell LODO."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from .hierarchy import domain_id_of, physical_cell_id_of, strategy_group_of


def _partition_inventory(dataset: Any) -> dict[str, Any]:
    domains: set[str] = set()
    cells: set[tuple[str, str]] = set()
    strategies: set[tuple[str, str]] = set()
    count = 0
    for index in range(len(dataset)):
        item = dataset[index]
        domain = domain_id_of(item, index)
        cell = physical_cell_id_of(item, index)
        strategy = strategy_group_of(item, index)
        domains.add(domain)
        cells.add((domain, cell))
        strategies.add((domain, strategy))
        count += 1
    return {
        "sample_count": count,
        "domains": sorted(domains),
        "strategies": [list(value) for value in sorted(strategies)],
        "physical_cells": [list(value) for value in sorted(cells)],
    }


def validate_lodo_provenance(
    config: Mapping[str, Any],
    *,
    loaders: Mapping[str, Any] | None = None,
    split_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and report the target-test-only boundary.

    This function does not infer a split.  It only audits an already resolved
    protocol and, when loaders are supplied, checks the emitted partitions.
    """

    experiment = dict(config.get("experiment", {}))
    reusability = dict(config.get("reusability", {}))
    source = [str(value) for value in experiment.get("source_domain_ids", [])]
    target_value = experiment.get("target_domain_id")
    if target_value is None:
        targets = [str(value) for value in experiment.get("target_domain_ids", [])]
    else:
        targets = [str(target_value)]
    if not source or not targets:
        raise ValueError("LODO provenance requires non-empty source and target domains.")
    overlap = sorted(set(source) & set(targets))
    if overlap:
        raise ValueError(f"LODO source/target overlap: {overlap}")
    if len(targets) != 1:
        raise ValueError(f"Paper-v2 zero-cell LODO requires one target domain, got {targets}")
    if reusability.get("protocol") != "leave_one_domain_out":
        raise ValueError("LODO provenance requires reusability.protocol='leave_one_domain_out'.")
    required_forbidden = {
        "target_train_and_validation_usage": "forbidden",
        "source_test_usage": "forbidden",
    }
    for field, expected in required_forbidden.items():
        if reusability.get(field) != expected:
            raise ValueError(f"reusability.{field} must be {expected!r} for zero-cell LODO.")
    for field, expected in {
        "source_train_split": "train",
        "source_validation_split": "val",
        "target_test_split": "test",
    }.items():
        if reusability.get(field) != expected:
            raise ValueError(f"reusability.{field} must be {expected!r}.")
    data = dict(config.get("data", {}))
    if bool(data.get("use_train_derived_statistics", False)):
        raise ValueError("LODO normalization cannot use target-fitted/train-derived statistics.")

    partitions: dict[str, Any] = {}
    if loaders is not None:
        for name in ("train", "val", "test"):
            if name not in loaders:
                raise ValueError(f"LODO loader is missing partition {name!r}.")
            partitions[name] = _partition_inventory(loaders[name].dataset)
        train_domains = set(partitions["train"]["domains"])
        val_domains = set(partitions["val"]["domains"])
        test_domains = set(partitions["test"]["domains"])
        if not train_domains <= set(source) or not val_domains <= set(source):
            raise ValueError(
                f"Target/undeclared domain entered source partitions: train={sorted(train_domains)}, "
                f"val={sorted(val_domains)}, source={source}"
            )
        if test_domains != {targets[0]}:
            raise ValueError(
                f"Target test partition must contain only {targets[0]!r}; got {sorted(test_domains)}"
            )
    audit = {
        "protocol": "leave_one_domain_out",
        "source_domain_ids": source,
        "target_domain_id": targets[0],
        "target_train_validation_usage": "forbidden",
        "source_test_usage": "forbidden",
        "normalization_target_fitted": False,
        "target_test_only": True,
        "partitions": partitions,
    }
    if split_info is not None:
        audit["loader_split_info"] = dict(split_info)
    return audit


def assert_cell_disjoint(left_items: Any, right_items: Any) -> None:
    left = {
        (domain_id_of(left_items[index], index), physical_cell_id_of(left_items[index], index))
        for index in range(len(left_items))
    }
    right = {
        (domain_id_of(right_items[index], index), physical_cell_id_of(right_items[index], index))
        for index in range(len(right_items))
    }
    overlap = sorted(left & right)
    if overlap:
        raise ValueError(f"Physical-cell overlap detected between episode sides: {overlap}")


validate_lodo_leakage = validate_lodo_provenance


__all__ = ["assert_cell_disjoint", "validate_lodo_leakage", "validate_lodo_provenance"]
