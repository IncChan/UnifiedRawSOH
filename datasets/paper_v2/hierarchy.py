"""Auditable dataset-domain-strategy-cell-cycle hierarchy utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def _metadata_value(item: Any, names: tuple[str, ...], *, field: str, index: int) -> str:
    if not isinstance(item, Mapping):
        raise TypeError(f"Dataset item {index} must be a mapping to build the Paper-v2 hierarchy.")
    for name in names:
        value = item.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(
        f"Dataset item {index} is missing hierarchy field {field!r}; "
        f"accepted metadata keys={names}."
    )


def domain_id_of(item: Mapping[str, Any], index: int = -1) -> str:
    return _metadata_value(item, ("domain_id", "dataset_id"), field="domain_id", index=index)


def strategy_group_of(item: Mapping[str, Any], index: int = -1) -> str:
    """Read explicit strategy metadata without inferring it from cycle order/files."""

    return _metadata_value(
        item,
        ("strategy_group", "condition", "batch_name"),
        field="strategy_group",
        index=index,
    )


def physical_cell_id_of(item: Mapping[str, Any], index: int = -1) -> str:
    return _metadata_value(
        item,
        ("physical_cell_id", "cell_id", "battery_id"),
        field="physical_cell_id",
        index=index,
    )


def cycle_id_of(item: Mapping[str, Any], index: int = -1) -> str:
    return _metadata_value(
        item,
        ("cycle_id", "cycle", "raw_cycle_order_index"),
        field="cycle_id",
        index=index,
    )


@dataclass(frozen=True)
class HierarchyMetadata:
    index: int
    domain_id: str
    strategy_group: str
    physical_cell_id: str
    cycle_id: str

    @property
    def environment(self) -> tuple[str, str]:
        return self.domain_id, self.strategy_group

    @property
    def cell_key(self) -> tuple[str, str]:
        return self.domain_id, self.physical_cell_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "domain_id": self.domain_id,
            "strategy_group": self.strategy_group,
            "physical_cell_id": self.physical_cell_id,
            "cycle_id": self.cycle_id,
        }


class HierarchyIndex:
    """An immutable index over the four sampling levels.

    The index is built from dataset metadata only.  Model-facing tensors are
    never inspected, and a missing strategy field is an error instead of an
    opportunity to guess from a filename or cycle order.
    """

    def __init__(self, items: Iterable[Mapping[str, Any]]) -> None:
        self.entries = [
            HierarchyMetadata(
                index=index,
                domain_id=domain_id_of(item, index),
                strategy_group=strategy_group_of(item, index),
                physical_cell_id=physical_cell_id_of(item, index),
                cycle_id=cycle_id_of(item, index),
            )
            for index, item in enumerate(items)
        ]
        if not self.entries:
            raise ValueError("Paper-v2 hierarchy cannot be built from an empty dataset.")
        self._by_domain: dict[str, list[int]] = defaultdict(list)
        self._by_environment: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._by_domain_strategy_cell: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for entry in self.entries:
            self._by_domain[entry.domain_id].append(entry.index)
            self._by_environment[entry.environment].append(entry.index)
            self._by_cell[entry.cell_key].append(entry.index)
            self._by_domain_strategy_cell[
                (entry.domain_id, entry.strategy_group, entry.physical_cell_id)
            ].append(entry.index)
        self._domains = tuple(sorted(self._by_domain))
        self._environments = tuple(sorted(self._by_environment))

    @classmethod
    def from_dataset(cls, dataset: Any) -> "HierarchyIndex":
        return cls(dataset[index] for index in range(len(dataset)))

    @property
    def domains(self) -> tuple[str, ...]:
        return self._domains

    @property
    def environments(self) -> tuple[tuple[str, str], ...]:
        return self._environments

    def domain_indices(self, domain_id: str) -> tuple[int, ...]:
        key = str(domain_id)
        if key not in self._by_domain:
            raise KeyError(f"Unknown hierarchy domain {key!r}.")
        return tuple(self._by_domain[key])

    def environment_indices(self, environment: tuple[str, str]) -> tuple[int, ...]:
        key = (str(environment[0]), str(environment[1]))
        if key not in self._by_environment:
            raise KeyError(f"Unknown hierarchy environment {key!r}.")
        return tuple(self._by_environment[key])

    def cell_indices(self, domain_id: str, physical_cell_id: str) -> tuple[int, ...]:
        key = (str(domain_id), str(physical_cell_id))
        if key not in self._by_cell:
            raise KeyError(f"Unknown hierarchy cell {key!r}.")
        return tuple(self._by_cell[key])

    def domain_strategies(self, domain_id: str) -> tuple[str, ...]:
        domain_id = str(domain_id)
        return tuple(sorted(strategy for domain, strategy in self._environments if domain == domain_id))

    def cells_in_environment(self, environment: tuple[str, str]) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    self.entries[index].physical_cell_id
                    for index in self.environment_indices(environment)
                }
            )
        )

    def metadata(self, index: int) -> HierarchyMetadata:
        return self.entries[int(index)]

    def inventory(self) -> dict[str, Any]:
        """Return a JSON-ready domain → strategy → cell inventory."""

        domains: dict[str, Any] = {}
        for domain in self.domains:
            strategies: dict[str, Any] = {}
            domain_cells = {
                self.entries[index].physical_cell_id
                for index in self.domain_indices(domain)
            }
            for strategy in self.domain_strategies(domain):
                env = (domain, strategy)
                cells: dict[str, Any] = {}
                for cell in self.cells_in_environment(env):
                    indices = self._by_domain_strategy_cell[(domain, strategy, cell)]
                    cells[cell] = {
                        "sample_count": len(indices),
                        "cycle_count": len(indices),
                    }
                strategies[strategy] = {
                    "sample_count": len(self.environment_indices(env)),
                    "cell_count": len(cells),
                    "cells": cells,
                }
            domains[domain] = {
                "sample_count": len(self.domain_indices(domain)),
                "strategy_count": len(strategies),
                "cell_count": len(domain_cells),
                "strategies": strategies,
            }
        return {
            "domain_count": len(self.domains),
            "domains": domains,
            "environment_count": len(self.environments),
            "sample_count": len(self.entries),
        }


def build_hierarchy_index(dataset_or_items: Any) -> HierarchyIndex:
    if hasattr(dataset_or_items, "__len__") and hasattr(dataset_or_items, "__getitem__"):
        return HierarchyIndex.from_dataset(dataset_or_items)
    return HierarchyIndex(dataset_or_items)


__all__ = [
    "HierarchyIndex",
    "HierarchyMetadata",
    "build_hierarchy_index",
    "cycle_id_of",
    "domain_id_of",
    "physical_cell_id_of",
    "strategy_group_of",
]
