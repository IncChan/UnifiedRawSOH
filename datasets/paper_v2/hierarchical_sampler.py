"""Four-level replacement sampler for Paper-v2 source training."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterator

import torch
from torch.utils.data import Sampler

from .hierarchy import HierarchyIndex, build_hierarchy_index


class HierarchicalReplacementSampler(Sampler[int]):
    """Uniformly sample domain → strategy → cell → cycle.

    Sampling is driven by a private ``torch.Generator``.  It is independent of
    DataLoader worker scheduling and is reproducible for a ``(seed, epoch)``
    pair.  The sampler is intended for training only; validation and test
    loaders must remain sequential.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.index: HierarchyIndex = build_hierarchy_index(dataset)
        self.num_samples = int(len(dataset) if num_samples is None else num_samples)
        if self.num_samples <= 0:
            raise ValueError("Hierarchical sampler num_samples must be positive.")
        self.seed = int(seed)
        self.epoch = 0
        self._last_counts: dict[str, Counter[str]] = {
            "domain": Counter(),
            "strategy": Counter(),
            "cell": Counter(),
        }
        self._epoch_history: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _choice(values: tuple[Any, ...], generator: torch.Generator) -> Any:
        position = int(torch.randint(len(values), (1,), generator=generator).item())
        return values[position]

    def __iter__(self) -> Iterator[int]:
        # A large odd stride avoids accidental overlap between neighboring
        # epochs while remaining deterministic for every Python process.
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch * 1_000_003) % (2**63 - 1))
        counts: dict[str, Counter[str]] = {
            "domain": Counter(),
            "strategy": Counter(),
            "cell": Counter(),
        }
        for _ in range(self.num_samples):
            domain = self._choice(self.index.domains, generator)
            strategy = self._choice(self.index.domain_strategies(domain), generator)
            cells = self.index.cells_in_environment((domain, strategy))
            if not cells:
                raise ValueError(
                    f"Hierarchy environment {(domain, strategy)!r} has no physical cells."
                )
            cell = self._choice(cells, generator)
            cycles = self.index._by_domain_strategy_cell[(domain, strategy, cell)]
            if not cycles:
                raise ValueError(
                    f"Hierarchy cell {(domain, strategy, cell)!r} has no cycles."
                )
            index = int(self._choice(tuple(cycles), generator))
            counts["domain"][domain] += 1
            counts["strategy"][f"{domain}|{strategy}"] += 1
            counts["cell"][f"{domain}|{cell}"] += 1
            yield index
        self._last_counts = counts
        self._epoch_history.append(self._counts_as_dict(counts))

    @staticmethod
    def _counts_as_dict(counts: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
        return {
            level: {str(key): int(value) for key, value in sorted(counter.items())}
            for level, counter in counts.items()
        }

    def audit(self) -> dict[str, Any]:
        return {
            "sampler": "domain_strategy_cell_cycle_replacement",
            "seed": self.seed,
            "epoch": self.epoch,
            "num_samples": self.num_samples,
            "selection_rule": [
                "uniform domain",
                "uniform strategy group within domain",
                "uniform physical cell within strategy group",
                "uniform cycle within physical cell and strategy group",
            ],
            "inventory": self.index.inventory(),
            "sampled_counts": self._counts_as_dict(self._last_counts),
            "epoch_history": list(self._epoch_history),
        }


HierarchicalSampler = HierarchicalReplacementSampler


def build_hierarchical_sampler(dataset: Any, *, num_samples: int | None = None, seed: int = 0) -> HierarchicalReplacementSampler:
    return HierarchicalReplacementSampler(dataset, num_samples=num_samples, seed=seed)


__all__ = [
    "HierarchicalReplacementSampler",
    "HierarchicalSampler",
    "build_hierarchical_sampler",
]
