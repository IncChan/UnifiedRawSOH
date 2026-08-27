"""Source-only pseudo-LODO episode construction for Paper-v2 MLDG."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .hierarchy import HierarchyIndex, build_hierarchy_index


EPISODE_TYPES = ("dataset", "strategy")
_EPISODE_TYPE_ALIASES = {
    "dataset": "dataset",
    "dataset_level": "dataset",
    "strategy": "strategy",
    "strategy_level": "strategy",
}


@dataclass(frozen=True)
class Episode:
    episode_type: str
    source_domain_ids: tuple[str, ...]
    meta_train_indices: tuple[int, ...]
    pseudo_target_indices: tuple[int, ...]
    pseudo_target_domains: tuple[str, ...]
    pseudo_target_strategies: tuple[str, ...]
    pseudo_target_cells: tuple[tuple[str, str], ...]
    cell_disjoint_expansion: bool
    audit: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_type": self.episode_type,
            "source_domain_ids": list(self.source_domain_ids),
            "meta_train_indices": list(self.meta_train_indices),
            "pseudo_target_indices": list(self.pseudo_target_indices),
            "pseudo_target_domains": list(self.pseudo_target_domains),
            "pseudo_target_strategies": list(self.pseudo_target_strategies),
            "pseudo_target_cells": [list(value) for value in self.pseudo_target_cells],
            "cell_disjoint_expansion": self.cell_disjoint_expansion,
            "audit": dict(self.audit),
        }


class SourceEpisodeBuilder:
    """Build complete held-out pseudo-environments from source train only."""

    def __init__(
        self,
        dataset: Any,
        source_domain_ids: list[str] | tuple[str, ...] | None = None,
        *,
        seed: int = 0,
        dataset_episode_probability: float = 0.5,
        strategy_episode_probability: float = 0.5,
    ) -> None:
        self.dataset = dataset
        self.index: HierarchyIndex = build_hierarchy_index(dataset)
        observed = set(self.index.domains)
        if source_domain_ids is None:
            source_domain_ids = list(self.index.domains)
        self.source_domain_ids = tuple(str(value) for value in source_domain_ids)
        if not self.source_domain_ids:
            raise ValueError("Source episode builder requires at least one source domain.")
        if len(set(self.source_domain_ids)) != len(self.source_domain_ids):
            raise ValueError(f"Source episode domains contain duplicates: {self.source_domain_ids}")
        missing = sorted(set(self.source_domain_ids) - observed)
        unexpected = sorted(observed - set(self.source_domain_ids))
        if missing:
            raise ValueError(f"Source episode domains are missing from source train: {missing}")
        if unexpected:
            raise ValueError(
                "Target or undeclared domains entered the source episode dataset: "
                f"{unexpected}"
            )
        self.dataset_probability = float(dataset_episode_probability)
        self.strategy_probability = float(strategy_episode_probability)
        if self.dataset_probability < 0.0 or self.strategy_probability < 0.0:
            raise ValueError("Episode probabilities must be non-negative.")
        if self.dataset_probability + self.strategy_probability <= 0.0:
            raise ValueError("At least one episode probability must be positive.")
        self.seed = int(seed)
        self.epoch = 0
        self._episode_number = 0
        self._type_counts: Counter[str] = Counter()
        self._heldout_counts: Counter[str] = Counter()
        self._episodes: list[dict[str, Any]] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._episode_number = 0

    def _rng(self) -> random.Random:
        value = self.seed + self.epoch * 1_000_003 + self._episode_number * 97_003
        self._episode_number += 1
        return random.Random(value)

    def _choose_type(self, rng: random.Random, episode_type: str | None) -> str:
        if episode_type is not None:
            episode_type = _EPISODE_TYPE_ALIASES.get(str(episode_type))
            if episode_type is None:
                raise ValueError(f"episode_type must be one of {EPISODE_TYPES}; got {episode_type!r}")
            return episode_type
        total = self.dataset_probability + self.strategy_probability
        return "dataset" if rng.random() < self.dataset_probability / total else "strategy"

    def sample_episode(self, episode_type: str | None = None) -> Episode:
        rng = self._rng()
        resolved_type = self._choose_type(rng, episode_type)
        all_indices = set(range(len(self.index.entries)))
        if resolved_type == "dataset":
            pseudo_target_domain = rng.choice(list(self.source_domain_ids))
            target_indices = set(self.index.domain_indices(pseudo_target_domain))
            target_domains = {pseudo_target_domain}
            target_environments = {
                (self.index.metadata(index).domain_id, self.index.metadata(index).strategy_group)
                for index in target_indices
            }
            expansion = False
            selection = {"pseudo_target_dataset": pseudo_target_domain}
        else:
            environments = [
                environment
                for environment in self.index.environments
                if environment[0] in self.source_domain_ids
            ]
            if not environments:
                raise ValueError("Source train has no strategy environments for strategy episodes.")
            selected_environment = rng.choice(environments)
            selected_indices = set(self.index.environment_indices(selected_environment))
            selected_cells = {
                (self.index.metadata(index).domain_id, self.index.metadata(index).physical_cell_id)
                for index in selected_indices
            }
            # If a physical cell appears under more than one strategy, remove
            # all of its cycles from meta-train.  This is stricter than merely
            # holding out the selected environment and is recorded explicitly.
            target_indices = {
                index
                for index in all_indices
                if self.index.metadata(index).cell_key in selected_cells
            }
            target_domains = {self.index.metadata(index).domain_id for index in target_indices}
            target_environments = {
                self.index.metadata(index).environment for index in target_indices
            }
            expansion = target_indices != selected_indices
            selection = {
                "pseudo_target_environment": list(selected_environment),
                "selected_environment_indices": sorted(selected_indices),
            }
        meta_train_indices = all_indices - target_indices
        if not target_indices or not meta_train_indices:
            raise ValueError(
                f"{resolved_type}-level episode cannot split source train into two non-empty sides: "
                f"meta={len(meta_train_indices)}, target={len(target_indices)}"
            )
        target_cells = sorted(
            {
                self.index.metadata(index).cell_key
                for index in target_indices
            }
        )
        target_strategies = sorted({environment[1] for environment in target_environments})
        overlap = {
            self.index.metadata(index).cell_key
            for index in meta_train_indices
        } & set(target_cells)
        if overlap:
            raise RuntimeError(
                f"Episode construction violated physical-cell disjointness: {sorted(overlap)}"
            )
        audit = {
            "selection": selection,
            "meta_train_count": len(meta_train_indices),
            "pseudo_target_count": len(target_indices),
            "pseudo_target_domains": sorted(target_domains),
            "pseudo_target_environments": [list(value) for value in sorted(target_environments)],
            "pseudo_target_cells": [list(value) for value in target_cells],
            "cell_disjoint": True,
            "cell_disjoint_expansion": bool(expansion),
            "source_only": True,
        }
        episode = Episode(
            episode_type=resolved_type,
            source_domain_ids=self.source_domain_ids,
            meta_train_indices=tuple(sorted(meta_train_indices)),
            pseudo_target_indices=tuple(sorted(target_indices)),
            pseudo_target_domains=tuple(sorted(target_domains)),
            pseudo_target_strategies=tuple(target_strategies),
            pseudo_target_cells=tuple(target_cells),
            cell_disjoint_expansion=bool(expansion),
            audit=audit,
        )
        self._type_counts[resolved_type] += 1
        for domain in target_domains:
            self._heldout_counts[domain] += 1
        self._episodes.append(audit)
        return episode

    def build_dataset_episode(self) -> Episode:
        return self.sample_episode("dataset")

    def build_strategy_episode(self) -> Episode:
        return self.sample_episode("strategy")

    def draw_indices(
        self,
        indices: tuple[int, ...] | list[int],
        count: int,
        *,
        seed_offset: int = 0,
        replacement: bool = True,
    ) -> list[int]:
        values = list(indices)
        if not values:
            raise ValueError("Cannot draw an episode batch from an empty index pool.")
        count = int(count)
        if count <= 0:
            raise ValueError("Episode batch size must be positive.")
        rng = random.Random(
            self.seed + self.epoch * 1_000_003 + self._episode_number * 97_003 + int(seed_offset)
        )
        if replacement:
            return [rng.choice(values) for _ in range(count)]
        if count > len(values):
            raise ValueError("replacement=False cannot draw more indices than the pool contains.")
        return rng.sample(values, count)

    def audit(self) -> dict[str, Any]:
        return {
            "sampler": "source_pseudo_lodo",
            "seed": self.seed,
            "epoch": self.epoch,
            "source_domain_ids": list(self.source_domain_ids),
            "dataset_episode_probability": self.dataset_probability,
            "strategy_episode_probability": self.strategy_probability,
            "inventory": self.index.inventory(),
            "episode_type_counts": {
                key: int(value) for key, value in sorted(self._type_counts.items())
            },
            "heldout_domain_counts": {
                key: int(value) for key, value in sorted(self._heldout_counts.items())
            },
            "episodes": list(self._episodes),
        }


HierarchicalEpisodeSampler = SourceEpisodeBuilder
EpisodeSampler = SourceEpisodeBuilder
EpisodeBuilder = SourceEpisodeBuilder


def build_dataset_level_episode(
    dataset: Any,
    source_domain_ids: list[str] | tuple[str, ...],
    *,
    seed: int = 0,
) -> Episode:
    return build_source_episode(
        dataset, source_domain_ids, episode_type="dataset", seed=seed
    )


def build_strategy_level_episode(
    dataset: Any,
    source_domain_ids: list[str] | tuple[str, ...],
    *,
    seed: int = 0,
) -> Episode:
    return build_source_episode(
        dataset, source_domain_ids, episode_type="strategy", seed=seed
    )


def build_source_episode(
    dataset: Any,
    source_domain_ids: list[str] | tuple[str, ...],
    *,
    episode_type: str,
    seed: int = 0,
) -> Episode:
    builder = SourceEpisodeBuilder(dataset, source_domain_ids, seed=seed)
    return builder.sample_episode(episode_type)


__all__ = [
    "EPISODE_TYPES",
    "Episode",
    "EpisodeBuilder",
    "EpisodeSampler",
    "HierarchicalEpisodeSampler",
    "SourceEpisodeBuilder",
    "build_source_episode",
    "build_dataset_level_episode",
    "build_strategy_level_episode",
]
