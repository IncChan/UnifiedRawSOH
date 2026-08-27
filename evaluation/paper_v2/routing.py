"""Routing diagnostics for Paper-v2 Residual MoE evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import torch


def _as_strings(batch: Mapping[str, Any], key: str, count: int, fallback: str = "unknown") -> list[str]:
    value = batch.get(key)
    if value is None:
        return [fallback] * count
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    else:
        values = list(value) if isinstance(value, (list, tuple)) else [value]
    if len(values) != count:
        raise ValueError(f"Routing metadata {key!r} has {len(values)} values for batch size {count}.")
    return [str(item) for item in values]


def _routing_statistics(probabilities: np.ndarray, topk_indices: np.ndarray, entropy: np.ndarray) -> dict[str, Any]:
    if probabilities.ndim != 2 or topk_indices.ndim != 2:
        raise ValueError("Routing probabilities and top-k indices must be rank-2 arrays.")
    if probabilities.shape[0] != topk_indices.shape[0]:
        raise ValueError("Routing arrays disagree on batch size.")
    num_experts = int(probabilities.shape[1])
    top_k = int(topk_indices.shape[1])
    if num_experts <= 0 or top_k <= 0 or top_k > num_experts:
        raise ValueError("Routing arrays contain invalid expert dimensions.")
    hard = np.zeros(num_experts, dtype=np.float64)
    for index in topk_indices.reshape(-1):
        index = int(index)
        if index < 0 or index >= num_experts:
            raise ValueError(f"Top-k expert index {index} is outside [0, {num_experts}).")
        hard[index] += 1.0
    hard /= float(topk_indices.size)
    importance = probabilities.astype(np.float64).mean(axis=0)
    entropy_value = float(np.asarray(entropy, dtype=np.float64).mean())
    return {
        "num_samples": int(probabilities.shape[0]),
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_load": hard.tolist(),
        "expert_importance": importance.tolist(),
        "routing_entropy": entropy_value,
        "topk_usage": hard.tolist(),
    }


class RoutingAccumulator:
    """Collect per-sample routing tensors and summarize by audit metadata."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def update(self, aux: Mapping[str, Any], batch: Mapping[str, Any]) -> None:
        probabilities = aux.get("router_probabilities")
        indices = aux.get("topk_indices")
        entropy = aux.get("routing_entropy")
        if probabilities is None or indices is None:
            return
        if not torch.is_tensor(probabilities) or not torch.is_tensor(indices):
            raise TypeError("Routing tensors must be torch tensors.")
        probabilities = probabilities.detach().cpu()
        indices = indices.detach().cpu()
        if entropy is None:
            entropy = -torch.sum(
                probabilities * torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).eps)),
                dim=-1,
            )
        elif torch.is_tensor(entropy):
            entropy = entropy.detach().cpu()
            if entropy.ndim == 0:
                entropy = entropy.repeat(probabilities.size(0))
            entropy = entropy.reshape(-1)
        else:
            entropy = torch.full((probabilities.size(0),), float(entropy), dtype=probabilities.dtype)
        if probabilities.ndim != 2 or indices.ndim != 2 or entropy.numel() != probabilities.size(0):
            raise ValueError("Routing tensors have incompatible shapes.")
        count = probabilities.size(0)
        domains = _as_strings(batch, "domain_id", count)
        strategies = _as_strings(batch, "strategy_group", count)
        if strategies == ["unknown"] * count:
            strategies = _as_strings(batch, "condition", count)
        cells = _as_strings(batch, "physical_cell_id", count)
        if cells == ["unknown"] * count:
            cells = _as_strings(batch, "battery_id", count)
        for row_index in range(count):
            self._rows.append(
                {
                    "domain_id": domains[row_index],
                    "strategy_group": strategies[row_index],
                    "physical_cell_id": cells[row_index],
                    "probabilities": probabilities[row_index].numpy().tolist(),
                    "topk_indices": indices[row_index].numpy().tolist(),
                    "entropy": float(entropy[row_index].item()),
                }
            )

    def summary(self) -> dict[str, Any]:
        if not self._rows:
            return {
                "num_samples": 0,
                "num_experts": 0,
                "top_k": 0,
                "expert_load": [],
                "expert_importance": [],
                "routing_entropy": None,
                "topk_usage": [],
                "by_domain": {},
                "by_strategy": {},
                "by_cell": {},
            }
        probabilities = np.asarray([row["probabilities"] for row in self._rows], dtype=np.float64)
        indices = np.asarray([row["topk_indices"] for row in self._rows], dtype=np.int64)
        entropy = np.asarray([row["entropy"] for row in self._rows], dtype=np.float64)
        result = _routing_statistics(probabilities, indices, entropy)
        for name, key_builder in (
            ("by_domain", lambda row: row["domain_id"]),
            ("by_strategy", lambda row: f"{row['domain_id']}|{row['strategy_group']}"),
            ("by_cell", lambda row: f"{row['domain_id']}|{row['physical_cell_id']}"),
        ):
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in self._rows:
                buckets[key_builder(row)].append(row)
            result[name] = {}
            for key, rows in sorted(buckets.items()):
                result[name][key] = _routing_statistics(
                    np.asarray([row["probabilities"] for row in rows], dtype=np.float64),
                    np.asarray([row["topk_indices"] for row in rows], dtype=np.int64),
                    np.asarray([row["entropy"] for row in rows], dtype=np.float64),
                )
        return result


def summarize_routing_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize JSON-ready per-sample routing records."""

    accumulator = RoutingAccumulator()
    for row in records:
        aux = {
            "router_probabilities": torch.as_tensor(row["probabilities"], dtype=torch.float32).unsqueeze(0),
            "topk_indices": torch.as_tensor(row["topk_indices"], dtype=torch.long).unsqueeze(0),
            "routing_entropy": torch.as_tensor([row.get("entropy", 0.0)], dtype=torch.float32),
        }
        accumulator.update(aux, row)
    return accumulator.summary()


__all__ = ["RoutingAccumulator", "summarize_routing_records"]
