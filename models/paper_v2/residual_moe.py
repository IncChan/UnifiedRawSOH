"""Residual mixture-of-experts adapters for Paper-v2.

The adapter deliberately operates on the representation produced by the
validated Paper-v1 raw encoder.  It has no knowledge of a domain, strategy,
cell, cycle number, or label provenance.  Routing is therefore a function of
``z_base`` only.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of trainable scalar parameters in ``module``."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class BottleneckResidualExpert(nn.Module):
    """A bottleneck MLP whose output can be initialized to exactly zero."""

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        *,
        dropout: float = 0.0,
        expert_init: str = "zero_output",
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0 or int(bottleneck_dim) <= 0:
            raise ValueError("Expert input and bottleneck dimensions must be positive.")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError("Expert dropout must be in [0, 1).")
        if str(expert_init) not in {"zero_output", "default"}:
            raise ValueError("expert_init must be 'zero_output' or 'default'.")
        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.expert_init = str(expert_init)
        self.down = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(self.bottleneck_dim, self.input_dim)
        if self.expert_init == "zero_output":
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, z_base: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.activation(self.down(z_base))))


class ResidualMoEAdapter(nn.Module):
    """Top-k residual MoE with differentiable importance/load balancing.

    For a batch ``z`` the returned representation is

    ``z + sum(topk_weight * expert(z))``.

    ``topk_weight`` is a softmax over the selected logits, so it sums to one
    even when ``top_k < num_experts``.  The balance loss is the classic
    importance/load product ``E * sum(mean(p) * mean(hard_topk))``.  The hard
    load is intentionally treated as a statistic; the importance term keeps
    the loss differentiable with respect to the router for batch size one as
    well as larger batches.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        num_experts: int = 8,
        top_k: int = 2,
        expert_bottleneck_dim: int = 16,
        dropout: float = 0.0,
        expert_init: str = "zero_output",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_bottleneck_dim = int(expert_bottleneck_dim)
        self.expert_init = str(expert_init)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                f"top_k must satisfy 1 <= top_k <= num_experts; got {self.top_k}, {self.num_experts}."
            )
        if self.expert_bottleneck_dim <= 0:
            raise ValueError("expert_bottleneck_dim must be positive.")
        self.router = nn.Linear(self.input_dim, self.num_experts)
        self.experts = nn.ModuleList(
            [
                BottleneckResidualExpert(
                    self.input_dim,
                    self.expert_bottleneck_dim,
                    dropout=dropout,
                    expert_init=self.expert_init,
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, z_base: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        if z_base.ndim != 2 or z_base.size(-1) != self.input_dim:
            raise ValueError(
                f"ResidualMoEAdapter expects z_base with shape [batch, {self.input_dim}], "
                f"got {tuple(z_base.shape)}."
            )
        if z_base.size(0) == 0:
            raise ValueError("ResidualMoEAdapter does not accept an empty batch.")

        router_logits = self.router(z_base)
        router_probabilities = torch.softmax(router_logits, dim=-1)
        _, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1, largest=True, sorted=True)
        selected_logits = router_logits.gather(dim=-1, index=topk_indices)
        topk_weights = torch.softmax(selected_logits, dim=-1)

        expert_outputs = torch.stack([expert(z_base) for expert in self.experts], dim=1)
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, self.input_dim)
        selected_outputs = expert_outputs.gather(dim=1, index=gather_index)
        residual_delta = (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)
        z_out = z_base + residual_delta

        hard_selected = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.num_experts
        ).to(dtype=z_base.dtype)
        expert_load = hard_selected.mean(dim=(0, 1))
        expert_importance = router_probabilities.mean(dim=0)
        # Keep the hard statistic detached: importance supplies the router
        # gradient while the reported load remains the actual top-k load.
        balance_loss = self.num_experts * torch.sum(
            expert_importance * expert_load.detach()
        )
        routing_entropy = -torch.sum(
            router_probabilities * torch.log(router_probabilities.clamp_min(torch.finfo(z_base.dtype).eps)),
            dim=-1,
        ).mean()
        topk_usage = hard_selected.mean(dim=(0, 1))
        aux: dict[str, torch.Tensor | None] = {
            "router_logits": router_logits,
            "router_probabilities": router_probabilities,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            "expert_load": expert_load,
            "expert_importance": expert_importance,
            "routing_entropy": routing_entropy,
            "topk_usage": topk_usage,
            "balance_loss": balance_loss,
        }
        return z_out, aux

    def parameter_summary(self) -> dict[str, Any]:
        return {
            "adapter": "residual_moe",
            "input_dim": self.input_dim,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_bottleneck_dim": self.expert_bottleneck_dim,
            "expert_init": self.expert_init,
            "trainable_parameters": count_trainable_parameters(self),
            "balance_loss": "num_experts * sum(mean(router_probabilities) * mean(hard_topk_load))",
        }


def residual_moe_parameter_count(
    input_dim: int,
    num_experts: int,
    expert_bottleneck_dim: int,
    *,
    router_bias: bool = True,
    expert_bias: bool = True,
) -> int:
    """Analytically count the trainable parameters added by a Residual MoE."""

    input_dim = int(input_dim)
    num_experts = int(num_experts)
    bottleneck = int(expert_bottleneck_dim)
    expert = input_dim * bottleneck + (bottleneck if expert_bias else 0)
    expert += bottleneck * input_dim + (input_dim if expert_bias else 0)
    router = input_dim * num_experts + (num_experts if router_bias else 0)
    return num_experts * expert + router


__all__ = [
    "BottleneckResidualExpert",
    "ResidualMoEAdapter",
    "count_trainable_parameters",
    "residual_moe_parameter_count",
]
