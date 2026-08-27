"""Parameter-matched dense residual adapter for Paper-v2 controls."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .residual_moe import count_trainable_parameters, residual_moe_parameter_count


class DenseResidualAdapter(nn.Module):
    """A domain-agnostic bottleneck residual MLP.

    Its final projection is zero initialized by default, making the adapter
    exactly identity at initialization just like the Residual MoE experts.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        *,
        dropout: float = 0.0,
        adapter_init: str = "zero_output",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.adapter_init = str(adapter_init)
        if self.input_dim <= 0 or self.bottleneck_dim <= 0:
            raise ValueError("Dense adapter dimensions must be positive.")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError("Dense adapter dropout must be in [0, 1).")
        if self.adapter_init not in {"zero_output", "default"}:
            raise ValueError("adapter_init must be 'zero_output' or 'default'.")
        self.down = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(self.bottleneck_dim, self.input_dim)
        if self.adapter_init == "zero_output":
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, z_base: torch.Tensor) -> torch.Tensor:
        if z_base.ndim != 2 or z_base.size(-1) != self.input_dim:
            raise ValueError(
                f"DenseResidualAdapter expects z_base with shape [batch, {self.input_dim}], "
                f"got {tuple(z_base.shape)}."
            )
        return self.up(self.dropout(self.activation(self.down(z_base))))

    def parameter_summary(self) -> dict[str, Any]:
        return {
            "adapter": "dense_residual",
            "input_dim": self.input_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "adapter_init": self.adapter_init,
            "trainable_parameters": count_trainable_parameters(self),
        }


def dense_adapter_parameter_count(
    input_dim: int,
    bottleneck_dim: int,
    *,
    bias: bool = True,
) -> int:
    input_dim = int(input_dim)
    bottleneck_dim = int(bottleneck_dim)
    return (
        input_dim * bottleneck_dim
        + (bottleneck_dim if bias else 0)
        + bottleneck_dim * input_dim
        + (input_dim if bias else 0)
    )


def choose_parameter_matched_dense_bottleneck(
    input_dim: int,
    *,
    num_experts: int = 8,
    top_k: int = 2,
    expert_bottleneck_dim: int = 16,
    max_bottleneck_dim: int | None = None,
) -> dict[str, int | float]:
    """Choose the integer dense width closest to the MoE adapter size.

    ``top_k`` is accepted as part of the public comparison contract.  The
    current router parameterization has the same parameter count regardless
    of top-k, but retaining it here makes the resolved comparison auditable.
    """

    del top_k
    target = residual_moe_parameter_count(
        input_dim,
        num_experts,
        expert_bottleneck_dim,
    )
    input_dim = int(input_dim)
    if max_bottleneck_dim is None:
        max_bottleneck_dim = max(1, int(target / max(1, 2 * input_dim))) + 4
    candidates = range(1, int(max_bottleneck_dim) + 1)
    chosen = min(
        candidates,
        key=lambda width: (
            abs(dense_adapter_parameter_count(input_dim, width) - target),
            width,
        ),
    )
    dense = dense_adapter_parameter_count(input_dim, chosen)
    return {
        "target_moe_parameters": int(target),
        "dense_bottleneck_dim": int(chosen),
        "dense_parameters": int(dense),
        "absolute_difference": int(abs(dense - target)),
        "relative_error": float(abs(dense - target) / max(target, 1)),
    }


__all__ = [
    "DenseResidualAdapter",
    "choose_parameter_matched_dense_bottleneck",
    "dense_adapter_parameter_count",
]
