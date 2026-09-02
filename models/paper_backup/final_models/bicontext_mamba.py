"""Dual phase Mambas with zero-initialized bidirectional context exchange."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...c5b_model import pool_hidden_states
from .common import IndependentPhaseMambaBase


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden.mean(dim=1)
    weights = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denominator


class BiContextMambaSOHModel(IndependentPhaseMambaBase):
    """Exchange global phase context between otherwise identical Mamba branches.

    The bridge is inserted after an early Mamba block.  It does not assume a
    point-to-point alignment between CC and CV: every local token receives a
    projected global context from the other phase, modulated by its own gate.
    Both cross projections are zero initialized, so the initial bridge output
    is exactly zero and the initial function is ordinary independent fusion.
    """

    model_id = "Ours-BiContext-Mamba-Final5"

    def __init__(
        self,
        *,
        bridge_after_layer: int = 1,
        bridge_gate_hidden_dim: int = 4,
        ordinary_fusion_hidden_dim: int = 27,
        **kwargs,
    ):
        num_layers = int(kwargs.get("num_layers", 3))
        bridge_after_layer = int(bridge_after_layer)
        bridge_gate_hidden_dim = int(bridge_gate_hidden_dim)
        ordinary_fusion_hidden_dim = int(ordinary_fusion_hidden_dim)
        if not 1 <= bridge_after_layer < num_layers:
            raise ValueError("bridge_after_layer must split the configured Mamba layers")
        if bridge_gate_hidden_dim < 1:
            raise ValueError("bridge_gate_hidden_dim must be positive")
        if ordinary_fusion_hidden_dim < 1:
            raise ValueError("ordinary_fusion_hidden_dim must be positive")

        super().__init__(**kwargs)
        self.bridge_after_layer = bridge_after_layer
        self.context_dim = self.phase_feature_dim // 2

        self.cc_context_norm = nn.LayerNorm(self.context_dim)
        self.cv_context_norm = nn.LayerNorm(self.context_dim)
        self.cv_to_cc_projection = nn.Linear(self.context_dim, self.context_dim)
        self.cc_to_cv_projection = nn.Linear(self.context_dim, self.context_dim)
        self.cc_bridge_gate = self._make_gate(bridge_gate_hidden_dim)
        self.cv_bridge_gate = self._make_gate(bridge_gate_hidden_dim)
        self.ordinary_fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim, ordinary_fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(ordinary_fusion_hidden_dim, self.fusion_dim),
        )

        # The bridge and fusion residual are exact no-ops at initialization.
        for projection in (self.cv_to_cc_projection, self.cc_to_cv_projection):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        for gate in (self.cc_bridge_gate, self.cv_bridge_gate):
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)
        nn.init.zeros_(self.ordinary_fusion_mlp[-1].weight)
        nn.init.zeros_(self.ordinary_fusion_mlp[-1].bias)

        self.last_cc_bridge_gate = None
        self.last_cv_bridge_gate = None
        self.last_cc_bridge_ratio = None
        self.last_cv_bridge_ratio = None

    def _make_gate(self, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2 * self.context_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    @staticmethod
    def _apply_mask(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return value
        return value * mask.to(device=value.device, dtype=value.dtype).unsqueeze(-1)

    @staticmethod
    def _contribution_ratio(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        numerator = delta.flatten(start_dim=1).norm(dim=-1)
        denominator = base.flatten(start_dim=1).norm(dim=-1).clamp_min(1e-12)
        return numerator / denominator

    def encode_phases(
        self,
        cc_signal,
        cv_signal,
        cc_mask=None,
        cv_mask=None,
        cc_time=None,
        cv_time=None,
        cc_temperature=None,
        cv_temperature=None,
    ):
        cc_hidden = self.cc_branch.input_encoder(
            cc_signal, cc_mask, cc_time, cc_temperature
        )
        cv_hidden = self.cv_branch.input_encoder(
            cv_signal, cv_mask, cv_time, cv_temperature
        )

        for layer_index in range(self.bridge_after_layer):
            cc_hidden = self.cc_branch.layers[layer_index](cc_hidden)
            cv_hidden = self.cv_branch.layers[layer_index](cv_hidden)

        cc_context = _masked_mean(self.cc_context_norm(cc_hidden), cc_mask)
        cv_context = _masked_mean(self.cv_context_norm(cv_hidden), cv_mask)
        expanded_cc_context = cc_context.unsqueeze(1).expand(-1, cv_hidden.size(1), -1)
        expanded_cv_context = cv_context.unsqueeze(1).expand(-1, cc_hidden.size(1), -1)

        cc_gate = torch.sigmoid(
            self.cc_bridge_gate(torch.cat((cc_hidden, expanded_cv_context), dim=-1))
        )
        cv_gate = torch.sigmoid(
            self.cv_bridge_gate(torch.cat((cv_hidden, expanded_cc_context), dim=-1))
        )
        cc_gate = self._apply_mask(cc_gate, cc_mask)
        cv_gate = self._apply_mask(cv_gate, cv_mask)
        cc_delta = self._apply_mask(
            cc_gate * self.cv_to_cc_projection(cv_context).unsqueeze(1), cc_mask
        )
        cv_delta = self._apply_mask(
            cv_gate * self.cc_to_cv_projection(cc_context).unsqueeze(1), cv_mask
        )

        with torch.no_grad():
            self.last_cc_bridge_gate = cc_gate.detach()
            self.last_cv_bridge_gate = cv_gate.detach()
            self.last_cc_bridge_ratio = self._contribution_ratio(
                cc_delta, self._apply_mask(cc_hidden, cc_mask)
            ).detach()
            self.last_cv_bridge_ratio = self._contribution_ratio(
                cv_delta, self._apply_mask(cv_hidden, cv_mask)
            ).detach()

        cc_hidden = cc_hidden + cc_delta
        cv_hidden = cv_hidden + cv_delta
        for layer_index in range(self.bridge_after_layer, len(self.cc_branch.layers)):
            cc_hidden = self.cc_branch.layers[layer_index](cc_hidden)
            cv_hidden = self.cv_branch.layers[layer_index](cv_hidden)

        cc_hidden = self.cc_branch.final_norm(cc_hidden)
        cv_hidden = self.cv_branch.final_norm(cv_hidden)
        return (
            pool_hidden_states(cc_hidden, cc_mask, self.pooling),
            pool_hidden_states(cv_hidden, cv_mask, self.pooling),
        )

    def fuse(self, z_cc: torch.Tensor, z_cv: torch.Tensor) -> torch.Tensor:
        base = 0.5 * self.cc_projection(z_cc) + 0.5 * self.cv_projection(z_cv)
        return self.fusion_norm(base + self.ordinary_fusion_mlp(base))

    def forward_with_aux(self, *args, **kwargs):
        output = super().forward_with_aux(*args, **kwargs)
        output.update(
            {
                "cc_bridge_gate": self.last_cc_bridge_gate,
                "cv_bridge_gate": self.last_cv_bridge_gate,
                "cc_bridge_contribution_ratio": self.last_cc_bridge_ratio,
                "cv_bridge_contribution_ratio": self.last_cv_bridge_ratio,
            }
        )
        return output


__all__ = ["BiContextMambaSOHModel"]
