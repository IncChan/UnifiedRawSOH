"""BiContext Mamba with zero-initialized sample-adaptive phase fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from .bicontext_mamba import BiContextMambaSOHModel


class BiContextAdaptiveFusionSOHModel(BiContextMambaSOHModel):
    """Retain both context bridges and learn the CC/CV fusion weight per cycle.

    The final gate is zero initialized, so every sample starts with the exact
    0.5/0.5 phase mixture used by :class:`BiContextMambaSOHModel`.  The model
    can subsequently adjust phase reliability from the two pooled phase
    representations without changing the bridge topology.
    """

    model_id = "Ours-BiContext-Adaptive-Fusion-Final5"

    def __init__(self, *, adaptive_fusion_gate_hidden_dim: int = 4, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = int(adaptive_fusion_gate_hidden_dim)
        if hidden_dim < 1:
            raise ValueError("adaptive_fusion_gate_hidden_dim must be positive")
        self.adaptive_fusion_gate = nn.Sequential(
            nn.Linear(2 * self.phase_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.adaptive_fusion_gate[-1].weight)
        nn.init.zeros_(self.adaptive_fusion_gate[-1].bias)
        self.last_adaptive_cc_weight = None

    def fuse(self, z_cc: torch.Tensor, z_cv: torch.Tensor) -> torch.Tensor:
        cc_weight = torch.sigmoid(
            self.adaptive_fusion_gate(torch.cat((z_cc, z_cv), dim=-1))
        )
        cc_value = self.cc_projection(z_cc)
        cv_value = self.cv_projection(z_cv)
        base = cc_weight * cc_value + (1.0 - cc_weight) * cv_value
        self.last_adaptive_cc_weight = cc_weight.detach()
        return self.fusion_norm(base + self.ordinary_fusion_mlp(base))

    def forward_with_aux(self, *args, **kwargs):
        output = super().forward_with_aux(*args, **kwargs)
        output["adaptive_fusion_cc_weight"] = self.last_adaptive_cc_weight
        output["adaptive_fusion_cv_weight"] = 1.0 - self.last_adaptive_cc_weight
        return output


__all__ = ["BiContextAdaptiveFusionSOHModel"]
