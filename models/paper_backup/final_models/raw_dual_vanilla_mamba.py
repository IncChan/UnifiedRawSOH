"""Two independent phase Mambas with ordinary, non-interacting fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import IndependentPhaseMambaBase


class RawDualVanillaMambaSOHModel(IndependentPhaseMambaBase):
    model_id = "Raw-Dual-Vanilla-Mamba-Final5"

    def __init__(self, *, ordinary_fusion_hidden_dim: int = 48, **kwargs):
        super().__init__(**kwargs)
        hidden = int(ordinary_fusion_hidden_dim)
        if hidden < 1:
            raise ValueError("ordinary_fusion_hidden_dim must be positive")
        self.ordinary_fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.fusion_dim),
        )
        # Start from pure independent equal fusion, then let the ordinary MLP
        # learn a post-fusion refinement. This also gives the control the same
        # trainable parameter budget as the explicit interaction model.
        nn.init.zeros_(self.ordinary_fusion_mlp[-1].weight)
        nn.init.zeros_(self.ordinary_fusion_mlp[-1].bias)

    def fuse(self, z_cc: torch.Tensor, z_cv: torch.Tensor) -> torch.Tensor:
        base = 0.5 * self.cc_projection(z_cc) + 0.5 * self.cv_projection(z_cv)
        return self.fusion_norm(base + self.ordinary_fusion_mlp(base))


__all__ = ["RawDualVanillaMambaSOHModel"]
