"""Independent phase Mambas with zero-initialized gated interaction fusion."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import IndependentPhaseMambaBase


class InteractionFusionMambaSOHModel(IndependentPhaseMambaBase):
    model_id = "Ours-Interaction-Fusion-Final5"

    def __init__(
        self,
        *,
        interaction_hidden_dim: int = 16,
        gate_hidden_dim: int = 8,
        gate_init_cc: float = 0.45,
        gate_init_cv: float = 0.45,
        gate_init_interaction: float = 0.10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        interaction_hidden_dim = int(interaction_hidden_dim)
        gate_hidden_dim = int(gate_hidden_dim)
        if interaction_hidden_dim < 1 or gate_hidden_dim < 1:
            raise ValueError("Interaction and gate hidden dimensions must be positive")
        probabilities = (float(gate_init_cc), float(gate_init_cv), float(gate_init_interaction))
        if any(value <= 0.0 for value in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8):
            raise ValueError("Initial gate probabilities must be positive and sum to one")
        interaction_input_dim = 4 * self.phase_feature_dim
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_input_dim, interaction_hidden_dim),
            nn.SiLU(),
            nn.Linear(interaction_hidden_dim, self.fusion_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * self.phase_feature_dim, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 3),
        )
        # The interaction output is exactly zero at initialization. The gate is
        # input-independent and starts at the conservative 0.45/0.45/0.10 mix.
        nn.init.zeros_(self.interaction_mlp[-1].weight)
        nn.init.zeros_(self.interaction_mlp[-1].bias)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        with torch.no_grad():
            self.gate_mlp[-1].bias.copy_(
                torch.tensor([math.log(value) for value in probabilities], dtype=self.gate_mlp[-1].bias.dtype)
            )
        self.last_gate = None
        self.last_interaction_ratio = None

    def fuse(self, z_cc: torch.Tensor, z_cv: torch.Tensor) -> torch.Tensor:
        interaction_input = torch.cat(
            (z_cc, z_cv, z_cc * z_cv, torch.abs(z_cc - z_cv)), dim=-1
        )
        z_interaction = self.interaction_mlp(interaction_input)
        gate = torch.softmax(self.gate_mlp(torch.cat((z_cc, z_cv), dim=-1)), dim=-1)
        cc_value = self.cc_projection(z_cc)
        cv_value = self.cv_projection(z_cv)
        fused = (
            gate[:, 0:1] * cc_value
            + gate[:, 1:2] * cv_value
            + gate[:, 2:3] * z_interaction
        )
        with torch.no_grad():
            base_norm = (gate[:, 0:1] * cc_value + gate[:, 1:2] * cv_value).norm(dim=-1).clamp_min(1e-12)
            self.last_gate = gate.detach()
            self.last_interaction_ratio = (
                (gate[:, 2:3] * z_interaction).norm(dim=-1) / base_norm
            ).detach()
        return self.fusion_norm(fused)


__all__ = ["InteractionFusionMambaSOHModel"]
