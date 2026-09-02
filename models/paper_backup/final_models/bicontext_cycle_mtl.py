"""BiContext Mamba with a training-only cycle-order auxiliary task."""

from __future__ import annotations

import torch
import torch.nn as nn

from .bicontext_mamba import BiContextMambaSOHModel


class BiContextCycleMTLSOHModel(BiContextMambaSOHModel):
    """Regularize the shared health representation with cycle-order supervision.

    The auxiliary prediction is produced from ``z_health`` and is never fed to
    the SOH head.  Consequently the cycle branch can shape the shared encoder
    during training without becoming an inference input or an SOH shortcut.
    """

    model_id = "Ours-BiContext-Cycle-MTL-Final5"

    def __init__(self, *, cycle_head_hidden_dim: int = 0, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = int(cycle_head_hidden_dim)
        if hidden_dim < 0:
            raise ValueError("cycle_head_hidden_dim must be non-negative")
        if hidden_dim == 0:
            # A linear auxiliary probe deliberately forces cycle order to be
            # directly decodable from the shared representation and adds only
            # fusion_dim + 1 trainable parameters (65 for the formal config).
            self.cycle_aux_head = nn.Linear(self.fusion_dim, 1)
        else:
            self.cycle_aux_head = nn.Sequential(
                nn.Linear(self.fusion_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward_with_aux(self, *args, **kwargs):
        output = super().forward_with_aux(*args, **kwargs)
        output["cycle_aux_pred"] = self.cycle_aux_head(output["z_health"])
        output["cycle_aux_is_soh_input"] = False
        return output

    def forward(self, **inputs):
        # The ordinary inference entry point bypasses the training-only head.
        return BiContextMambaSOHModel.forward_with_aux(self, **inputs)["soh_pred"]


__all__ = ["BiContextCycleMTLSOHModel"]
