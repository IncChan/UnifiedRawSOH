"""Late latent-token bidirectional context exchange for the final E1 model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...c5b_model import pool_hidden_states
from .raw_dual_vanilla_mamba import RawDualVanillaMambaSOHModel


class LateLatentTokenBiContextSOHModel(RawDualVanillaMambaSOHModel):
    """Function-preserving late cross-phase extension of Raw Dual Vanilla.

    Each phase executes two independent Mamba blocks, is summarized into four
    learned latent tokens, and lets its original tokens read the other phase's
    latents once.  Both supported residual modes are exact no-ops at
    initialization, so the whole model initially computes the same function
    as :class:`RawDualVanillaMambaSOHModel`.  The formal follow-up uses
    pre-normalized cross inputs and bounded bidirectional ReZero scalars.
    """

    model_id = "Ours-Late-LatentToken-BiContext"

    def __init__(
        self,
        *,
        num_latents: int = 4,
        cross_num_heads: int = 4,
        cross_after_layer: int = 2,
        cross_residual_mode: str = "zero_init_out_proj",
        cross_max_scale: float = 0.1,
        **kwargs,
    ):
        d_model = int(kwargs.get("d_model", 32))
        num_layers = int(kwargs.get("num_layers", 3))
        num_latents = int(num_latents)
        cross_num_heads = int(cross_num_heads)
        cross_after_layer = int(cross_after_layer)
        cross_residual_mode = str(cross_residual_mode)
        cross_max_scale = float(cross_max_scale)

        if d_model != 32:
            raise ValueError("Late Latent-Token BiContext requires d_model=32")
        if num_layers != 3:
            raise ValueError("Late Latent-Token BiContext requires num_layers=3")
        if num_latents != 4:
            raise ValueError("Late Latent-Token BiContext fixes num_latents=4")
        if cross_num_heads != 4:
            raise ValueError("Late Latent-Token BiContext fixes cross_num_heads=4")
        if cross_after_layer != 2:
            raise ValueError("Late Latent-Token BiContext fixes cross_after_layer=2")
        if d_model % cross_num_heads:
            raise ValueError("d_model must be divisible by cross_num_heads")
        if cross_residual_mode not in {
            "zero_init_out_proj",
            "prenorm_bounded_rezero",
        }:
            raise ValueError(
                "cross_residual_mode must be zero_init_out_proj or "
                "prenorm_bounded_rezero"
            )
        if not 0.0 < cross_max_scale <= 1.0:
            raise ValueError("cross_max_scale must be in (0, 1]")

        # Construct the complete Raw Dual Vanilla model first.  With an equal
        # random seed, every inherited parameter therefore receives exactly
        # the same initialization as the baseline before new modules are made.
        super().__init__(**kwargs)
        self.num_latents = num_latents
        self.cross_num_heads = cross_num_heads
        self.cross_after_layer = cross_after_layer
        self.cross_residual_mode = cross_residual_mode
        self.cross_max_scale = cross_max_scale

        # CC and CV intentionally learn independent token-to-latent poolers.
        self.cc_latent_score = nn.Linear(d_model, num_latents)
        self.cv_latent_score = nn.Linear(d_model, num_latents)
        self.cc_read_cv = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=cross_num_heads,
            batch_first=True,
        )
        self.cv_read_cc = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=cross_num_heads,
            batch_first=True,
        )

        if self.cross_residual_mode == "prenorm_bounded_rezero":
            # The MHA projections retain their ordinary initialization.  The
            # two zero ReZero scalars alone make both residuals exact no-ops.
            # This avoids the unstable, unconstrained injection seen in the
            # original late-token experiment while preserving the baseline
            # function exactly at initialization.
            self.cc_cross_norm = nn.LayerNorm(d_model)
            self.cv_cross_norm = nn.LayerNorm(d_model)
            self.cc_cross_alpha = nn.Parameter(torch.zeros(()))
            self.cv_cross_alpha = nn.Parameter(torch.zeros(()))
        else:
            # Backward-compatible construction for archived checkpoints and
            # the completed first late-token experiment.
            self.cc_cross_norm = None
            self.cv_cross_norm = None
            self.register_parameter("cc_cross_alpha", None)
            self.register_parameter("cv_cross_alpha", None)
            for attention in (self.cc_read_cv, self.cv_read_cc):
                nn.init.zeros_(attention.out_proj.weight)
                if attention.out_proj.bias is not None:
                    nn.init.zeros_(attention.out_proj.bias)

        self.last_latent_cc_attention = None
        self.last_latent_cv_attention = None
        self.last_cc_cross_contribution_ratio = None
        self.last_cv_cross_contribution_ratio = None
        self.last_cc_cross_delta = None
        self.last_cv_cross_delta = None
        self.last_cc_cross_scale = None
        self.last_cv_cross_scale = None

    @staticmethod
    def _mask_tokens(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return value
        return value * mask.to(device=value.device, dtype=value.dtype).unsqueeze(-1)

    @staticmethod
    def _contribution_ratio(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        numerator = delta.flatten(start_dim=1).norm(dim=-1)
        denominator = base.flatten(start_dim=1).norm(dim=-1).clamp_min(1e-12)
        return numerator / denominator

    @staticmethod
    def _latent_pool(
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        scorer: nn.Linear,
    ) -> torch.Tensor:
        """Pool ``[B, T, D]`` tokens into ``[B, K, D]`` learned latents."""

        scores = scorer(hidden)  # [B, T, K]
        if mask is not None:
            valid = mask.to(device=hidden.device, dtype=torch.bool)
            if valid.shape != hidden.shape[:2]:
                raise ValueError(
                    f"Latent-pooling mask shape {tuple(valid.shape)} does not match "
                    f"tokens {tuple(hidden.shape[:2])}"
                )
            scores = scores.masked_fill(~valid.unsqueeze(-1), -torch.inf)
        weights = torch.softmax(scores, dim=1)
        # Keep the operation finite for a defensive all-padding row.  Formal
        # data contain valid phase tokens, so this does not affect E1 samples.
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.bmm(weights.transpose(1, 2), hidden)

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

        # Fixed placement: Mamba 1 -> Mamba 2 -> cross -> Mamba 3.
        for layer_index in range(self.cross_after_layer):
            cc_hidden = self.cc_branch.layers[layer_index](cc_hidden)
            cv_hidden = self.cv_branch.layers[layer_index](cv_hidden)

        if self.cross_residual_mode == "prenorm_bounded_rezero":
            cc_cross_input = self.cc_cross_norm(cc_hidden)
            cv_cross_input = self.cv_cross_norm(cv_hidden)
        else:
            cc_cross_input = cc_hidden
            cv_cross_input = cv_hidden

        cc_latent = self._latent_pool(
            cc_cross_input, cc_mask, self.cc_latent_score
        )
        cv_latent = self._latent_pool(
            cv_cross_input, cv_mask, self.cv_latent_score
        )
        cc_delta, cc_attention = self.cc_read_cv(
            query=cc_cross_input,
            key=cv_latent,
            value=cv_latent,
            need_weights=True,
            average_attn_weights=True,
        )
        cv_delta, cv_attention = self.cv_read_cc(
            query=cv_cross_input,
            key=cc_latent,
            value=cc_latent,
            need_weights=True,
            average_attn_weights=True,
        )
        if self.cross_residual_mode == "prenorm_bounded_rezero":
            cc_scale = self.cross_max_scale * torch.tanh(self.cc_cross_alpha)
            cv_scale = self.cross_max_scale * torch.tanh(self.cv_cross_alpha)
            cc_delta = cc_scale * cc_delta
            cv_delta = cv_scale * cv_delta
        else:
            cc_scale = cc_hidden.new_tensor(1.0)
            cv_scale = cv_hidden.new_tensor(1.0)
        cc_delta = self._mask_tokens(cc_delta, cc_mask)
        cv_delta = self._mask_tokens(cv_delta, cv_mask)
        masked_cc_hidden = self._mask_tokens(cc_hidden, cc_mask)
        masked_cv_hidden = self._mask_tokens(cv_hidden, cv_mask)

        if cc_mask is not None:
            cc_attention = self._mask_tokens(cc_attention, cc_mask)
        if cv_mask is not None:
            cv_attention = self._mask_tokens(cv_attention, cv_mask)
        with torch.no_grad():
            # ``latent_cc_attention`` means CC tokens reading CV latents; the
            # corresponding CV key follows the same query-phase convention.
            self.last_latent_cc_attention = cc_attention.detach()
            self.last_latent_cv_attention = cv_attention.detach()
            self.last_cc_cross_contribution_ratio = self._contribution_ratio(
                cc_delta, masked_cc_hidden
            ).detach()
            self.last_cv_cross_contribution_ratio = self._contribution_ratio(
                cv_delta, masked_cv_hidden
            ).detach()
            self.last_cc_cross_delta = cc_delta.detach()
            self.last_cv_cross_delta = cv_delta.detach()
            self.last_cc_cross_scale = cc_scale.detach()
            self.last_cv_cross_scale = cv_scale.detach()

        cc_hidden = cc_hidden + cc_delta
        cv_hidden = cv_hidden + cv_delta
        cc_hidden = self.cc_branch.layers[2](cc_hidden)
        cv_hidden = self.cv_branch.layers[2](cv_hidden)
        cc_hidden = self.cc_branch.final_norm(cc_hidden)
        cv_hidden = self.cv_branch.final_norm(cv_hidden)
        return (
            pool_hidden_states(cc_hidden, cc_mask, self.pooling),
            pool_hidden_states(cv_hidden, cv_mask, self.pooling),
        )

    def forward_with_aux(self, *args, **kwargs):
        output = super().forward_with_aux(*args, **kwargs)
        output.update(
            {
                "latent_cc_attention": self.last_latent_cc_attention,
                "latent_cv_attention": self.last_latent_cv_attention,
                "cc_cross_contribution_ratio": self.last_cc_cross_contribution_ratio,
                "cv_cross_contribution_ratio": self.last_cv_cross_contribution_ratio,
                "cc_cross_scale": self.last_cc_cross_scale,
                "cv_cross_scale": self.last_cv_cross_scale,
            }
        )
        return output


__all__ = ["LateLatentTokenBiContextSOHModel"]
