"""Paper-v2 RawMamba wrappers around the stable Paper-v1 encoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from UnifiedRawSOH.models.c5b_model import build_c5b_model

from .dense_adapter import (
    DenseResidualAdapter,
    choose_parameter_matched_dense_bottleneck,
    dense_adapter_parameter_count,
)
from .residual_moe import (
    ResidualMoEAdapter,
    count_trainable_parameters,
    residual_moe_parameter_count,
)


V2_MODEL_VARIANTS = ("base", "dense_adapter", "residual_moe")


def _zero_balance(z_base: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=z_base.device, dtype=z_base.dtype)


class PaperV2RawMambaModel(nn.Module):
    """Base/Dense/MoE model with a common, explicit forward contract.

    The wrapped ``base_model`` is not modified.  ``encode_base`` delegates to
    its existing ``encode`` method, and all adapters consume only that tensor.
    """

    input_contract = "standard_cccv"

    def __init__(
        self,
        model_config: dict[str, Any],
        *,
        variant: str | None = None,
        backend_override: str | None = None,
    ) -> None:
        super().__init__()
        cfg = dict(model_config)
        resolved_variant = str(variant or cfg.get("variant", "")).strip()
        if resolved_variant not in V2_MODEL_VARIANTS:
            raise ValueError(
                "Paper-v2 model.variant must be one of "
                f"{list(V2_MODEL_VARIANTS)}; got {resolved_variant!r}."
            )
        if bool(cfg.get("use_cycle_prediction", False)):
            raise ValueError("Paper-v2 RawMamba requires model.use_cycle_prediction=false.")
        if bool(cfg.get("use_predicted_cycle_for_soh", False)):
            raise ValueError("Paper-v2 RawMamba requires model.use_predicted_cycle_for_soh=false.")
        if cfg.get("detach_predicted_cycle_for_soh", False):
            raise ValueError("Paper-v2 RawMamba does not accept predicted-cycle SOH inputs.")

        self.variant = resolved_variant
        self.base_config = cfg
        base_config = dict(cfg)
        base_config.pop("variant", None)
        base_config.pop("num_experts", None)
        base_config.pop("top_k", None)
        base_config.pop("expert_bottleneck_dim", None)
        base_config.pop("expert_init", None)
        base_config.pop("router_input", None)
        base_config.pop("adapter_bottleneck_dim", None)
        base_config.pop("adapter_init", None)
        base_config.pop("adapter_dropout", None)
        base_config["use_cycle_prediction"] = False
        base_config["use_predicted_cycle_for_soh"] = False
        base_config["detach_predicted_cycle_for_soh"] = False
        self.base_model = build_c5b_model(base_config, backend_override=backend_override)
        self.z_base_dim = int(self.base_model.fusion_dim)
        self.adapter: nn.Module | None
        if self.variant == "base":
            self.adapter = None
        elif self.variant == "dense_adapter":
            width = cfg.get("adapter_bottleneck_dim")
            if width is None:
                width = choose_parameter_matched_dense_bottleneck(
                    self.z_base_dim,
                    num_experts=int(cfg.get("num_experts", 8)),
                    top_k=int(cfg.get("top_k", 2)),
                    expert_bottleneck_dim=int(cfg.get("expert_bottleneck_dim", 16)),
            )["dense_bottleneck_dim"]
            self.adapter = DenseResidualAdapter(
                self.z_base_dim,
                int(width),
                dropout=float(cfg.get("adapter_dropout", 0.0)),
                adapter_init=str(cfg.get("adapter_init", "zero_output")),
            )
        else:
            if str(cfg.get("router_input", "z_base")) != "z_base":
                raise ValueError("Paper-v2 ResidualMoE router_input must be exactly 'z_base'.")
            self.adapter = ResidualMoEAdapter(
                self.z_base_dim,
                num_experts=int(cfg.get("num_experts", 8)),
                top_k=int(cfg.get("top_k", 2)),
                expert_bottleneck_dim=int(cfg.get("expert_bottleneck_dim", 16)),
                dropout=float(cfg.get("adapter_dropout", 0.0)),
                expert_init=str(cfg.get("expert_init", "zero_output")),
            )

    def encode_base(
        self,
        cc_signal: torch.Tensor,
        cv_signal: torch.Tensor,
        cc_mask: torch.Tensor | None = None,
        cv_mask: torch.Tensor | None = None,
        cc_time: torch.Tensor | None = None,
        cv_time: torch.Tensor | None = None,
        cc_temperature: torch.Tensor | None = None,
        cv_temperature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base_model.encode(
            cc_signal=cc_signal,
            cv_signal=cv_signal,
            cc_mask=cc_mask,
            cv_mask=cv_mask,
            cc_time=cc_time,
            cv_time=cv_time,
            cc_temperature=cc_temperature,
            cv_temperature=cv_temperature,
        )

    def compose(self, z_base: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if z_base.ndim != 2 or z_base.size(-1) != self.z_base_dim:
            raise ValueError(
                f"compose expects z_base with shape [batch, {self.z_base_dim}], got {tuple(z_base.shape)}."
            )
        if self.variant == "base":
            return z_base, {
                "balance_loss": _zero_balance(z_base),
                "router_logits": None,
                "router_probabilities": None,
                "topk_indices": None,
                "topk_weights": None,
                "expert_load": None,
                "expert_importance": None,
                "routing_entropy": None,
                "topk_usage": None,
            }
        if self.variant == "dense_adapter":
            return z_base + self.adapter(z_base), {
                "balance_loss": _zero_balance(z_base),
                "router_logits": None,
                "router_probabilities": None,
                "topk_indices": None,
                "topk_weights": None,
                "expert_load": None,
                "expert_importance": None,
                "routing_entropy": None,
                "topk_usage": None,
            }
        z_out, routing = self.adapter(z_base)
        return z_out, routing

    def predict_from_composed_feature(
        self,
        z_out: torch.Tensor,
        t0_temperature_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base_model.predict_from_signal_feature(z_out, t0_temperature_norm)

    def forward_with_aux(
        self,
        cc_signal: torch.Tensor,
        cv_signal: torch.Tensor,
        cc_mask: torch.Tensor | None = None,
        cv_mask: torch.Tensor | None = None,
        cc_time: torch.Tensor | None = None,
        cv_time: torch.Tensor | None = None,
        cc_temperature: torch.Tensor | None = None,
        cv_temperature: torch.Tensor | None = None,
        t0_temperature_norm: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        z_base = self.encode_base(
            cc_signal=cc_signal,
            cv_signal=cv_signal,
            cc_mask=cc_mask,
            cv_mask=cv_mask,
            cc_time=cc_time,
            cv_time=cv_time,
            cc_temperature=cc_temperature,
            cv_temperature=cv_temperature,
        )
        z_out, routing = self.compose(z_base)
        soh_pred = self.predict_from_composed_feature(z_out, t0_temperature_norm)
        return {
            "soh_pred": soh_pred,
            "z_base": z_base,
            "z_out": z_out,
            "balance_loss": routing.get("balance_loss"),
            "router_logits": routing.get("router_logits"),
            "router_probabilities": routing.get("router_probabilities"),
            "topk_indices": routing.get("topk_indices"),
            "topk_weights": routing.get("topk_weights"),
            "expert_load": routing.get("expert_load"),
            "expert_importance": routing.get("expert_importance"),
            "routing_entropy": routing.get("routing_entropy"),
            "topk_usage": routing.get("topk_usage"),
        }

    def forward(
        self,
        cc_signal: torch.Tensor,
        cv_signal: torch.Tensor,
        cc_mask: torch.Tensor | None = None,
        cv_mask: torch.Tensor | None = None,
        cc_time: torch.Tensor | None = None,
        cv_time: torch.Tensor | None = None,
        cc_temperature: torch.Tensor | None = None,
        cv_temperature: torch.Tensor | None = None,
        t0_temperature_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_aux(
            cc_signal=cc_signal,
            cv_signal=cv_signal,
            cc_mask=cc_mask,
            cv_mask=cv_mask,
            cc_time=cc_time,
            cv_time=cv_time,
            cc_temperature=cc_temperature,
            cv_temperature=cv_temperature,
            t0_temperature_norm=t0_temperature_norm,
        )["soh_pred"]

    def parameter_summary(self) -> dict[str, Any]:
        base_parameters = count_trainable_parameters(self.base_model)
        adapter_parameters = count_trainable_parameters(self.adapter) if self.adapter is not None else 0
        summary: dict[str, Any] = {
            "variant": self.variant,
            "z_base_dim": self.z_base_dim,
            "base_trainable_parameters": base_parameters,
            "adapter_trainable_parameters": adapter_parameters,
            "total_trainable_parameters": count_trainable_parameters(self),
        }
        if self.adapter is not None and hasattr(self.adapter, "parameter_summary"):
            summary["adapter"] = self.adapter.parameter_summary()
        if self.variant == "dense_adapter":
            target = residual_moe_parameter_count(
                self.z_base_dim,
                int(self.base_config.get("num_experts", 8)),
                int(self.base_config.get("expert_bottleneck_dim", 16)),
            )
            dense = dense_adapter_parameter_count(
                self.z_base_dim,
                int(self.adapter.bottleneck_dim),
            )
            summary["parameter_match"] = {
                "target_moe_parameters": int(target),
                "dense_parameters": int(dense),
                "absolute_difference": int(abs(dense - target)),
                "relative_error": float(abs(dense - target) / max(target, 1)),
            }
        return summary


def build_paper_v2_model(model_config: dict[str, Any], backend_override: str | None = None) -> PaperV2RawMambaModel:
    """Build a V2 RawMamba variant without a silent Base fallback."""

    variant = str(model_config.get("variant", "")).strip()
    if variant not in V2_MODEL_VARIANTS:
        raise ValueError(
            "A Paper-v2 raw model requires an explicit model.variant; "
            f"expected {list(V2_MODEL_VARIANTS)}, got {variant!r}."
        )
    return PaperV2RawMambaModel(
        model_config,
        variant=variant,
        backend_override=backend_override,
    )


build_raw_mamba_moe = build_paper_v2_model


__all__ = [
    "PaperV2RawMambaModel",
    "V2_MODEL_VARIANTS",
    "build_paper_v2_model",
    "build_raw_mamba_moe",
]
