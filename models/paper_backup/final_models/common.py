"""Shared, architecture-neutral pieces for the isolated final models."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...c5b_model import StandardMambaPhaseBranch


def make_regression_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    hidden_dim = int(hidden_dim)
    return nn.Sequential(
        nn.Linear(int(input_dim), hidden_dim),
        nn.SiLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
        nn.SiLU(),
        nn.Linear(max(1, hidden_dim // 2), 1),
    )


def init_identity(linear: nn.Linear) -> None:
    if linear.in_features != linear.out_features:
        raise ValueError("Identity initialization requires a square Linear layer")
    nn.init.eye_(linear.weight)
    nn.init.zeros_(linear.bias)


class IndependentPhaseMambaBase(nn.Module):
    """Two independent Mamba encoders with no token/state cross-injection."""

    input_contract = "phase_separated_terminal_raw"
    model_family = "final_phase_mamba"

    def __init__(
        self,
        *,
        input_dim: int = 5,
        signal_input_dim: int = 3,
        d_model: int = 32,
        num_layers: int = 3,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank="auto",
        dropout: float = 0.1,
        pooling: str = "last_mean",
        fusion_dim: int = 64,
        head_hidden_dim: int = 128,
        use_time_as_input: bool = True,
        temperature_injection: str = "input_concat",
        temperature_features: str = "delta",
        use_t0_temperature_meta: bool = True,
        t0_temperature_meta_dim: int = 1,
        time_embedding_time_scale_min: float = 10.0,
        backend: str = "mamba_ssm.Mamba",
    ):
        super().__init__()
        if pooling != "last_mean":
            raise ValueError("Final phase models require last_mean pooling")
        expected_input_dim = int(signal_input_dim) + 1 + (
            1 if temperature_injection == "input_concat" and temperature_features in {"delta", "absolute"}
            else 2 if temperature_injection == "input_concat" and temperature_features == "absolute_delta"
            else 0 if temperature_injection == "none" and temperature_features == "none"
            else -1000
        )
        if int(input_dim) != expected_input_dim:
            raise ValueError(
                f"input_dim={input_dim} does not match signal/time/temperature contract ({expected_input_dim})"
            )
        if use_t0_temperature_meta and int(t0_temperature_meta_dim) != 1:
            raise ValueError("T0 temperature metadata must have one channel")
        if not use_t0_temperature_meta and int(t0_temperature_meta_dim) != 0:
            raise ValueError("Disabled T0 metadata requires t0_temperature_meta_dim=0")
        self.backend = str(backend)
        self.pooling = str(pooling)
        self.phase_feature_dim = 2 * int(d_model)
        self.fusion_dim = int(fusion_dim)
        self.use_t0_temperature_meta = bool(use_t0_temperature_meta)
        self.t0_temperature_meta_dim = int(t0_temperature_meta_dim)
        branch = {
            "input_dim": int(input_dim),
            "signal_input_dim": int(signal_input_dim),
            "d_model": int(d_model),
            "num_layers": int(num_layers),
            "d_state": int(d_state),
            "d_conv": int(d_conv),
            "expand": int(expand),
            "dt_rank": dt_rank,
            "dropout": float(dropout),
            "pooling": str(pooling),
            "use_time_as_input": bool(use_time_as_input),
            "temperature_injection": str(temperature_injection),
            "temperature_features": str(temperature_features),
            "time_scale_min": float(time_embedding_time_scale_min),
            "backend": self.backend,
            "phase_input_fusion": "standard",
        }
        self.cc_branch = StandardMambaPhaseBranch(**branch, phase_kind="cc")
        self.cv_branch = StandardMambaPhaseBranch(**branch, phase_kind="cv")
        self.cc_projection = nn.Linear(self.phase_feature_dim, self.fusion_dim)
        self.cv_projection = nn.Linear(self.phase_feature_dim, self.fusion_dim)
        if self.phase_feature_dim == self.fusion_dim:
            init_identity(self.cc_projection)
            init_identity(self.cv_projection)
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)
        self.head = make_regression_head(
            self.fusion_dim + self.t0_temperature_meta_dim,
            int(head_hidden_dim),
            float(dropout),
        )

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
        z_cc = self.cc_branch(cc_signal, cc_mask, cc_time, cc_temperature)
        z_cv = self.cv_branch(cv_signal, cv_mask, cv_time, cv_temperature)
        return z_cc, z_cv

    def fuse(self, z_cc: torch.Tensor, z_cv: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_signal_feature(self, **inputs) -> torch.Tensor:
        z_cc, z_cv = self.encode_phases(**inputs)
        return self.fuse(z_cc, z_cv)

    def _append_t0(self, feature: torch.Tensor, t0_temperature_norm) -> torch.Tensor:
        if not self.use_t0_temperature_meta:
            return feature
        if t0_temperature_norm is None:
            raise ValueError("Final phase model requires t0_temperature_norm")
        t0 = t0_temperature_norm.to(device=feature.device, dtype=feature.dtype).reshape(feature.size(0), -1)
        if t0.size(1) != self.t0_temperature_meta_dim:
            raise ValueError("Unexpected T0 temperature metadata shape")
        return torch.cat((feature, t0), dim=-1)

    def forward_with_aux(
        self,
        cc_signal,
        cv_signal,
        cc_mask=None,
        cv_mask=None,
        cc_time=None,
        cv_time=None,
        cc_temperature=None,
        cv_temperature=None,
        t0_temperature_norm=None,
    ):
        feature = self.encode_signal_feature(
            cc_signal=cc_signal,
            cv_signal=cv_signal,
            cc_mask=cc_mask,
            cv_mask=cv_mask,
            cc_time=cc_time,
            cv_time=cv_time,
            cc_temperature=cc_temperature,
            cv_temperature=cv_temperature,
        )
        prediction = self.head(self._append_t0(feature, t0_temperature_norm))
        return {
            "soh_pred": prediction,
            "z_health": feature,
            "signal_feature": feature,
            "cycle_life_hat": None,
            "cycle_life_hat_unit": None,
        }

    def forward(self, **inputs):
        return self.forward_with_aux(**inputs)["soh_pred"]
