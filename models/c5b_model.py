"""Clean C5B extraction used by the paper-oriented codebase.

The default path is the stable C5B model:

* independent CC and CV Mamba branches;
* fixed-size last/mean pooling;
* zero-initialized CC-to-CV bridge;
* post-fusion T0 metadata;
* shared SOH/cycle multitask head with no-detach predicted-cycle injection.

E1 temperature controls use a small, explicit input-mode surface rather than
a second model implementation; they do not alter the default state-dict
contract. The module intentionally has no C1-C9 feature-flag surface. A
``torch_reference`` backend exists solely for CPU smoke tests when the
official CUDA Mamba backend cannot execute; paper runs must use the default
``mamba_ssm.Mamba`` backend.
"""

from __future__ import annotations

from importlib import metadata

import torch
import torch.nn as nn


def _patch_transformers_generation_outputs():
    """Keep mamba-ssm importable across supported transformers versions."""

    try:
        import transformers.generation as generation
    except Exception:
        return
    output_cls = getattr(generation, "GenerateDecoderOnlyOutput", None)
    if output_cls is None:
        return
    for name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
        if not hasattr(generation, name):
            setattr(generation, name, output_cls)


_patch_transformers_generation_outputs()

try:
    from mamba_ssm.modules.mamba_simple import Mamba as OfficialMamba
except (ImportError, OSError) as exc:  # pragma: no cover - depends on environment
    OfficialMamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


def get_mamba_backend_info():
    info = {
        "package": "mamba-ssm",
        "version": None,
        "mamba_imported": OfficialMamba is not None,
        "selective_scan_cuda_imported": False,
        "causal_conv1d_imported": False,
        "cuda_available": bool(torch.cuda.is_available()),
        "import_error": None if _MAMBA_IMPORT_ERROR is None else repr(_MAMBA_IMPORT_ERROR),
    }
    try:
        info["version"] = metadata.version("mamba-ssm")
    except metadata.PackageNotFoundError:
        pass
    try:
        import selective_scan_cuda  # noqa: F401
    except (ImportError, OSError):
        pass
    else:
        info["selective_scan_cuda_imported"] = True
    try:
        import causal_conv1d  # noqa: F401
    except (ImportError, OSError):
        pass
    else:
        info["causal_conv1d_imported"] = True
    return info


def require_official_mamba():
    info = get_mamba_backend_info()
    if not info["mamba_imported"] or not info["selective_scan_cuda_imported"]:
        raise RuntimeError(
            "Paper-v1 raw SOH requires the official mamba-ssm package and compiled "
            f"selective_scan_cuda extension. Backend diagnostics: {info}"
        )
    return info


class TorchReferenceMamba(nn.Module):
    """Shape-compatible CPU reference used only by smoke tests."""

    def __init__(self, d_model):
        super().__init__()
        self.input = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, hidden):
        return self.output(torch.tanh(self.input(hidden)))


def _make_mixer(d_model, d_state, d_conv, expand, dt_rank, layer_idx, backend):
    if backend == "torch_reference":
        return TorchReferenceMamba(d_model)
    if backend != "mamba_ssm.Mamba":
        raise ValueError("backend must be 'mamba_ssm.Mamba' or 'torch_reference'.")
    require_official_mamba()
    return OfficialMamba(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dt_rank=dt_rank,
        use_fast_path=True,
        layer_idx=layer_idx,
    )


def get_pooling_multiplier(pooling):
    if pooling == "mean":
        return 1
    if pooling == "last_mean":
        return 2
    raise ValueError("C5B supports pooling='last_mean' only in the Paper-v1 path.")


def pool_hidden_states(hidden, mask=None, pooling="last_mean"):
    if mask is None:
        last_hidden = hidden[:, -1, :]
        mean_hidden = hidden.mean(dim=1)
    else:
        mask = mask.to(device=hidden.device, dtype=hidden.dtype)
        lengths = mask.sum(dim=1).clamp_min(1.0)
        last_indices = (lengths.long() - 1).clamp_min(0)
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_indices, last_indices, :]
        mean_hidden = (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
    if pooling == "last_mean":
        return torch.cat([last_hidden, mean_hidden], dim=-1)
    if pooling == "mean":
        return mean_hidden
    raise ValueError(f"Unsupported pooling mode: {pooling}")


class PreNormMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dt_rank, dropout, layer_idx, backend):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = _make_mixer(d_model, d_state, d_conv, expand, dt_rank, layer_idx, backend)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        return hidden_states + self.dropout(self.mixer(self.norm(hidden_states)))


class StandardMambaPhaseInputEncoder(nn.Module):
    """Phase token construction used by the C5B main model and E1 controls.

    The default is deliberately unchanged from C5B: ``signal + time/10 min
    + DeltaT``.  The small set of alternate temperature modes only exists so
    that E1 can remove or isolate temperature information without maintaining
    a second raw-model implementation.  They change the input projection
    shape, so they are ablation-only checkpoints and cannot be loaded into the
    default Paper-v1 model (or vice versa).
    """

    def __init__(self, input_dim, signal_input_dim, d_model, use_time_as_input, temperature_injection, temperature_features, time_scale_min):
        super().__init__()
        self.input_dim = int(input_dim)
        self.signal_input_dim = int(signal_input_dim)
        self.use_time_as_input = bool(use_time_as_input)
        self.temperature_injection = str(temperature_injection)
        self.temperature_features = str(temperature_features)
        self.time_scale_min = float(time_scale_min)
        if not self.use_time_as_input:
            raise ValueError("The current Paper-v1 raw model requires real relative time as a token input.")
        valid_temperature_modes = {
            ("none", "none"): 0,
            ("input_concat", "delta"): 1,
            ("input_concat", "absolute"): 1,
            ("input_concat", "absolute_delta"): 2,
        }
        key = (self.temperature_injection, self.temperature_features)
        if key not in valid_temperature_modes:
            raise ValueError(
                "temperature mode must be one of "
                "('none', 'none'), ('input_concat', 'delta'), "
                "('input_concat', 'absolute'), or ('input_concat', 'absolute_delta')."
            )
        self.temperature_channels = valid_temperature_modes[key]
        expected_input_dim = self.signal_input_dim + 1 + self.temperature_channels
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"Phase token expects input_dim={expected_input_dim} for signal_input_dim="
                f"{self.signal_input_dim} and {key}; got {self.input_dim}."
            )
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

    def forward(self, signal, mask=None, time=None, temperature=None):
        if time is None:
            raise ValueError("C5B requires a relative-time tensor.")
        time_feature = (time.to(device=signal.device, dtype=signal.dtype) / self.time_scale_min).unsqueeze(-1)
        pieces = [signal, time_feature]
        if self.temperature_channels:
            if temperature is None or temperature.size(-1) != 2:
                raise ValueError("Temperature input must contain [T_abs_norm, DeltaT_norm].")
            temperature = temperature.to(device=signal.device, dtype=signal.dtype)
            if self.temperature_features == "delta":
                pieces.append(temperature[..., 1:2])
            elif self.temperature_features == "absolute":
                pieces.append(temperature[..., 0:1])
            else:
                pieces.append(temperature)
        combined = torch.cat(pieces, dim=-1)
        if combined.size(-1) != self.input_dim:
            raise ValueError(f"C5B token has {combined.size(-1)} channels; expected {self.input_dim}.")
        return self.input_norm(self.input_proj(combined))


class GatedResidualMambaPhaseInputEncoder(nn.Module):
    """Dominant phase input with a zero-initialized, sample-gated residual.

    The rich schema-v2 signal order is ``[voltage, C-rate, phase_tau]`` for
    both phases.  CC treats voltage as dominant and C-rate as secondary; CV
    treats C-rate as dominant and voltage as secondary.  Absolute time and
    temperature remain in the dominant path, so the gate is conditioned only
    on physical sample content and never on dataset/strategy/cell metadata.

    The secondary projection is initialized to exactly zero.  Consequently,
    the initial forward function is the Dominant model even though the gate
    starts at 0.5 and the secondary projection receives a non-zero gradient
    on the first optimization step.
    """

    def __init__(
        self,
        input_dim,
        signal_input_dim,
        d_model,
        use_time_as_input,
        temperature_injection,
        temperature_features,
        time_scale_min,
        phase_kind,
        gate_hidden_dim,
        gate_context,
        secondary_residual_init,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.signal_input_dim = int(signal_input_dim)
        self.use_time_as_input = bool(use_time_as_input)
        self.temperature_injection = str(temperature_injection)
        self.temperature_features = str(temperature_features)
        self.time_scale_min = float(time_scale_min)
        self.phase_kind = str(phase_kind).lower()
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.gate_context = str(gate_context)
        self.secondary_residual_init = str(secondary_residual_init)
        if not self.use_time_as_input:
            raise ValueError("Gated FullVI requires real relative time as a token input.")
        if self.phase_kind not in {"cc", "cv"}:
            raise ValueError("Gated FullVI phase_kind must be 'cc' or 'cv'.")
        if self.signal_input_dim != 3:
            raise ValueError("Gated FullVI requires [voltage, C-rate, phase_tau].")
        if self.gate_hidden_dim < 1:
            raise ValueError("gate_hidden_dim must be positive.")
        if self.gate_context != "masked_mean":
            raise ValueError("Gated FullVI currently requires gate_context='masked_mean'.")
        if self.secondary_residual_init != "zero":
            raise ValueError("Gated FullVI requires secondary_residual_init='zero'.")
        valid_temperature_modes = {
            ("none", "none"): 0,
            ("input_concat", "delta"): 1,
            ("input_concat", "absolute"): 1,
            ("input_concat", "absolute_delta"): 2,
        }
        key = (self.temperature_injection, self.temperature_features)
        if key not in valid_temperature_modes:
            raise ValueError(
                "temperature mode must be one of ('none', 'none'), "
                "('input_concat', 'delta'), ('input_concat', 'absolute'), "
                "or ('input_concat', 'absolute_delta')."
            )
        self.temperature_channels = valid_temperature_modes[key]
        expected_input_dim = self.signal_input_dim + 1 + self.temperature_channels
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"Gated FullVI expects input_dim={expected_input_dim}; got {self.input_dim}."
            )
        self.primary_input_dim = 2 + 1 + self.temperature_channels
        self.input_proj = nn.Linear(self.primary_input_dim, int(d_model))
        self.input_norm = nn.LayerNorm(int(d_model))

        # Do not advance the global RNG used by the following Mamba blocks.
        # This preserves paired-seed comparability with the Dominant control.
        with torch.random.fork_rng(devices=[]):
            self.secondary_proj = nn.Linear(1, int(d_model), bias=False)
            self.gate_net = nn.Sequential(
                nn.Linear(self.primary_input_dim, self.gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.gate_hidden_dim, 1),
            )
        nn.init.zeros_(self.secondary_proj.weight)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)
        self.last_gate = None

    def _primary_secondary(self, signal):
        if signal.ndim != 3 or signal.size(-1) != self.signal_input_dim:
            raise ValueError(
                "Gated FullVI signal must have shape [batch, length, 3] "
                f"for [voltage, C-rate, phase_tau], got {tuple(signal.shape)}."
            )
        if self.phase_kind == "cc":
            return signal[..., [0, 2]], signal[..., 1:2]
        return signal[..., [1, 2]], signal[..., 0:1]

    def _masked_mean(self, values, mask):
        if mask is None:
            return values.mean(dim=1)
        mask = mask.to(device=values.device, dtype=values.dtype)
        if mask.shape != values.shape[:2]:
            raise ValueError(
                f"phase mask must have shape {tuple(values.shape[:2])}, got {tuple(mask.shape)}"
            )
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (values * mask.unsqueeze(-1)).sum(dim=1) / denominator

    def forward(self, signal, mask=None, time=None, temperature=None):
        if time is None:
            raise ValueError("Gated FullVI requires a relative-time tensor.")
        primary_signal, secondary_signal = self._primary_secondary(signal)
        time_feature = (
            time.to(device=signal.device, dtype=signal.dtype) / self.time_scale_min
        ).unsqueeze(-1)
        pieces = [primary_signal, time_feature]
        if self.temperature_channels:
            if temperature is None or temperature.size(-1) != 2:
                raise ValueError("Temperature input must contain [T_abs_norm, DeltaT_norm].")
            temperature = temperature.to(device=signal.device, dtype=signal.dtype)
            if self.temperature_features == "delta":
                pieces.append(temperature[..., 1:2])
            elif self.temperature_features == "absolute":
                pieces.append(temperature[..., 0:1])
            else:
                pieces.append(temperature)
        primary = torch.cat(pieces, dim=-1)
        if primary.size(-1) != self.primary_input_dim:
            raise ValueError(
                f"Gated FullVI primary token has {primary.size(-1)} channels; "
                f"expected {self.primary_input_dim}."
            )
        context = self._masked_mean(primary, mask)
        gate = torch.sigmoid(self.gate_net(context)).unsqueeze(1)
        self.last_gate = gate.detach()
        hidden = self.input_proj(primary) + gate * self.secondary_proj(secondary_signal)
        return self.input_norm(hidden)


class StandardMambaPhaseBranch(nn.Module):
    def __init__(
        self,
        input_dim,
        signal_input_dim,
        d_model,
        num_layers,
        d_state,
        d_conv,
        expand,
        dt_rank,
        dropout,
        pooling,
        use_time_as_input,
        temperature_injection,
        temperature_features,
        time_scale_min,
        backend,
        phase_input_fusion="standard",
        phase_kind=None,
        gate_hidden_dim=4,
        gate_context="masked_mean",
        secondary_residual_init="zero",
    ):
        super().__init__()
        self.pooling = pooling
        self.phase_input_fusion = str(phase_input_fusion)
        if self.phase_input_fusion == "standard":
            self.input_encoder = StandardMambaPhaseInputEncoder(
                input_dim=input_dim,
                signal_input_dim=signal_input_dim,
                d_model=d_model,
                use_time_as_input=use_time_as_input,
                temperature_injection=temperature_injection,
                temperature_features=temperature_features,
                time_scale_min=time_scale_min,
            )
        elif self.phase_input_fusion == "gated_residual_full_vi":
            self.input_encoder = GatedResidualMambaPhaseInputEncoder(
                input_dim=input_dim,
                signal_input_dim=signal_input_dim,
                d_model=d_model,
                use_time_as_input=use_time_as_input,
                temperature_injection=temperature_injection,
                temperature_features=temperature_features,
                time_scale_min=time_scale_min,
                phase_kind=phase_kind,
                gate_hidden_dim=gate_hidden_dim,
                gate_context=gate_context,
                secondary_residual_init=secondary_residual_init,
            )
        else:
            raise ValueError(
                "phase_input_fusion must be 'standard' or 'gated_residual_full_vi'."
            )
        self.layers = nn.ModuleList(
            [
                PreNormMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    dropout=dropout,
                    layer_idx=layer_idx,
                    backend=backend,
                )
                for layer_idx in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, signal, mask=None, time=None, temperature=None):
        hidden = self.input_encoder(signal, mask, time, temperature)
        for layer in self.layers:
            hidden = layer(hidden)
        return pool_hidden_states(self.final_norm(hidden), mask, self.pooling)


def _make_mlp_head(input_dim, hidden_dim, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.SiLU(),
        nn.Linear(hidden_dim // 2, 1),
    )


class PaperRawSOHModel(nn.Module):
    """Paper-v1 phase-specific raw CC/CV/T SOH model.

    The validated C5B parameterisation and state-dict layout are preserved,
    while the public model name is detached from the historical experiment
    label.  ``encode`` is the stable health-representation entry point.
    """

    input_contract = "standard_cccv"

    def __init__(
        self,
        input_dim=4,
        signal_input_dim=2,
        d_model=32,
        num_layers=3,
        d_state=8,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dropout=0.1,
        pooling="last_mean",
        fusion_type="concat",
        fusion_phase_dim=64,
        head_hidden_dim=128,
        use_time_as_input=True,
        temperature_injection="input_concat",
        temperature_features="delta",
        use_t0_temperature_meta=True,
        t0_temperature_meta_dim=1,
        use_cc_to_cv_bridge=True,
        cc_to_cv_bridge_type="zero_init_linear",
        cc_to_cv_bridge_input_dim=64,
        cc_to_cv_bridge_output_dim=32,
        use_cycle_prediction=True,
        cycle_target="cycle_life_norm",
        cycle_head_hidden_dim=64,
        cycle_output_activation="sigmoid",
        use_predicted_cycle_for_soh=True,
        detach_predicted_cycle_for_soh=False,
        time_embedding_time_scale_min=10.0,
        phase_input_fusion="standard",
        gate_hidden_dim=4,
        gate_context="masked_mean",
        secondary_residual_init="zero",
        backend="mamba_ssm.Mamba",
    ):
        super().__init__()
        temperature_token_channels = {
            ("none", "none"): 0,
            ("input_concat", "delta"): 1,
            ("input_concat", "absolute"): 1,
            ("input_concat", "absolute_delta"): 2,
        }.get((str(temperature_injection), str(temperature_features)))
        if temperature_token_channels is None:
            raise ValueError("Unsupported temperature token configuration.")
        expected_input_dim = int(signal_input_dim) + 1 + temperature_token_channels
        if int(input_dim) != expected_input_dim:
            raise ValueError(
                f"input_dim must be {expected_input_dim}: signal({int(signal_input_dim)})+time(1)+configured temperature channels "
                f"({temperature_token_channels})."
            )
        if pooling != "last_mean" or fusion_type != "concat":
            raise ValueError("Paper-v1 raw SOH requires last_mean pooling and concat fusion.")
        if int(fusion_phase_dim) != 64 or int(head_hidden_dim) != 128:
            raise ValueError("Paper-v1 raw SOH requires phase_dim=64 and head_hidden_dim=128.")
        if use_t0_temperature_meta and int(t0_temperature_meta_dim) != 1:
            raise ValueError("T0 metadata must have one post-fusion channel when enabled.")
        if not use_t0_temperature_meta and int(t0_temperature_meta_dim) != 0:
            raise ValueError("t0_temperature_meta_dim must be zero when T0 metadata is disabled.")
        allowed_bridge_types = {
            "zero_init_linear",
            "adaptive_pointwise_zero_init",
        }
        if use_cc_to_cv_bridge and (
            cc_to_cv_bridge_type not in allowed_bridge_types
            or int(cc_to_cv_bridge_input_dim) != 64
            or int(cc_to_cv_bridge_output_dim) != int(d_model)
        ):
            raise ValueError(
                "C5B bridge must be zero-init Linear(64, 32), optionally with "
                "the adaptive pointwise gate."
            )
        if use_cycle_prediction and (
            cycle_target != "cycle_life_norm"
            or int(cycle_head_hidden_dim) != 64
            or cycle_output_activation != "sigmoid"
        ):
            raise ValueError("C5B cycle supervision requires cycle_life_norm and a 128->64->1 sigmoid head.")
        if use_predicted_cycle_for_soh and not use_cycle_prediction:
            raise ValueError("Predicted-cycle injection requires cycle prediction to be enabled.")

        self.backend = backend
        self.use_cc_to_cv_bridge = bool(use_cc_to_cv_bridge)
        self.cc_to_cv_bridge_type = str(cc_to_cv_bridge_type)
        self.use_t0_temperature_meta = bool(use_t0_temperature_meta)
        self.t0_temperature_meta_dim = int(t0_temperature_meta_dim)
        self.use_cycle_prediction = bool(use_cycle_prediction)
        self.use_predicted_cycle_for_soh = bool(use_predicted_cycle_for_soh)
        self.detach_predicted_cycle_for_soh = bool(detach_predicted_cycle_for_soh)
        self.cycle_target = str(cycle_target)
        self.pooling = pooling
        self.fusion_type = fusion_type
        self.fusion_dim = 2 * int(fusion_phase_dim)
        self.phase_input_fusion = str(phase_input_fusion)
        if self.phase_input_fusion == "gated_residual_full_vi" and (
            int(signal_input_dim) != 3 or int(input_dim) != expected_input_dim
        ):
            raise ValueError("Gated FullVI requires the rich three-signal phase input.")

        branch_kwargs = {
            "input_dim": int(input_dim),
            "signal_input_dim": int(signal_input_dim),
            "d_model": int(d_model),
            "num_layers": int(num_layers),
            "d_state": int(d_state),
            "d_conv": int(d_conv),
            "expand": int(expand),
            "dt_rank": dt_rank,
            "dropout": float(dropout),
            "pooling": pooling,
            "use_time_as_input": bool(use_time_as_input),
            "temperature_injection": temperature_injection,
            "temperature_features": temperature_features,
            "time_scale_min": float(time_embedding_time_scale_min),
            "backend": backend,
            "phase_input_fusion": self.phase_input_fusion,
            "gate_hidden_dim": int(gate_hidden_dim),
            "gate_context": str(gate_context),
            "secondary_residual_init": str(secondary_residual_init),
        }
        self.cc_branch = StandardMambaPhaseBranch(**branch_kwargs, phase_kind="cc")
        self.cv_branch = StandardMambaPhaseBranch(**branch_kwargs, phase_kind="cv")
        self.cc_fusion_proj = nn.Identity()
        self.cv_fusion_proj = nn.Identity()
        if self.use_cc_to_cv_bridge:
            self.cc_to_cv_bridge = nn.Linear(int(cc_to_cv_bridge_input_dim), int(cc_to_cv_bridge_output_dim))
            nn.init.zeros_(self.cc_to_cv_bridge.weight)
            nn.init.zeros_(self.cc_to_cv_bridge.bias)
            self.cc_to_cv_point_gate = None
            if self.cc_to_cv_bridge_type == "adaptive_pointwise_zero_init":
                # One gate per CV time point, conditioned on both the local CV
                # token and the pooled CC representation.  Construct it without
                # advancing the RNG used by the prediction head, then zero it so
                # 2*sigmoid(0)=1.  The new model therefore has the exact FullVI
                # initial function and the same first-step bridge gradient.
                point_gate_input_dim = int(cc_to_cv_bridge_input_dim) + int(d_model)
                with torch.random.fork_rng(devices=[]):
                    self.cc_to_cv_point_gate = nn.Linear(point_gate_input_dim, 1)
                nn.init.zeros_(self.cc_to_cv_point_gate.weight)
                nn.init.zeros_(self.cc_to_cv_point_gate.bias)
        else:
            self.cc_to_cv_bridge = None
            self.cc_to_cv_point_gate = None
        self.last_cc_to_cv_point_gate = None

        head_input_dim = self.fusion_dim + self.t0_temperature_meta_dim
        self.head = _make_mlp_head(head_input_dim, int(head_hidden_dim), float(dropout))
        self.cycle_head = None
        self.cycle_adapter = None
        if self.use_cycle_prediction:
            self.cycle_head = nn.Sequential(
                nn.Linear(self.fusion_dim, int(cycle_head_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(cycle_head_hidden_dim), 1),
                nn.Sigmoid(),
            )
            if self.use_predicted_cycle_for_soh:
                self.cycle_adapter = nn.Linear(1, int(head_hidden_dim), bias=False)
                nn.init.zeros_(self.cycle_adapter.weight)

    def encode_signal_feature(
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
        if self.use_cc_to_cv_bridge:
            cc_sequence = self.cc_branch.input_encoder(cc_signal, cc_mask, cc_time, cc_temperature)
            for layer in self.cc_branch.layers:
                cc_sequence = layer(cc_sequence)
            cc_sequence = self.cc_branch.final_norm(cc_sequence)
            cc_feature = pool_hidden_states(cc_sequence, cc_mask, self.pooling)

            cv_sequence = self.cv_branch.input_encoder(cv_signal, cv_mask, cv_time, cv_temperature)
            cc_context = self.cc_to_cv_bridge(cc_feature)
            if self.cc_to_cv_point_gate is None:
                self.last_cc_to_cv_point_gate = None
                cv_sequence = cv_sequence + cc_context.unsqueeze(1)
            else:
                expanded_cc = cc_feature.unsqueeze(1).expand(-1, cv_sequence.size(1), -1)
                gate_input = torch.cat([cv_sequence, expanded_cc], dim=-1)
                point_gate = 2.0 * torch.sigmoid(self.cc_to_cv_point_gate(gate_input))
                if cv_mask is not None:
                    point_gate = point_gate * cv_mask.to(
                        device=point_gate.device, dtype=point_gate.dtype
                    ).unsqueeze(-1)
                self.last_cc_to_cv_point_gate = point_gate.detach()
                cv_sequence = cv_sequence + point_gate * cc_context.unsqueeze(1)
            for layer in self.cv_branch.layers:
                cv_sequence = layer(cv_sequence)
            cv_sequence = self.cv_branch.final_norm(cv_sequence)
            cv_feature = pool_hidden_states(cv_sequence, cv_mask, self.pooling)
        else:
            cc_feature = self.cc_branch(cc_signal, cc_mask, cc_time, cc_temperature)
            cv_feature = self.cv_branch(cv_signal, cv_mask, cv_time, cv_temperature)
        cc_feature = self.cc_fusion_proj(cc_feature)
        cv_feature = self.cv_fusion_proj(cv_feature)
        signal_feature = torch.cat([cc_feature, cv_feature], dim=-1)
        if signal_feature.size(-1) != self.fusion_dim:
            raise RuntimeError(f"C5B signal feature has {signal_feature.size(-1)} dimensions, expected {self.fusion_dim}.")
        return signal_feature

    def encode(
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
        """Encode the current cycle into the stable ``z_health`` tensor."""

        return self.encode_signal_feature(
            cc_signal=cc_signal,
            cv_signal=cv_signal,
            cc_mask=cc_mask,
            cv_mask=cv_mask,
            cc_time=cc_time,
            cv_time=cv_time,
            cc_temperature=cc_temperature,
            cv_temperature=cv_temperature,
        )

    def _append_t0(self, signal_feature, t0_temperature_norm):
        if not self.use_t0_temperature_meta:
            return signal_feature
        if t0_temperature_norm is None:
            raise ValueError("This model configuration requires t0_temperature_norm.")
        t0 = t0_temperature_norm.to(device=signal_feature.device, dtype=signal_feature.dtype)
        t0 = t0.view(t0.size(0), -1)
        if t0.size(1) != self.t0_temperature_meta_dim:
            raise ValueError(f"T0 metadata must have shape [batch, {self.t0_temperature_meta_dim}].")
        return torch.cat([signal_feature, t0], dim=-1)

    def predict_cycle_from_signal_feature(self, signal_feature, return_unit=False):
        if self.cycle_head is None:
            raise RuntimeError("C5B cycle prediction is disabled.")
        unit = self.cycle_head(signal_feature)
        return unit if return_unit else 2.0 * unit - 1.0

    def predict_from_signal_feature(self, signal_feature, t0_temperature_norm=None, cycle_life_hat=None):
        fusion = self._append_t0(signal_feature, t0_temperature_norm)
        if self.cycle_adapter is not None:
            if cycle_life_hat is None:
                raise ValueError("C5B SOH prediction requires the predicted cycle coordinate.")
            cycle_source = cycle_life_hat.to(device=fusion.device, dtype=fusion.dtype).view(fusion.size(0), -1)
            if self.detach_predicted_cycle_for_soh:
                cycle_source = cycle_source.detach()
            base_hidden = self.head[0](fusion)
            cycle_hidden = self.cycle_adapter(cycle_source)
            return self.head[1:](base_hidden + cycle_hidden)
        return self.head(fusion)

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
        z_health = self.encode(
            cc_signal=cc_signal,
            cv_signal=cv_signal,
            cc_mask=cc_mask,
            cv_mask=cv_mask,
            cc_time=cc_time,
            cv_time=cv_time,
            cc_temperature=cc_temperature,
            cv_temperature=cv_temperature,
        )
        cycle_life_hat_unit = (
            self.predict_cycle_from_signal_feature(z_health, return_unit=True)
            if self.cycle_head is not None
            else None
        )
        cycle_life_hat = 2.0 * cycle_life_hat_unit - 1.0 if cycle_life_hat_unit is not None else None
        soh_pred = self.predict_from_signal_feature(z_health, t0_temperature_norm, cycle_life_hat)
        output = {
            "soh_pred": soh_pred,
            "z_health": z_health,
            "signal_feature": z_health,
            "cycle_life_hat": cycle_life_hat,
            "cycle_life_hat_unit": cycle_life_hat_unit,
        }
        if self.phase_input_fusion == "gated_residual_full_vi":
            output["cc_secondary_gate"] = self.cc_branch.input_encoder.last_gate
            output["cv_secondary_gate"] = self.cv_branch.input_encoder.last_gate
        if self.cc_to_cv_point_gate is not None:
            output["cc_to_cv_point_gate"] = self.last_cc_to_cv_point_gate
        return output

    def forward(
        self,
        cc_signal=None,
        cv_signal=None,
        cc_mask=None,
        cv_mask=None,
        cc_time=None,
        cv_time=None,
        cc_temperature=None,
        cv_temperature=None,
        t0_temperature_norm=None,
        cycle_life_norm=None,
    ):
        # ``cycle_life_norm`` is accepted for the historical call contract but
        # is deliberately not consumed: C5B uses only its predicted cycle
        # coordinate for SOH and keeps the true target in the loss path.
        del cycle_life_norm
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


# Compatibility alias for existing local callers and state-dict contract tests.
StandardSingleCycleMamba = PaperRawSOHModel


def build_c5b_model(model_config, backend_override=None):
    cfg = dict(model_config)
    backend = backend_override or cfg.get("backend", "mamba_ssm.Mamba")
    return PaperRawSOHModel(
        input_dim=cfg.get("input_dim", 4),
        signal_input_dim=cfg.get("signal_input_dim", 2),
        d_model=cfg.get("d_model", 32),
        num_layers=cfg.get("num_layers", 3),
        d_state=cfg.get("d_state", 8),
        d_conv=cfg.get("d_conv", 4),
        expand=cfg.get("expand", 2),
        dt_rank=cfg.get("dt_rank", "auto"),
        dropout=cfg.get("dropout", 0.1),
        pooling=cfg.get("pooling", "last_mean"),
        fusion_type=cfg.get("fusion_type", "concat"),
        fusion_phase_dim=cfg.get("fusion_phase_dim", 64),
        head_hidden_dim=cfg.get("head_hidden_dim", 128),
        use_time_as_input=cfg.get("use_time_as_input", True),
        temperature_injection=cfg.get("temperature_injection", "input_concat"),
        temperature_features=cfg.get("temperature_features", "delta"),
        use_t0_temperature_meta=cfg.get("use_t0_temperature_meta", True),
        t0_temperature_meta_dim=cfg.get("t0_temperature_meta_dim", 1),
        use_cc_to_cv_bridge=cfg.get("use_cc_to_cv_bridge", True),
        cc_to_cv_bridge_type=cfg.get("cc_to_cv_bridge_type", "zero_init_linear"),
        cc_to_cv_bridge_input_dim=cfg.get("cc_to_cv_bridge_input_dim", 64),
        cc_to_cv_bridge_output_dim=cfg.get("cc_to_cv_bridge_output_dim", 32),
        use_cycle_prediction=cfg.get("use_cycle_prediction", True),
        cycle_target=cfg.get("cycle_target", "cycle_life_norm"),
        cycle_head_hidden_dim=cfg.get("cycle_head_hidden_dim", 64),
        cycle_output_activation=cfg.get("cycle_output_activation", "sigmoid"),
        use_predicted_cycle_for_soh=cfg.get("use_predicted_cycle_for_soh", True),
        detach_predicted_cycle_for_soh=cfg.get("detach_predicted_cycle_for_soh", False),
        time_embedding_time_scale_min=cfg.get("time_embedding_time_scale_min", 10.0),
        phase_input_fusion=cfg.get("phase_input_fusion", "standard"),
        gate_hidden_dim=cfg.get("gate_hidden_dim", 4),
        gate_context=cfg.get("gate_context", "masked_mean"),
        secondary_residual_init=cfg.get("secondary_residual_init", "zero"),
        backend=backend,
    )
