#!/usr/bin/env python3
"""Validate the Late Latent-Token BiContext function-preserving contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from UnifiedRawSOH.models.paper_backup.final_models import (  # noqa: E402
    LateLatentTokenBiContextSOHModel,
    RawDualVanillaMambaSOHModel,
)
from UnifiedRawSOH.models.paper_backup.model_factory import build_model  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_contract import validate_config  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402


RAW_CONFIG_DIR = REPO_ROOT / "configs/paper_backup/e1_final_interaction_5seed/raw_dual_vanilla"
LATE_CONFIG_DIR = REPO_ROOT / (
    "configs/paper_backup/e1_late_latent_token_bicontext_5seed/"
    "ours_late_latent_token_bicontext"
)
FILES = (
    "xjtu.json",
    "mit.json",
    "smarthealth_lishen40.json",
    "smarthealth_catl280.json",
    "smarthealth_eve280.json",
)
EXTRA_MODEL_KEYS = {"num_latents", "cross_num_heads", "cross_after_layer"}


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _without(mapping: dict, keys: set[str]) -> dict:
    return {key: value for key, value in mapping.items() if key not in keys}


def _inputs() -> dict[str, torch.Tensor]:
    batch, cc_length, cv_length = 3, 13, 11
    cc_mask = torch.ones(batch, cc_length, dtype=torch.bool)
    cv_mask = torch.ones(batch, cv_length, dtype=torch.bool)
    cc_mask[1, -2:] = False
    cc_mask[2, -4:] = False
    cv_mask[1, -3:] = False
    cv_mask[2, -1:] = False
    return {
        "cc_signal": torch.randn(batch, cc_length, 3),
        "cv_signal": torch.randn(batch, cv_length, 3),
        "cc_time": torch.arange(cc_length, dtype=torch.float32).unsqueeze(0).expand(batch, -1),
        "cv_time": torch.arange(cv_length, dtype=torch.float32).unsqueeze(0).expand(batch, -1),
        "cc_temperature": torch.randn(batch, cc_length, 2),
        "cv_temperature": torch.randn(batch, cv_length, 2),
        "t0_temperature_norm": torch.randn(batch, 1),
        "cc_mask": cc_mask,
        "cv_mask": cv_mask,
    }


def _validate_config_fairness() -> None:
    for filename in FILES:
        raw = load_config(RAW_CONFIG_DIR / filename)
        late = load_config(LATE_CONFIG_DIR / filename)
        validate_config(late, REPO_ROOT, check_files=True)
        raw_model = _without(raw["model"], {"type"})
        late_model = _without(late["model"], {"type", *EXTRA_MODEL_KEYS})
        if raw_model != late_model:
            raise ValueError(f"Inherited Raw Dual model settings differ for {filename}")
        for section in (
            "data",
            "normalization",
            "train",
            "optimizer",
            "scheduler",
            "formal_protocol",
        ):
            if raw.get(section) != late.get(section):
                raise ValueError(f"{section} differs from Raw Dual for {filename}")


def main() -> int:
    _validate_config_fairness()
    raw_config = load_config(RAW_CONFIG_DIR / "xjtu.json")
    late_config = load_config(LATE_CONFIG_DIR / "xjtu.json")

    # Formal construction verifies the official Mamba parameter/state layout;
    # its CUDA kernels are not needed for this read-only architecture check.
    torch.manual_seed(2026)
    raw_formal = build_model(raw_config["model"])
    torch.manual_seed(2026)
    late_formal = build_model(late_config["model"])
    formal_shared_mismatch = [
        name
        for name, value in raw_formal.state_dict().items()
        if name not in late_formal.state_dict()
        or not torch.equal(value, late_formal.state_dict()[name])
    ]
    if formal_shared_mismatch:
        raise ValueError(
            "Formal same-seed inherited initialization differs: "
            f"{formal_shared_mismatch}"
        )

    # Resetting the seed is intentionally sufficient: the subclass constructs
    # the complete Raw Dual parent before allocating any new cross modules.
    torch.manual_seed(2026)
    raw = build_model(raw_config["model"], backend_override="torch_reference")
    torch.manual_seed(2026)
    late = build_model(late_config["model"], backend_override="torch_reference")
    if not isinstance(late, LateLatentTokenBiContextSOHModel):
        raise TypeError("Factory did not build LateLatentTokenBiContextSOHModel")
    if not isinstance(late, RawDualVanillaMambaSOHModel):
        raise TypeError("Late model does not inherit RawDualVanillaMambaSOHModel")
    if len(late.cc_branch.layers) != 3 or len(late.cv_branch.layers) != 3:
        raise ValueError("Late model must retain three Mamba blocks per phase")
    if sum(isinstance(module, nn.MultiheadAttention) for module in late.modules()) != 2:
        raise ValueError("Late model must contain exactly two cross-attention modules")

    raw_state = raw.state_dict()
    late_state = late.state_dict()
    shared_mismatch = [
        name
        for name, value in raw_state.items()
        if name not in late_state or not torch.equal(value, late_state[name])
    ]
    if shared_mismatch:
        raise ValueError(f"Same-seed inherited initialization differs: {shared_mismatch}")
    for attention in (late.cc_read_cv, late.cv_read_cc):
        if torch.count_nonzero(attention.out_proj.weight).item():
            raise ValueError("Cross-attention out_proj weight is not zero initialized")
        if attention.out_proj.bias is not None and torch.count_nonzero(attention.out_proj.bias).item():
            raise ValueError("Cross-attention out_proj bias is not zero initialized")
    if not torch.count_nonzero(late.cc_latent_score.weight).item():
        raise ValueError("CC latent pooler must not be zero initialized")
    if not torch.count_nonzero(late.cv_latent_score.weight).item():
        raise ValueError("CV latent pooler must not be zero initialized")

    raw.eval()
    late.eval()
    torch.manual_seed(420)
    inputs = _inputs()
    with torch.no_grad():
        raw_output = raw.forward_with_aux(**inputs)
        late_output = late.forward_with_aux(**inputs)
    prediction = late_output["soh_pred"]
    if prediction.shape != (3, 1):
        raise ValueError(f"Unexpected forward shape: {tuple(prediction.shape)}")
    if late_output["latent_cc_attention"].shape != (3, 13, 4):
        raise ValueError("Unexpected CC-to-CV-latent attention shape")
    if late_output["latent_cv_attention"].shape != (3, 11, 4):
        raise ValueError("Unexpected CV-to-CC-latent attention shape")
    if torch.count_nonzero(late.last_cc_cross_delta).item():
        raise ValueError("Initial cc_delta is not exactly zero")
    if torch.count_nonzero(late.last_cv_cross_delta).item():
        raise ValueError("Initial cv_delta is not exactly zero")
    for key in ("cc_cross_contribution_ratio", "cv_cross_contribution_ratio"):
        if torch.count_nonzero(late_output[key]).item():
            raise ValueError(f"Initial {key} is not exactly zero")
    if not torch.equal(raw_output["soh_pred"], prediction):
        difference = float((raw_output["soh_pred"] - prediction).abs().max())
        raise ValueError(f"Initial function differs from Raw Dual: max_abs={difference}")

    # A zero output projection must still receive a gradient on step one.
    late.train()
    late.zero_grad(set_to_none=True)
    late.forward_with_aux(**inputs)["soh_pred"].sum().backward()
    first_step_gradients = {
        "cc_read_cv_out_proj": float(late.cc_read_cv.out_proj.weight.grad.norm()),
        "cv_read_cc_out_proj": float(late.cv_read_cc.out_proj.weight.grad.norm()),
    }
    if any(value <= 0.0 for value in first_step_gradients.values()):
        raise ValueError(f"Cross pathway did not receive gradients: {first_step_gradients}")

    payload = {
        "status": "PASS",
        "model_id": late.model_id,
        "datasets": list(FILES),
        "formal_config_matches_raw_dual_except_cross_fields": True,
        "forward_shape": list(prediction.shape),
        "cc_cross_delta_exactly_zero": True,
        "cv_cross_delta_exactly_zero": True,
        "same_seed_shared_initialization_exact": True,
        "initial_function_matches_raw_dual_exactly": True,
        "formal_backend": "mamba_ssm.Mamba",
        "raw_dual_parameters": _parameter_count(raw_formal),
        "late_latent_token_parameters": _parameter_count(late_formal),
        "added_parameters": (
            _parameter_count(late_formal) - _parameter_count(raw_formal)
        ),
        "num_latents": late.num_latents,
        "cross_num_heads": late.cross_num_heads,
        "cross_after_layer": late.cross_after_layer,
        "first_backward_out_proj_gradient_norms": first_step_gradients,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
