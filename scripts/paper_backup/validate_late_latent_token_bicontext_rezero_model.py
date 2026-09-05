#!/usr/bin/env python3
"""Validate Pre-Norm + bounded bidirectional ReZero late-token cross."""

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
REZERO_CONFIG_DIR = REPO_ROOT / (
    "configs/paper_backup/e1_late_latent_token_bicontext_rezero_5seed/"
    "ours_late_latent_token_bicontext_rezero"
)
FILES = (
    "xjtu.json",
    "mit.json",
    "smarthealth_lishen40.json",
    "smarthealth_catl280.json",
    "smarthealth_eve280.json",
)
EXTRA_MODEL_KEYS = {
    "num_latents",
    "cross_num_heads",
    "cross_after_layer",
    "cross_residual_mode",
    "cross_max_scale",
}


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
        rezero = load_config(REZERO_CONFIG_DIR / filename)
        validate_config(rezero, REPO_ROOT, check_files=True)
        raw_model = _without(raw["model"], {"type"})
        rezero_model = _without(rezero["model"], {"type", *EXTRA_MODEL_KEYS})
        if raw_model != rezero_model:
            raise ValueError(f"Inherited Raw Dual settings differ for {filename}")
        for section in ("data", "normalization", "train", "optimizer", "scheduler"):
            if raw.get(section) != rezero.get(section):
                raise ValueError(f"{section} differs from Raw Dual for {filename}")


def _assert_shared_initialization(raw: nn.Module, rezero: nn.Module) -> None:
    rezero_state = rezero.state_dict()
    mismatch = [
        name
        for name, value in raw.state_dict().items()
        if name not in rezero_state or not torch.equal(value, rezero_state[name])
    ]
    if mismatch:
        raise ValueError(f"Same-seed inherited initialization differs: {mismatch}")


def main() -> int:
    _validate_config_fairness()
    raw_config = load_config(RAW_CONFIG_DIR / "xjtu.json")
    rezero_config = load_config(REZERO_CONFIG_DIR / "xjtu.json")

    torch.manual_seed(2026)
    raw_formal = build_model(raw_config["model"])
    torch.manual_seed(2026)
    rezero_formal = build_model(rezero_config["model"])
    _assert_shared_initialization(raw_formal, rezero_formal)

    torch.manual_seed(2026)
    raw = build_model(raw_config["model"], backend_override="torch_reference")
    torch.manual_seed(2026)
    rezero = build_model(rezero_config["model"], backend_override="torch_reference")
    _assert_shared_initialization(raw, rezero)
    if not isinstance(rezero, LateLatentTokenBiContextSOHModel):
        raise TypeError("Factory did not build the late latent-token model")
    if not isinstance(rezero, RawDualVanillaMambaSOHModel):
        raise TypeError("ReZero model does not inherit Raw Dual Vanilla")
    if rezero.cross_residual_mode != "prenorm_bounded_rezero":
        raise ValueError("Unexpected cross residual mode")
    if not isinstance(rezero.cc_cross_norm, nn.LayerNorm):
        raise TypeError("CC cross Pre-Norm is missing")
    if not isinstance(rezero.cv_cross_norm, nn.LayerNorm):
        raise TypeError("CV cross Pre-Norm is missing")
    for alpha in (rezero.cc_cross_alpha, rezero.cv_cross_alpha):
        if torch.count_nonzero(alpha).item():
            raise ValueError("ReZero alpha is not exactly zero initialized")
    for attention in (rezero.cc_read_cv, rezero.cv_read_cc):
        if not torch.count_nonzero(attention.out_proj.weight).item():
            raise ValueError("ReZero MHA out_proj must use ordinary initialization")

    raw.eval()
    rezero.eval()
    torch.manual_seed(420)
    inputs = _inputs()
    with torch.no_grad():
        raw_output = raw.forward_with_aux(**inputs)
        output = rezero.forward_with_aux(**inputs)
    if output["soh_pred"].shape != (3, 1):
        raise ValueError(f"Unexpected forward shape: {tuple(output['soh_pred'].shape)}")
    if not torch.equal(raw_output["soh_pred"], output["soh_pred"]):
        difference = float((raw_output["soh_pred"] - output["soh_pred"]).abs().max())
        raise ValueError(f"Initial function differs from Raw Dual: max_abs={difference}")
    for key in (
        "cc_cross_scale",
        "cv_cross_scale",
        "cc_cross_contribution_ratio",
        "cv_cross_contribution_ratio",
    ):
        if torch.count_nonzero(output[key]).item():
            raise ValueError(f"Initial {key} is not exactly zero")
    if torch.count_nonzero(rezero.last_cc_cross_delta).item():
        raise ValueError("Initial scaled cc_delta is not exactly zero")
    if torch.count_nonzero(rezero.last_cv_cross_delta).item():
        raise ValueError("Initial scaled cv_delta is not exactly zero")

    # At exact ReZero initialization only the two scalar gates should receive
    # cross-path gradients.  Once they open, MHA and pooler gradients follow.
    rezero.train()
    rezero.zero_grad(set_to_none=True)
    rezero.forward_with_aux(**inputs)["soh_pred"].sum().backward()
    first_step_alpha_gradients = {
        "cc": float(rezero.cc_cross_alpha.grad.abs()),
        "cv": float(rezero.cv_cross_alpha.grad.abs()),
    }
    if any(value <= 0.0 for value in first_step_alpha_gradients.values()):
        raise ValueError(f"ReZero gates did not receive gradients: {first_step_alpha_gradients}")
    first_step_out_proj_gradients = {
        "cc": float(rezero.cc_read_cv.out_proj.weight.grad.norm()),
        "cv": float(rezero.cv_read_cc.out_proj.weight.grad.norm()),
    }
    if any(value != 0.0 for value in first_step_out_proj_gradients.values()):
        raise ValueError("Cross MHA should remain frozen on the exact first ReZero step")

    with torch.no_grad():
        rezero.cc_cross_alpha.fill_(0.01)
        rezero.cv_cross_alpha.fill_(0.01)
    rezero.zero_grad(set_to_none=True)
    rezero.forward_with_aux(**inputs)["soh_pred"].sum().backward()
    opened_out_proj_gradients = {
        "cc": float(rezero.cc_read_cv.out_proj.weight.grad.norm()),
        "cv": float(rezero.cv_read_cc.out_proj.weight.grad.norm()),
    }
    if any(value <= 0.0 for value in opened_out_proj_gradients.values()):
        raise ValueError(f"Opened cross pathway has no gradients: {opened_out_proj_gradients}")

    payload = {
        "status": "PASS",
        "model_id": rezero_config["output"]["model_id"],
        "datasets": list(FILES),
        "forward_shape": list(output["soh_pred"].shape),
        "prenorm_enabled": True,
        "cross_max_scale": rezero.cross_max_scale,
        "initial_scales_exactly_zero": True,
        "initial_scaled_deltas_exactly_zero": True,
        "same_seed_shared_initialization_exact": True,
        "initial_function_matches_raw_dual_exactly": True,
        "raw_dual_parameters": _parameter_count(raw_formal),
        "rezero_parameters": _parameter_count(rezero_formal),
        "added_parameters": _parameter_count(rezero_formal) - _parameter_count(raw_formal),
        "first_step_alpha_gradient_magnitudes": first_step_alpha_gradients,
        "first_step_out_proj_gradient_norms": first_step_out_proj_gradients,
        "opened_out_proj_gradient_norms": opened_out_proj_gradients,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
