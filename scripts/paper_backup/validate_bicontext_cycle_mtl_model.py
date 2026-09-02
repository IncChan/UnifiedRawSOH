#!/usr/bin/env python3
"""Validate that Cycle MTL regularizes BiContext without entering the SOH head."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from UnifiedRawSOH.models.paper_backup.model_factory import build_model  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402


BASE_CONFIG = REPO_ROOT / "configs/paper_backup/e1_bicontext_5seed/ours_bicontext/xjtu.json"
MTL_CONFIG = REPO_ROOT / "configs/paper_backup/e1_bicontext_cycle_mtl_5seed/ours_bicontext_cycle_mtl/xjtu.json"


def _inputs() -> dict[str, torch.Tensor]:
    batch, length = 3, 11
    mask = torch.ones(batch, length, dtype=torch.bool)
    mask[1, -2:] = False
    mask[2, -4:] = False
    time = torch.arange(length, dtype=torch.float32).unsqueeze(0).expand(batch, -1)
    return {
        "cc_signal": torch.randn(batch, length, 3),
        "cv_signal": torch.randn(batch, length, 3),
        "cc_time": time.clone(),
        "cv_time": time.clone(),
        "cc_temperature": torch.randn(batch, length, 2),
        "cv_temperature": torch.randn(batch, length, 2),
        "t0_temperature_norm": torch.randn(batch, 1),
        "cc_mask": mask.clone(),
        "cv_mask": mask.clone(),
    }


def main() -> int:
    base_config = load_config(BASE_CONFIG)
    mtl_config = load_config(MTL_CONFIG)
    base_formal = build_model(base_config["model"])
    mtl_formal = build_model(mtl_config["model"])
    base_count = sum(parameter.numel() for parameter in base_formal.parameters())
    mtl_count = sum(parameter.numel() for parameter in mtl_formal.parameters())
    cycle_head_count = sum(
        parameter.numel() for parameter in mtl_formal.cycle_aux_head.parameters()
    )
    if mtl_count - base_count != cycle_head_count:
        raise ValueError("Formal Cycle MTL parameter delta is not isolated to its head")
    if {
        name: tuple(parameter.shape)
        for name, parameter in base_formal.named_parameters()
    } != {
        name: tuple(parameter.shape)
        for name, parameter in mtl_formal.named_parameters()
        if not name.startswith("cycle_aux_head.")
    }:
        raise ValueError("Formal Cycle MTL changed the BiContext backbone shapes")

    base = build_model(base_config["model"], backend_override="torch_reference")
    mtl = build_model(mtl_config["model"], backend_override="torch_reference")

    base_state = base.state_dict()
    mtl_state = mtl.state_dict()
    shared_names = {
        name for name in base_state
        if name in mtl_state and base_state[name].shape == mtl_state[name].shape
    }
    if shared_names != set(base_state):
        raise ValueError("Cycle MTL changed or removed a BiContext parameter")
    mtl.load_state_dict({name: base_state[name] for name in shared_names}, strict=False)

    inputs = _inputs()
    base.eval()
    mtl.eval()
    with torch.no_grad():
        base_soh = base.forward_with_aux(**inputs)["soh_pred"]
        output = mtl.forward_with_aux(**inputs)
        inference_soh = mtl(**inputs)
    if not torch.allclose(base_soh, output["soh_pred"], atol=1e-7, rtol=1e-7):
        raise ValueError("Adding the auxiliary head changed the initial SOH path")
    if output["cycle_aux_pred"].shape != base_soh.shape:
        raise ValueError("Cycle auxiliary output shape does not match SOH batch shape")
    if not torch.allclose(inference_soh, output["soh_pred"], atol=1e-7, rtol=1e-7):
        raise ValueError("Inference-only SOH path differs from the multitask SOH path")
    if output["cycle_aux_is_soh_input"] is not False:
        raise ValueError("Cycle auxiliary branch is incorrectly marked as an SOH input")

    mtl.train()
    mtl.zero_grad(set_to_none=True)
    mtl.forward_with_aux(**inputs)["soh_pred"].sum().backward()
    if any(parameter.grad is not None for parameter in mtl.cycle_aux_head.parameters()):
        raise ValueError("SOH loss unexpectedly backpropagates through the cycle head")

    mtl.zero_grad(set_to_none=True)
    mtl.forward_with_aux(**inputs)["cycle_aux_pred"].sum().backward()
    backbone_gradient = sum(
        float(parameter.grad.norm())
        for name, parameter in mtl.named_parameters()
        if name.startswith(("cc_branch.", "cv_branch.")) and parameter.grad is not None
    )
    if backbone_gradient <= 0.0:
        raise ValueError("Cycle auxiliary loss did not reach the shared Mamba backbone")

    payload = {
        "status": "PASS",
        "parameter_count_backend": "mamba_ssm.Mamba",
        "base_bicontext_parameters": base_count,
        "cycle_mtl_parameters": mtl_count,
        "cycle_head_parameters": cycle_head_count,
        "shared_bicontext_state_identical": True,
        "initial_soh_prediction_identical": True,
        "cycle_prediction_used_by_soh_head": False,
        "ordinary_inference_bypasses_cycle_head": True,
        "cycle_aux_gradient_reaches_backbone": True,
        "backbone_gradient_norm_sum": backbone_gradient,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
