#!/usr/bin/env python3
"""Validate BiContext parameter fairness, initialization, and forward contract."""

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


DUAL_CONFIG = REPO_ROOT / "configs/paper_backup/e1_final_interaction_5seed/raw_dual_vanilla/xjtu.json"
BICONTEXT_CONFIG = REPO_ROOT / "configs/paper_backup/e1_bicontext_5seed/ours_bicontext/xjtu.json"


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _branch_shapes(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(parameter.shape)
        for name, parameter in model.named_parameters()
        if name.startswith(("cc_branch.", "cv_branch."))
    }


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
    dual_config = load_config(DUAL_CONFIG)
    bicontext_config = load_config(BICONTEXT_CONFIG)
    dual = build_model(dual_config["model"])
    bicontext = build_model(bicontext_config["model"])
    dual_count = _parameter_count(dual)
    bicontext_count = _parameter_count(bicontext)
    gap_percent = 100.0 * abs(bicontext_count - dual_count) / dual_count
    if gap_percent > 1.0:
        raise ValueError(f"BiContext parameter gap is {gap_percent:.6f}% (>1%)")
    if _branch_shapes(dual) != _branch_shapes(bicontext):
        raise ValueError("Raw Dual and BiContext Mamba branch parameter shapes differ")
    if len(bicontext.cc_branch.layers) != 3 or len(bicontext.cv_branch.layers) != 3:
        raise ValueError("BiContext must retain three Mamba blocks per phase")
    for projection in (bicontext.cv_to_cc_projection, bicontext.cc_to_cv_projection):
        if torch.count_nonzero(projection.weight).item() or torch.count_nonzero(projection.bias).item():
            raise ValueError("BiContext cross projection is not zero initialized")
    for gate in (bicontext.cc_bridge_gate, bicontext.cv_bridge_gate):
        if torch.count_nonzero(gate[-1].weight).item() or torch.count_nonzero(gate[-1].bias).item():
            raise ValueError("BiContext point gate output is not zero initialized")
    if torch.count_nonzero(bicontext.ordinary_fusion_mlp[-1].weight).item():
        raise ValueError("BiContext fusion residual is not zero initialized")

    # The reference backend provides a device-independent forward/backward
    # smoke test while the formal parameter count above uses mamba_ssm.Mamba.
    torch.manual_seed(2026)
    dual_ref = build_model(dual_config["model"], backend_override="torch_reference")
    torch.manual_seed(2026)
    bicontext_ref = build_model(
        bicontext_config["model"], backend_override="torch_reference"
    )
    source = dual_ref.state_dict()
    target = bicontext_ref.state_dict()
    compatible = {
        name: tensor
        for name, tensor in source.items()
        if name in target and tensor.shape == target[name].shape
    }
    bicontext_ref.load_state_dict(compatible, strict=False)
    dual_ref.eval()
    bicontext_ref.eval()
    inputs = _inputs()
    with torch.no_grad():
        dual_prediction = dual_ref.forward_with_aux(**inputs)["soh_pred"]
        output = bicontext_ref.forward_with_aux(**inputs)
    if not torch.allclose(dual_prediction, output["soh_pred"], atol=1e-6, rtol=1e-6):
        raise ValueError("Zero-initialized BiContext does not match ordinary fusion")
    for key in ("cc_bridge_gate", "cv_bridge_gate"):
        gate = output[key]
        mask = inputs["cc_mask"] if key.startswith("cc") else inputs["cv_mask"]
        if not torch.allclose(gate[mask], torch.full_like(gate[mask], 0.5)):
            raise ValueError(f"Unexpected initial valid-token gate for {key}")
        if torch.count_nonzero(gate[~mask]).item():
            raise ValueError(f"Padding tokens are not masked for {key}")
    for key in ("cc_bridge_contribution_ratio", "cv_bridge_contribution_ratio"):
        if torch.count_nonzero(output[key]).item():
            raise ValueError(f"Initial bridge contribution is not zero for {key}")

    bicontext_ref.train()
    bicontext_ref.zero_grad(set_to_none=True)
    bicontext_ref.forward_with_aux(**inputs)["soh_pred"].sum().backward()
    projection_gradients = {
        "cv_to_cc": float(bicontext_ref.cv_to_cc_projection.weight.grad.norm()),
        "cc_to_cv": float(bicontext_ref.cc_to_cv_projection.weight.grad.norm()),
    }
    if any(value <= 0.0 for value in projection_gradients.values()):
        raise ValueError(f"Cross projections did not receive gradients: {projection_gradients}")

    bridge_parameter_count = sum(
        parameter.numel()
        for name, parameter in bicontext.named_parameters()
        if name.startswith(
            (
                "cc_context_norm.",
                "cv_context_norm.",
                "cv_to_cc_projection.",
                "cc_to_cv_projection.",
                "cc_bridge_gate.",
                "cv_bridge_gate.",
            )
        )
    )
    payload = {
        "status": "PASS",
        "formal_backend": "mamba_ssm.Mamba",
        "raw_dual_parameters": dual_count,
        "bicontext_parameters": bicontext_count,
        "parameter_gap": bicontext_count - dual_count,
        "parameter_gap_percent": gap_percent,
        "bridge_parameters": bridge_parameter_count,
        "mamba_branch_shapes_identical": True,
        "mamba_layers_per_phase": 3,
        "bridge_after_layer": bicontext.bridge_after_layer,
        "initial_valid_gate": 0.5,
        "initial_bridge_contribution": 0.0,
        "initial_function_matches_raw_dual": True,
        "first_backward_projection_gradient_norms": projection_gradients,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
