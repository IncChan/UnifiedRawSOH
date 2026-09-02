#!/usr/bin/env python3
"""Validate BiContext Adaptive Fusion parameter and initialization contracts."""

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
ADAPTIVE_CONFIG = REPO_ROOT / (
    "configs/paper_backup/e1_bicontext_adaptive_fusion_5seed/"
    "ours_bicontext_adaptive_fusion/xjtu.json"
)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _backbone_shapes(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    prefixes = (
        "cc_branch.",
        "cv_branch.",
        "cc_context_norm.",
        "cv_context_norm.",
        "cv_to_cc_projection.",
        "cc_to_cv_projection.",
        "cc_bridge_gate.",
        "cv_bridge_gate.",
        "cc_projection.",
        "cv_projection.",
        "ordinary_fusion_mlp.",
        "fusion_norm.",
        "head.",
    )
    return {
        name: tuple(parameter.shape)
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
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
    base_config = load_config(BASE_CONFIG)
    adaptive_config = load_config(ADAPTIVE_CONFIG)
    base_formal = build_model(base_config["model"])
    adaptive_formal = build_model(adaptive_config["model"])
    base_count = _parameter_count(base_formal)
    adaptive_count = _parameter_count(adaptive_formal)
    gate_count = _parameter_count(adaptive_formal.adaptive_fusion_gate)
    parameter_delta = adaptive_count - base_count
    gap_percent = 100.0 * parameter_delta / base_count
    if parameter_delta != gate_count:
        raise ValueError(
            "Adaptive Fusion parameter delta is not isolated to its gate: "
            f"delta={parameter_delta}, gate={gate_count}"
        )
    if gap_percent >= 1.0:
        raise ValueError(f"Adaptive Fusion parameter increase is {gap_percent:.6f}% (>=1%)")
    if _backbone_shapes(base_formal) != _backbone_shapes(adaptive_formal):
        raise ValueError("Adaptive Fusion changed the BiContext backbone parameter shapes")
    if torch.count_nonzero(adaptive_formal.adaptive_fusion_gate[-1].weight).item():
        raise ValueError("Adaptive Fusion gate output weights are not zero initialized")
    if torch.count_nonzero(adaptive_formal.adaptive_fusion_gate[-1].bias).item():
        raise ValueError("Adaptive Fusion gate output bias is not zero initialized")

    torch.manual_seed(2026)
    base = build_model(base_config["model"], backend_override="torch_reference")
    torch.manual_seed(2026)
    adaptive = build_model(
        adaptive_config["model"], backend_override="torch_reference"
    )
    base_state = base.state_dict()
    adaptive_state = adaptive.state_dict()
    shared = {
        name: tensor
        for name, tensor in base_state.items()
        if name in adaptive_state and tensor.shape == adaptive_state[name].shape
    }
    if set(shared) != set(base_state):
        raise ValueError("Adaptive Fusion changed or removed a BiContext parameter")
    adaptive.load_state_dict(shared, strict=False)

    inputs = _inputs()
    base.eval()
    adaptive.eval()
    with torch.no_grad():
        base_output = base.forward_with_aux(**inputs)
        adaptive_output = adaptive.forward_with_aux(**inputs)
    if not torch.allclose(
        base_output["soh_pred"], adaptive_output["soh_pred"], atol=1e-7, rtol=1e-7
    ):
        raise ValueError("Zero-initialized Adaptive Fusion changed the initial SOH function")
    cc_weight = adaptive_output["adaptive_fusion_cc_weight"]
    cv_weight = adaptive_output["adaptive_fusion_cv_weight"]
    if cc_weight.shape != (inputs["cc_signal"].size(0), 1):
        raise ValueError(f"Unexpected adaptive phase-gate shape: {tuple(cc_weight.shape)}")
    if not torch.allclose(cc_weight, torch.full_like(cc_weight, 0.5)):
        raise ValueError("Adaptive Fusion does not start from a 0.5 CC weight")
    if not torch.allclose(cc_weight + cv_weight, torch.ones_like(cc_weight)):
        raise ValueError("Adaptive CC/CV fusion weights do not sum to one")

    adaptive.train()
    adaptive.zero_grad(set_to_none=True)
    adaptive.forward_with_aux(**inputs)["soh_pred"].sum().backward()
    output_gradient = adaptive.adaptive_fusion_gate[-1].weight.grad
    if output_gradient is None or float(output_gradient.norm()) <= 0.0:
        raise ValueError("Adaptive Fusion gate output did not receive a gradient")

    payload = {
        "status": "PASS",
        "formal_backend": "mamba_ssm.Mamba",
        "base_bicontext_parameters": base_count,
        "adaptive_fusion_parameters": adaptive_count,
        "adaptive_gate_parameters": gate_count,
        "parameter_increase": parameter_delta,
        "parameter_increase_percent": gap_percent,
        "bicontext_backbone_shapes_identical": True,
        "ordinary_fusion_hidden_dim_unchanged": True,
        "initial_cc_weight": 0.5,
        "initial_cv_weight": 0.5,
        "initial_function_matches_bicontext": True,
        "first_backward_gate_output_gradient_norm": float(output_gradient.norm()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
