#!/usr/bin/env python3
"""Validate architecture isolation, initialization, and the ~82.8k raw-model budget."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from UnifiedRawSOH.models.paper_backup.model_factory import build_model  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402


def _count(path: Path):
    config = load_config(path)
    model = build_model(config["model"])
    total = sum(parameter.numel() for parameter in model.parameters())
    return config, model, total


def main() -> int:
    e1_root = REPO_ROOT / "configs/paper_backup/e1_final_interaction_5seed"
    paths = {path.parent.name: path for path in sorted(e1_root.glob("*/xjtu.json"))}
    built = {name: _count(path) for name, path in paths.items()}
    target = built["ours_interaction"][2]
    counts = {}
    for name, (config, model, total) in built.items():
        counts[name] = total
        if name != "hi_mlp":
            gap = 100.0 * abs(total - target) / target
            if gap > 1.0:
                raise ValueError(f"{name} differs from Ours by {gap:.4f}% (>1%)")
            if (
                "Mamba" in str(config["model"].get("type", ""))
                and str(config["model"].get("backend")) != "mamba_ssm.Mamba"
            ):
                raise ValueError(f"{name} is not configured for the formal Mamba backend")

    ours = built["ours_interaction"][1]
    dual = built["raw_dual_vanilla"][1]
    if hasattr(ours, "point_bridge") or hasattr(ours, "bridge"):
        raise ValueError("Final Ours must not contain PointBridge")
    if hasattr(dual, "interaction_mlp") or hasattr(dual, "gate_mlp"):
        raise ValueError("Raw Dual control leaked interaction/gating modules")
    if not torch.count_nonzero(ours.interaction_mlp[-1].weight).item() == 0:
        raise ValueError("Interaction output layer is not zero initialized")
    gate = torch.softmax(ours.gate_mlp[-1].bias.detach(), dim=0).tolist()
    expected = (0.45, 0.45, 0.10)
    if any(not math.isclose(value, reference, abs_tol=1e-6) for value, reference in zip(gate, expected)):
        raise ValueError(f"Unexpected interaction gate initialization: {gate}")

    e2_root = REPO_ROOT / "configs/paper_backup/e2_final_interaction_5seed"
    e2_counts = {path.parent.name: _count(path)[2] for path in sorted(e2_root.glob("*/xjtu.json"))}
    if e2_counts["raw_dual_vanilla"] != counts["raw_dual_vanilla"] or e2_counts["ours_interaction"] != target:
        raise ValueError("E1/E2 terminal model definitions are not identical")
    payload = {
        "status": "PASS",
        "backend": "mamba_ssm.Mamba",
        "target_ours_parameters": target,
        "e1_registered_parameters": counts,
        "e2_registered_parameters": e2_counts,
        "feature_mlp_budget_exception": "HI feature baseline is intentionally not width-inflated",
        "gate_initial_probabilities": gate,
        "interaction_output_zero_initialized": True,
        "pointbridge_present": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
