#!/usr/bin/env python3
"""Bounded CPU structural smoke for the requested E1 model set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from UnifiedRawSOH.models.paper_backup.model_factory import build_model  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402


def main() -> int:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    cases = {
        "HI-MLP": REPO_ROOT / "configs/paper_backup/e1_main_estimation/hi_mlp/xjtu.json",
        "Transformer": REPO_ROOT / "configs/paper_backup/e1_main_estimation/transformer/xjtu.json",
        "Ours": REPO_ROOT / "configs/paper_backup/e1_main_estimation/ours/xjtu.json",
    }
    result = {"status": "PASS", "backend": "torch_reference", "models": {}}
    for model_type, path in cases.items():
        config = load_config(path)
        model = build_model(config["model"], backend_override="torch_reference").to(device)
        if model_type == "HI-MLP":
            prediction = model(torch.randn(2, 24, device=device))
        elif model_type == "Ours":
            common = {
                "cc_signal": torch.randn(2, 8, 2, device=device),
                "cv_signal": torch.randn(2, 10, 2, device=device),
                "cc_mask": torch.ones(2, 8, dtype=torch.bool, device=device),
                "cv_mask": torch.ones(2, 10, dtype=torch.bool, device=device),
                "cc_time": torch.rand(2, 8, device=device),
                "cv_time": torch.rand(2, 10, device=device),
                "cc_temperature": torch.randn(2, 8, 2, device=device),
                "cv_temperature": torch.randn(2, 10, 2, device=device),
                "t0_temperature_norm": torch.randn(2, 1, device=device),
            }
            prediction = model.forward_with_aux(**common)["soh_pred"]
        else:
            prediction = model(torch.randn(2, 12, 5, device=device), torch.ones(2, 12, dtype=torch.bool, device=device))
        loss = prediction.square().mean()
        loss.backward()
        result["models"][model_type] = {"output_shape": list(prediction.shape), "finite_loss": bool(torch.isfinite(loss))}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
