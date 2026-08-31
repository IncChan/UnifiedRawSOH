#!/usr/bin/env python3
"""Validate the predeclared E2 Vanilla/Ours parameter match without training."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from UnifiedRawSOH.models.paper_backup.model_factory import build_model  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402


def count(config_path: Path) -> tuple[int, int]:
    config = load_config(config_path)
    model = build_model(config["model"])
    return (
        sum(parameter.numel() for parameter in model.parameters()),
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )


def main() -> int:
    root = REPO_ROOT / "configs/paper_backup/e2_final_256budget"
    paths = {
        "full_vanilla": root / "full_vanilla_256/xjtu.json",
        "terminal_vanilla": root / "terminal_vanilla_sep_128x128/xjtu.json",
        "ours": root / "ours_pointbridge_128x128/xjtu.json",
        "ours_cc_only": root / "ours_cc_only_128/xjtu.json",
        "ours_cv_only": root / "ours_cv_only_128/xjtu.json",
    }
    counts = {name: count(path) for name, path in paths.items()}
    if counts["full_vanilla"] != counts["terminal_vanilla"]:
        raise ValueError("FULL and Terminal Vanilla parameter counts differ")
    vanilla_total = counts["full_vanilla"][0]
    ours_total = counts["ours"][0]
    relative_percent = 100.0 * abs(vanilla_total - ours_total) / ours_total
    if relative_percent > 1.0:
        raise ValueError(
            f"E2 parameter mismatch is {relative_percent:.4f}%, above 1%"
        )
    payload = {
        "backend": "mamba_ssm.Mamba",
        "counts": {
            name: {"registered": value[0], "trainable": value[1]}
            for name, value in counts.items()
        },
        "vanilla_vs_ours_registered_difference": vanilla_total - ours_total,
        "vanilla_vs_ours_relative_difference_percent": relative_percent,
        "status": "PASS",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
