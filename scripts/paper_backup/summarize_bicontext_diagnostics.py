#!/usr/bin/env python3
"""Run best-checkpoint BiContext gate diagnostics on each formal test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from UnifiedRawSOH.models.paper_backup.model_factory import build_model  # noqa: E402
from summarize_results import (  # noqa: E402
    discover_runs,
    expected_jobs,
    select_latest_runs,
)
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.trainer import PHASE_KEYS, _build_loaders  # noqa: E402
from UnifiedRawSOH.utils.seed import set_random_seed  # noqa: E402


COLUMNS = (
    "dataset", "seed", "n_samples",
    "cc_gate_mean", "cc_gate_std", "cc_gate_q10", "cc_gate_q50", "cc_gate_q90",
    "cv_gate_mean", "cv_gate_std", "cv_gate_q10", "cv_gate_q50", "cv_gate_q90",
    "cc_bridge_ratio_mean", "cc_bridge_ratio_q10", "cc_bridge_ratio_q50", "cc_bridge_ratio_q90",
    "cv_bridge_ratio_mean", "cv_bridge_ratio_q10", "cv_bridge_ratio_q50", "cv_bridge_ratio_q90",
    "cv_to_cc_projection_weight_norm", "cc_to_cv_projection_weight_norm",
    "cc_gate_last_weight_norm", "cv_gate_last_weight_norm",
    "run_dir",
)


def _parse_seeds(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.replace(",", " ").split())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique integer list")
    return values


def _stats(values: list[np.ndarray], prefix: str) -> dict[str, float]:
    merged = np.concatenate(values).astype(np.float64, copy=False)
    return {
        f"{prefix}_mean": float(merged.mean()),
        f"{prefix}_std": float(merged.std()),
        f"{prefix}_q10": float(np.quantile(merged, 0.10)),
        f"{prefix}_q50": float(np.quantile(merged, 0.50)),
        f"{prefix}_q90": float(np.quantile(merged, 0.90)),
    }


def _ratio_stats(values: list[np.ndarray], prefix: str) -> dict[str, float]:
    merged = np.concatenate(values).astype(np.float64, copy=False)
    return {
        f"{prefix}_mean": float(merged.mean()),
        f"{prefix}_q10": float(np.quantile(merged, 0.10)),
        f"{prefix}_q50": float(np.quantile(merged, 0.50)),
        f"{prefix}_q90": float(np.quantile(merged, 0.90)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("42 52 62 72 82"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BiContext diagnostics requested CUDA but CUDA is unavailable")

    expected, _ = expected_jobs("e1_bicontext_5seed", args.seeds)
    selected, _ = select_latest_runs(discover_runs(args.root.resolve()), "e1_bicontext_5seed")
    missing = sorted(set(expected) - set(selected))
    if missing:
        raise ValueError(f"BiContext diagnostics require a complete matrix; missing={missing}")

    rows = []
    for key in sorted(expected):
        spec, run = expected[key], selected[key]
        seed = int(spec["seed"])
        set_random_seed(seed)
        config = load_config(REPO_ROOT / spec["config"])
        loaders, _ = _build_loaders(config, REPO_ROOT, seed)
        model = build_model(config["model"]).to(device)
        checkpoint = torch.load(Path(run["run_dir"]) / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        cc_gates: list[np.ndarray] = []
        cv_gates: list[np.ndarray] = []
        cc_ratios: list[np.ndarray] = []
        cv_ratios: list[np.ndarray] = []
        sample_count = 0
        with torch.no_grad():
            for batch in loaders["test"]:
                inputs = {
                    name: batch[name].to(device, non_blocking=device.type == "cuda")
                    for name in PHASE_KEYS
                }
                output = model.forward_with_aux(**inputs)
                cc_mask = inputs["cc_mask"].bool()
                cv_mask = inputs["cv_mask"].bool()
                cc_gates.append(output["cc_bridge_gate"].squeeze(-1)[cc_mask].cpu().numpy())
                cv_gates.append(output["cv_bridge_gate"].squeeze(-1)[cv_mask].cpu().numpy())
                cc_ratios.append(output["cc_bridge_contribution_ratio"].cpu().numpy())
                cv_ratios.append(output["cv_bridge_contribution_ratio"].cpu().numpy())
                sample_count += int(inputs["cc_signal"].size(0))
        row = {"dataset": spec["family"], "seed": seed, "n_samples": sample_count}
        row.update(_stats(cc_gates, "cc_gate"))
        row.update(_stats(cv_gates, "cv_gate"))
        row.update(_ratio_stats(cc_ratios, "cc_bridge_ratio"))
        row.update(_ratio_stats(cv_ratios, "cv_bridge_ratio"))
        row.update(
            {
                "cv_to_cc_projection_weight_norm": float(model.cv_to_cc_projection.weight.norm()),
                "cc_to_cv_projection_weight_norm": float(model.cc_to_cv_projection.weight.norm()),
                "cc_gate_last_weight_norm": float(model.cc_bridge_gate[-1].weight.norm()),
                "cv_gate_last_weight_norm": float(model.cv_bridge_gate[-1].weight.norm()),
                "run_dir": run["run_dir"],
            }
        )
        rows.append(row)
        print(
            f"[diagnostics] dataset={spec['family']} seed={seed} "
            f"cc_gate={row['cc_gate_mean']:.4f} cv_gate={row['cv_gate_mean']:.4f} "
            f"cc_ratio={row['cc_bridge_ratio_mean']:.4f} cv_ratio={row['cv_bridge_ratio_mean']:.4f}",
            flush=True,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "e1_bicontext_gate_diagnostics.csv"
    json_path = output_dir / "e1_bicontext_gate_diagnostics.json"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    json_path.write_text(
        json.dumps({"experiment": "E1_BICONTEXT_5SEED", "rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
