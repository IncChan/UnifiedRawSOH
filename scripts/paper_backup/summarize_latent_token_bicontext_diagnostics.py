#!/usr/bin/env python3
"""Summarize learned Late Latent-Token BiContext use on formal test sets."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from summarize_results import discover_runs, expected_jobs, select_latest_runs  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.trainer import PHASE_KEYS, _build_loaders  # noqa: E402
from UnifiedRawSOH.utils.seed import set_random_seed  # noqa: E402


DEFAULT_EXPERIMENT = "e1_late_latent_token_bicontext_5seed"
COLUMNS = (
    "dataset",
    "seed",
    "n_samples",
    "cc_cross_ratio_mean",
    "cc_cross_ratio_q50",
    "cc_cross_ratio_q90",
    "cv_cross_ratio_mean",
    "cv_cross_ratio_q50",
    "cv_cross_ratio_q90",
    "cc_attention_entropy_normalized_mean",
    "cv_attention_entropy_normalized_mean",
    "cc_attention_max_mean",
    "cv_attention_max_mean",
    "cc_cross_scale",
    "cv_cross_scale",
    "cc_read_cv_out_proj_weight_norm",
    "cv_read_cc_out_proj_weight_norm",
    "run_dir",
)


def _parse_seeds(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.replace(",", " ").split())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique integer list")
    return values


def _ratio_stats(values: list[np.ndarray], prefix: str) -> dict[str, float]:
    merged = np.concatenate(values).astype(np.float64, copy=False)
    return {
        f"{prefix}_mean": float(merged.mean()),
        f"{prefix}_q50": float(np.quantile(merged, 0.50)),
        f"{prefix}_q90": float(np.quantile(merged, 0.90)),
    }


def _attention_stats(
    attention: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = attention[mask.bool()].float().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1) / math.log(
        probabilities.size(-1)
    )
    maximum = probabilities.max(dim=-1).values
    return entropy.cpu().numpy(), maximum.cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("42 52 62 72 82"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Latent-token diagnostics requested CUDA but CUDA is unavailable")

    experiment = str(args.experiment)
    expected, _ = expected_jobs(experiment, args.seeds)
    selected, _ = select_latest_runs(discover_runs(args.root.resolve()), experiment)
    missing = sorted(set(expected) - set(selected))
    if missing:
        raise ValueError(f"Latent-token diagnostics require a complete matrix; missing={missing}")

    rows = []
    for key in sorted(expected):
        spec, run = expected[key], selected[key]
        seed = int(spec["seed"])
        set_random_seed(seed)
        config = load_config(REPO_ROOT / spec["config"])
        loaders, _ = _build_loaders(config, REPO_ROOT, seed)
        model = build_model(config["model"]).to(device)
        checkpoint = torch.load(
            Path(run["run_dir"]) / "best.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()
        cc_ratios: list[np.ndarray] = []
        cv_ratios: list[np.ndarray] = []
        cc_entropies: list[np.ndarray] = []
        cv_entropies: list[np.ndarray] = []
        cc_maxima: list[np.ndarray] = []
        cv_maxima: list[np.ndarray] = []
        sample_count = 0
        with torch.no_grad():
            for batch in loaders["test"]:
                inputs = {
                    name: batch[name].to(device, non_blocking=device.type == "cuda")
                    for name in PHASE_KEYS
                }
                output = model.forward_with_aux(**inputs)
                cc_ratios.append(output["cc_cross_contribution_ratio"].cpu().numpy())
                cv_ratios.append(output["cv_cross_contribution_ratio"].cpu().numpy())
                cc_entropy, cc_maximum = _attention_stats(
                    output["latent_cc_attention"], inputs["cc_mask"]
                )
                cv_entropy, cv_maximum = _attention_stats(
                    output["latent_cv_attention"], inputs["cv_mask"]
                )
                cc_entropies.append(cc_entropy)
                cv_entropies.append(cv_entropy)
                cc_maxima.append(cc_maximum)
                cv_maxima.append(cv_maximum)
                sample_count += int(inputs["cc_signal"].size(0))
        row = {"dataset": spec["family"], "seed": seed, "n_samples": sample_count}
        row.update(_ratio_stats(cc_ratios, "cc_cross_ratio"))
        row.update(_ratio_stats(cv_ratios, "cv_cross_ratio"))
        row.update(
            {
                "cc_attention_entropy_normalized_mean": float(np.concatenate(cc_entropies).mean()),
                "cv_attention_entropy_normalized_mean": float(np.concatenate(cv_entropies).mean()),
                "cc_attention_max_mean": float(np.concatenate(cc_maxima).mean()),
                "cv_attention_max_mean": float(np.concatenate(cv_maxima).mean()),
                "cc_cross_scale": float(model.last_cc_cross_scale),
                "cv_cross_scale": float(model.last_cv_cross_scale),
                "cc_read_cv_out_proj_weight_norm": float(model.cc_read_cv.out_proj.weight.norm()),
                "cv_read_cc_out_proj_weight_norm": float(model.cv_read_cc.out_proj.weight.norm()),
                "run_dir": run["run_dir"],
            }
        )
        rows.append(row)
        print(
            f"[diagnostics] dataset={spec['family']} seed={seed} "
            f"cc_ratio={row['cc_cross_ratio_mean']:.4f} "
            f"cv_ratio={row['cv_cross_ratio_mean']:.4f}",
            flush=True,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{experiment}_diagnostics.csv"
    json_path = output_dir / f"{experiment}_diagnostics.json"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    json_path.write_text(
        json.dumps({"experiment": experiment, "rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
