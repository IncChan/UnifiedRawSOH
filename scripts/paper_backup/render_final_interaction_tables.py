#!/usr/bin/env python3
"""Render the formal five-seed battery-macro MAPE/RMSE tables as Markdown."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASETS = ("XJTU", "MIT", "LISHEN", "CATL", "EVE")
FAMILY = {
    "xjtu": "XJTU", "mit": "MIT", "smarthealth_lishen40": "LISHEN",
    "smarthealth_catl280": "CATL", "smarthealth_eve280": "EVE",
}
E1_MODELS = (
    ("PINN4SOH-like MLP", "Final-PINN4SOH-like-MLP"),
    ("Raw CNN", "Final-Raw-CNN"),
    ("Raw LSTM", "Final-Raw-LSTM"),
    ("Raw Transformer", "Final-Raw-Transformer"),
    ("Raw Vanilla Mamba", "Final-Raw-Vanilla-Mamba"),
    ("Raw CC Vanilla Mamba", "Final-Raw-CC-Vanilla-Mamba"),
    ("Raw CV Vanilla Mamba", "Final-Raw-CV-Vanilla-Mamba"),
    ("Raw Dual Vanilla Mamba", "Final-Raw-Dual-Vanilla-Mamba"),
    ("Ours", "Final-Ours-Interaction-Mamba"),
)
E2_MODELS = (
    ("FULL Vanilla Mamba", "Final-FULL-Vanilla-Mamba"),
    ("Raw Dual Vanilla Mamba", "Final-Raw-Dual-Vanilla-Mamba"),
    ("Ours", "Final-Ours-Interaction-Mamba"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("e1", "e2"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["model"], FAMILY.get(row["dataset"], row["dataset"])): row for row in rows}
    models = E1_MODELS if args.experiment == "e1" else E2_MODELS
    lines = [
        "| MAPE / RMSE (%) | " + " | ".join(DATASETS) + " |",
        "| --- | " + " | ".join("---" for _ in DATASETS) + " |",
    ]
    for label, model_id in models:
        cells = []
        for dataset in DATASETS:
            row = indexed.get((model_id, dataset))
            if row is None:
                cells.append("—")
                continue
            cells.append(
                f"{float(row['battery_macro_mape_percent_mean']):.5f} ± "
                f"{float(row['battery_macro_mape_percent_std']):.5f} / "
                f"{float(row['battery_macro_rmse_soh_percent_mean']):.5f} ± "
                f"{float(row['battery_macro_rmse_soh_percent_std']):.5f}"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
