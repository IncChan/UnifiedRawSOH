#!/usr/bin/env python3
"""Render the formal four-model table for the ReZero late-token follow-up."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPERIMENT = "e1_late_latent_token_bicontext_rezero_5seed"
REZERO_MODEL = "Ours-Late-LatentToken-BiContext-PreNorm-ReZero"
RAW_MODEL = "Final-Raw-Dual-Vanilla-Mamba"
MEAN_MODEL = "Final-Ours-BiContext-Mamba"
LATE_MODEL = "Ours-Late-LatentToken-BiContext"
DATASETS = ("XJTU", "MIT", "LISHEN", "CATL", "EVE", "ALL_DATASETS_MACRO")
FAMILY = {
    "xjtu": "XJTU",
    "mit": "MIT",
    "smarthealth_lishen40": "LISHEN",
    "smarthealth_catl280": "CATL",
    "smarthealth_eve280": "EVE",
    "ALL_DATASETS_MACRO": "ALL_DATASETS_MACRO",
}


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cell(row: dict[str, str] | None) -> str:
    if row is None:
        return "—"
    return (
        f"{float(row['battery_macro_mape_percent_mean']):.5f} ± "
        f"{float(row['battery_macro_mape_percent_std']):.5f} / "
        f"{float(row['battery_macro_rmse_soh_percent_mean']):.5f} ± "
        f"{float(row['battery_macro_rmse_soh_percent_std']):.5f}"
    )


def _coverage(
    rows: list[dict[str, str]], model: str
) -> dict[tuple[str, int], tuple[int, int]]:
    return {
        (FAMILY.get(row["dataset"], row["dataset"]), int(row["seed"])): (
            int(row["n_cycles"]),
            int(row["n_batteries"]),
        )
        for row in rows
        if row["model"] == model and row["dataset"] != "ALL_DATASETS_MACRO"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-summary-dir", required=True, type=Path)
    parser.add_argument("--raw-dual-summary-dir", required=True, type=Path)
    parser.add_argument("--mean-bicontext-summary-dir", required=True, type=Path)
    parser.add_argument("--late-token-summary-dir", required=True, type=Path)
    args = parser.parse_args()

    sources = (
        (
            args.raw_dual_summary_dir.resolve(),
            "e1_final_interaction_5seed",
            RAW_MODEL,
            "Raw Dual Vanilla Mamba",
        ),
        (
            args.mean_bicontext_summary_dir.resolve(),
            "e1_bicontext_5seed",
            MEAN_MODEL,
            "Current Mean-BiContext",
        ),
        (
            args.late_token_summary_dir.resolve(),
            "e1_late_latent_token_bicontext_5seed",
            LATE_MODEL,
            "Late Latent-Token BiContext",
        ),
        (
            args.new_summary_dir.resolve(),
            EXPERIMENT,
            REZERO_MODEL,
            "Late Latent + PreNorm + ReZero",
        ),
    )
    mean_rows: list[dict[str, str]] = []
    coverages = {}
    labels = []
    for directory, experiment, model, label in sources:
        mean_rows.extend(_read(directory / f"{experiment}_metrics_mean_std.csv"))
        seed_rows = _read(directory / f"{experiment}_metrics_per_seed.csv")
        coverages[model] = _coverage(seed_rows, model)
        labels.append((label, model))
    reference = coverages[RAW_MODEL]
    for model, coverage in coverages.items():
        if coverage != reference:
            raise ValueError(
                f"Test coverage differs between {RAW_MODEL} and {model}: "
                f"reference={reference}, candidate={coverage}"
            )

    indexed = {
        (row["model"], FAMILY.get(row["dataset"], row["dataset"])): row
        for row in mean_rows
    }
    lines = [
        "| Battery-macro MAPE / RMSE (%) | " + " | ".join(DATASETS) + " |",
        "| --- | " + " | ".join("---" for _ in DATASETS) + " |",
    ]
    for label, model in labels:
        lines.append(
            f"| {label} | "
            + " | ".join(_cell(indexed.get((model, dataset))) for dataset in DATASETS)
            + " |"
        )
    output = args.new_summary_dir.resolve() / f"{EXPERIMENT}_four_model_macro_table.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
