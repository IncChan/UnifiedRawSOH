#!/usr/bin/env python3
"""Render standalone and archived-E1 comparison tables for BiContext."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


DATASETS = ("XJTU", "MIT", "LISHEN", "CATL", "EVE")
FAMILY = {
    "xjtu": "XJTU",
    "mit": "MIT",
    "smarthealth_lishen40": "LISHEN",
    "smarthealth_catl280": "CATL",
    "smarthealth_eve280": "EVE",
}
MODELS = (
    ("Feature MLP (PINN4SOH F-only structure)", "Final-Feature-MLP-PINN4SOH-Structure"),
    ("Raw CNN", "Final-Raw-CNN"),
    ("Raw LSTM", "Final-Raw-LSTM"),
    ("Raw Transformer", "Final-Raw-Transformer"),
    ("Raw Vanilla Mamba", "Final-Raw-Vanilla-Mamba"),
    ("Raw CC Vanilla Mamba", "Final-Raw-CC-Vanilla-Mamba"),
    ("Raw CV Vanilla Mamba", "Final-Raw-CV-Vanilla-Mamba"),
    ("Raw Dual Vanilla Mamba", "Final-Raw-Dual-Vanilla-Mamba"),
    ("Ours Interaction", "Final-Ours-Interaction-Mamba"),
    ("Ours BiContext", "Final-Ours-BiContext-Mamba"),
)
NEW_MODEL = "Final-Ours-BiContext-Mamba"
REFERENCE_MODEL = "Final-Raw-Dual-Vanilla-Mamba"


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _cell(row: dict[str, str] | None) -> str:
    if row is None:
        return "—"
    return (
        f"{float(row['battery_macro_mape_percent_mean']):.5f} ± "
        f"{float(row['battery_macro_mape_percent_std']):.5f} / "
        f"{float(row['battery_macro_rmse_soh_percent_mean']):.5f} ± "
        f"{float(row['battery_macro_rmse_soh_percent_std']):.5f}"
    )


def _render(path: Path, rows: list[dict[str, str]], models) -> None:
    indexed = {(row["model"], FAMILY.get(row["dataset"], row["dataset"])): row for row in rows}
    lines = [
        "| MAPE / RMSE (%) | " + " | ".join(DATASETS) + " |",
        "| --- | " + " | ".join("---" for _ in DATASETS) + " |",
    ]
    for label, model in models:
        lines.append(
            f"| {label} | "
            + " | ".join(_cell(indexed.get((model, dataset))) for dataset in DATASETS)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-summary-dir", type=Path, required=True)
    parser.add_argument("--reference-summary-dir", type=Path)
    args = parser.parse_args()
    new_dir = args.new_summary_dir.resolve()
    new_mean = _read(new_dir / "e1_bicontext_5seed_metrics_mean_std.csv")
    new_seed = _read(new_dir / "e1_bicontext_5seed_metrics_per_seed.csv")

    standalone = new_dir / "e1_bicontext_5seed_macro_table.md"
    _render(standalone, new_mean, (("Ours BiContext", NEW_MODEL),))
    print(standalone)
    if args.reference_summary_dir is None:
        return 0
    reference_dir = args.reference_summary_dir.resolve()
    reference_mean_path = reference_dir / "e1_final_interaction_5seed_metrics_mean_std.csv"
    reference_seed_path = reference_dir / "e1_final_interaction_5seed_metrics_per_seed.csv"
    if not reference_mean_path.is_file() or not reference_seed_path.is_file():
        print(
            "[warning] archived E1 summary is unavailable; wrote the standalone "
            "BiContext table but skipped comparison outputs"
        )
        return 0
    reference_mean = _read(reference_mean_path)
    reference_seed = _read(reference_seed_path)
    comparison = new_dir / "e1_bicontext_5seed_comparison_macro_table.md"
    _render(comparison, reference_mean + new_mean, MODELS)

    new_index = {
        (row["dataset"], int(row["seed"])): row
        for row in new_seed
        if row["model"] == NEW_MODEL and row["dataset"] != "ALL_DATASETS_MACRO"
    }
    ref_index = {
        (row["dataset"], int(row["seed"])): row
        for row in reference_seed
        if row["model"] == REFERENCE_MODEL and row["dataset"] != "ALL_DATASETS_MACRO"
    }
    if set(new_index) != set(ref_index):
        raise ValueError(
            "BiContext and archived Raw Dual do not have identical dataset/seed coverage: "
            f"new_only={sorted(set(new_index) - set(ref_index))}, "
            f"reference_only={sorted(set(ref_index) - set(new_index))}"
        )
    delta_rows = []
    for key in sorted(new_index):
        new, ref = new_index[key], ref_index[key]
        if (new["n_cycles"], new["n_batteries"]) != (ref["n_cycles"], ref["n_batteries"]):
            raise ValueError(f"Test coverage differs for {key}")
        delta_rows.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "bicontext_minus_rawdual_macro_mape_percent": float(new["battery_macro_mape_percent"]) - float(ref["battery_macro_mape_percent"]),
                "bicontext_minus_rawdual_macro_rmse_percent": float(new["battery_macro_rmse_soh_percent"]) - float(ref["battery_macro_rmse_soh_percent"]),
                "n_cycles": int(new["n_cycles"]),
                "n_batteries": int(new["n_batteries"]),
            }
        )
    delta_columns = (
        "dataset", "seed",
        "bicontext_minus_rawdual_macro_mape_percent",
        "bicontext_minus_rawdual_macro_rmse_percent",
        "n_cycles", "n_batteries",
    )
    _write_csv(new_dir / "e1_bicontext_vs_rawdual_per_seed.csv", delta_rows, delta_columns)

    aggregate_rows = []
    for dataset in sorted({row["dataset"] for row in delta_rows}):
        selected = [row for row in delta_rows if row["dataset"] == dataset]
        mape = [float(row["bicontext_minus_rawdual_macro_mape_percent"]) for row in selected]
        rmse = [float(row["bicontext_minus_rawdual_macro_rmse_percent"]) for row in selected]
        aggregate_rows.append(
            {
                "dataset": dataset,
                "seed_count": len(selected),
                "macro_mape_delta_mean": mean(mape),
                "macro_mape_delta_std": stdev(mape) if len(mape) > 1 else 0.0,
                "macro_mape_win_seeds": sum(value < 0.0 for value in mape),
                "macro_rmse_delta_mean": mean(rmse),
                "macro_rmse_delta_std": stdev(rmse) if len(rmse) > 1 else 0.0,
                "macro_rmse_win_seeds": sum(value < 0.0 for value in rmse),
            }
        )
    aggregate_columns = (
        "dataset", "seed_count",
        "macro_mape_delta_mean", "macro_mape_delta_std", "macro_mape_win_seeds",
        "macro_rmse_delta_mean", "macro_rmse_delta_std", "macro_rmse_win_seeds",
    )
    _write_csv(new_dir / "e1_bicontext_vs_rawdual_mean_std.csv", aggregate_rows, aggregate_columns)
    for path in (
        comparison,
        new_dir / "e1_bicontext_vs_rawdual_per_seed.csv",
        new_dir / "e1_bicontext_vs_rawdual_mean_std.csv",
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
