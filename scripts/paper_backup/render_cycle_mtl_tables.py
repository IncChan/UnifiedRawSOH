#!/usr/bin/env python3
"""Render Cycle MTL macro metrics and paired deltas against plain BiContext."""

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
MTL_MODEL = "Final-Ours-BiContext-Cycle-MTL"
BASE_MODEL = "Final-Ours-BiContext-Mamba"


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


def _write_table(path: Path, rows: list[dict[str, str]], models) -> None:
    indexed = {
        (row["model"], FAMILY.get(row["dataset"], row["dataset"])): row
        for row in rows
    }
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


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-summary-dir", required=True, type=Path)
    parser.add_argument("--reference-summary-dir", type=Path)
    args = parser.parse_args()
    new_dir = args.new_summary_dir.resolve()
    new_mean = _read(new_dir / "e1_bicontext_cycle_mtl_5seed_metrics_mean_std.csv")
    new_seed = _read(new_dir / "e1_bicontext_cycle_mtl_5seed_metrics_per_seed.csv")
    standalone = new_dir / "e1_bicontext_cycle_mtl_5seed_macro_table.md"
    _write_table(standalone, new_mean, (("Ours BiContext Cycle MTL", MTL_MODEL),))
    print(standalone)

    if args.reference_summary_dir is None:
        return 0
    reference_dir = args.reference_summary_dir.resolve()
    reference_mean_path = reference_dir / "e1_bicontext_5seed_metrics_mean_std.csv"
    reference_seed_path = reference_dir / "e1_bicontext_5seed_metrics_per_seed.csv"
    if not reference_mean_path.is_file() or not reference_seed_path.is_file():
        print("[warning] plain BiContext summary unavailable; skipped paired comparison")
        return 0
    reference_mean = _read(reference_mean_path)
    reference_seed = _read(reference_seed_path)
    comparison = new_dir / "e1_bicontext_cycle_mtl_vs_bicontext_macro_table.md"
    _write_table(
        comparison,
        reference_mean + new_mean,
        (("Ours BiContext", BASE_MODEL), ("Ours BiContext Cycle MTL", MTL_MODEL)),
    )

    mtl_index = {
        (row["dataset"], int(row["seed"])): row
        for row in new_seed
        if row["model"] == MTL_MODEL and row["dataset"] != "ALL_DATASETS_MACRO"
    }
    base_index = {
        (row["dataset"], int(row["seed"])): row
        for row in reference_seed
        if row["model"] == BASE_MODEL and row["dataset"] != "ALL_DATASETS_MACRO"
    }
    if set(mtl_index) != set(base_index):
        raise ValueError("Cycle MTL and BiContext do not have identical dataset/seed coverage")
    paired = []
    for key in sorted(mtl_index):
        mtl, base = mtl_index[key], base_index[key]
        if (mtl["n_cycles"], mtl["n_batteries"]) != (base["n_cycles"], base["n_batteries"]):
            raise ValueError(f"Test coverage differs for {key}")
        paired.append({
            "dataset": key[0],
            "seed": key[1],
            "cycle_mtl_minus_bicontext_macro_mape_percent": float(mtl["battery_macro_mape_percent"]) - float(base["battery_macro_mape_percent"]),
            "cycle_mtl_minus_bicontext_macro_rmse_percent": float(mtl["battery_macro_rmse_soh_percent"]) - float(base["battery_macro_rmse_soh_percent"]),
        })
    paired_columns = (
        "dataset", "seed",
        "cycle_mtl_minus_bicontext_macro_mape_percent",
        "cycle_mtl_minus_bicontext_macro_rmse_percent",
    )
    paired_path = new_dir / "e1_cycle_mtl_vs_bicontext_per_seed.csv"
    _write_csv(paired_path, paired, paired_columns)

    aggregate = []
    for dataset in sorted({row["dataset"] for row in paired}):
        selected = [row for row in paired if row["dataset"] == dataset]
        mape = [row["cycle_mtl_minus_bicontext_macro_mape_percent"] for row in selected]
        rmse = [row["cycle_mtl_minus_bicontext_macro_rmse_percent"] for row in selected]
        aggregate.append({
            "dataset": dataset,
            "seed_count": len(selected),
            "macro_mape_delta_mean": mean(mape),
            "macro_mape_delta_std": stdev(mape) if len(mape) > 1 else 0.0,
            "macro_mape_win_seeds": sum(value < 0.0 for value in mape),
            "macro_rmse_delta_mean": mean(rmse),
            "macro_rmse_delta_std": stdev(rmse) if len(rmse) > 1 else 0.0,
            "macro_rmse_win_seeds": sum(value < 0.0 for value in rmse),
        })
    aggregate_columns = (
        "dataset", "seed_count", "macro_mape_delta_mean", "macro_mape_delta_std",
        "macro_mape_win_seeds", "macro_rmse_delta_mean", "macro_rmse_delta_std",
        "macro_rmse_win_seeds",
    )
    aggregate_path = new_dir / "e1_cycle_mtl_vs_bicontext_mean_std.csv"
    _write_csv(aggregate_path, aggregate, aggregate_columns)
    for path in (comparison, paired_path, aggregate_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
