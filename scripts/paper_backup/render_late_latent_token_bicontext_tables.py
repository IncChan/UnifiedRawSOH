#!/usr/bin/env python3
"""Compare Late Latent-Token BiContext with Raw Dual and Mean-BiContext."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


EXPERIMENT = "e1_late_latent_token_bicontext_5seed"
LATE_MODEL = "Ours-Late-LatentToken-BiContext"
RAW_MODEL = "Final-Raw-Dual-Vanilla-Mamba"
MEAN_MODEL = "Final-Ours-BiContext-Mamba"
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


def _render_table(path: Path, rows: list[dict[str, str]]) -> None:
    indexed = {
        (row["model"], FAMILY.get(row["dataset"], row["dataset"])): row
        for row in rows
    }
    models = (
        ("Raw Dual Vanilla Mamba", RAW_MODEL),
        ("Current Mean-BiContext", MEAN_MODEL),
        ("Late Latent-Token BiContext", LATE_MODEL),
    )
    lines = [
        "| Battery-macro MAPE / RMSE (%) | " + " | ".join(DATASETS) + " |",
        "| --- | " + " | ".join("---" for _ in DATASETS) + " |",
    ]
    for label, model in models:
        lines.append(
            f"| {label} | "
            + " | ".join(_cell(indexed.get((model, dataset))) for dataset in DATASETS)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_index(rows: list[dict[str, str]], model: str) -> dict[tuple[str, int], dict[str, str]]:
    return {
        (FAMILY.get(row["dataset"], row["dataset"]), int(row["seed"])): row
        for row in rows
        if row["model"] == model and row["dataset"] != "ALL_DATASETS_MACRO"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-summary-dir", required=True, type=Path)
    parser.add_argument("--raw-dual-summary-dir", required=True, type=Path)
    parser.add_argument("--mean-bicontext-summary-dir", required=True, type=Path)
    args = parser.parse_args()
    new_dir = args.new_summary_dir.resolve()
    raw_dir = args.raw_dual_summary_dir.resolve()
    mean_dir = args.mean_bicontext_summary_dir.resolve()

    late_mean = _read(new_dir / f"{EXPERIMENT}_metrics_mean_std.csv")
    late_seed = _read(new_dir / f"{EXPERIMENT}_metrics_per_seed.csv")
    raw_mean = _read(raw_dir / "e1_final_interaction_5seed_metrics_mean_std.csv")
    raw_seed = _read(raw_dir / "e1_final_interaction_5seed_metrics_per_seed.csv")
    mean_mean = _read(mean_dir / "e1_bicontext_5seed_metrics_mean_std.csv")
    mean_seed = _read(mean_dir / "e1_bicontext_5seed_metrics_per_seed.csv")

    table_path = new_dir / f"{EXPERIMENT}_three_model_macro_table.md"
    _render_table(table_path, raw_mean + mean_mean + late_mean)

    late_index = _seed_index(late_seed, LATE_MODEL)
    raw_index = _seed_index(raw_seed, RAW_MODEL)
    mean_index = _seed_index(mean_seed, MEAN_MODEL)
    if set(late_index) != set(raw_index) or set(late_index) != set(mean_index):
        raise ValueError("The three models do not have identical dataset/seed coverage")

    paired = []
    for key in sorted(late_index):
        late, raw, current_mean = late_index[key], raw_index[key], mean_index[key]
        coverage = (late["n_cycles"], late["n_batteries"])
        if coverage != (raw["n_cycles"], raw["n_batteries"]):
            raise ValueError(f"Late/Raw test coverage differs for {key}")
        if coverage != (current_mean["n_cycles"], current_mean["n_batteries"]):
            raise ValueError(f"Late/Mean-BiContext test coverage differs for {key}")
        paired.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "late_minus_raw_macro_mape_percent": float(late["battery_macro_mape_percent"]) - float(raw["battery_macro_mape_percent"]),
                "late_minus_raw_macro_rmse_percent": float(late["battery_macro_rmse_soh_percent"]) - float(raw["battery_macro_rmse_soh_percent"]),
                "late_minus_mean_bicontext_macro_mape_percent": float(late["battery_macro_mape_percent"]) - float(current_mean["battery_macro_mape_percent"]),
                "late_minus_mean_bicontext_macro_rmse_percent": float(late["battery_macro_rmse_soh_percent"]) - float(current_mean["battery_macro_rmse_soh_percent"]),
                "n_cycles": int(late["n_cycles"]),
                "n_batteries": int(late["n_batteries"]),
            }
        )
    paired_columns = tuple(paired[0])
    paired_path = new_dir / f"{EXPERIMENT}_paired_deltas_per_seed.csv"
    _write_csv(paired_path, paired, paired_columns)

    aggregate = []
    delta_columns = paired_columns[2:6]
    for dataset in sorted({row["dataset"] for row in paired}):
        selected = [row for row in paired if row["dataset"] == dataset]
        row = {"dataset": dataset, "seed_count": len(selected)}
        for column in delta_columns:
            values = [float(item[column]) for item in selected]
            row[f"{column}_mean"] = mean(values)
            row[f"{column}_std"] = stdev(values) if len(values) > 1 else 0.0
            row[f"{column}_win_seeds"] = sum(value < 0.0 for value in values)
        aggregate.append(row)
    aggregate_columns = tuple(aggregate[0])
    aggregate_path = new_dir / f"{EXPERIMENT}_paired_deltas_mean_std.csv"
    _write_csv(aggregate_path, aggregate, aggregate_columns)
    for path in (table_path, paired_path, aggregate_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
