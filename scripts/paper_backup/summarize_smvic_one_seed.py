#!/usr/bin/env python3
"""Summarize the one-seed SMVIC MLP/Raw-Mamba/BiContext comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402


CONFIG_ROOT = REPO_ROOT / "configs/paper_backup/e4_industrial_external/smvic"
DEFAULT_RESULT_ROOT = REPO_ROOT / "outputs/Paper-Backup/E4-SMVIC-Curated-OneSeed"
MODEL_NAMES = {
    "Final-PINN4SOH-like-MLP": "PINN4SOH-like MLP",
    "Final-Raw-Vanilla-Mamba": "Raw Vanilla Mamba",
    "Final-Ours-BiContext-Mamba": "Ours bicontext",
}
DOMAIN_COUNT = 6
PROTOCOLS_PER_DOMAIN = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {name}: {value!r}")
    return number


def _expected(seed: int) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    output = {}
    for path in sorted(CONFIG_ROOT.glob("*/*.json")):
        config = load_config(path)
        values = config["output"]
        domain_id = str(config["data"]["domain_id"])
        protocols = (
            ("test_seed420", "test_seed421")
            if domain_id == "smvic_type3_108ah"
            else ("test_cell01", "test_cell02")
        )
        for protocol_id in protocols:
            data_id = f"{domain_id}__{protocol_id}"
            key = (
                str(values["experiment_id"]),
                str(values["model_id"]),
                data_id,
                int(seed),
            )
            if key in output:
                raise ValueError(f"Duplicate SMVIC task config: {key}")
            output[key] = {
                "domain_id": domain_id,
                "protocol_id": protocol_id,
                "model": MODEL_NAMES.get(str(values["model_id"]), str(values["model_id"])),
                "config": str(path.relative_to(REPO_ROOT)),
                "split": f"splits/smvic/{domain_id}__{protocol_id}.json",
            }
    expected_count = DOMAIN_COUNT * len(MODEL_NAMES) * PROTOCOLS_PER_DOMAIN
    if len(output) != expected_count:
        raise ValueError(
            f"Expected {expected_count} SMVIC model/domain/protocol tasks, "
            f"found {len(output)}"
        )
    return output


def _discover(root: Path, seed: int) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for metrics_path in sorted(root.rglob("test_metrics.json")):
        run_dir = metrics_path.parent
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        if int(manifest.get("seed", -1)) != int(seed):
            continue
        if str(manifest.get("status", "completed")) != "completed":
            continue
        key = (
            str(manifest.get("experiment_id", "")),
            str(manifest.get("model_id", "")),
            str(manifest.get("data_id", "")),
            int(seed),
        )
        grouped.setdefault(key, []).append({
            "metrics": _read_json(metrics_path),
            "run_dir": str(run_dir),
            "modified_ns": max(metrics_path.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns),
        })
    return {
        key: max(values, key=lambda item: (int(item["modified_ns"]), item["run_dir"]))
        for key, values in grouped.items()
    }


def _metric_row(spec: Mapping[str, Any], run: Mapping[str, Any], seed: int) -> dict[str, Any]:
    metrics = run["metrics"]
    macro = metrics.get("battery_macro", {})
    macro_mape = macro.get("mape")
    if macro_mape is None:
        values = [
            float(item["mape"])
            for item in metrics.get("per_battery", {}).values()
            if "mape" in item and math.isfinite(float(item["mape"]))
        ]
        macro_mape = sum(values) / len(values) if values else float("nan")
    return {
        "domain_id": spec["domain_id"],
        "protocol_id": spec["protocol_id"],
        "model": spec["model"],
        "seed": int(seed),
        "mape_percent": 100.0 * _finite(metrics["mape"], "mape"),
        "rmse_soh_percent": 100.0 * _finite(metrics["rmse"], "rmse"),
        "battery_macro_mape_percent": 100.0 * _finite(macro_mape, "battery_macro.mape"),
        "battery_macro_rmse_soh_percent": 100.0 * _finite(macro["rmse"], "battery_macro.rmse"),
        "n_cycles": int(metrics.get("n_cycles", 0)),
        "n_batteries": int(metrics.get("n_batteries", 0)),
        "run_dir": str(run["run_dir"]),
        "config": spec["config"],
        "split": spec["split"],
    }


def _average_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["domain_id"]), str(row["model"])), []).append(row)
    output = []
    metric_names = (
        "mape_percent",
        "rmse_soh_percent",
        "battery_macro_mape_percent",
        "battery_macro_rmse_soh_percent",
    )
    for (domain_id, model), values in sorted(grouped.items()):
        if len(values) != 2:
            raise ValueError(f"Expected two SMVIC evaluations for {domain_id}/{model}, got {len(values)}")
        item = {
            "domain_id": domain_id,
            "model": model,
            "seed": int(seed),
            "evaluations": 2,
            "protocol_ids": ",".join(sorted(str(row["protocol_id"]) for row in values)),
            "total_test_cycles": sum(int(row["n_cycles"]) for row in values),
            "total_test_batteries": sum(int(row["n_batteries"]) for row in values),
        }
        for name in metric_names:
            item[name] = sum(float(row[name]) for row in values) / 2.0
        output.append(item)
    expected_count = DOMAIN_COUNT * len(MODEL_NAMES)
    if len(output) != expected_count:
        raise ValueError(
            f"Expected {expected_count} averaged domain/model rows, got {len(output)}"
        )
    return output


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "summaries" / f"seed_{args.seed}").resolve()
    expected = _expected(args.seed)
    discovered = _discover(root, args.seed)
    missing = [key for key in expected if key not in discovered]
    if missing:
        raise ValueError(f"Missing {len(missing)} SMVIC completed tasks: {missing}")
    detail_rows = [
        _metric_row(expected[key], discovered[key], args.seed)
        for key in sorted(expected, key=lambda item: (item[2], item[1]))
    ]
    rows = _average_rows(detail_rows, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    csv_path = output_dir / "smvic_one_seed_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    detail_path = output_dir / "smvic_one_seed_details.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    payload = {
        "status": "PASS",
        "seed": args.seed,
        "result_root": str(root),
        "tasks": len(detail_rows),
        "averaged_rows": rows,
        "details": detail_rows,
    }
    (output_dir / "smvic_one_seed_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    table = [
        "| Domain | Model | Eval mean | MAPE (%) | RMSE (%SOH) | Battery-macro MAPE (%) | Battery-macro RMSE (%SOH) | Total test cycles |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['domain_id']} | {row['model']} | {row['evaluations']} | {row['mape_percent']:.4f} | "
            f"{row['rmse_soh_percent']:.4f} | {row['battery_macro_mape_percent']:.4f} | "
            f"{row['battery_macro_rmse_soh_percent']:.4f} | {row['total_test_cycles']} |"
        )
    (output_dir / "smvic_one_seed_metrics.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "seed": args.seed,
        "tasks": len(detail_rows),
        "averaged_rows": len(rows),
        "csv": str(csv_path),
        "details_csv": str(detail_path),
        "markdown": str(output_dir / "smvic_one_seed_metrics.md"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
