#!/usr/bin/env python3
"""Aggregate completed Paper-Backup runs without treating blocked jobs as zero."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from UnifiedRawSOH.evaluation.paper_backup.aggregation import aggregate_seed_metrics, metrics_from_rows  # noqa: E402
from UnifiedRawSOH.evaluation.paper_backup.comparisons import e2_comparisons, e3_strategy_comparison, view_coverage  # noqa: E402


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _family(manifest: dict, data_id: str) -> str:
    loader = manifest.get("loader_info", {})
    if loader.get("domain_id"):
        return str(loader["domain_id"])
    for suffix in ("_terminal_ours", "_terminal_cc", "_terminal_cv", "_terminal", "_full"):
        if data_id.endswith(suffix):
            return data_id[: -len(suffix)]
    return data_id.split("_strategy_", 1)[0].removesuffix("_pooled")


def discover_runs(root: Path) -> list[dict]:
    runs = []
    for metrics_path in sorted(root.rglob("test_metrics.json")):
        run_dir = metrics_path.parent
        manifest_path = run_dir / "run_manifest.json"
        predictions_path = run_dir / "predictions.json"
        if not manifest_path.is_file() or not predictions_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        rows = _read_json(predictions_path)
        metrics = metrics_from_rows(rows)
        metrics["status"] = manifest.get("status", "completed")
        runs.append(
            {
                "run_dir": str(run_dir),
                "experiment_id": str(manifest.get("experiment_id", "")),
                "model_id": str(manifest.get("model_id", "")),
                "data_id": str(manifest.get("data_id", "")),
                "family": _family(manifest, str(manifest.get("data_id", ""))),
                "seed": int(manifest.get("seed", 0)),
                "metrics": metrics,
                "rows": rows,
            }
        )
    return runs


def _seed_summary(items: list[dict]) -> dict:
    seed_metrics = {str(item["seed"]): item["metrics"] for item in items}
    return {
        "runs": [{"run_dir": item["run_dir"], "seed": item["seed"], "metrics": item["metrics"]} for item in items],
        "seed_aggregate": aggregate_seed_metrics(seed_metrics),
    }


def summarize(runs: list[dict]) -> dict:
    output = {
        "status": "ok",
        "completed_run_count": len(runs),
        "blocked_or_missing_are_not_zero": True,
        "e1_main_estimation": {},
        "e2_charging_information": {},
        "e3_strategy_pooling": {},
    }
    for experiment_id in output:
        if not experiment_id.startswith("e"):
            continue
        selected = [item for item in runs if item["experiment_id"] == experiment_id]
        if experiment_id == "e1_main_estimation":
            groups = defaultdict(list)
            for item in selected:
                groups[(item["family"], item["model_id"])].append(item)
            output[experiment_id] = {f"{family}::{model}": _seed_summary(values) for (family, model), values in sorted(groups.items())}
        elif experiment_id == "e2_charging_information":
            groups = defaultdict(list)
            for item in selected:
                groups[(item["family"], item["seed"])].append(item)
            e2_output = {}
            for (family, seed), values in sorted(groups.items()):
                named = {item["model_id"].replace("Full-", "full_").replace("Terminal-", "terminal_").replace("VanillaMamba", "vanilla").replace("Ours", "ours").replace("CCOnly-Mamba", "cc_only").replace("CVOnly-Mamba", "cv_only"): item["rows"] for item in values}
                # Normalize the two one-phase names to the keys consumed by
                # e2_comparisons; unrecognized names simply remain audit data.
                named = {
                    ("full_vanilla" if key == "full_vanilla" else
                     "terminal_vanilla" if key == "terminal_vanilla" else
                     "terminal_ours" if key == "terminal_ours" else key): rows
                    for key, rows in named.items()
                }
                e2_output[f"{family}::seed_{seed}"] = {
                    "metrics": {item["model_id"]: item["metrics"] for item in values},
                    "paired": e2_comparisons(named),
                    "view_coverage": view_coverage(named.get("terminal_vanilla", []), named.get("full_vanilla", [])),
                }
            output[experiment_id] = e2_output
        elif experiment_id == "e3_strategy_pooling":
            groups = defaultdict(list)
            for item in selected:
                groups[(item["family"], item["seed"])].append(item)
            e3_output = {}
            for (family, seed), values in sorted(groups.items()):
                pooled = next((item for item in values if item["data_id"].endswith("_pooled")), None)
                comparisons = {}
                if pooled is not None:
                    for item in values:
                        if item is pooled or "_strategy_" not in item["data_id"]:
                            continue
                        strategy = item["data_id"].split("_strategy_", 1)[1]
                        comparisons[strategy] = e3_strategy_comparison(item["rows"], pooled["rows"])
                e3_output[f"{family}::seed_{seed}"] = {
                    "specific_and_pooled_metrics": {item["data_id"]: item["metrics"] for item in values},
                    "paired": comparisons,
                }
            output[experiment_id] = e3_output
    return output


def main() -> int:
    parser = argparse.ArgumentParser("Summarize Paper-Backup output")
    parser.add_argument("--root", default=str(REPO_ROOT / "outputs/Paper-Backup"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    result = summarize(discover_runs(root))
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
