#!/usr/bin/env python3
"""Write compact, seed-aware Paper-Backup result tables.

The summarizer intentionally does not select runs by ``runtime_*`` directory.
For each experiment/model/data/seed task it selects the newest completed run,
records duplicate resolution in a small status file, and refuses to publish a
formal table until the configured experiment matrix is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from UnifiedRawSOH.evaluation.paper_backup.aggregation import metrics_from_rows  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402


SUMMARY_VERSION = "paper_backup_compact_metrics_v1"
EXPERIMENTS = {
    "e1": {
        "experiment_id": "e1_main_estimation",
        "config_dir": "configs/paper_backup/e1_main_estimation",
    },
    "e1_crate": {
        "experiment_id": "e1_shared_crate_fullvi",
        "config_dir": "configs/paper_backup/e1_shared_crate_fullvi",
    },
    "e1_crate_128x128": {
        "experiment_id": "e1_shared_crate_128x128",
        "config_dir": "configs/paper_backup/e1_shared_crate_128x128",
    },
    "e1_core3_128x128": {
        "experiment_id": "e1_shared_crate_128x128",
        "config_dir": "configs/paper_backup/e1_core3_128x128",
    },
    "e2": {
        "experiment_id": "e2_charging_information",
        "config_dir": "configs/paper_backup/e2_charging_information",
    },
    "e2_final_256budget": {
        "experiment_id": "e2_final_256budget",
        "config_dir": "configs/paper_backup/e2_final_256budget",
    },
    "e1_final_interaction_5seed": {
        "experiment_id": "e1_final_interaction_5seed",
        "config_dir": "configs/paper_backup/e1_final_interaction_5seed",
    },
    "e2_final_interaction_5seed": {
        "experiment_id": "e2_final_interaction_5seed",
        "config_dir": "configs/paper_backup/e2_final_interaction_5seed",
    },
    "e3": {
        "experiment_id": "e3_strategy_pooling",
        "config_dir": "configs/paper_backup/e3_strategy_pooling",
    },
}
METRIC_COLUMNS = (
    "mape_percent",
    "rmse_soh_percent",
    "battery_macro_mape_percent",
    "battery_macro_rmse_soh_percent",
)
PER_SEED_COLUMNS = (
    "experiment",
    "dataset",
    "strategy",
    "model",
    "data_id",
    "seed",
    *METRIC_COLUMNS,
    "n_cycles",
    "n_batteries",
)
MEAN_STD_COLUMNS = (
    "experiment",
    "dataset",
    "strategy",
    "model",
    "data_id",
    "seed_count",
    *tuple(
        column
        for metric in METRIC_COLUMNS
        for column in (f"{metric}_mean", f"{metric}_std")
    ),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _parse_seeds(value: str) -> tuple[int, ...]:
    parts = str(value).replace(",", " ").split()
    if not parts:
        raise argparse.ArgumentTypeError("at least one seed is required")
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be integers") from exc
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must not contain duplicates")
    return seeds


def _family(manifest: Mapping[str, Any], data_id: str) -> str:
    loader = manifest.get("loader_info", {})
    if isinstance(loader, Mapping):
        source = loader.get("source", {})
        if isinstance(source, Mapping) and source.get("domain_id"):
            return str(source["domain_id"])
        if loader.get("domain_id"):
            return str(loader["domain_id"])
    for suffix in (
        "_terminal_ours",
        "_terminal_cc",
        "_terminal_cv",
        "_terminal",
        "_full",
    ):
        if data_id.endswith(suffix):
            return data_id[: -len(suffix)]
    return data_id.split("_strategy_", 1)[0].removesuffix("_pooled")


def _task_key(item: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(item["experiment_id"]),
        str(item["model_id"]),
        str(item["data_id"]),
        int(item["seed"]),
    )


def discover_runs(root: Path) -> list[dict[str, Any]]:
    """Discover completed runs using compact metrics, not prediction payloads."""

    runs: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("test_metrics.json")):
        run_dir = metrics_path.parent
        manifest_path = run_dir / "run_manifest.json"
        resolved_config_path = run_dir / "resolved_config.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        if str(manifest.get("status", "completed")) != "completed":
            continue
        if resolved_config_path.is_file():
            resolved_config = _read_json(resolved_config_path)
            debug_samples = int(
                resolved_config.get("debug", {}).get("debug_num_samples", 0) or 0
            )
            if debug_samples > 0:
                continue
        data_id = str(manifest.get("data_id", ""))
        runs.append(
            {
                "run_dir": str(run_dir),
                "manifest_path": str(manifest_path),
                "metrics_path": str(metrics_path),
                "predictions_path": str(run_dir / "predictions.json"),
                "modified_ns": max(
                    manifest_path.stat().st_mtime_ns,
                    metrics_path.stat().st_mtime_ns,
                ),
                "experiment_id": str(manifest.get("experiment_id", "")),
                "model_id": str(manifest.get("model_id", "")),
                "data_id": data_id,
                "family": _family(manifest, data_id),
                "seed": int(manifest.get("seed", 0)),
                "metrics": _read_json(metrics_path),
            }
        )
    return runs


def select_latest_runs(
    runs: Iterable[Mapping[str, Any]], experiment_id: str
) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    """Choose the newest completed result when the same task was rerun."""

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in runs:
        if str(item.get("experiment_id")) == str(experiment_id):
            grouped[_task_key(item)].append(dict(item))
    selected: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    duplicate_resolutions: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda item: (int(item["modified_ns"]), str(item["run_dir"])),
        )
        selected[key] = ordered[-1]
        if len(ordered) > 1:
            duplicate_resolutions.append(
                {
                    "task": {
                        "experiment_id": key[0],
                        "model_id": key[1],
                        "data_id": key[2],
                        "seed": key[3],
                    },
                    "selected": ordered[-1]["run_dir"],
                    "ignored": [item["run_dir"] for item in ordered[:-1]],
                    "rule": "newest_completed_test_metrics_mtime",
                }
            )
    return selected, duplicate_resolutions


def expected_jobs(
    experiment: str, seeds: Iterable[int]
) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], list[str]]:
    """Build the expected task matrix directly from the checked-in configs."""

    definition = EXPERIMENTS[experiment]
    config_root = REPO_ROOT / definition["config_dir"]
    specs: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    config_paths = sorted(config_root.rglob("*.json"))
    for config_path in config_paths:
        config = load_config(config_path)
        output = config.get("output", {})
        experiment_id = str(output.get("experiment_id", ""))
        if experiment_id != definition["experiment_id"]:
            continue
        model_id = str(output.get("model_id", config.get("model", {}).get("type", "")))
        data_id = str(output.get("data_id", config.get("data", {}).get("domain_id", "")))
        family = str(
            config.get("experiment", {}).get(
                "family_id", config.get("data", {}).get("domain_id", "")
            )
        )
        strategy = str(config.get("experiment", {}).get("strategy_id", "all"))
        pooling_mode = str(config.get("experiment", {}).get("pooling_mode", ""))
        for seed in seeds:
            key = (experiment_id, model_id, data_id, int(seed))
            if key in specs:
                raise ValueError(f"Duplicate expected Paper-Backup task in configs: {key}")
            specs[key] = {
                "experiment": experiment,
                "experiment_id": experiment_id,
                "model_id": model_id,
                "data_id": data_id,
                "family": family,
                "strategy": strategy,
                "pooling_mode": pooling_mode,
                "seed": int(seed),
                "config": str(config_path.relative_to(REPO_ROOT)),
            }
    if not specs:
        raise ValueError(f"No configs found for {experiment} under {config_root}")
    return specs, [str(path.relative_to(REPO_ROOT)) for path in config_paths]


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric cannot be summarized: {value!r}")
    return result


def _macro_from_metric_groups(metrics: Mapping[str, Any], metric: str) -> float:
    values = []
    for item in metrics.get("per_battery", {}).values():
        if metric in item and math.isfinite(float(item[metric])):
            values.append(float(item[metric]))
    return mean(values) if values else float("nan")


def _metric_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    battery_macro = metrics.get("battery_macro", {})
    battery_mape = battery_macro.get("mape")
    if battery_mape is None:
        battery_mape = _macro_from_metric_groups(metrics, "mape")
    return {
        "mape_percent": 100.0 * _finite(metrics["mape"]),
        "rmse_soh_percent": 100.0 * _finite(metrics["rmse"]),
        "battery_macro_mape_percent": 100.0 * _finite(battery_mape),
        "battery_macro_rmse_soh_percent": 100.0
        * _finite(battery_macro["rmse"]),
    }


def _base_row(
    experiment: str,
    spec: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    model: str | None = None,
    strategy: str | None = None,
    data_id: str | None = None,
) -> dict[str, Any]:
    row = {
        "experiment": experiment.upper(),
        "dataset": str(spec["family"]),
        "strategy": str(strategy if strategy is not None else spec.get("strategy", "all")),
        "model": str(model if model is not None else spec["model_id"]),
        "data_id": str(data_id if data_id is not None else spec["data_id"]),
        "seed": int(spec["seed"]),
        "n_cycles": int(metrics.get("n_cycles", 0)),
        "n_batteries": int(metrics.get("n_batteries", 0)),
    }
    row.update(_metric_values(metrics))
    return row


def _prediction_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    result = metrics_from_rows(rows)
    result["battery_macro"]["mape"] = _macro_from_metric_groups(result, "mape")
    return result


def _paired_strategy_rows(
    specific_rows: Iterable[Mapping[str, Any]],
    pooled_rows: Iterable[Mapping[str, Any]],
    strategy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match E3 specific/pooled predictions on the same physical cycles."""

    def keyed(rows: Iterable[Mapping[str, Any]], *, filter_strategy: bool) -> dict[tuple[str, str], dict[str, Any]]:
        output: dict[tuple[str, str], dict[str, Any]] = {}
        for raw in rows:
            row = dict(raw)
            if filter_strategy and str(row.get("strategy_id")) != str(strategy):
                continue
            key = (str(row.get("battery_id")), str(row.get("cycle_id")))
            if key in output:
                raise ValueError(f"Duplicate E3 prediction cycle for {strategy}: {key}")
            output[key] = row
        return output

    specific = keyed(specific_rows, filter_strategy=False)
    pooled = keyed(pooled_rows, filter_strategy=True)
    common = sorted(set(specific) & set(pooled))
    if not common:
        raise ValueError(f"No matched E3 cycles for strategy {strategy}")
    left, right = [], []
    for key in common:
        lhs, rhs = specific[key], pooled[key]
        if not math.isclose(
            float(lhs["y_true"]), float(rhs["y_true"]), rel_tol=1e-5, abs_tol=1e-6
        ):
            raise ValueError(f"E3 labels differ for strategy={strategy}, cycle={key}")
        left.append(lhs)
        right.append(rhs)
    if len(common) != len(specific) or len(common) != len(pooled):
        raise ValueError(
            f"E3 prediction coverage differs for strategy={strategy}: "
            f"specific={len(specific)}, pooled={len(pooled)}, common={len(common)}"
        )
    return left, right


def _load_predictions(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(run["predictions_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"E3 summary requires predictions: {path}")
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Predictions must be a list: {path}")
    return [dict(item) for item in rows]


def per_seed_rows(
    experiment: str,
    expected: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
    selected: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create compact rows and optional E3 paired-difference rows."""

    if experiment != "e3":
        rows = [
            _base_row(experiment, expected[key], selected[key]["metrics"])
            for key in sorted(expected)
        ]
        return _append_dataset_macro_rows(rows), []

    by_family_seed: dict[tuple[str, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for key, spec in expected.items():
        by_family_seed[(str(spec["family"]), int(spec["seed"]))].append(
            (spec, selected[key])
        )
    rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for (family, seed), values in sorted(by_family_seed.items()):
        pooled = [(spec, run) for spec, run in values if spec["pooling_mode"] == "pooled"]
        specific = [(spec, run) for spec, run in values if spec["pooling_mode"] == "specific"]
        if len(pooled) != 1 or not specific:
            raise ValueError(
                f"E3 requires one pooled run and strategy-specific runs for {family}, seed={seed}"
            )
        pooled_spec, pooled_run = pooled[0]
        pooled_predictions = _load_predictions(pooled_run)
        for specific_spec, specific_run in sorted(specific, key=lambda item: str(item[0]["strategy"])):
            strategy = str(specific_spec["strategy"])
            matched_specific, matched_pooled = _paired_strategy_rows(
                _load_predictions(specific_run), pooled_predictions, strategy
            )
            specific_metrics = _prediction_metrics(matched_specific)
            pooled_metrics = _prediction_metrics(matched_pooled)
            specific_row = _base_row(
                experiment,
                specific_spec,
                specific_metrics,
                model="Ours-strategy-specific",
                strategy=strategy,
            )
            pooled_row = _base_row(
                experiment,
                pooled_spec,
                pooled_metrics,
                model="Ours-dataset-pooled",
                strategy=strategy,
                data_id=f"{pooled_spec['data_id']}::{strategy}",
            )
            rows.extend((specific_row, pooled_row))
            comparison_rows.append(
                {
                    "experiment": "E3",
                    "dataset": family,
                    "strategy": strategy,
                    "seed": seed,
                    "specific_mape_percent": specific_row["mape_percent"],
                    "pooled_mape_percent": pooled_row["mape_percent"],
                    "mape_delta_pooled_minus_specific_percent": pooled_row["mape_percent"]
                    - specific_row["mape_percent"],
                    "specific_rmse_soh_percent": specific_row["rmse_soh_percent"],
                    "pooled_rmse_soh_percent": pooled_row["rmse_soh_percent"],
                    "rmse_delta_pooled_minus_specific_soh_percent": pooled_row["rmse_soh_percent"]
                    - specific_row["rmse_soh_percent"],
                    "common_cycles": int(specific_metrics["n_cycles"]),
                }
            )
    return _append_strategy_macro_rows(rows), comparison_rows


def _mean_metric_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    experiment: str,
    dataset: str,
    strategy: str,
    model: str,
    data_id: str,
    seed: int,
) -> dict[str, Any]:
    values = list(rows)
    output: dict[str, Any] = {
        "experiment": experiment,
        "dataset": dataset,
        "strategy": strategy,
        "model": model,
        "data_id": data_id,
        "seed": seed,
        "n_cycles": sum(int(item.get("n_cycles", 0)) for item in values),
        "n_batteries": sum(int(item.get("n_batteries", 0)) for item in values),
    }
    for metric in METRIC_COLUMNS:
        output[metric] = mean(float(item[metric]) for item in values)
    return output


def _append_dataset_macro_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["experiment"]), str(row["model"]), int(row["seed"]))].append(row)
    output = list(rows)
    for (experiment, model, seed), values in sorted(grouped.items()):
        output.append(
            _mean_metric_rows(
                values,
                experiment=experiment,
                dataset="ALL_DATASETS_MACRO",
                strategy="all",
                model=model,
                data_id="ALL_DATASETS_MACRO",
                seed=seed,
            )
        )
    return sorted(output, key=lambda row: (row["dataset"], row["model"], row["strategy"], row["seed"]))


def _append_strategy_macro_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["experiment"]), str(row["dataset"]), str(row["model"]), int(row["seed"]))].append(row)
    output = list(rows)
    for (experiment, dataset, model, seed), values in sorted(grouped.items()):
        output.append(
            _mean_metric_rows(
                values,
                experiment=experiment,
                dataset=dataset,
                strategy="ALL_STRATEGIES_MACRO",
                model=model,
                data_id=f"{dataset}::ALL_STRATEGIES_MACRO",
                seed=seed,
            )
        )
    return sorted(output, key=lambda row: (row["dataset"], row["model"], row["strategy"], row["seed"]))


def aggregate_seed_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["experiment"]),
            str(row["dataset"]),
            str(row["strategy"]),
            str(row["model"]),
            str(row["data_id"]),
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        seeds = {int(item["seed"]) for item in values}
        if len(seeds) != len(values):
            raise ValueError(f"Duplicate seed while aggregating compact summary: {key}")
        summary: dict[str, Any] = {
            "experiment": key[0],
            "dataset": key[1],
            "strategy": key[2],
            "model": key[3],
            "data_id": key[4],
            "seed_count": len(values),
        }
        for metric in METRIC_COLUMNS:
            metric_values = [float(item[metric]) for item in values]
            summary[f"{metric}_mean"] = mean(metric_values)
            summary[f"{metric}_std"] = stdev(metric_values) if len(metric_values) > 1 else 0.0
        output.append(summary)
    return output


def _missing_payload(key: tuple[str, str, str, int], spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": key[1],
        "data_id": key[2],
        "dataset": spec["family"],
        "strategy": spec["strategy"],
        "seed": key[3],
        "config": spec["config"],
    }


E2_FINAL_COMPARISONS = (
    (
        "terminal_vanilla_minus_full_vanilla",
        "Terminal-VanillaMamba-Matched-SEP-128x128",
        "Full-VanillaMamba-Matched-256",
    ),
    (
        "ours_minus_terminal_vanilla",
        "Ours-FullVI-PointBridge-128x128",
        "Terminal-VanillaMamba-Matched-SEP-128x128",
    ),
    (
        "ours_minus_cc_only",
        "Ours-FullVI-PointBridge-128x128",
        "Ours-CC-Only-FullVI-128",
    ),
    (
        "ours_minus_cv_only",
        "Ours-FullVI-PointBridge-128x128",
        "Ours-CV-Only-FullVI-128",
    ),
    (
        "ours_minus_full_vanilla",
        "Ours-FullVI-PointBridge-128x128",
        "Full-VanillaMamba-Matched-256",
    ),
)
E2_FINAL_INTERACTION_COMPARISONS = (
    (
        "ours_minus_full_vanilla",
        "Final-Ours-Interaction-Mamba",
        "Final-FULL-Vanilla-Mamba",
    ),
    (
        "ours_minus_raw_dual_vanilla",
        "Final-Ours-Interaction-Mamba",
        "Final-Raw-Dual-Vanilla-Mamba",
    ),
    (
        "raw_dual_minus_full_vanilla",
        "Final-Raw-Dual-Vanilla-Mamba",
        "Final-FULL-Vanilla-Mamba",
    ),
)


def _prediction_keyed(run: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _load_predictions(run):
        key = (str(row["battery_id"]), int(row["cycle_id"]))
        if key in output:
            raise ValueError(f"Duplicate E2 final prediction cycle: {key}")
        output[key] = row
    return output


def e2_final_paired_rows(
    expected: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
    selected: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
    *,
    experiment_label: str = "E2_FINAL_256BUDGET",
    comparisons=E2_FINAL_COMPARISONS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for key, spec in expected.items():
        grouped[(str(spec["family"]), int(spec["seed"]))][str(spec["model_id"])] = selected[key]
    rows: list[dict[str, Any]] = []
    for (family, seed), model_runs in sorted(grouped.items()):
        keyed = {model: _prediction_keyed(run) for model, run in model_runs.items()}
        reference_keys: set[tuple[str, int]] | None = None
        reference_truth: dict[tuple[str, int], float] = {}
        for model, predictions in sorted(keyed.items()):
            keys = set(predictions)
            if reference_keys is None:
                reference_keys = keys
                reference_truth = {key: float(value["y_true"]) for key, value in predictions.items()}
            elif keys != reference_keys:
                raise ValueError(
                    f"E2 final test coverage differs for {family}, seed={seed}, model={model}"
                )
            for cycle_key, value in predictions.items():
                if not math.isclose(
                    float(value["y_true"]), reference_truth[cycle_key], rel_tol=1e-6, abs_tol=1e-7
                ):
                    raise ValueError(
                        f"E2 final labels differ for {family}, seed={seed}, cycle={cycle_key}"
                    )
        metrics = {
            model: _prediction_metrics(predictions.values())
            for model, predictions in keyed.items()
        }
        for comparison, left, right in comparisons:
            if left not in metrics or right not in metrics:
                raise ValueError(f"E2 final comparison is missing {left!r} or {right!r}")
            left_values = _metric_values(metrics[left])
            right_values = _metric_values(metrics[right])
            rows.append(
                {
                    "experiment": experiment_label,
                    "dataset": family,
                    "seed": seed,
                    "comparison": comparison,
                    "left_model": left,
                    "right_model": right,
                    "battery_macro_mape_delta_percent": left_values["battery_macro_mape_percent"]
                    - right_values["battery_macro_mape_percent"],
                    "battery_macro_rmse_delta_percent": left_values["battery_macro_rmse_soh_percent"]
                    - right_values["battery_macro_rmse_soh_percent"],
                    "common_cycles": len(reference_keys or ()),
                    "common_batteries": len({key[0] for key in reference_keys or ()}),
                }
            )
    macro_groups: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        macro_groups[(int(row["seed"]), str(row["comparison"]), str(row["left_model"]), str(row["right_model"]))].append(row)
    for (seed, comparison, left, right), values in sorted(macro_groups.items()):
        rows.append(
            {
                "experiment": experiment_label,
                "dataset": "ALL_DATASETS_MACRO",
                "seed": seed,
                "comparison": comparison,
                "left_model": left,
                "right_model": right,
                "battery_macro_mape_delta_percent": mean(float(item["battery_macro_mape_delta_percent"]) for item in values),
                "battery_macro_rmse_delta_percent": mean(float(item["battery_macro_rmse_delta_percent"]) for item in values),
                "common_cycles": sum(int(item["common_cycles"]) for item in values),
                "common_batteries": sum(int(item["common_batteries"]) for item in values),
            }
        )
    aggregate_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        aggregate_groups[(str(row["dataset"]), str(row["comparison"]), str(row["left_model"]), str(row["right_model"]))].append(row)
    summaries = []
    for (dataset, comparison, left, right), values in sorted(aggregate_groups.items()):
        mape = [float(item["battery_macro_mape_delta_percent"]) for item in values]
        rmse = [float(item["battery_macro_rmse_delta_percent"]) for item in values]
        summaries.append(
            {
                "experiment": experiment_label,
                "dataset": dataset,
                "comparison": comparison,
                "left_model": left,
                "right_model": right,
                "seed_count": len(values),
                "battery_macro_mape_delta_percent_mean": mean(mape),
                "battery_macro_mape_delta_percent_std": stdev(mape) if len(mape) > 1 else 0.0,
                "battery_macro_rmse_delta_percent_mean": mean(rmse),
                "battery_macro_rmse_delta_percent_std": stdev(rmse) if len(rmse) > 1 else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (row["dataset"], row["comparison"], row["seed"])), summaries


def summarize_experiment(
    experiment: str,
    *,
    root: Path,
    output_dir: Path,
    seeds: Iterable[int],
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in seeds)
    expected, config_paths = expected_jobs(experiment, seeds)
    discovered = discover_runs(root)
    selected, duplicate_resolutions = select_latest_runs(
        discovered, EXPERIMENTS[experiment]["experiment_id"]
    )
    relevant_selected = {key: value for key, value in selected.items() if key in expected}
    missing_keys = sorted(set(expected) - set(relevant_selected))
    unexpected_keys = sorted(set(selected) - set(expected))
    status = {
        "summary_version": SUMMARY_VERSION,
        "experiment": experiment.upper(),
        "experiment_id": EXPERIMENTS[experiment]["experiment_id"],
        "status": (
            "waiting_for_strategy_specific_runs"
            if experiment == "e3" and missing_keys
            else "incomplete"
            if missing_keys
            else "complete"
        ),
        "selection_rule": "newest completed non-debug run per experiment/model/data/seed; runtime names are ignored",
        "expected_seeds": list(seeds),
        "expected_job_count": len(expected),
        "selected_job_count": len(relevant_selected),
        "missing_job_count": len(missing_keys),
        "missing_jobs": [_missing_payload(key, expected[key]) for key in missing_keys],
        "unexpected_completed_jobs": [
            {"model": key[1], "data_id": key[2], "seed": key[3]}
            for key in unexpected_keys
        ],
        "duplicate_resolutions": duplicate_resolutions,
        "config_files": config_paths,
        "blocked_or_missing_are_not_zero": True,
        "formal_summary_written": False,
        "outputs": {},
    }
    status_path = output_dir / f"{experiment}_status.json"
    if missing_keys:
        _write_json(status_path, status)
        return status

    seed_rows, comparison_rows = per_seed_rows(
        experiment, expected, relevant_selected
    )
    mean_std_rows = aggregate_seed_rows(seed_rows)
    paired_rows: list[dict[str, Any]] = []
    paired_summary: list[dict[str, Any]] = []
    if experiment in {"e2_final_256budget", "e2_final_interaction_5seed"}:
        # Validate matched physical-cycle coverage and labels before publishing
        # any formal table for the final E2 matrix.
        paired_rows, paired_summary = e2_final_paired_rows(
            expected,
            relevant_selected,
            experiment_label=experiment.upper(),
            comparisons=(
                E2_FINAL_INTERACTION_COMPARISONS
                if experiment == "e2_final_interaction_5seed"
                else E2_FINAL_COMPARISONS
            ),
        )
    seed_csv = output_dir / f"{experiment}_metrics_per_seed.csv"
    seed_json = output_dir / f"{experiment}_metrics_per_seed.json"
    mean_csv = output_dir / f"{experiment}_metrics_mean_std.csv"
    mean_json = output_dir / f"{experiment}_metrics_mean_std.json"
    _write_csv(seed_csv, seed_rows, PER_SEED_COLUMNS)
    _write_json(
        seed_json,
        {
            "summary_version": SUMMARY_VERSION,
            "experiment": experiment.upper(),
            "units": {"mape": "percent", "rmse": "SOH percentage points"},
            "rows": seed_rows,
        },
    )
    _write_csv(mean_csv, mean_std_rows, MEAN_STD_COLUMNS)
    _write_json(
        mean_json,
        {
            "summary_version": SUMMARY_VERSION,
            "experiment": experiment.upper(),
            "aggregation": "arithmetic mean and sample standard deviation across seeds",
            "units": {"mape": "percent", "rmse": "SOH percentage points"},
            "rows": mean_std_rows,
        },
    )
    outputs = {
        "metrics_per_seed_csv": str(seed_csv),
        "metrics_per_seed_json": str(seed_json),
        "metrics_mean_std_csv": str(mean_csv),
        "metrics_mean_std_json": str(mean_json),
    }
    if experiment == "e3":
        comparison_columns = (
            "experiment",
            "dataset",
            "strategy",
            "seed",
            "specific_mape_percent",
            "pooled_mape_percent",
            "mape_delta_pooled_minus_specific_percent",
            "specific_rmse_soh_percent",
            "pooled_rmse_soh_percent",
            "rmse_delta_pooled_minus_specific_soh_percent",
            "common_cycles",
        )
        comparison_csv = output_dir / "e3_strategy_comparison_per_seed.csv"
        comparison_json = output_dir / "e3_strategy_comparison_per_seed.json"
        _write_csv(comparison_csv, comparison_rows, comparison_columns)
        _write_json(
            comparison_json,
            {
                "summary_version": SUMMARY_VERSION,
                "experiment": "E3",
                "comparison": "dataset pooled minus strategy specific on matched cycles",
                "rows": comparison_rows,
            },
        )
        outputs.update(
            {
                "strategy_comparison_per_seed_csv": str(comparison_csv),
                "strategy_comparison_per_seed_json": str(comparison_json),
            }
        )
    if experiment in {"e2_final_256budget", "e2_final_interaction_5seed"}:
        paired_columns = (
            "experiment",
            "dataset",
            "seed",
            "comparison",
            "left_model",
            "right_model",
            "battery_macro_mape_delta_percent",
            "battery_macro_rmse_delta_percent",
            "common_cycles",
            "common_batteries",
        )
        paired_summary_columns = (
            "experiment",
            "dataset",
            "comparison",
            "left_model",
            "right_model",
            "seed_count",
            "battery_macro_mape_delta_percent_mean",
            "battery_macro_mape_delta_percent_std",
            "battery_macro_rmse_delta_percent_mean",
            "battery_macro_rmse_delta_percent_std",
        )
        paired_csv = output_dir / f"{experiment}_paired_gaps_per_seed.csv"
        paired_mean_csv = output_dir / f"{experiment}_paired_gaps_mean_std.csv"
        _write_csv(paired_csv, paired_rows, paired_columns)
        _write_csv(paired_mean_csv, paired_summary, paired_summary_columns)
        outputs.update(
            {
                "paired_gaps_per_seed_csv": str(paired_csv),
                "paired_gaps_mean_std_csv": str(paired_mean_csv),
            }
        )
    status["formal_summary_written"] = True
    status["outputs"] = outputs
    _write_json(status_path, status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        "Write compact Paper-Backup MAPE/RMSE summaries"
    )
    parser.add_argument(
        "--experiment",
        choices=(*EXPERIMENTS, "all"),
        default="e1",
        help="Experiment to summarize; defaults to E1.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=_parse_seeds("42 52 62"),
        help='Expected seeds, for example --seeds "42 52 62".',
    )
    parser.add_argument(
        "--root", default=str(REPO_ROOT / "outputs/Paper-Backup")
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs/Paper-Backup/summaries"),
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    experiments = tuple(EXPERIMENTS) if args.experiment == "all" else (args.experiment,)
    statuses = []
    for experiment in experiments:
        status = summarize_experiment(
            experiment,
            root=root,
            output_dir=output_dir,
            seeds=args.seeds,
        )
        statuses.append(status)
        print(
            f"[{experiment.upper()} summary] status={status['status']} "
            f"selected={status['selected_job_count']}/{status['expected_job_count']}"
        )
        for name, path in status.get("outputs", {}).items():
            print(f"  {name}: {path}")
    return 0 if all(item["status"] == "complete" for item in statuses) else 2


if __name__ == "__main__":
    raise SystemExit(main())
