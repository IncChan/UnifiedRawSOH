"""Shared Paper-v2 loader, epoch, metric, and artifact helpers."""

from __future__ import annotations

import copy
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from UnifiedRawSOH.datasets.loaders import (
    build_lodo_loaders,
    build_single_domain_loaders,
    build_unified_loaders,
)
from UnifiedRawSOH.datasets.paper_v2.hierarchical_sampler import HierarchicalReplacementSampler
from UnifiedRawSOH.datasets.paper_v2.leakage import validate_lodo_provenance
from UnifiedRawSOH.evaluation.metrics import compute_metrics, grouped_metrics, macro_rmse_by_group
from UnifiedRawSOH.evaluation.paper_v2.routing import RoutingAccumulator
from UnifiedRawSOH.evaluation.paper_v2_metrics import build_hierarchical_metric_tables, test_metrics_payload
from UnifiedRawSOH.utils.config import save_json

from .config_contract import (
    PAPER_VERSION,
    build_v2_seed_output_dir,
    validate_data_readiness,
    validate_v2_config,
    v2_output_identity,
)


MODEL_BATCH_KEYS = (
    "cc_signal",
    "cv_signal",
    "cc_mask",
    "cv_mask",
    "cc_time",
    "cv_time",
    "cc_temperature",
    "cv_temperature",
    "t0_temperature_norm",
)


def runtime_directory_name(value: str | None = None) -> str:
    raw = str(value or datetime.now().strftime("%y%m%d-%H%M%S"))
    raw = raw.replace("/", "_").replace("\\", "_")
    return raw if raw.startswith("runtime_") else f"runtime_{raw}"


def resolve_path(project_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def device_from_config(config: Mapping[str, Any], override: str | None = None) -> torch.device:
    requested = str(override or config.get("train", {}).get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Paper-v2 config requests CUDA but it is unavailable. Use --device_override cpu "
            "only for bounded torch_reference smoke tests."
        )
    return torch.device(requested)


def model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in ("cc_signal", "cv_signal") if key not in batch]
    if missing:
        raise ValueError(f"Paper-v2 batch is missing model inputs: {missing}")
    return {key: batch[key] for key in MODEL_BATCH_KEYS if key in batch}


def move_tensors(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _metadata_list(batch: Mapping[str, Any], key: str, count: int, default: str = "unknown") -> list[Any]:
    value = batch.get(key)
    if value is None:
        return [default] * count
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if len(values) != count:
        raise ValueError(f"Batch metadata {key!r} has {len(values)} values; expected {count}.")
    return values


def prediction_rows_from_batch(
    batch: Mapping[str, Any],
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> list[dict[str, Any]]:
    count = int(truth.size(0))
    domains = _metadata_list(batch, "domain_id", count)
    strategies = _metadata_list(batch, "strategy_group", count, default="unknown")
    if strategies == ["unknown"] * count:
        strategies = _metadata_list(batch, "condition", count)
    cells = _metadata_list(batch, "physical_cell_id", count, default="unknown")
    if cells == ["unknown"] * count:
        cells = _metadata_list(batch, "battery_id", count)
    cycles = _metadata_list(batch, "cycle_id", count, default=None)
    truths = truth.detach().cpu().reshape(-1).tolist()
    predictions = prediction.detach().cpu().reshape(-1).tolist()
    return [
        {
            "domain_id": str(domains[index]),
            "group_id": str(strategies[index]),
            "cell_id": str(cells[index]),
            "cycle_id": cycles[index],
            "y_true": float(truths[index]),
            "y_pred": float(predictions[index]),
        }
        for index in range(count)
    ]


def _balance_value(aux: Mapping[str, Any], reference: torch.Tensor) -> torch.Tensor:
    value = aux.get("balance_loss")
    if value is None:
        return torch.zeros((), device=reference.device, dtype=reference.dtype)
    if not torch.is_tensor(value):
        raise TypeError("Model balance_loss must be a tensor or None.")
    return value


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    lambda_balance: float = 0.0,
    grad_clip_norm: float = 0.0,
    collect_predictions: bool = False,
    collect_routing: bool = False,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    truths: list[float] = []
    predictions: list[float] = []
    domains: list[str] = []
    strategies: list[str] = []
    cells: list[str] = []
    prediction_rows: list[dict[str, Any]] = []
    routing = RoutingAccumulator() if collect_routing else None
    total_loss = 0.0
    total_soh_loss = 0.0
    total_balance_loss = 0.0
    total_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = move_tensors(raw_batch, device)
            aux = model.forward_with_aux(**model_inputs(batch))
            if "soh" not in batch:
                raise ValueError("Paper-v2 batch is missing the soh target.")
            soh_loss = criterion(aux["soh_pred"], batch["soh"])
            balance_loss = _balance_value(aux, soh_loss)
            loss = soh_loss + float(lambda_balance) * balance_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()
            count = int(batch["soh"].size(0))
            total_count += count
            total_loss += float(loss.detach().item()) * count
            total_soh_loss += float(soh_loss.detach().item()) * count
            total_balance_loss += float(balance_loss.detach().item()) * count
            batch_truth = batch["soh"].detach().reshape(-1).cpu().numpy().tolist()
            batch_prediction = aux["soh_pred"].detach().reshape(-1).cpu().numpy().tolist()
            truths.extend(float(value) for value in batch_truth)
            predictions.extend(float(value) for value in batch_prediction)
            row_batch = prediction_rows_from_batch(raw_batch, batch["soh"], aux["soh_pred"])
            row_domains = [row["domain_id"] for row in row_batch]
            row_strategies = [row["group_id"] for row in row_batch]
            row_cells = [row["cell_id"] for row in row_batch]
            domains.extend(row_domains)
            strategies.extend(row_strategies)
            cells.extend(row_cells)
            if collect_predictions:
                prediction_rows.extend(row_batch)
            if routing is not None:
                routing.update(aux, raw_batch)

    metrics = compute_metrics(truths, predictions)
    metrics.update(
        {
            "loss": total_loss / max(total_count, 1),
            "soh_loss": total_soh_loss / max(total_count, 1),
            "balance_loss": total_balance_loss / max(total_count, 1),
            "condition_macro_rmse": macro_rmse_by_group(truths, predictions, strategies),
            "battery_macro_rmse": macro_rmse_by_group(truths, predictions, cells),
            "domain_macro_rmse": macro_rmse_by_group(truths, predictions, domains),
            "per_condition": grouped_metrics(truths, predictions, strategies),
            "per_battery": grouped_metrics(truths, predictions, cells),
            "per_domain": grouped_metrics(truths, predictions, domains),
        }
    )
    tables = build_hierarchical_metric_tables(prediction_rows) if collect_predictions else None
    if tables is not None:
        metrics["hierarchical_tables"] = tables
        metrics["valid_domain_macro_rmse"] = float(tables["overall"]["rmse"])
    if collect_predictions:
        metrics["_prediction_rows"] = prediction_rows
    if routing is not None:
        metrics["_routing_summary"] = routing.summary()
    return metrics


def _loader_from_dataset(
    dataset: Any,
    config: Mapping[str, Any],
    *,
    sampler: Any = None,
    shuffle: bool = False,
) -> DataLoader:
    data = config["data"]
    train = config["train"]
    return DataLoader(
        dataset,
        batch_size=int(train.get("batch_size", 64)),
        shuffle=bool(shuffle),
        sampler=sampler,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def build_paper_v2_loaders(
    config: Mapping[str, Any],
    project_root: str | Path,
    seed: int,
) -> tuple[dict[str, DataLoader], dict[str, Any], HierarchicalReplacementSampler | None]:
    """Reuse validated raw adapters, replacing only the V2 train sampler."""

    loader_name = str(config.get("experiment", {}).get("loader", ""))
    if loader_name == "unified_multi_dataset":
        legacy_loaders, split_info = build_unified_loaders(config, project_root, seed)
    elif loader_name == "leave_one_domain_out":
        legacy_loaders, split_info = build_lodo_loaders(config, project_root, seed)
    elif loader_name == "single_domain":
        legacy_loaders, split_info = build_single_domain_loaders(config, project_root, seed)
    else:
        raise ValueError(
            "Paper-v2 raw entry point supports loader values single_domain, "
            f"unified_multi_dataset, and leave_one_domain_out; got {loader_name!r}."
        )
    sampler_cfg = config["data"].get("sampler", {})
    sampler_kind = str(sampler_cfg.get("kind", "sequential"))
    train_sampler = None
    if sampler_kind == "hierarchical":
        requested_num_samples = sampler_cfg.get("num_samples")
        train_sampler = HierarchicalReplacementSampler(
            legacy_loaders["train"].dataset,
            num_samples=(None if requested_num_samples is None else int(requested_num_samples)),
            seed=int(seed),
        )
    elif sampler_kind != "sequential":
        raise ValueError(f"Unsupported Paper-v2 data.sampler.kind={sampler_kind!r}.")
    loaders = {
        "train": _loader_from_dataset(
            legacy_loaders["train"].dataset,
            config,
            sampler=train_sampler,
            shuffle=train_sampler is None,
        ),
        "val": _loader_from_dataset(legacy_loaders["val"].dataset, config, shuffle=False),
        "test": _loader_from_dataset(legacy_loaders["test"].dataset, config, shuffle=False),
    }
    if str(config.get("experiment", {}).get("loader", "")) == "leave_one_domain_out":
        split_info = dict(split_info)
        split_info["paper_v2_leakage"] = validate_lodo_provenance(
            config,
            loaders=loaders,
            split_info=split_info,
        )
    split_info = copy.deepcopy(split_info)
    split_info["paper_v2_sampler"] = {
        "kind": sampler_kind,
        "train_only": True,
        "validation_test_sequential": True,
        "seed": int(seed),
    }
    return loaders, split_info, train_sampler


def set_train_epoch(sampler: Any, epoch: int) -> None:
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))


def standard_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    train = config["train"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train.get("lr", 1e-3)),
        weight_decay=float(train.get("weight_decay", 1e-4)),
    )


def standard_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any]) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    scheduler = config["train"].get("scheduler", {})
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler.get("mode", "min")),
        factor=float(scheduler.get("factor", 0.5)),
        patience=int(scheduler.get("patience", 10)),
        threshold=float(scheduler.get("threshold", 1e-5)),
        min_lr=float(scheduler.get("min_lr", 1e-5)),
    )


def build_run_manifest(config: Mapping[str, Any], output_root: str | Path, run_time: str, seed: int) -> dict[str, Any]:
    return {
        "paper_version": PAPER_VERSION,
        "output": v2_output_identity(config),
        "output_root": str(output_root),
        "runtime": str(run_time),
        "seed": int(seed),
        "model_variant": config["model"]["variant"],
        "trainer_variant": config["trainer"]["variant"],
        "experiment": copy.deepcopy(config.get("experiment", {})),
        "data_protocol": {
            "label_mode": config["data"]["label_mode"],
            "bol_reference_rule": config["data"]["bol_reference_rule"],
            "q_ref_is_model_input": config["data"].get("q_ref_is_model_input", False),
            "q_ref_in_normalization": config["data"].get("q_ref_in_normalization", False),
        },
        "scope": {
            "formal_training_executed": False,
            "p3_one_trajectory_adaptation": False,
            "p4_enterprise": False,
        },
    }


def save_training_artifacts(
    config: dict[str, Any],
    *,
    output_root: str | Path,
    run_time: str,
    seed: int,
    model: torch.nn.Module,
    split_info: Mapping[str, Any],
    history: list[dict[str, Any]],
    test_metrics: dict[str, Any],
    parameter_summary: Mapping[str, Any],
    sampling_audit: Mapping[str, Any],
    episode_audit: Mapping[str, Any] | None = None,
    routing_summary: Mapping[str, Any] | None = None,
) -> Path:
    validate_v2_config(config, require_runnable=True)
    run_dir = build_v2_seed_output_dir(output_root, config, run_time, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "resolved_config.json", config)
    save_json(run_dir / "run_manifest.json", build_run_manifest(config, output_root, run_time, seed))
    save_json(run_dir / "split_info.json", split_info)
    save_json(run_dir / "history.json", history)
    save_json(run_dir / "parameter_summary.json", parameter_summary)
    save_json(run_dir / "sampling_audit.json", sampling_audit)
    if episode_audit is not None:
        save_json(run_dir / "episode_audit.json", episode_audit)
    if routing_summary is not None:
        save_json(run_dir / "routing_summary.json", routing_summary)
    tables = test_metrics.pop("_hierarchical_tables", None)
    if tables is None:
        rows = test_metrics.pop("_prediction_rows", [])
        tables = build_hierarchical_metric_tables(rows)
    from UnifiedRawSOH.evaluation.paper_v2_metrics import write_metric_tables

    write_metric_tables(run_dir, tables)
    # The primary V2 test metrics are hierarchical.  Preserve the legacy
    # sample-micro values as a secondary diagnostic instead of overwriting the
    # domain-macro payload returned by ``test_metrics_payload``.
    payload = test_metrics_payload(tables)
    payload["sample_micro_metrics"] = {
        key: test_metrics[key]
        for key in ("mae", "mape", "mse", "rmse")
        if key in test_metrics
    }
    protected_hierarchical_keys = {
        "mae",
        "mape",
        "mse",
        "rmse",
        "aggregation",
        "per_domain",
        "hierarchical_metrics",
    }
    for key, value in list(test_metrics.items()):
        if not key.startswith("_") and key not in protected_hierarchical_keys:
            payload[key] = value
    payload["model_variant"] = config["model"]["variant"]
    payload["trainer_variant"] = config["trainer"]["variant"]
    save_json(run_dir / "test_metrics.json", payload)
    save_json(
        run_dir / "test_metrics_by_domain.json",
        {"aggregation": payload["aggregation"], "domains": payload["per_domain"]},
    )
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "split_info": split_info,
            "model_variant": config["model"]["variant"],
            "trainer_variant": config["trainer"]["variant"],
        },
        run_dir / "best.pt",
    )
    (run_dir / "completed.status").write_text("completed\n", encoding="utf-8")
    return run_dir


__all__ = [
    "MODEL_BATCH_KEYS",
    "build_paper_v2_loaders",
    "build_run_manifest",
    "device_from_config",
    "model_inputs",
    "move_tensors",
    "run_epoch",
    "runtime_directory_name",
    "save_training_artifacts",
    "set_train_epoch",
    "standard_optimizer",
    "standard_scheduler",
]
