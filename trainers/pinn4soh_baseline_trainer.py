"""Training loop for the independent PINN4SOH-noLeak-OnlyF baseline."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from UnifiedRawSOH.datasets.baseline_loaders import build_feature_loaders
from UnifiedRawSOH.evaluation.metrics import compute_metrics, grouped_metrics, macro_rmse_by_group
from UnifiedRawSOH.evaluation.paper_v2_metrics import (
    build_hierarchical_metric_tables,
    test_metrics_payload,
    write_metric_tables,
)
from UnifiedRawSOH.datasets.soh_labels import is_bol_label_mode
from UnifiedRawSOH.models.baselines.pinn4soh_no_leak_onlyf import PINNFOnlyMLP
from UnifiedRawSOH.utils.config import save_json
from UnifiedRawSOH.utils.output_layout import build_run_manifest, build_seed_output_dir
from UnifiedRawSOH.utils.seed import set_random_seed


def _resolve_path(repo_root, value):
    value = Path(value)
    return value if value.is_absolute() else (Path(repo_root) / value).resolve()


def _runtime_directory_name(value=None):
    """Return the single runtime directory label used by Paper-v1 outputs."""

    raw = str(value or datetime.now().strftime("%y%m%d-%H%M%S"))
    return raw if raw.startswith("runtime_") else f"runtime_{raw}"


def _device_from_config(config, override=None):
    requested = override or config["train"].get("device", "cuda")
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The baseline configuration requests CUDA, but CUDA is unavailable.")
    return torch.device(requested)


def _run_epoch(model, loader, criterion, device, optimizer=None, collect_predictions=False):
    training = optimizer is not None
    model.train(training)
    truths, predictions, batteries, conditions, domains = [], [], [], [], []
    prediction_rows = []
    total_loss = 0.0
    total_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            features = batch["features"].to(device)
            truth = batch["soh"].to(device)
            prediction = model(features)
            loss = criterion(prediction, truth)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip = float(getattr(optimizer, "_unified_grad_clip", 0.0))
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            count = int(truth.size(0))
            total_count += count
            total_loss += float(loss.detach().item()) * count
            truths.extend(truth.detach().cpu().numpy().reshape(-1).tolist())
            predictions.extend(prediction.detach().cpu().numpy().reshape(-1).tolist())
            batch_batteries = list(batch["battery_id"])
            batch_conditions = list(batch["condition"])
            batch_domains = list(batch.get("domain_id", batch["dataset_id"]))
            batteries.extend(batch_batteries)
            conditions.extend(batch_conditions)
            domains.extend(batch_domains)
            if collect_predictions:
                batch_cycle_ids = batch.get("cycle_id")
                if torch.is_tensor(batch_cycle_ids):
                    batch_cycle_ids = batch_cycle_ids.detach().cpu().reshape(-1).tolist()
                elif batch_cycle_ids is None:
                    batch_cycle_ids = [None] * count
                else:
                    batch_cycle_ids = list(batch_cycle_ids)
                batch_truths = truth.detach().cpu().numpy().reshape(-1).tolist()
                batch_predictions = prediction.detach().cpu().numpy().reshape(-1).tolist()
                for index in range(count):
                    prediction_rows.append(
                        {
                            "domain_id": str(batch_domains[index]),
                            "group_id": str(batch_conditions[index]),
                            "cell_id": str(batch_batteries[index]),
                            "cycle_id": batch_cycle_ids[index],
                            "y_true": float(batch_truths[index]),
                            "y_pred": float(batch_predictions[index]),
                        }
                    )
    metrics = compute_metrics(truths, predictions)
    condition_macro_rmse = macro_rmse_by_group(truths, predictions, conditions)
    battery_macro_rmse = macro_rmse_by_group(truths, predictions, batteries)
    domain_macro_rmse = macro_rmse_by_group(truths, predictions, domains)
    metrics.update(
        {
            "loss": total_loss / max(total_count, 1),
            # E1 selects both the statistical baseline and RawMamba with the
            # same condition-balanced validation criterion.
            "macro_rmse": condition_macro_rmse,
            "condition_macro_rmse": condition_macro_rmse,
            "battery_macro_rmse": battery_macro_rmse,
            "domain_macro_rmse": domain_macro_rmse,
            "per_condition": grouped_metrics(truths, predictions, conditions),
            "per_battery": grouped_metrics(truths, predictions, batteries),
            "per_domain": grouped_metrics(truths, predictions, domains),
        }
    )
    if collect_predictions:
        metrics["_prediction_rows"] = prediction_rows
    return metrics


def build_model(config):
    model_cfg = config["model"]
    if model_cfg.get("type") != "PINNFOnlyMLP":
        raise ValueError("OnlyF baseline requires model.type='PINNFOnlyMLP'")
    return PINNFOnlyMLP(
        input_dim=int(model_cfg.get("input_dim", 24)),
        encoder_hidden_dim=int(model_cfg.get("encoder_hidden_dim", 60)),
        encoder_output_dim=int(model_cfg.get("encoder_output_dim", 32)),
        encoder_layers_num=int(model_cfg.get("encoder_layers_num", 3)),
        predictor_hidden_dim=int(model_cfg.get("predictor_hidden_dim", 32)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )


def train_from_config(config, repo_root, device_override=None):
    seed = int(config["train"].get("seed", 42))
    set_random_seed(seed, bool(config.get("debug", {}).get("deterministic", True)))
    device = _device_from_config(config, device_override)
    loaders, split_info = build_feature_loaders(config, repo_root, seed=seed)
    model = build_model(config).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"].get("lr", 1e-3)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    optimizer._unified_grad_clip = float(config["train"].get("grad_clip_norm", 1.0))
    scheduler_cfg = config["train"].get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 10)),
        threshold=float(scheduler_cfg.get("threshold", 1e-5)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
    )
    monitor_name = str(config["train"].get("monitor", "valid_condition_macro_rmse"))
    monitor_keys = {
        "valid_condition_macro_rmse": "condition_macro_rmse",
        # ``valid_macro_rmse`` remains a V2-compatible alias and now means
        # the condition-balanced metric in the unified Paper-v1 protocol.
        "valid_macro_rmse": "condition_macro_rmse",
        "valid_battery_macro_rmse": "battery_macro_rmse",
        "valid_domain_macro_rmse": "domain_macro_rmse",
        "valid_rmse": "rmse",
        "valid_loss": "loss",
    }
    if monitor_name not in monitor_keys:
        raise ValueError(
            "train.monitor must be one of "
            f"{sorted(monitor_keys)}, got {monitor_name!r}"
        )
    monitor_key = monitor_keys[monitor_name]
    best_metric = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(int(config["train"].get("epochs", 500))):
        train_metrics = _run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = _run_epoch(model, loaders["val"], criterion, device)
        monitor_value = float(val_metrics[monitor_key])
        scheduler.step(monitor_value)
        history.append(
            {
                "epoch": epoch + 1,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "monitor_name": monitor_name,
                "monitor_value": monitor_value,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        if monitor_value < best_metric:
            best_metric = monitor_value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(config["train"].get("patience", 30)):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    paper_v2 = str(config.get("output", {}).get("paper_version", "")) == "Paper-v2" or is_bol_label_mode(config)
    test_metrics = _run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        collect_predictions=paper_v2,
    )
    prediction_rows = test_metrics.pop("_prediction_rows", []) if paper_v2 else []
    metric_tables = build_hierarchical_metric_tables(prediction_rows) if paper_v2 else None
    output_root = _resolve_path(repo_root, config["experiment"].get("output_root", "UnifiedRawSOH/outputs"))
    run_time = _runtime_directory_name(config["experiment"].get("run_time"))
    config.setdefault("experiment", {})["run_time"] = run_time
    run_dir = build_seed_output_dir(output_root, config, run_time, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "resolved_config.json", config)
    save_json(run_dir / "run_manifest.json", build_run_manifest(config, output_root, run_time, seed=seed))
    save_json(run_dir / "split_info.json", split_info)
    save_json(run_dir / "history.json", history)
    if paper_v2:
        write_metric_tables(run_dir, metric_tables)
        hierarchical_payload = test_metrics_payload(metric_tables)
        sample_metrics = dict(test_metrics)
        test_metrics.update(hierarchical_payload)
        test_metrics["loss"] = float(test_metrics["mse"])
        test_metrics["domain_macro_rmse"] = float(metric_tables["overall"]["rmse"])
        test_metrics["condition_macro_rmse"] = float(metric_tables["overall"]["rmse"])
        test_metrics["sample_micro_metrics"] = {
            key: sample_metrics[key]
            for key in ("mae", "mape", "mse", "rmse", "loss", "condition_macro_rmse", "domain_macro_rmse")
            if key in sample_metrics
        }
    save_json(run_dir / "test_metrics.json", test_metrics)
    torch.save({"model": model.state_dict(), "config": config, "split_info": split_info}, run_dir / "best.pt")
    return {
        "run_dir": str(run_dir),
        "monitor": monitor_name,
        "best_monitor_value": best_metric,
        "best_valid_macro_rmse": best_metric,
        "test": test_metrics,
        "split_info": split_info,
    }
