"""Paper-v1 raw SOH training loop with mixed-cycle evaluation."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from UnifiedRawSOH.datasets.loaders import build_single_domain_loaders, build_unified_loaders
from UnifiedRawSOH.evaluation.metrics import compute_metrics, grouped_metrics, macro_rmse_by_group
from UnifiedRawSOH.models.raw_soh_model import build_raw_soh_model
from UnifiedRawSOH.utils.config import save_json
from UnifiedRawSOH.utils.output_layout import build_run_manifest, build_seed_output_dir
from UnifiedRawSOH.utils.seed import set_random_seed


def _resolve_path(repo_root, value):
    value = Path(value)
    return value if value.is_absolute() else (repo_root / value).resolve()


def _runtime_directory_name(value=None):
    """Return the single runtime directory label used by Paper-v1 outputs."""

    raw = str(value or datetime.now().strftime("%y%m%d-%H%M%S"))
    return raw if raw.startswith("runtime_") else f"runtime_{raw}"


def _device_from_config(config, override=None):
    requested = override or config["train"].get("device", "cuda")
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "The paper configuration requests CUDA, but torch.cuda.is_available() is false. "
            "Use a CUDA environment for formal C5B runs or the CPU reference backend only for smoke tests."
        )
    return torch.device(requested)


def _model_inputs(batch):
    return {
        "cc_signal": batch["cc_signal"],
        "cv_signal": batch["cv_signal"],
        "cc_mask": batch["cc_mask"],
        "cv_mask": batch["cv_mask"],
        "cc_time": batch["cc_time"],
        "cv_time": batch["cv_time"],
        "cc_temperature": batch["cc_temperature"],
        "cv_temperature": batch["cv_temperature"],
        "t0_temperature_norm": batch["t0_temperature_norm"],
    }


def run_epoch(model, loader, criterion, device, optimizer=None, lambda_cycle=0.0):
    training = optimizer is not None
    model.train(training)
    truths, predictions, batteries, conditions, domains, cycle_truths, cycle_predictions = [], [], [], [], [], [], []
    total_loss = 0.0
    total_soh_loss = 0.0
    total_cycle_loss = 0.0
    total_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            tensors = {
                key: value.to(device)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            aux = model.forward_with_aux(**_model_inputs(tensors))
            soh_loss = criterion(aux["soh_pred"], tensors["soh"])
            if aux["cycle_life_hat"] is not None and "cycle_life_norm_target" in tensors:
                cycle_loss = criterion(aux["cycle_life_hat"], tensors["cycle_life_norm_target"])
            else:
                cycle_loss = torch.zeros((), device=device, dtype=soh_loss.dtype)
            loss = soh_loss + (float(lambda_cycle) * cycle_loss if training else 0.0)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = float(getattr(optimizer, "_unified_grad_clip", 0.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            count = int(tensors["soh"].size(0))
            total_count += count
            total_loss += float(loss.detach().item()) * count
            total_soh_loss += float(soh_loss.detach().item()) * count
            total_cycle_loss += float(cycle_loss.detach().item()) * count
            truths.extend(tensors["soh"].detach().cpu().numpy().reshape(-1).tolist())
            predictions.extend(aux["soh_pred"].detach().cpu().numpy().reshape(-1).tolist())
            batteries.extend(list(batch["battery_id"]))
            conditions.extend(list(batch["condition"]))
            domains.extend(list(batch.get("domain_id", batch["dataset_id"])))
            if aux["cycle_life_hat"] is not None and "cycle_life_norm_target" in tensors:
                cycle_truths.extend(tensors["cycle_life_norm_target"].detach().cpu().numpy().reshape(-1).tolist())
                cycle_predictions.extend(aux["cycle_life_hat"].detach().cpu().numpy().reshape(-1).tolist())

    metrics = compute_metrics(truths, predictions)
    condition_macro_rmse = macro_rmse_by_group(truths, predictions, conditions)
    battery_macro_rmse = macro_rmse_by_group(truths, predictions, batteries)
    domain_macro_rmse = macro_rmse_by_group(truths, predictions, domains)
    metrics.update(
        {
            "loss": total_loss / max(total_count, 1),
            "soh_loss": total_soh_loss / max(total_count, 1),
            "cycle_loss": total_cycle_loss / max(total_count, 1),
            # ``macro_rmse`` deliberately follows V2 C5B's all-batch
            # checkpoint-selection definition: each charge condition has
            # equal weight, regardless of its cycle or battery count.
            "macro_rmse": condition_macro_rmse,
            "condition_macro_rmse": condition_macro_rmse,
            "battery_macro_rmse": battery_macro_rmse,
            "domain_macro_rmse": domain_macro_rmse,
            "per_condition": grouped_metrics(truths, predictions, conditions),
            "per_battery": grouped_metrics(truths, predictions, batteries),
            "per_domain": grouped_metrics(truths, predictions, domains),
        }
    )
    if cycle_truths:
        metrics["cycle_metrics"] = compute_metrics(cycle_truths, cycle_predictions)
    return metrics


def train_from_config(config, repo_root, backend_override=None, device_override=None):
    seed = int(config["train"].get("seed", 42))
    set_random_seed(seed, bool(config.get("debug", {}).get("deterministic", True)))
    device = _device_from_config(config, device_override)
    if config.get("experiment", {}).get("loader", "single_domain") == "unified_multi_dataset":
        loaders, split_info = build_unified_loaders(config, repo_root, seed)
    else:
        loaders, split_info = build_single_domain_loaders(config, repo_root, seed)
    model = build_raw_soh_model(config["model"], backend_override=backend_override).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 1e-3)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    optimizer._unified_grad_clip = float(config["train"].get("grad_clip_norm", 1.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["train"].get("scheduler", {}).get("factor", 0.5)),
        patience=int(config["train"].get("scheduler", {}).get("patience", 10)),
        threshold=float(config["train"].get("scheduler", {}).get("threshold", 1e-5)),
        min_lr=float(config["train"].get("scheduler", {}).get("min_lr", 1e-5)),
    )
    epochs = int(config["train"].get("epochs", 500))
    patience = int(config["train"].get("patience", 40))
    monitor_name = str(config["train"].get("monitor", "valid_condition_macro_rmse"))
    monitor_keys = {
        "valid_condition_macro_rmse": "condition_macro_rmse",
        # Compatibility alias for the V2 C5B field name.
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
    lambda_cycle = float(config["train"].get("lambda_cycle", 0.0035))
    warmup = max(1, int(config["train"].get("cycle_loss_warmup_epochs", 10)))
    best_metric = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(epochs):
        # Match V2 C5B exactly: epoch 1 starts with no cycle auxiliary loss
        # and epoch ``warmup`` reaches the configured weight.
        current_lambda = (
            lambda_cycle
            if warmup <= 1
            else lambda_cycle * min(1.0, float(epoch) / float(warmup - 1))
        )
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer, current_lambda)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        monitor_value = float(val_metrics[monitor_key])
        scheduler.step(monitor_value)
        row = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "lambda_cycle": current_lambda,
            "monitor_name": monitor_name,
            "monitor_value": monitor_value,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        if monitor_value < best_metric:
            best_metric = monitor_value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = run_epoch(model, loaders["test"], criterion, device)
    output_root = _resolve_path(repo_root, config["experiment"].get("output_root", "UnifiedRawSOH/outputs"))
    run_time = _runtime_directory_name(config["experiment"].get("run_time"))
    config.setdefault("experiment", {})["run_time"] = run_time
    run_dir = build_seed_output_dir(output_root, config, run_time, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "resolved_config.json", config)
    save_json(run_dir / "run_manifest.json", build_run_manifest(config, output_root, run_time, seed=seed))
    save_json(run_dir / "split_info.json", split_info)
    save_json(run_dir / "history.json", history)
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
