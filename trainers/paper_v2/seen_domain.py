"""Paper-v2 ERM and first-order MLDG training orchestration."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import default_collate

from UnifiedRawSOH.datasets.paper_v2.episodic_sampler import SourceEpisodeBuilder
from UnifiedRawSOH.datasets.paper_v2.leakage import validate_lodo_provenance
from UnifiedRawSOH.models.paper_v2.raw_mamba_moe import build_paper_v2_model
from UnifiedRawSOH.utils.seed import set_random_seed

from .common import (
    build_paper_v2_loaders,
    device_from_config,
    move_tensors,
    run_epoch,
    runtime_directory_name,
    save_training_artifacts,
    set_train_epoch,
    standard_optimizer,
    standard_scheduler,
)
from .config_contract import (
    is_lodo_config,
    validate_data_readiness,
    validate_v2_config,
)
from .mldg import first_order_mldg_step


def _monitor_value(metrics: Mapping[str, Any], name: str) -> float:
    aliases = {
        "valid_domain_macro_rmse": "valid_domain_macro_rmse",
        "valid_rmse": "rmse",
        "valid_loss": "loss",
        "valid_condition_macro_rmse": "condition_macro_rmse",
    }
    key = aliases.get(str(name), str(name))
    if key not in metrics:
        raise ValueError(
            f"Paper-v2 monitor {name!r} is unavailable; available keys include {sorted(metrics)}"
        )
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"Paper-v2 validation monitor {name!r} is not finite: {value}")
    return value


def _public_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not key.startswith("_") and key not in {"hierarchical_tables"}
    }


def _is_moe(config: Mapping[str, Any]) -> bool:
    return str(config["model"]["variant"]) == "residual_moe"


def _train_erm(
    model: torch.nn.Module,
    loaders: Mapping[str, Any],
    train_sampler: Any,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    criterion = torch.nn.MSELoss()
    optimizer = standard_optimizer(model, config)
    scheduler = standard_scheduler(optimizer, config)
    train_cfg = config["train"]
    epochs = int(train_cfg.get("epochs", 1))
    patience = int(train_cfg.get("patience", max(1, epochs)))
    if epochs <= 0 or patience <= 0:
        raise ValueError("Paper-v2 train.epochs and train.patience must be positive.")
    lambda_balance = float(train_cfg.get("lambda_balance", 0.0))
    if lambda_balance < 0.0:
        raise ValueError("train.lambda_balance must be non-negative.")
    monitor_name = str(train_cfg.get("monitor", "valid_domain_macro_rmse"))
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("inf")
    stale = 0
    for epoch in range(epochs):
        set_train_epoch(train_sampler, epoch)
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            lambda_balance=lambda_balance,
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
            collect_routing=False,
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            collect_predictions=True,
            collect_routing=False,
        )
        monitor_value = _monitor_value(val_metrics, monitor_name)
        scheduler.step(monitor_value)
        row = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "lambda_balance": lambda_balance,
            "monitor_name": monitor_name,
            "monitor_value": monitor_value,
            "train": _public_metrics(train_metrics),
            "val": _public_metrics(val_metrics),
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
    if best_state is None:
        raise RuntimeError("Paper-v2 ERM did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state, strict=True)
    test_metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        collect_predictions=True,
        collect_routing=_is_moe(config),
    )
    routing_summary = test_metrics.pop("_routing_summary", None)
    tables = test_metrics.pop("hierarchical_tables", None)
    prediction_rows = test_metrics.pop("_prediction_rows", [])
    test_metrics["_hierarchical_tables"] = tables
    test_metrics["_prediction_rows"] = prediction_rows
    return history, test_metrics, routing_summary, {
        "best_monitor_value": best_metric,
        "monitor": monitor_name,
        "optimizer": "AdamW",
    }


def _episode_batch(dataset: Any, indices: list[int], device: torch.device) -> dict[str, Any]:
    batch = default_collate([dataset[index] for index in indices])
    return move_tensors(batch, device)


def _train_mldg(
    model: torch.nn.Module,
    loaders: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    if not is_lodo_config(config):
        raise ValueError("Paper-v2 first_order_mldg is reserved for a source-only LODO config.")
    source_domains = [str(value) for value in config["experiment"]["source_domain_ids"]]
    trainer_cfg = config["trainer"]
    builder = SourceEpisodeBuilder(
        loaders["train"].dataset,
        source_domains,
        seed=int(config["train"].get("seed", 42)),
        dataset_episode_probability=float(trainer_cfg["dataset_episode_probability"]),
        strategy_episode_probability=float(trainer_cfg["strategy_episode_probability"]),
    )
    optimizer = standard_optimizer(model, config)
    scheduler = standard_scheduler(optimizer, config)
    criterion = torch.nn.MSELoss()
    train_cfg = config["train"]
    epochs = int(train_cfg.get("epochs", 1))
    patience = int(train_cfg.get("patience", max(1, epochs)))
    if epochs <= 0 or patience <= 0:
        raise ValueError("Paper-v2 train.epochs and train.patience must be positive.")
    batch_size = int(trainer_cfg.get("episode_batch_size", train_cfg.get("batch_size", 64)))
    episodes_per_epoch = int(
        trainer_cfg.get(
            "episodes_per_epoch",
            max(1, math.ceil(len(loaders["train"].dataset) / max(batch_size, 1))),
        )
    )
    if batch_size <= 0 or episodes_per_epoch <= 0:
        raise ValueError("MLDG episode_batch_size and episodes_per_epoch must be positive.")
    alpha = float(trainer_cfg["inner_learning_rate"])
    beta = float(trainer_cfg["beta"])
    lambda_balance = float(train_cfg.get("lambda_balance", 0.0))
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("inf")
    stale = 0
    monitor_name = str(train_cfg.get("monitor", "valid_domain_macro_rmse"))
    for epoch in range(epochs):
        builder.set_epoch(epoch)
        model.train(True)
        sums = {"erm_loss": 0.0, "meta_train_loss": 0.0, "pseudo_target_loss": 0.0, "balance_loss": 0.0, "total_loss": 0.0}
        changed = 0
        for episode_number in range(episodes_per_epoch):
            episode = builder.sample_episode()
            meta_indices = builder.draw_indices(
                episode.meta_train_indices,
                batch_size,
                seed_offset=episode_number * 2,
                replacement=True,
            )
            target_indices = builder.draw_indices(
                episode.pseudo_target_indices,
                batch_size,
                seed_offset=episode_number * 2 + 1,
                replacement=True,
            )
            meta_batch = _episode_batch(loaders["train"].dataset, meta_indices, device)
            target_batch = _episode_batch(loaders["train"].dataset, target_indices, device)
            result = first_order_mldg_step(
                model,
                meta_batch,
                target_batch,
                criterion,
                inner_learning_rate=alpha,
                beta=beta,
                lambda_balance=lambda_balance,
                inner_steps=int(trainer_cfg["inner_steps"]),
            )
            if float(train_cfg.get("grad_clip_norm", 1.0)) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            optimizer.step()
            for key in sums:
                sums[key] += float(result[key])
            changed += int(result["fast_parameters_changed"])
        val_metrics = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            collect_predictions=True,
            collect_routing=False,
        )
        monitor_value = _monitor_value(val_metrics, monitor_name)
        scheduler.step(monitor_value)
        denominator = float(episodes_per_epoch)
        train_metrics = {
            key: value / denominator for key, value in sums.items()
        }
        train_metrics.update(
            {
                "episodes": episodes_per_epoch,
                "fast_parameters_changed_episodes": changed,
                "inner_learning_rate": alpha,
                "beta": beta,
                "lambda_balance": lambda_balance,
            }
        )
        history.append(
            {
                "epoch": epoch + 1,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "monitor_name": monitor_name,
                "monitor_value": monitor_value,
                "train": train_metrics,
                "val": _public_metrics(val_metrics),
            }
        )
        if monitor_value < best_metric:
            best_metric = monitor_value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Paper-v2 MLDG did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state, strict=True)
    test_metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        collect_predictions=True,
        collect_routing=_is_moe(config),
    )
    routing_summary = test_metrics.pop("_routing_summary", None)
    tables = test_metrics.pop("hierarchical_tables", None)
    prediction_rows = test_metrics.pop("_prediction_rows", [])
    test_metrics["_hierarchical_tables"] = tables
    test_metrics["_prediction_rows"] = prediction_rows
    return history, test_metrics, routing_summary, {
        "best_monitor_value": best_metric,
        "monitor": monitor_name,
        "optimizer": "AdamW",
        "mldg": {
            "first_order": True,
            "inner_steps": 1,
            "inner_learning_rate": alpha,
            "beta": beta,
            "dataset_episode_probability": float(trainer_cfg["dataset_episode_probability"]),
            "strategy_episode_probability": float(trainer_cfg["strategy_episode_probability"]),
        },
        "episode_audit": builder.audit(),
    }


def train_from_config(
    config: dict[str, Any],
    project_root: str,
    *,
    backend_override: str | None = None,
    device_override: str | None = None,
    check_data: bool = True,
) -> dict[str, Any]:
    """Run one V2 seed and write only the Paper-v2 artifact namespace."""

    validate_v2_config(config, require_runnable=True)
    if str(config["model"]["variant"]) not in {"base", "dense_adapter", "residual_moe"}:
        raise ValueError(
            "The independent Paper-v2 raw entry point does not implement feature_mlp; "
            "use the existing FeatureMLP launcher for that compatibility path."
        )
    if check_data:
        readiness = validate_data_readiness(config, project_root)
    else:
        readiness = {"ready": False, "skipped": True}
    seed = int(config["train"].get("seed", 42))
    config.setdefault("train", {})["seed"] = seed
    set_random_seed(seed, bool(config.get("debug", {}).get("deterministic", True)))
    device = device_from_config(config, device_override)
    loaders, split_info, train_sampler = build_paper_v2_loaders(config, project_root, seed)
    if is_lodo_config(config):
        split_info = dict(split_info)
        split_info["paper_v2_leakage"] = validate_lodo_provenance(
            config,
            loaders=loaders,
            split_info=split_info,
        )
    model = build_paper_v2_model(config["model"], backend_override=backend_override).to(device)
    parameter_summary = model.parameter_summary()
    if str(config["trainer"]["variant"]) == "erm":
        history, test_metrics, routing_summary, train_summary = _train_erm(
            model, loaders, train_sampler, config, device
        )
        episode_audit = None
    elif str(config["trainer"]["variant"]) == "first_order_mldg":
        history, test_metrics, routing_summary, train_summary = _train_mldg(
            model, loaders, config, device
        )
        episode_audit = train_summary.pop("episode_audit")
    else:
        raise ValueError(
            f"Unknown trainer.variant {config['trainer']['variant']!r}; no fallback is permitted."
        )
    run_time = runtime_directory_name(config.get("experiment", {}).get("run_time"))
    config.setdefault("experiment", {})["run_time"] = run_time
    output_root = Path(config["experiment"].get("output_root", "UnifiedRawSOH/outputs"))
    if not output_root.is_absolute():
        output_root = Path(project_root) / output_root
    config["experiment"]["output_root"] = str(output_root)
    if episode_audit is not None:
        sampling_audit = {
            "sampler": "episodic_source_hierarchy",
            "train_only": True,
            "validation_test_sequential": True,
            "inventory": episode_audit.get("inventory"),
            "episode_type_counts": episode_audit.get("episode_type_counts", {}),
            "heldout_domain_counts": episode_audit.get("heldout_domain_counts", {}),
            "actual_episode_count": len(episode_audit.get("episodes", [])),
            "seed": seed,
        }
    elif train_sampler is not None:
        sampling_audit = train_sampler.audit()
    else:
        sampling_audit = {
            "sampler": "sequential",
            "train_only": True,
            "inventory": None,
            "sampled_counts": {},
            "seed": seed,
        }
    hierarchical_overall = {}
    tables = test_metrics.get("_hierarchical_tables")
    if isinstance(tables, Mapping) and isinstance(tables.get("overall"), Mapping):
        hierarchical_overall = dict(tables["overall"])
    test_metrics["readiness"] = readiness
    test_metrics["training"] = train_summary
    run_dir = save_training_artifacts(
        config,
        output_root=output_root,
        run_time=run_time,
        seed=seed,
        model=model,
        split_info=split_info,
        history=history,
        test_metrics=test_metrics,
        parameter_summary=parameter_summary,
        sampling_audit=sampling_audit,
        episode_audit=episode_audit,
        routing_summary=routing_summary,
    )
    return {
        "status": "completed",
        "run_dir": str(run_dir),
        "paper_version": "Paper-v2",
        "model_variant": config["model"]["variant"],
        "trainer_variant": config["trainer"]["variant"],
        "seed": seed,
        "best_monitor_value": train_summary["best_monitor_value"],
        "test": {
            key: float(hierarchical_overall[key])
            for key in ("mae", "mape", "mse", "rmse")
            if key in hierarchical_overall
        },
    }


__all__ = ["train_from_config"]
