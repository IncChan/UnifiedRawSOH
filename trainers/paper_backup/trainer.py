"""Small, auditable SOH-only training runner for Paper-Backup."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ...evaluation.paper_backup.aggregation import metrics_from_rows
from ...models.paper_backup.model_factory import build_model, model_input_kind
from ...utils.seed import set_random_seed
from .config_contract import validate_config


PHASE_KEYS = (
    "cc_signal",
    "cv_signal",
    "cc_time",
    "cv_time",
    "cc_temperature",
    "cv_temperature",
    "t0_temperature_norm",
    "cc_mask",
    "cv_mask",
)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _as_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
        return value[0] if len(value) == 1 else value
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
        return value[0] if len(value) == 1 else value
    return value


def _batch_values(batch: Mapping[str, Any], key: str, count: int, default: Any = "unknown") -> list[Any]:
    fallback = batch.get("condition", default) if key == "strategy_id" else default
    value = batch.get(key, fallback)
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value] * count
    if len(values) == 1 and count > 1:
        values *= count
    if len(values) != count:
        raise ValueError(f"Batch metadata {key!r} has {len(values)} values for batch size {count}")
    return values


def _forward_prediction(model_type: str, model: torch.nn.Module, batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    kind = model_input_kind(model_type)
    if kind == "features":
        return model(batch["features"].to(device, non_blocking=device.type == "cuda"))
    if kind == "sequence":
        mask = batch.get("mask")
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device, non_blocking=device.type == "cuda")
        sequence = batch["sequence"].to(
            device, non_blocking=device.type == "cuda"
        )
        boundary_index = batch.get("boundary_index")
        if isinstance(boundary_index, torch.Tensor):
            boundary_index = boundary_index.to(
                device, non_blocking=device.type == "cuda"
            )
            return model(sequence, mask, boundary_index=boundary_index)
        return model(sequence, mask)
    inputs = {
        key: batch[key].to(device, non_blocking=device.type == "cuda")
        for key in PHASE_KEYS
    }
    return model.forward_with_aux(**inputs)["soh_pred"]


def _rows_from_batch(batch: Mapping[str, Any], truth: torch.Tensor, prediction: torch.Tensor) -> list[dict[str, Any]]:
    count = int(truth.shape[0])
    fields = {
        key: _batch_values(batch, key, count)
        for key in ("domain_id", "strategy_id", "battery_id", "cycle_id", "view_id", "raw_point_count", "duration_min")
    }
    rows = []
    truth_values = truth.detach().cpu().reshape(-1).tolist()
    prediction_values = prediction.detach().cpu().reshape(-1).tolist()
    for index in range(count):
        row = {
            "domain_id": str(fields["domain_id"][index]),
            "strategy_id": str(fields["strategy_id"][index]),
            "battery_id": str(fields["battery_id"][index]),
            "cycle_id": int(float(fields["cycle_id"][index])),
            "view_id": str(fields["view_id"][index]),
            "y_true": float(truth_values[index]),
            "y_pred": float(prediction_values[index]),
        }
        for key in ("raw_point_count", "duration_min"):
            try:
                row[key] = float(fields[key][index])
            except (TypeError, ValueError):
                row[key] = float("nan")
        rows.append(row)
    return rows


def run_epoch(
    model_type: str,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    collect_predictions: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    loss_function = torch.nn.MSELoss()
    # Keep the running loss on the accelerator and synchronize only once at
    # epoch end. Calling loss.detach().cpu() for every batch serializes the CPU
    # and GPU and is especially expensive for these small SOH models.
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    sample_count = 0
    rows: list[dict[str, Any]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            truth = batch["soh"].to(
                device, non_blocking=device.type == "cuda"
            ).reshape(-1, 1)
            prediction = _forward_prediction(model_type, model, batch, device).reshape(-1, 1)
            loss = loss_function(prediction, truth)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            batch_size = int(truth.shape[0])
            loss_sum.add_(loss.detach().to(dtype=torch.float64), alpha=batch_size)
            sample_count += batch_size
            if collect_predictions:
                rows.extend(_rows_from_batch(batch, truth, prediction))
    metrics = metrics_from_rows(rows)
    if not collect_predictions:
        metrics["n_cycles"] = sample_count
    metrics["prediction_metrics_collected"] = bool(collect_predictions)
    metrics["loss"] = (
        float(loss_sum.item()) / sample_count if sample_count else float("nan")
    )
    metrics["soh_loss"] = metrics["loss"]
    metrics["battery_macro_rmse"] = metrics["battery_macro"]["rmse"]
    metrics["strategy_macro_rmse"] = metrics["strategy_macro"]["rmse"]
    return metrics, rows


def _metric_at(metrics: Mapping[str, Any], monitor: str) -> float:
    value: Any = metrics
    for part in str(monitor).split("."):
        if not isinstance(value, Mapping):
            return float("nan")
        value = value.get(part)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _unique_run_dir(root: Path, experiment_id: str, model_id: str, data_id: str, run_time: str, seed: int) -> Path:
    base = root / experiment_id / model_id / data_id / f"runtime_{run_time}"
    candidate = base
    suffix = 1
    while (candidate / f"seed_{int(seed)}").exists():
        candidate = Path(f"{base}-{suffix}")
        suffix += 1
    return candidate / f"seed_{int(seed)}"


def _resolve_root(config: Mapping[str, Any], repo_root: Path, output_root: str | Path | None) -> Path:
    raw = output_root if output_root is not None else config["output"]["root"]
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    if "Paper-Backup" not in root.parts and "Paper-Backup" not in root.as_posix().split("/"):
        raise ValueError(f"Training output must be inside Paper-Backup namespace: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_loaders(config: Mapping[str, Any], repo_root: Path, seed: int):
    model_type = str(config["model"]["type"])
    if model_type == "HI-MLP":
        if str(config.get("data", {}).get("source_mode", "legacy_runtime")) in {
            "preprocessed_v1", "preprocessed_v2"
        }:
            from ...datasets.paper_backup.preprocessed import build_preprocessed_feature_loaders

            return build_preprocessed_feature_loaders(config, repo_root, seed=seed)
        from ...models.baselines.pinn4soh_no_leak_onlyf import build_feature_loaders

        return build_feature_loaders(dict(config), repo_root, seed=seed)
    if str(config["output"]["experiment_id"]) == "e3_strategy_pooling":
        from ...datasets.paper_backup.strategy_pooling import build_strategy_loaders

        return build_strategy_loaders(dict(config), repo_root, seed=seed)
    from ...datasets.paper_backup.sequence_views import build_sequence_loaders

    return build_sequence_loaders(dict(config), repo_root, seed=seed)


def train_from_config(
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    seed: int | None = None,
    device: str | None = None,
    backend: str | None = None,
    output_root: str | Path | None = None,
    run_time: str | None = None,
) -> dict[str, Any]:
    """Train one config and persist a non-overwriting, provenance-rich run."""

    repo_root = Path(repo_root).resolve()
    config = copy.deepcopy(dict(config))
    validate_config(config, repo_root, check_files=True)
    seed_value = int(seed if seed is not None else config.get("train", {}).get("seed", 42))
    set_random_seed(seed_value)
    requested_device = str(device or config.get("train", {}).get("device", "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Paper-Backup requested CUDA but CUDA is unavailable")
    actual_device = torch.device(requested_device)
    loaders, loader_info = _build_loaders(config, repo_root, seed_value)
    model_type = str(config["model"]["type"])
    model = build_model(config["model"], backend_override=backend).to(actual_device)
    train_cfg = config.get("train", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", train_cfg.get("lr", 1e-3))),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    max_epochs = int(train_cfg.get("epochs", 1))
    patience = max(1, int(train_cfg.get("patience", max_epochs)))
    monitor = str(train_cfg.get("monitor", "battery_macro.rmse"))
    history = []
    best_metric = float("inf")
    best_epoch = 0
    best_state = None
    no_improvement = 0
    for epoch in range(1, max_epochs + 1):
        # The optimizer only needs the training loss. Battery/strategy metrics
        # are computed on validation and final test data, so materializing one
        # Python prediction row per training cycle here is pure overhead.
        train_metrics, _ = run_epoch(
            model_type,
            model,
            loaders["train"],
            actual_device,
            optimizer=optimizer,
            collect_predictions=False,
        )
        with torch.no_grad():
            val_metrics, _ = run_epoch(model_type, model, loaders["val"], actual_device, collect_predictions=True)
        monitored = _metric_at(val_metrics, monitor)
        if not np.isfinite(monitored):
            monitored = float(val_metrics.get("loss", float("inf")))
        improved = monitored < best_metric
        if improved:
            best_metric = monitored
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "monitor": monitored,
                "best": improved,
            }
        )
        print(
            f"[epoch] model={model_type} epoch={epoch}/{max_epochs} "
            f"train_loss={float(train_metrics['loss']):.8f} "
            f"val_loss={float(val_metrics['loss']):.8f} "
            f"{monitor}={monitored:.8f} best={str(improved).lower()} "
            f"patience={no_improvement}/{patience}",
            flush=True,
        )
        if no_improvement >= patience:
            print(
                f"[early-stop] model={model_type} epoch={epoch} "
                f"best_epoch={best_epoch} best_{monitor}={best_metric:.8f}",
                flush=True,
            )
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        test_metrics, prediction_rows = run_epoch(model_type, model, loaders["test"], actual_device, collect_predictions=True)
    output = config["output"]
    root = _resolve_root(config, repo_root, output_root)
    runtime = str(run_time or config.get("run", {}).get("run_time") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir = _unique_run_dir(root, str(output["experiment_id"]), str(output.get("model_id", model_type)), str(output.get("data_id", config.get("data", {}).get("domain_id", "dataset"))), runtime, seed_value)
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = copy.deepcopy(config)
    resolved_config.setdefault("run", {})["run_time"] = run_dir.parent.name.removeprefix("runtime_")
    resolved_config["run"]["seed"] = seed_value
    resolved_config["run"]["actual_device"] = str(actual_device)
    resolved_config["run"]["backend_override"] = backend
    manifest = {
        "status": "completed",
        "paper_version": "Paper-Backup",
        "experiment_id": output["experiment_id"],
        "model_id": output.get("model_id", model_type),
        "data_id": output.get("data_id", config.get("data", {}).get("domain_id", "dataset")),
        "seed": seed_value,
        "device": str(actual_device),
        "model_type": model_type,
        "model_input_kind": model_input_kind(model_type),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "soh_only": True,
        "metadata_in_forward": False,
        "cycle_lifetime_auxiliary": False,
        "best_epoch": best_epoch,
        "monitor": monitor,
        "runtime_optimizations": {
            "train_prediction_rows_per_epoch": False,
            "loss_device_sync": "once_per_epoch",
            "non_blocking_device_transfer": actual_device.type == "cuda",
        },
        "loader_info": loader_info,
    }
    _write_json(run_dir / "resolved_config.json", resolved_config)
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "history.json", history)
    _write_json(run_dir / "test_metrics.json", test_metrics)
    _write_json(run_dir / "predictions.json", prediction_rows)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_epoch": best_epoch,
        },
        run_dir / "best.pt",
    )
    return {"status": "completed", "run_dir": str(run_dir), "manifest": manifest, "test_metrics": test_metrics}


__all__ = ["PHASE_KEYS", "run_epoch", "train_from_config"]
