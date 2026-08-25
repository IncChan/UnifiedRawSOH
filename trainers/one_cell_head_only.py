"""Head-only one-reference-cell adaptation for Paper-v1 E3."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from UnifiedRawSOH.datasets.one_cell import build_one_cell_loaders
from UnifiedRawSOH.evaluation.metrics import compute_metrics
from UnifiedRawSOH.models.raw_soh_model import build_raw_soh_model
from UnifiedRawSOH.utils.config import load_config, save_json
from UnifiedRawSOH.utils.seed import set_random_seed


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _encoder_hash(model):
    return _tensor_hash(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("head.")
    )


def load_strict_lodo_model(
    config,
    checkpoint_path,
    target_domain_id,
    backend_override=None,
):
    """Strictly validate and load a no-cycle LODO source checkpoint."""

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LODO checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload or "config" not in payload:
        raise ValueError(
            "LODO checkpoint must contain model and resolved config payloads"
        )
    source_config = payload["config"]
    source_model = source_config.get("model", {})
    if source_model.get("use_cycle_prediction") is not False:
        raise ValueError("Source checkpoint is not a no-cycle model")
    if source_model.get("use_predicted_cycle_for_soh") is not False:
        raise ValueError("Source checkpoint still injects predicted cycle")
    if float(source_config.get("train", {}).get("lambda_cycle", -1.0)) != 0.0:
        raise ValueError("Source checkpoint has nonzero cycle auxiliary weight")

    experiment = source_config.get("experiment", {})
    source_domains = [
        str(value) for value in experiment.get("source_domain_ids", [])
    ]
    source_target = experiment.get("target_domain_id")
    if not source_domains:
        raise ValueError("Source checkpoint does not record LODO source_domain_ids")
    if str(target_domain_id) in source_domains:
        raise ValueError(
            f"Source checkpoint trained on target domain {target_domain_id!r}"
        )
    if source_target is not None and str(source_target) != str(target_domain_id):
        raise ValueError(
            f"Checkpoint target {source_target!r} does not match job target "
            f"{target_domain_id!r}"
        )

    current_model = config["model"]
    if current_model.get("use_cycle_prediction") is not False:
        raise ValueError("One-cell config must disable cycle prediction")
    if current_model.get("use_predicted_cycle_for_soh") is not False:
        raise ValueError("One-cell config must disable predicted-cycle injection")
    model = build_raw_soh_model(
        current_model,
        backend_override=backend_override,
    )
    model.load_state_dict(payload["model"], strict=True)
    if model.cycle_head is not None or model.cycle_adapter is not None:
        raise ValueError("No-cycle model unexpectedly contains cycle modules")
    return model, payload, {
        "input_path": str(checkpoint_path),
        "resolved_path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "source_domain_ids": source_domains,
        "source_target_domain_id": source_target,
        "strict_load": True,
    }


def freeze_head_only(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any(not name.startswith("head.") for name in trainable):
        raise RuntimeError(f"Invalid head-only trainable parameters: {trainable}")
    return trainable


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
    }


def extract_frozen_features(model, loader, device, retain_metadata=False):
    model.eval()
    features = []
    temperatures = []
    truths = []
    metadata = []
    with torch.no_grad():
        for batch in loader:
            tensors = {
                key: value.to(device)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            feature = model.encode(**_model_inputs(tensors))
            features.append(feature.detach().cpu())
            temperatures.append(tensors["t0_temperature_norm"].detach().cpu())
            truths.append(tensors["soh"].detach().cpu())
            if retain_metadata:
                size = int(tensors["soh"].size(0))
                for index in range(size):
                    metadata.append(
                        {
                            "battery_id": str(batch["battery_id"][index]),
                            "condition": str(batch["condition"][index]),
                            "cycle_id": int(batch["cycle_id"][index]),
                        }
                    )
    if not features:
        raise ValueError("Feature extraction received an empty loader")
    return {
        "feature": torch.cat(features, dim=0),
        "t0": torch.cat(temperatures, dim=0),
        "soh": torch.cat(truths, dim=0),
        "metadata": metadata,
    }


def _feature_loader(data, batch_size, shuffle):
    return DataLoader(
        TensorDataset(data["feature"], data["t0"], data["soh"]),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
    )


def _head_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.head.train(training)
    total = 0.0
    count = 0
    criterion = torch.nn.MSELoss()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for feature, t0, truth in loader:
            feature = feature.to(device)
            t0 = t0.to(device)
            truth = truth.to(device)
            prediction = model.predict_from_signal_feature(feature, t0)
            loss = criterion(prediction, truth)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            size = int(truth.size(0))
            total += float(loss.detach().item()) * size
            count += size
    return total / max(count, 1)


def fit_head_only(model, train_data, val_data, config, device):
    one_cell = config["one_cell"]
    batch_size = int(one_cell["batch_size"])
    train_loader = _feature_loader(train_data, batch_size, True)
    val_loader = _feature_loader(val_data, batch_size, False)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(one_cell["learning_rate"]),
        weight_decay=float(one_cell["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(one_cell["scheduler_factor"]),
        patience=int(one_cell["scheduler_patience"]),
        threshold=float(one_cell["scheduler_threshold"]),
        min_lr=float(one_cell["scheduler_min_lr"]),
    )
    epochs = int(one_cell["epochs"])
    patience = int(one_cell["patience"])
    best_loss = float("inf")
    best_head = None
    stale = 0
    history = []
    for epoch in range(epochs):
        train_loss = _head_epoch(model, train_loader, device, optimizer)
        val_loss = _head_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "support_train_mse": train_loss,
                "support_validation_mse": val_loss,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_head = copy.deepcopy(model.head.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_head is None:
        raise RuntimeError("Head-only fitting did not produce a checkpoint")
    model.head.load_state_dict(best_head, strict=True)
    return history, best_loss


def _mean_metrics(rows):
    result = {}
    for metric in ("mae", "mape", "mse", "rmse"):
        values = [float(row[metric]) for row in rows]
        result[metric] = float(np.mean(values))
    return result


def evaluate_all_test(model, test_data, device):
    model.eval()
    feature = test_data["feature"].to(device)
    t0 = test_data["t0"].to(device)
    with torch.no_grad():
        prediction = model.predict_from_signal_feature(feature, t0)
    truths = test_data["soh"].numpy().reshape(-1)
    predictions = prediction.detach().cpu().numpy().reshape(-1)
    rows = []
    for truth, predicted, metadata in zip(
        truths, predictions, test_data["metadata"]
    ):
        rows.append(
            {
                **metadata,
                "truth": float(truth),
                "prediction": float(predicted),
                "error": float(predicted - truth),
            }
        )

    cells = {}
    for row in rows:
        cells.setdefault((row["condition"], row["battery_id"]), []).append(row)
    metrics_by_cell = []
    for (group, cell), values in sorted(cells.items()):
        metrics = compute_metrics(
            [value["truth"] for value in values],
            [value["prediction"] for value in values],
        )
        metrics_by_cell.append(
            {
                "test_group": group,
                "test_cell": cell,
                "n_samples": len(values),
                **metrics,
            }
        )
    grouped = {}
    for row in metrics_by_cell:
        grouped.setdefault(row["test_group"], []).append(row)
    metrics_by_group = []
    for group, values in sorted(grouped.items()):
        metrics_by_group.append(
            {
                "test_group": group,
                "n_test_cells": len(values),
                "n_samples": sum(value["n_samples"] for value in values),
                **_mean_metrics(values),
            }
        )
    overall = {
        "aggregation": "equal test-cell mean within group, then equal test-group macro",
        "n_test_groups": len(metrics_by_group),
        "n_test_cells": len(metrics_by_cell),
        "n_samples": len(rows),
        **_mean_metrics(metrics_by_group),
    }
    return rows, metrics_by_cell, metrics_by_group, overall


def _write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_job(job_spec, backend_override=None, device_override=None):
    output_dir = Path(job_spec["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "status.json", {"status": "running"})
    config = load_config(job_spec["config_path"])
    target = config["one_cell"]["target_domain_id"]
    seed_text = str(job_spec["support_choice"])
    seed = (
        int(seed_text)
        if seed_text.lstrip("-").isdigit()
        else int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    )
    set_random_seed(seed, True)
    requested_device = device_override or config["train"].get("device", "cuda")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for one-cell adaptation")
    device = torch.device(requested_device)

    model, _, checkpoint_manifest = load_strict_lodo_model(
        config,
        job_spec["checkpoint_path"],
        target,
        backend_override=backend_override,
    )
    if checkpoint_manifest["sha256"] != job_spec["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after the job manifest was generated")
    model = model.to(device)
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    encoder_hash_before = _encoder_hash(model)
    trainable_names = freeze_head_only(model)
    save_json(
        output_dir / "trainable_parameter_names.json",
        {"trainable_parameter_names": trainable_names},
    )

    loaders, selection, support_split, inventory = build_one_cell_loaders(
        config,
        repo_root=Path(__file__).resolve().parents[2],
        support_group=job_spec["support_group"],
        support_choice=job_spec["support_choice"],
        selected_cell=job_spec["support_cell"],
    )
    save_json(output_dir / "support_selection.json", selection)
    save_json(output_dir / "support_split.json", support_split)

    train_features = extract_frozen_features(model, loaders["train"], device)
    val_features = extract_frozen_features(model, loaders["val"], device)
    test_features = extract_frozen_features(
        model, loaders["test"], device, retain_metadata=True
    )
    history, best_val = fit_head_only(
        model, train_features, val_features, config, device
    )
    encoder_hash_after = _encoder_hash(model)
    if encoder_hash_before != encoder_hash_after:
        raise RuntimeError("Frozen encoder parameter hash changed")

    changed = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(initial_parameters[name], parameter.detach().cpu())
    ]
    illegal = [name for name in changed if not name.startswith("head.")]
    if illegal:
        raise RuntimeError(f"Non-head parameters changed: {illegal}")

    predictions, by_cell, by_group, overall = evaluate_all_test(
        model, test_features, device
    )
    expected_groups = set(config["one_cell"]["support_groups"])
    if {row["test_group"] for row in by_group} != expected_groups:
        raise RuntimeError("Adapted model was not evaluated on every test group")
    if overall["n_samples"] != inventory["all_test_sample_count"]:
        raise RuntimeError("Target test sample count changed across jobs")

    _write_csv(output_dir / "training_history.csv", history)
    _write_csv(output_dir / "predictions.csv", predictions)
    _write_csv(output_dir / "metrics_by_test_cell.csv", by_cell)
    _write_csv(output_dir / "metrics_by_test_group.csv", by_group)
    save_json(
        output_dir / "metrics_overall.json",
        {
            **overall,
            "checkpoint_seed": job_spec.get("checkpoint_seed"),
            "support_group": job_spec["support_group"],
            "support_choice": str(job_spec["support_choice"]),
            "support_cell": selection["support_cell"],
            "best_support_validation_mse": best_val,
            "encoder_sha256_before": encoder_hash_before,
            "encoder_sha256_after": encoder_hash_after,
            "changed_parameter_names": changed,
            "source_checkpoint": checkpoint_manifest,
        },
    )
    torch.save(
        {
            "head": model.head.state_dict(),
            "checkpoint_seed": job_spec.get("checkpoint_seed"),
            "source_checkpoint_sha256": checkpoint_manifest["sha256"],
            "support_selection": selection,
        },
        output_dir / "best_head.pt",
    )
    status = {
        "status": "completed",
        "metrics_file": "metrics_overall.json",
        "best_head_file": "best_head.pt",
    }
    save_json(output_dir / "status.json", status)
    return status


def parse_args():
    parser = argparse.ArgumentParser("Paper-v1 one-cell head-only job")
    parser.add_argument("--job-spec", required=True)
    parser.add_argument("--backend-override", default=None)
    parser.add_argument("--device-override", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    job_spec = json.loads(Path(args.job_spec).read_text(encoding="utf-8"))
    output_dir = Path(job_spec["output_dir"])
    try:
        result = run_job(
            job_spec,
            backend_override=args.backend_override,
            device_override=args.device_override,
        )
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            output_dir / "status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
