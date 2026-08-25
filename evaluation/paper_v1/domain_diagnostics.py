#!/usr/bin/env python3
"""Post-training diagnostics for the Paper-v1 E2 shared models.

The diagnostics are deliberately read-only with respect to the source run and
use validation samples only. They test representation domain dominance,
domain-wise residual calibration, and shared-encoder gradient conflict.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UnifiedRawSOH.datasets.loaders import build_unified_loaders  # noqa: E402
from UnifiedRawSOH.evaluation.metrics import compute_metrics  # noqa: E402
from UnifiedRawSOH.models.raw_soh_model import build_raw_soh_model  # noqa: E402
from UnifiedRawSOH.utils.config import load_config, save_json  # noqa: E402
from UnifiedRawSOH.utils.seed import set_random_seed  # noqa: E402


MODEL_INPUT_KEYS = (
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
ENCODER_PREFIXES = (
    "cc_branch.",
    "cv_branch.",
    "cc_fusion_proj.",
    "cv_fusion_proj.",
    "cc_to_cv_bridge.",
)


def _resolve_path(repo_root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(repo_root) / path).resolve()


def _write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metadata_values(batch, key, count):
    value = batch[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().reshape(-1).tolist()
    values = list(value)
    if len(values) != count:
        raise ValueError(f"Batch field {key!r} has {len(values)} values, expected {count}")
    return values


def collect_features(model, loader, device, max_samples_per_domain=0):
    """Extract z_health and predictions while preserving sample identities."""

    model.eval()
    features, truths, predictions = [], [], []
    domains, batteries, conditions, cycle_ids = [], [], [], []
    kept = Counter()
    limit = int(max_samples_per_domain or 0)
    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: value.to(device)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            aux = model.forward_with_aux(**{key: tensors[key] for key in MODEL_INPUT_KEYS})
            batch_features = aux["z_health"].detach().cpu().numpy()
            batch_truths = tensors["soh"].detach().cpu().numpy().reshape(-1)
            batch_predictions = aux["soh_pred"].detach().cpu().numpy().reshape(-1)
            count = int(batch_truths.size)
            batch_domains = [str(value) for value in _metadata_values(batch, "domain_id", count)]
            batch_batteries = [str(value) for value in _metadata_values(batch, "battery_id", count)]
            batch_conditions = [str(value) for value in _metadata_values(batch, "condition", count)]
            batch_cycle_ids = _metadata_values(batch, "cycle_id", count)
            for index, domain in enumerate(batch_domains):
                if limit > 0 and kept[domain] >= limit:
                    continue
                features.append(batch_features[index])
                truths.append(float(batch_truths[index]))
                predictions.append(float(batch_predictions[index]))
                domains.append(domain)
                batteries.append(batch_batteries[index])
                conditions.append(batch_conditions[index])
                cycle_ids.append(int(batch_cycle_ids[index]))
                kept[domain] += 1
    if not features:
        raise ValueError("No validation features were collected")
    return {
        "features": np.asarray(features, dtype=np.float32),
        "truth": np.asarray(truths, dtype=np.float64),
        "prediction": np.asarray(predictions, dtype=np.float64),
        "domain": np.asarray(domains, dtype=str),
        "battery": np.asarray(batteries, dtype=str),
        "condition": np.asarray(conditions, dtype=str),
        "cycle_id": np.asarray(cycle_ids, dtype=np.int64),
    }


def battery_group_split(domains, batteries, test_fraction=0.25, seed=42):
    """Return battery-disjoint train/test masks, stratified by domain."""

    domains = np.asarray(domains, dtype=str)
    batteries = np.asarray(batteries, dtype=str)
    if domains.shape != batteries.shape:
        raise ValueError("domains and batteries must have the same shape")
    rng = np.random.default_rng(int(seed))
    train = np.zeros(domains.size, dtype=bool)
    test = np.zeros(domains.size, dtype=bool)
    assignment = {}
    for domain in sorted(set(domains.tolist())):
        domain_mask = domains == domain
        unique = np.unique(batteries[domain_mask])
        if unique.size < 2:
            raise ValueError(
                f"Domain {domain!r} needs at least two validation batteries for a battery-disjoint split"
            )
        shuffled = unique.copy()
        rng.shuffle(shuffled)
        n_test = min(unique.size - 1, max(1, int(round(unique.size * float(test_fraction)))))
        test_batteries = set(shuffled[:n_test].tolist())
        domain_test = domain_mask & np.isin(batteries, list(test_batteries))
        domain_train = domain_mask & ~domain_test
        train |= domain_train
        test |= domain_test
        assignment[domain] = {
            "train_batteries": sorted(set(batteries[domain_train].tolist())),
            "test_batteries": sorted(test_batteries),
        }
    if np.any(train & test) or not np.all(train | test):
        raise RuntimeError("Invalid battery-group split")
    return train, test, assignment


def matched_health_indices(
    truths,
    domains,
    bin_width=0.02,
    bin_origin=0.0,
    max_per_domain_bin=0,
    seed=42,
    require_all_domains=True,
):
    """Balance domains within SOH bins so a domain probe cannot use SOH alone."""

    truths = np.asarray(truths, dtype=np.float64)
    domains = np.asarray(domains, dtype=str)
    if truths.shape != domains.shape:
        raise ValueError("truths and domains must have the same shape")
    if float(bin_width) <= 0:
        raise ValueError("bin_width must be positive")
    bin_ids = np.floor((truths - float(bin_origin)) / float(bin_width)).astype(np.int64)
    all_domains = sorted(set(domains.tolist()))
    rng = np.random.default_rng(int(seed))
    selected = []
    rows = []
    cap = int(max_per_domain_bin or 0)
    for bin_id in sorted(set(bin_ids.tolist())):
        present = {}
        for domain in all_domains:
            indices = np.flatnonzero((bin_ids == bin_id) & (domains == domain))
            if indices.size:
                present[domain] = indices
        if require_all_domains and len(present) != len(all_domains):
            continue
        if len(present) < 2:
            continue
        target = min(values.size for values in present.values())
        if cap > 0:
            target = min(target, cap)
        if target <= 0:
            continue
        for domain, indices in sorted(present.items()):
            chosen = rng.choice(indices, size=target, replace=False)
            selected.extend(chosen.tolist())
            rows.append(
                {
                    "soh_bin": int(bin_id),
                    "soh_low": float(bin_origin + bin_id * bin_width),
                    "soh_high": float(bin_origin + (bin_id + 1) * bin_width),
                    "domain": domain,
                    "available": int(indices.size),
                    "selected": int(target),
                }
            )
    selected = np.asarray(sorted(selected), dtype=np.int64)
    if selected.size == 0:
        raise ValueError("SOH-bin matching selected no samples; widen bins or relax domain coverage")
    return selected, rows


def _classification_metrics(labels, predictions, domain_names):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    confusion = np.zeros((len(domain_names), len(domain_names)), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        confusion[int(truth), int(prediction)] += 1
    per_domain = {}
    f1_values = []
    for index, domain in enumerate(domain_names):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_domain[domain] = {
            "support": support,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(np.mean(f1_values)),
        "per_domain": per_domain,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": list(domain_names),
    }


def _pca_projection(features):
    features = np.asarray(features, dtype=np.float64)
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    components = min(2, right.shape[0])
    projection = centered @ right[:components].T
    if components == 1:
        projection = np.column_stack([projection[:, 0], np.zeros(features.shape[0])])
    variance = singular_values**2
    explained = variance[:components] / max(float(variance.sum()), np.finfo(float).eps)
    return projection[:, :2], explained.tolist()


def _fit_linear_probe(features, domains, train_mask, test_mask, config, seed):
    """Fit one standardized linear domain probe and return test metrics."""

    features = np.asarray(features, dtype=np.float32)
    domains = np.asarray(domains, dtype=str)
    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Domain probe needs non-empty train and test samples")
    domain_names = sorted(set(domains.tolist()))
    if len(domain_names) < 2:
        raise ValueError("Domain probe needs at least two domains")
    domain_to_label = {domain: index for index, domain in enumerate(domain_names)}
    labels = np.asarray([domain_to_label[value] for value in domains], dtype=np.int64)

    mean = features[train_mask].mean(axis=0, keepdims=True)
    std = features[train_mask].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    normalized = (features - mean) / std
    probe_device = torch.device(str(config.get("probe_device", "cpu")))
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        probe = torch.nn.Linear(normalized.shape[1], len(domain_names)).to(probe_device)
        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=float(config.get("learning_rate", 0.03)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )
        x_train = torch.as_tensor(
            normalized[train_mask], dtype=torch.float32, device=probe_device
        )
        y_train = torch.as_tensor(
            labels[train_mask], dtype=torch.long, device=probe_device
        )
        for _ in range(int(config.get("epochs", 200))):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(probe(x_train), y_train)
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.inference_mode():
            x_test = torch.as_tensor(
                normalized[test_mask], dtype=torch.float32, device=probe_device
            )
            predicted = probe(x_test).argmax(dim=1).cpu().numpy()
    finally:
        torch.random.set_rng_state(generator_state)
    return (
        _classification_metrics(labels[test_mask], predicted, domain_names),
        normalized,
        domain_names,
    )


def _pairwise_matched_split(data, domain_a, domain_b, config, seed):
    """Find a deterministic battery split with matched train/test SOH support."""

    pair_mask = np.isin(data["domain"], [domain_a, domain_b])
    pair_indices = np.flatnonzero(pair_mask)
    pair_domains = data["domain"][pair_indices]
    pair_batteries = data["battery"][pair_indices]
    attempts = int(config.get("pairwise_split_search_attempts", 64))
    if attempts <= 0:
        raise ValueError(
            "representation_probe.pairwise_split_search_attempts must be positive"
        )
    match_options = {
        "bin_width": float(config.get("soh_bin_width", 0.02)),
        "bin_origin": float(config.get("soh_bin_origin", 0.0)),
        "max_per_domain_bin": int(config.get("max_per_domain_bin", 0)),
        "require_all_domains": True,
    }
    best = None
    failures = []
    for attempt in range(attempts):
        split_seed = int(seed) + attempt
        train_mask, test_mask, assignment = battery_group_split(
            pair_domains,
            pair_batteries,
            test_fraction=float(config.get("test_battery_fraction", 0.25)),
            seed=split_seed,
        )
        train_pool = np.flatnonzero(train_mask)
        test_pool = np.flatnonzero(test_mask)
        try:
            train_local, train_rows = matched_health_indices(
                data["truth"][pair_indices[train_pool]],
                pair_domains[train_pool],
                seed=split_seed,
                **match_options,
            )
            test_local, test_rows = matched_health_indices(
                data["truth"][pair_indices[test_pool]],
                pair_domains[test_pool],
                seed=split_seed + 1,
                **match_options,
            )
        except ValueError as error:
            failures.append(str(error))
            continue
        train_selected = pair_indices[train_pool[train_local]]
        test_selected = pair_indices[test_pool[test_local]]
        train_bins = {int(row["soh_bin"]) for row in train_rows}
        test_bins = {int(row["soh_bin"]) for row in test_rows}
        score = (
            min(int(train_selected.size), int(test_selected.size)),
            len(test_bins),
            int(test_selected.size),
            len(train_bins),
            int(train_selected.size),
            -attempt,
        )
        candidate = {
            "score": score,
            "attempt": attempt,
            "split_seed": split_seed,
            "assignment": assignment,
            "train_selected": train_selected,
            "test_selected": test_selected,
            "train_rows": train_rows,
            "test_rows": test_rows,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        detail = failures[0] if failures else "no valid split candidate"
        raise ValueError(
            f"No battery-disjoint SOH overlap for {domain_a} vs {domain_b} "
            f"after {attempts} split attempts: {detail}"
        )
    return best


def run_pairwise_representation_probe(data, config, seed, output_dir):
    """Run binary domain probes on each pair's battery-disjoint SOH overlap."""

    domain_names = sorted(set(data["domain"].tolist()))
    pair_reports = []
    match_rows = []
    for pair_index, (domain_a, domain_b) in enumerate(combinations(domain_names, 2)):
        pair_id = f"{domain_a}__vs__{domain_b}"
        try:
            split = _pairwise_matched_split(
                data, domain_a, domain_b, config, seed + pair_index * 1000
            )
            train_selected = split["train_selected"]
            test_selected = split["test_selected"]
            selected = np.concatenate([train_selected, test_selected])
            train_mask = np.arange(selected.size) < train_selected.size
            test_mask = ~train_mask
            metrics, _, _ = _fit_linear_probe(
                data["features"][selected],
                data["domain"][selected],
                train_mask,
                test_mask,
                config,
                seed + pair_index,
            )
            pair_report = {
                "pair_id": pair_id,
                "domain_a": domain_a,
                "domain_b": domain_b,
                "status": "completed",
                "split_seed": int(split["split_seed"]),
                "split_search_attempt": int(split["attempt"]),
                "n_train_samples": int(train_selected.size),
                "n_test_samples": int(test_selected.size),
                "chance_accuracy": 0.5,
                "battery_assignment": split["assignment"],
                **metrics,
            }
            for probe_split, rows in (
                ("train", split["train_rows"]),
                ("test", split["test_rows"]),
            ):
                match_rows.extend(
                    {
                        "pair_id": pair_id,
                        "domain_a": domain_a,
                        "domain_b": domain_b,
                        "split_seed": int(split["split_seed"]),
                        "probe_split": probe_split,
                        **row,
                    }
                    for row in rows
                )
        except Exception as error:  # Keep other domain pairs usable.
            pair_report = {
                "pair_id": pair_id,
                "domain_a": domain_a,
                "domain_b": domain_b,
                "status": "unavailable",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        pair_reports.append(pair_report)

    completed = [row for row in pair_reports if row["status"] == "completed"]
    if completed:
        status = "completed" if len(completed) == len(pair_reports) else "partial"
        result = {
            "status": status,
            "definition": (
                "macro average of pairwise battery-disjoint binary linear domain probes "
                "on each pair's SOH-bin overlap"
            ),
            "split_policy": (
                "pair-specific battery-disjoint split selected by SOH-support "
                "coverage only"
            ),
            "accuracy": float(np.mean([row["accuracy"] for row in completed])),
            "macro_f1": float(np.mean([row["macro_f1"] for row in completed])),
            "chance_accuracy": 0.5,
            "n_pairs_total": len(pair_reports),
            "n_pairs_completed": len(completed),
            "n_pairs_unavailable": len(pair_reports) - len(completed),
            "n_train_samples_across_pairs": int(
                sum(row["n_train_samples"] for row in completed)
            ),
            "n_test_samples_across_pairs": int(
                sum(row["n_test_samples"] for row in completed)
            ),
            "pairs": pair_reports,
        }
    else:
        result = {
            "status": "unavailable",
            "definition": (
                "battery-disjoint binary linear domain probes on pairwise "
                "SOH-bin overlap"
            ),
            "n_pairs_total": len(pair_reports),
            "n_pairs_completed": 0,
            "n_pairs_unavailable": len(pair_reports),
            "pairs": pair_reports,
        }
    output_dir = Path(output_dir)
    _write_csv(
        output_dir / "representation_pairwise_health_matching.csv",
        match_rows,
        fieldnames=[
            "pair_id",
            "domain_a",
            "domain_b",
            "split_seed",
            "probe_split",
            "soh_bin",
            "soh_low",
            "soh_high",
            "domain",
            "available",
            "selected",
        ],
    )
    save_json(output_dir / "representation_pairwise_probe.json", result)
    return result

def run_strict_representation_probe(data, config, seed, output_dir):
    """Fit the supplemental five-domain probe on their common SOH bins."""


    full_train, full_test, assignment = battery_group_split(
        data["domain"],
        data["battery"],
        test_fraction=float(config.get("test_battery_fraction", 0.25)),
        seed=seed,
    )
    match_options = {
        "bin_width": float(config.get("soh_bin_width", 0.02)),
        "bin_origin": float(config.get("soh_bin_origin", 0.0)),
        "max_per_domain_bin": int(config.get("max_per_domain_bin", 0)),
        "require_all_domains": bool(config.get("require_all_domains", True)),
    }
    train_pool = np.flatnonzero(full_train)
    test_pool = np.flatnonzero(full_test)
    train_local, train_rows = matched_health_indices(
        data["truth"][train_pool],
        data["domain"][train_pool],
        seed=seed,
        **match_options,
    )
    test_local, test_rows = matched_health_indices(
        data["truth"][test_pool],
        data["domain"][test_pool],
        seed=seed + 1,
        **match_options,
    )
    train_selected = train_pool[train_local]
    test_selected = test_pool[test_local]
    selected = np.concatenate([train_selected, test_selected])
    train_mask = np.arange(selected.size) < train_selected.size
    test_mask = ~train_mask
    match_rows = [
        {**row, "probe_split": split}
        for split, rows in (("train", train_rows), ("test", test_rows))
        for row in rows
    ]
    domains = data["domain"][selected]
    result, normalized, domain_names = _fit_linear_probe(
        data["features"][selected],
        domains,
        train_mask,
        test_mask,
        config,
        seed,
    )
    result.update(
        {
            "status": "completed",
            "definition": "strict five-domain linear probe on common SOH-bin-matched z_health",
            "split_policy": "battery-disjoint within each validation domain",
            "n_matched_samples": int(selected.size),
            "n_train_samples": int(train_mask.sum()),
            "n_test_samples": int(test_mask.sum()),
            "chance_accuracy": 1.0 / len(domain_names),
            "domain_labels": domain_names,
            "battery_assignment": assignment,
            "matched_counts_by_domain": dict(Counter(domains.tolist())),
        }
    )
    projection, explained = _pca_projection(normalized)
    projection_rows = []
    for local_index, original_index in enumerate(selected):
        projection_rows.append(
            {
                "pc1": float(projection[local_index, 0]),
                "pc2": float(projection[local_index, 1]),
                "domain": str(data["domain"][original_index]),
                "battery_id": str(data["battery"][original_index]),
                "condition": str(data["condition"][original_index]),
                "cycle_id": int(data["cycle_id"][original_index]),
                "soh": float(data["truth"][original_index]),
                "probe_split": "train" if train_mask[local_index] else "test",
            }
        )
    output_dir = Path(output_dir)
    _write_csv(output_dir / "representation_strict_health_matching.csv", match_rows)
    _write_csv(output_dir / "representation_strict_pca.csv", projection_rows)
    result["pca_explained_variance_ratio"] = explained
    save_json(output_dir / "representation_strict_probe.json", result)
    return result


def fit_affine_calibration(predictions, truths, ridge=1e-6):
    """Fit truth = scale * prediction + bias with a small ridge penalty."""

    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    truths = np.asarray(truths, dtype=np.float64).reshape(-1)
    if predictions.size != truths.size or predictions.size < 2:
        raise ValueError("Affine calibration needs at least two paired samples")
    design = np.column_stack([predictions, np.ones_like(predictions)])
    penalty = np.diag([float(ridge), 0.0])
    scale, bias = np.linalg.solve(design.T @ design + penalty, design.T @ truths)
    return float(scale), float(bias)


def run_residual_calibration(data, config, seed, output_dir):
    """Measure removable per-domain affine bias on held-out validation batteries."""

    train_mask, test_mask, assignment = battery_group_split(
        data["domain"],
        data["battery"],
        test_fraction=float(config.get("test_battery_fraction", 0.25)),
        seed=seed,
    )
    width = float(config.get("soh_bin_width", 0.02))
    origin = float(config.get("soh_bin_origin", 0.0))
    ridge = float(config.get("ridge", 1e-6))
    per_domain = {}
    bin_rows = []
    all_truth, all_before, all_after = [], [], []
    for domain in sorted(set(data["domain"].tolist())):
        domain_mask = data["domain"] == domain
        fit_mask = train_mask & domain_mask
        eval_mask = test_mask & domain_mask
        scale, bias = fit_affine_calibration(
            data["prediction"][fit_mask], data["truth"][fit_mask], ridge=ridge
        )
        truth = data["truth"][eval_mask]
        before = data["prediction"][eval_mask]
        after = scale * before + bias
        before_metrics = compute_metrics(truth, before)
        after_metrics = compute_metrics(truth, after)
        per_domain[domain] = {
            "scale": scale,
            "bias": bias,
            "n_fit_samples": int(fit_mask.sum()),
            "n_eval_samples": int(eval_mask.sum()),
            "before": before_metrics,
            "after": after_metrics,
            "rmse_change": float(after_metrics["rmse"] - before_metrics["rmse"]),
        }
        all_truth.extend(truth.tolist())
        all_before.extend(before.tolist())
        all_after.extend(after.tolist())
        bin_ids = np.floor((truth - origin) / width).astype(np.int64)
        for bin_id in sorted(set(bin_ids.tolist())):
            mask = bin_ids == bin_id
            bin_rows.append(
                {
                    "domain": domain,
                    "soh_bin": int(bin_id),
                    "soh_low": float(origin + bin_id * width),
                    "soh_high": float(origin + (bin_id + 1) * width),
                    "n_samples": int(mask.sum()),
                    "mean_residual_before": float(np.mean(before[mask] - truth[mask])),
                    "mean_residual_after": float(np.mean(after[mask] - truth[mask])),
                    "rmse_before": float(compute_metrics(truth[mask], before[mask])["rmse"]),
                    "rmse_after": float(compute_metrics(truth[mask], after[mask])["rmse"]),
                }
            )
    before_micro = compute_metrics(all_truth, all_before)
    after_micro = compute_metrics(all_truth, all_after)
    before_macro = float(np.mean([values["before"]["rmse"] for values in per_domain.values()]))
    after_macro = float(np.mean([values["after"]["rmse"] for values in per_domain.values()]))
    result = {
        "definition": "per-domain affine calibration fitted on validation batteries and evaluated on disjoint validation batteries",
        "split_policy": "battery-disjoint within each validation domain",
        "battery_assignment": assignment,
        "before_micro": before_micro,
        "after_micro": after_micro,
        "before_domain_macro_rmse": before_macro,
        "after_domain_macro_rmse": after_macro,
        "domain_macro_rmse_change": after_macro - before_macro,
        "per_domain": per_domain,
    }
    output_dir = Path(output_dir)
    _write_csv(output_dir / "residual_by_soh_bin.csv", bin_rows)
    save_json(output_dir / "residual_calibration.json", result)
    return result


def gradient_cosine_report(gradients):
    """Build the pairwise cosine matrix and conflict summary."""

    domains = sorted(gradients)
    vectors = {
        domain: torch.as_tensor(gradients[domain], dtype=torch.float64).reshape(-1)
        for domain in domains
    }
    matrix = np.eye(len(domains), dtype=np.float64)
    pairs = []
    for left_index, left in enumerate(domains):
        for right_index in range(left_index + 1, len(domains)):
            right = domains[right_index]
            denominator = torch.linalg.vector_norm(vectors[left]) * torch.linalg.vector_norm(vectors[right])
            cosine = 0.0 if float(denominator) == 0.0 else float(
                torch.dot(vectors[left], vectors[right]) / denominator
            )
            matrix[left_index, right_index] = cosine
            matrix[right_index, left_index] = cosine
            pairs.append({"domain_a": left, "domain_b": right, "cosine": cosine, "conflict": cosine < 0.0})
    cosines = [row["cosine"] for row in pairs]
    return {
        "domain_order": domains,
        "cosine_matrix": matrix.tolist(),
        "pairs": pairs,
        "n_pairs": len(pairs),
        "negative_pair_count": sum(bool(row["conflict"]) for row in pairs),
        "negative_pair_fraction": float(np.mean([value < 0.0 for value in cosines])) if cosines else 0.0,
        "mean_pairwise_cosine": float(np.mean(cosines)) if cosines else float("nan"),
    }


def _encoder_parameters(model):
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(ENCODER_PREFIXES)
    ]
    if not parameters:
        raise ValueError("No shared encoder parameters matched the diagnostic prefixes")
    return parameters


def run_gradient_conflict(model, loader, device, config, output_dir, seed=42):
    """Compute equal-budget per-domain SOH gradients on the shared encoder."""

    model.eval()
    named_parameters = _encoder_parameters(model)
    parameters = [parameter for _, parameter in named_parameters]
    max_samples = int(config.get("max_samples_per_domain", 64))
    if max_samples <= 0:
        raise ValueError("gradient_conflict.max_samples_per_domain must be positive")
    gradient_sums = {}
    counts = Counter()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    randomized_loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )
    for batch in randomized_loader:
        batch_domains = np.asarray([str(value) for value in batch["domain_id"]], dtype=str)
        for domain in sorted(set(batch_domains.tolist())):
            remaining = max_samples - counts[domain]
            if remaining <= 0:
                continue
            indices = np.flatnonzero(batch_domains == domain)[:remaining]
            if indices.size == 0:
                continue
            index_tensor = torch.as_tensor(indices, dtype=torch.long)
            tensors = {
                key: value.index_select(0, index_tensor).to(device)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            model.zero_grad(set_to_none=True)
            aux = model.forward_with_aux(**{key: tensors[key] for key in MODEL_INPUT_KEYS})
            loss = torch.nn.functional.mse_loss(aux["soh_pred"], tensors["soh"])
            grads = torch.autograd.grad(loss, parameters, allow_unused=True)
            flat = torch.cat(
                [
                    torch.zeros_like(parameter).reshape(-1)
                    if gradient is None
                    else gradient.detach().reshape(-1)
                    for parameter, gradient in zip(parameters, grads)
                ]
            ).cpu().to(torch.float64)
            weight = int(indices.size)
            if domain not in gradient_sums:
                gradient_sums[domain] = torch.zeros_like(flat)
            gradient_sums[domain] += flat * weight
            counts[domain] += weight
    gradients = {
        domain: value / counts[domain]
        for domain, value in gradient_sums.items()
        if counts[domain] > 0
    }
    if len(gradients) < 2:
        raise ValueError("Gradient conflict requires at least two validation domains")
    report = gradient_cosine_report(gradients)
    report.update(
        {
            "definition": "pairwise cosine of per-domain SOH-loss gradients",
            "parameter_scope": "shared CC/CV encoder, fusion projections, and CC-to-CV bridge",
            "loss_scope": "SOH MSE only on a deterministic random equal-budget sample; no cycle auxiliary loss",
            "samples_per_domain": {domain: int(counts[domain]) for domain in sorted(gradients)},
            "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "parameter_tensors": [name for name, _ in named_parameters],
        }
    )
    output_dir = Path(output_dir)
    _write_csv(output_dir / "gradient_pairs.csv", report["pairs"])
    matrix_rows = []
    for index, domain in enumerate(report["domain_order"]):
        row = {"domain": domain}
        row.update(
            {
                other: float(report["cosine_matrix"][index][other_index])
                for other_index, other in enumerate(report["domain_order"])
            }
        )
        matrix_rows.append(row)
    _write_csv(
        output_dir / "gradient_cosine_matrix.csv",
        matrix_rows,
        fieldnames=["domain", *report["domain_order"]],
    )
    save_json(output_dir / "gradient_conflict.json", report)
    return report


def _load_seed_run(run_root, seed, device, backend_override=None):
    seed_dir = Path(run_root) / f"seed_{int(seed)}"
    config_path = seed_dir / "resolved_config.json"
    checkpoint_path = seed_dir / "best.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing seed artifacts under {seed_dir}; expected resolved_config.json and best.pt"
        )
    with config_path.open(encoding="utf-8") as handle:
        training_config = json.load(handle)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_raw_soh_model(
        training_config["model"], backend_override=backend_override
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, training_config, seed_dir, checkpoint_path


def _save_features(path, data):
    np.savez_compressed(
        path,
        features=data["features"],
        truth=data["truth"],
        prediction=data["prediction"],
        domain=data["domain"],
        battery=data["battery"],
        condition=data["condition"],
        cycle_id=data["cycle_id"],
    )


def _load_features(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def _run_diagnostic_safely(name, function):
    """Run one diagnostic without preventing the remaining diagnostics."""

    try:
        result = function()
        return result, {
            "status": str(result.get("status", "completed")),
        }
    except Exception as error:
        status = {
            "status": "unavailable",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        print(
            f"[diagnostic unavailable] {name}: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return None, status


def _upgrade_legacy_representation_report(seed_report, seed_output, config, seed):
    """Add the pairwise probe to old reports using their preserved feature cache."""

    representation = seed_report.get("representation_probe")
    if not isinstance(representation, dict):
        return seed_report
    if representation.get("definition") != "linear domain probe on SOH-bin-matched z_health":
        return seed_report
    feature_path = Path(seed_output) / "validation_features.npz"
    if not feature_path.is_file():
        return seed_report

    seed_report["representation_strict_probe"] = representation
    save_json(
        Path(seed_output) / "representation_strict_probe.json", representation
    )
    pairwise, pairwise_status = _run_diagnostic_safely(
        "representation_pairwise_probe_cached_upgrade",
        lambda: run_pairwise_representation_probe(
            _load_features(feature_path), config, int(seed), seed_output
        ),
    )
    seed_report["representation_probe"] = pairwise
    statuses = seed_report.setdefault("diagnostic_status", {})
    statuses["representation_pairwise_probe"] = pairwise_status
    statuses["representation_strict_probe"] = {
        "status": "completed",
        "source": "legacy_representation_probe",
    }
    save_json(Path(seed_output) / "diagnostic_report.json", seed_report)
    return seed_report


def _aggregate_seed_reports(seed_reports):
    scalar_paths = {
        "domain_probe_accuracy": ("representation_probe", "accuracy"),
        "domain_probe_macro_f1": ("representation_probe", "macro_f1"),
        "pairwise_domain_probe_accuracy": ("representation_probe", "accuracy"),
        "pairwise_domain_probe_macro_f1": ("representation_probe", "macro_f1"),
        "strict_domain_probe_accuracy": ("representation_strict_probe", "accuracy"),
        "strict_domain_probe_macro_f1": ("representation_strict_probe", "macro_f1"),
        "calibration_before_domain_macro_rmse": (
            "residual_calibration",
            "before_domain_macro_rmse",
        ),
        "calibration_after_domain_macro_rmse": (
            "residual_calibration",
            "after_domain_macro_rmse",
        ),
        "calibration_domain_macro_rmse_change": (
            "residual_calibration",
            "domain_macro_rmse_change",
        ),
        "gradient_negative_pair_fraction": ("gradient_conflict", "negative_pair_fraction"),
        "gradient_mean_pairwise_cosine": ("gradient_conflict", "mean_pairwise_cosine"),
    }
    summary = {}
    for name, (section, metric) in scalar_paths.items():
        values = [
            float(report[section][metric])
            for report in seed_reports
            if isinstance(report.get(section), dict) and metric in report[section]
        ]
        if values:
            summary[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "values": values,
            }
    return summary


def _diagnostic_summary(config, repo_root, seeds, skip_gradients, seed_reports=None):
    diagnostic = config["diagnostic"]
    split_name = str(diagnostic.get("split", "val"))
    if split_name != "val":
        raise ValueError("Paper-v1 diagnostics are restricted to the validation split")
    run_root = _resolve_path(repo_root, diagnostic["run_root"])
    output_root = _resolve_path(repo_root, diagnostic["output_root"])
    if seed_reports is None:
        seed_reports = []
        for seed in seeds:
            report_path = output_root / f"seed_{int(seed)}" / "diagnostic_report.json"
            if not report_path.is_file():
                raise FileNotFoundError(f"Missing completed diagnostic worker report: {report_path}")
            with report_path.open(encoding="utf-8") as handle:
                report = json.load(handle)
            seed_reports.append(
                _upgrade_legacy_representation_report(
                    report,
                    report_path.parent,
                    diagnostic.get("representation_probe", {}),
                    seed,
                )
            )
    summary = {
        "diagnostic_name": str(diagnostic.get("name", "V1_E2_Diagnostics")),
        "source_run_root": str(run_root),
        "output_root": str(output_root),
        "split": split_name,
        "seeds": [int(seed) for seed in seeds],
        "diagnostics": [
            "representation_pairwise_domain_probe",
            "representation_strict_five_domain_probe_supplemental",
            "residual_affine_calibration",
            *([] if skip_gradients else ["shared_encoder_gradient_conflict"]),
        ],
        "diagnostic_status_by_seed": {
            str(seed): report.get("diagnostic_status", {})
            for seed, report in zip(seeds, seed_reports)
        },
        "aggregate": _aggregate_seed_reports(seed_reports),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(output_root / "resolved_diagnostic_config.json", config)
    save_json(output_root / "diagnostic_summary.json", summary)
    return summary


def aggregate_from_config(config, repo_root=REPO_ROOT, seed_override=None, skip_gradients=False):
    """Aggregate workers without loading a model or rebuilding a dataset.

    Legacy strict-probe reports are upgraded from their saved feature caches.
    """

    if config.get("status", "runnable") != "runnable":
        raise ValueError("The diagnostic config is not marked runnable")
    diagnostic = config["diagnostic"]
    seeds = (
        [int(value) for value in seed_override]
        if seed_override
        else [int(value) for value in diagnostic.get("seeds", [42, 52, 62])]
    )
    if not seeds:
        raise ValueError("diagnostic.seeds cannot be empty")
    return _diagnostic_summary(config, repo_root, seeds, skip_gradients)


def run_from_config(
    config,
    repo_root=REPO_ROOT,
    device_override=None,
    backend_override=None,
    seed_override=None,
    max_samples_override=None,
    skip_gradients=False,
    worker_mode=False,
):
    """Run all configured V1 diagnostics without modifying source artifacts."""

    if config.get("status", "runnable") != "runnable":
        raise ValueError("The diagnostic config is not marked runnable")
    diagnostic = config["diagnostic"]
    split_name = str(diagnostic.get("split", "val"))
    if split_name != "val":
        raise ValueError("Paper-v1 diagnostics are restricted to the validation split")
    run_root = _resolve_path(repo_root, diagnostic["run_root"])
    output_root = _resolve_path(repo_root, diagnostic["output_root"])
    if not run_root.is_dir():
        raise FileNotFoundError(f"Source E2 runtime does not exist: {run_root}")
    if (
        output_root == run_root
        or run_root in output_root.parents
        or output_root in run_root.parents
    ):
        raise ValueError("Diagnostic output_root must be separate from the source E2 runtime")
    requested_device = str(device_override or diagnostic.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Formal E2 diagnostics require CUDA. Use --device_override cpu with "
            "--backend_override torch_reference only for structural smoke tests."
        )
    device = torch.device(requested_device)
    seeds = (
        [int(value) for value in seed_override]
        if seed_override
        else [int(value) for value in diagnostic.get("seeds", [42, 52, 62])]
    )
    if not seeds:
        raise ValueError("diagnostic.seeds cannot be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    if not worker_mode:
        save_json(output_root / "resolved_diagnostic_config.json", config)

    seed_reports = []
    for seed in seeds:
        set_random_seed(seed, True)
        model, training_config, seed_dir, checkpoint_path = _load_seed_run(
            run_root, seed, device, backend_override=backend_override
        )
        if training_config.get("experiment", {}).get("loader") != "unified_multi_dataset":
            raise ValueError(f"Seed {seed} is not a unified E2 run")
        loaders, split_info = build_unified_loaders(training_config, repo_root, seed)
        loader = loaders[split_name]
        feature_limit = (
            int(max_samples_override)
            if max_samples_override is not None
            else int(diagnostic.get("max_samples_per_domain", 0))
        )
        data = collect_features(model, loader, device, max_samples_per_domain=feature_limit)
        seed_output = output_root / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        if bool(diagnostic.get("save_features", True)):
            _save_features(seed_output / "validation_features.npz", data)
        probe_config = diagnostic.get("representation_probe", {})
        representation, representation_status = _run_diagnostic_safely(
            "representation_pairwise_probe",
            lambda: run_pairwise_representation_probe(
                data,
                probe_config,
                seed,
                seed_output,
            ),
        )
        strict_representation, strict_status = _run_diagnostic_safely(
            "representation_strict_probe",
            lambda: run_strict_representation_probe(
                data,
                probe_config,
                seed,
                seed_output,
            ),
        )
        if strict_representation is None:
            strict_representation = dict(strict_status)
            save_json(
                seed_output / "representation_strict_probe.json",
                strict_representation,
            )
        calibration, calibration_status = _run_diagnostic_safely(
            "residual_calibration",
            lambda: run_residual_calibration(
                data,
                diagnostic.get("residual_calibration", {}),
                seed,
                seed_output,
            ),
        )
        gradient = None
        gradient_status = {"status": "skipped"}
        if not skip_gradients:
            gradient, gradient_status = _run_diagnostic_safely(
                "gradient_conflict",
                lambda: run_gradient_conflict(
                    model,
                    loader,
                    device,
                    diagnostic.get("gradient_conflict", {}),
                    seed_output,
                    seed=seed,
                ),
            )
        diagnostic_status = {
            "representation_pairwise_probe": representation_status,
            "representation_strict_probe": strict_status,
            "residual_calibration": calibration_status,
            "gradient_conflict": gradient_status,
        }
        seed_report = {
            "seed": seed,
            "source_seed_dir": str(seed_dir),
            "source_checkpoint": str(checkpoint_path),
            "split": split_name,
            "sample_counts_by_domain": dict(Counter(data["domain"].tolist())),
            "battery_counts_by_domain": {
                domain: len(set(data["battery"][data["domain"] == domain].tolist()))
                for domain in sorted(set(data["domain"].tolist()))
            },
            "representation_probe": representation,
            "representation_strict_probe": strict_representation,
            "residual_calibration": calibration,
            "gradient_conflict": gradient,
            "diagnostic_status": diagnostic_status,
            "split_info": split_info,
        }
        save_json(seed_output / "diagnostic_report.json", seed_report)
        seed_reports.append(seed_report)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if worker_mode:
        return {
            "worker_mode": True,
            "output_root": str(output_root),
            "completed_seeds": seeds,
        }
    return _diagnostic_summary(
        config,
        repo_root,
        seeds,
        skip_gradients,
        seed_reports=seed_reports,
    )


def parse_args():
    parser = argparse.ArgumentParser("Paper-v1 E2 domain diagnostics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device_override", default=None)
    parser.add_argument(
        "--backend_override",
        choices=("mamba_ssm.Mamba", "torch_reference"),
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Run only this seed; repeat to select multiple seeds.",
    )
    parser.add_argument(
        "--max_samples_per_domain",
        type=int,
        default=None,
        help="Feature-extraction cap for quick smoke tests; zero in formal configs means all samples.",
    )
    parser.add_argument(
        "--skip_gradients",
        action="store_true",
        help="Skip the expensive gradient diagnostic in a smoke test.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--worker_mode",
        action="store_true",
        help="Write only per-seed outputs; a later aggregate-only call writes shared summaries.",
    )
    mode.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Aggregate completed per-seed reports without loading models or datasets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.aggregate_only:
        result = aggregate_from_config(
            config,
            repo_root=REPO_ROOT,
            seed_override=args.seed,
            skip_gradients=args.skip_gradients,
        )
    else:
        result = run_from_config(
            config,
            repo_root=REPO_ROOT,
            device_override=args.device_override,
            backend_override=args.backend_override,
            seed_override=args.seed,
            max_samples_override=args.max_samples_per_domain,
            skip_gradients=args.skip_gradients,
            worker_mode=args.worker_mode,
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
