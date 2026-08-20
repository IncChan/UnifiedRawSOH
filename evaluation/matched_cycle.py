#!/usr/bin/env python3
"""Matched-cycle E1 evaluation for existing RawMamba and Only-F checkpoints.

This intentionally does not train or select checkpoints.  It reconstructs the
saved E1 test preprocessing, intersects physical cycle keys, verifies labels,
and evaluates the already selected checkpoints on that shared test subset.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.baseline_loaders import build_feature_loaders
from UnifiedRawSOH.datasets.loaders import build_single_domain_loaders
from UnifiedRawSOH.datasets.mit import read_mit_raw_file
from UnifiedRawSOH.datasets.xjtu import read_xjtu_file
from UnifiedRawSOH.evaluation.metrics import compute_metrics
from UnifiedRawSOH.models.baselines.pinn4soh_no_leak_onlyf import PINNFOnlyMLP
from UnifiedRawSOH.models.raw_soh_model import build_raw_soh_model
from UnifiedRawSOH.utils.config import load_config, save_json


# The v2 physical-124 data were deliberately renamed to the stable paper
# names ``MIT_raw`` / ``MIT_features`` after these E1 weights were trained.
# Saved resolved configs are immutable experiment artifacts, so translate only
# their directory references in memory while reconstructing evaluation data.
_RENAMED_MIT_PATHS = (
    ("UnifiedRawSOH/datasets/MIT_raw_physical124", "UnifiedRawSOH/datasets/MIT_raw"),
    ("UnifiedRawSOH/datasets/MIT_features_physical124", "UnifiedRawSOH/datasets/MIT_features"),
    ("PINN4SOH/data/MIT_raw_t_v2_physical124", "PINN4SOH/data/MIT_raw"),
    ("PINN4SOH/data/MIT_t_v2_physical124", "PINN4SOH/data/MIT_features"),
)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate E1 checkpoints on matched physical test cycles")
    parser.add_argument("--xjtu-raw-run-dir", required=True, type=Path)
    parser.add_argument("--xjtu-onlyf-run-dir", required=True, type=Path)
    parser.add_argument("--mit-raw-run-dir", required=True, type=Path)
    parser.add_argument("--mit-onlyf-run-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "UnifiedRawSOH/outputs/e1_matched_cycle/result.json",
    )
    parser.add_argument("--label-atol", type=float, default=1e-6)
    parser.add_argument("--label-rtol", type=float, default=1e-5)
    return parser.parse_args()


def _canonicalize_saved_mit_paths(value):
    """Return a copied saved config with pre-rename MIT paths updated.

    This compatibility layer is intentionally local to checkpoint evaluation:
    it never writes the historical ``resolved_config.json`` artifact.
    """

    if isinstance(value, dict):
        return {key: _canonicalize_saved_mit_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize_saved_mit_paths(item) for item in value]
    if isinstance(value, str):
        for legacy_path, canonical_path in _RENAMED_MIT_PATHS:
            value = value.replace(legacy_path, canonical_path)
    return value


def _load_seed_artifacts(run_dir, seed):
    run_dir = Path(run_dir)
    seed_dir = run_dir / f"seed_{int(seed)}"
    config_path = seed_dir / "resolved_config.json"
    checkpoint_path = seed_dir / "best.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Expected E1 seed artifacts under {seed_dir}: resolved_config.json and best.pt"
        )
    return _canonicalize_saved_mit_paths(load_config(config_path)), checkpoint_path


def _physical_key(domain, sample):
    return (str(domain), str(sample["battery_id"]), int(sample["cycle_id"]))


def _index_unique_samples(samples, domain):
    indexed = {}
    for index, sample in enumerate(samples):
        key = _physical_key(domain, sample)
        if key in indexed:
            raise ValueError(f"Duplicate physical cycle key in {domain} evaluation data: {key}")
        indexed[key] = index
    return indexed


def _sample_soh(sample):
    return float(np.asarray(sample["soh"], dtype=np.float64).reshape(-1)[0])


def _resolve_data_root(config):
    root = Path(config["data"]["data_root"])
    return root if root.is_absolute() else (PROJECT_ROOT / root).resolve()


def _raw_label_settings(domain, raw_config):
    data_cfg = raw_config["data"]
    nominal = float(
        data_cfg.get("nominal_capacities", {}).get(
            domain, data_cfg.get("nominal_capacity", 2.0)
        )
    )
    scale_mode = data_cfg.get("label_scale_modes", {}).get(
        domain, data_cfg.get("label_scale_mode", "auto_capacity_to_soh")
    )
    return nominal, scale_mode


def _read_raw_records_for_feature_file(domain, feature_file, raw_config):
    raw_file = _resolve_data_root(raw_config) / Path(feature_file).name
    if not raw_file.is_file():
        raise FileNotFoundError(
            f"Missing raw counterpart for Only-F feature file {feature_file}: {raw_file}"
        )
    nominal_capacity, scale_mode = _raw_label_settings(domain, raw_config)
    if domain == "xjtu":
        return read_xjtu_file(raw_file, nominal_capacity, scale_mode)
    if domain == "mit":
        return read_mit_raw_file(raw_file, nominal_capacity, scale_mode)
    raise ValueError(f"Unsupported matched-cycle domain: {domain}")


def _source_feature_capacity_crosswalk(
    domain, feature_file, needed_indices, raw_config, onlyf_config, label_atol, label_rtol
):
    """Map feature-source rows to physical raw cycles using monotone capacity order.

    Only-F source CSVs do not carry a physical cycle column.  Their legacy
    row ordinal cannot be compared to raw ids directly.  Instead, this follows
    the repository's existing XJTU/MIT alignment provenance: walk the ordered
    capacity sequence and resolve each feature record to the next
    label-consistent raw physical cycle.  The source row index is used only to
    retrieve that derived crosswalk for a surviving Only-F sample; it is never
    treated as a raw cycle id or used as a matching key.
    """

    raw_records = _read_raw_records_for_feature_file(domain, feature_file, raw_config)
    feature_nominal = float(onlyf_config["data"].get("nominal_capacity", 2.0))
    if feature_nominal <= 0:
        raise ValueError("Only-F nominal_capacity must be positive")
    crosswalk = {}
    cursor = 0
    with Path(feature_file).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "capacity" not in reader.fieldnames:
            raise ValueError(f"Only-F feature source is missing capacity: {feature_file}")
        for source_index, source_row in enumerate(reader):
            try:
                feature_soh = float(source_row["capacity"]) / feature_nominal
            except (TypeError, ValueError):
                if source_index in needed_indices:
                    raise ValueError(
                        f"Non-numeric capacity for retained Only-F source row "
                        f"{source_index} in {feature_file}"
                    )
                continue
            if not np.isfinite(feature_soh):
                if source_index in needed_indices:
                    raise ValueError(
                        f"Non-finite capacity for retained Only-F source row "
                        f"{source_index} in {feature_file}"
                    )
                continue
            matched_index = None
            while cursor < len(raw_records):
                raw_soh = float(raw_records[cursor]["soh"])
                if np.isclose(feature_soh, raw_soh, atol=label_atol, rtol=label_rtol):
                    matched_index = cursor
                    break
                cursor += 1
            if matched_index is None:
                if source_index in needed_indices:
                    raise ValueError(
                        f"Could not map retained Only-F source row {source_index} in "
                        f"{feature_file} to a later physical raw cycle by capacity/SOH."
                    )
                continue
            record = raw_records[matched_index]
            crosswalk[source_index] = _physical_key(domain, record)
            cursor = matched_index + 1
    missing = sorted(set(needed_indices) - set(crosswalk))
    if missing:
        raise ValueError(
            f"{domain} feature-to-raw crosswalk is missing {len(missing)} retained "
            f"Only-F source rows for {feature_file}; examples={missing[:5]}"
        )
    return crosswalk


def _map_onlyf_rows_to_raw_physical_keys(
    domain, onlyf_dataset, raw_config, onlyf_config, label_atol, label_rtol
):
    """Resolve retained Only-F test rows to ``dataset+battery+physical cycle`` keys."""

    rows_by_source = {}
    matched = {}
    for onlyf_index, row in enumerate(onlyf_dataset.rows):
        direct_cycle_id = row.get("raw_cycle_id")
        if direct_cycle_id is not None:
            key = (str(domain), str(row["battery_id"]), int(direct_cycle_id))
            if key in matched:
                raise ValueError(
                    f"{domain} canonical feature rows map multiple Only-F rows "
                    f"to physical cycle {key}; source identity is not one-to-one."
                )
            matched[key] = onlyf_index
            continue
        source_file = row.get("source_file")
        source_index = row.get("source_row_index")
        if source_file is None or source_index is None:
            raise ValueError(
                "Only-F rows lack source provenance required for physical matched-cycle evaluation."
            )
        rows_by_source.setdefault(str(source_file), []).append((onlyf_index, int(source_index)))

    for source_file, indexed_rows in rows_by_source.items():
        crosswalk = _source_feature_capacity_crosswalk(
            domain,
            source_file,
            {source_index for _, source_index in indexed_rows},
            raw_config,
            onlyf_config,
            label_atol,
            label_rtol,
        )
        for onlyf_index, source_index in indexed_rows:
            key = crosswalk[source_index]
            if key in matched:
                raise ValueError(
                    f"{domain} feature-to-raw crosswalk maps multiple Only-F rows "
                    f"to physical cycle {key}; source identity is not one-to-one."
                )
            matched[key] = onlyf_index
    return matched


def _build_matched_test_subsets(domain, raw_config, onlyf_config, label_atol, label_rtol):
    """Recreate each saved pipeline, then return test subsets with identical keys."""

    raw_loaders, _ = build_single_domain_loaders(
        raw_config,
        PROJECT_ROOT,
        seed=int(raw_config["train"].get("seed", 42)),
    )
    onlyf_loaders, _ = build_feature_loaders(
        onlyf_config,
        PROJECT_ROOT,
        seed=int(onlyf_config["train"].get("seed", 42)),
    )
    raw_dataset = raw_loaders["test"].dataset
    onlyf_dataset = onlyf_loaders["test"].dataset
    raw_index = _index_unique_samples(raw_dataset.samples, domain)
    onlyf_index = _map_onlyf_rows_to_raw_physical_keys(
        domain, onlyf_dataset, raw_config, onlyf_config, label_atol, label_rtol
    )

    raw_batteries = {key[1] for key in raw_index}
    onlyf_batteries = {str(row["battery_id"]) for row in onlyf_dataset.rows}
    if raw_batteries != onlyf_batteries:
        raise ValueError(
            f"{domain} test-battery mismatch between raw and Only-F pipelines: "
            f"raw_only={sorted(raw_batteries - onlyf_batteries)}, "
            f"onlyf_only={sorted(onlyf_batteries - raw_batteries)}"
        )

    matched_keys = sorted(set(raw_index) & set(onlyf_index))
    if not matched_keys:
        raise ValueError(f"No matched physical test cycles were found for {domain}")

    mismatches = []
    for key in matched_keys:
        raw_soh = _sample_soh(raw_dataset.samples[raw_index[key]])
        onlyf_soh = _sample_soh(onlyf_dataset.rows[onlyf_index[key]])
        if not np.isclose(raw_soh, onlyf_soh, atol=label_atol, rtol=label_rtol):
            mismatches.append((key, raw_soh, onlyf_soh))
    if mismatches:
        preview = "; ".join(
            f"{key}: raw={raw_soh:.10g}, onlyf={onlyf_soh:.10g}"
            for key, raw_soh, onlyf_soh in mismatches[:5]
        )
        raise ValueError(
            f"{domain} matched cycle labels disagree for {len(mismatches)} keys; "
            f"check physical cycle identity or label normalization. Examples: {preview}"
        )

    raw_subset = Subset(raw_dataset, [raw_index[key] for key in matched_keys])
    onlyf_subset = Subset(onlyf_dataset, [onlyf_index[key] for key in matched_keys])
    return raw_subset, onlyf_subset, matched_keys


def _loader(dataset, config):
    return DataLoader(
        dataset,
        batch_size=int(config["train"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def _raw_predictions(model, loader, device):
    truths, predictions = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            tensors = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
            output = model.forward_with_aux(
                cc_signal=tensors["cc_signal"],
                cv_signal=tensors["cv_signal"],
                cc_mask=tensors["cc_mask"],
                cv_mask=tensors["cv_mask"],
                cc_time=tensors["cc_time"],
                cv_time=tensors["cv_time"],
                cc_temperature=tensors["cc_temperature"],
                cv_temperature=tensors["cv_temperature"],
                t0_temperature_norm=tensors["t0_temperature_norm"],
            )
            truths.extend(tensors["soh"].cpu().numpy().reshape(-1).tolist())
            predictions.extend(output["soh_pred"].cpu().numpy().reshape(-1).tolist())
    return compute_metrics(truths, predictions)


def _onlyf_model(model_config):
    return PINNFOnlyMLP(
        input_dim=int(model_config.get("input_dim", 24)),
        encoder_hidden_dim=int(model_config.get("encoder_hidden_dim", 60)),
        encoder_output_dim=int(model_config.get("encoder_output_dim", 32)),
        encoder_layers_num=int(model_config.get("encoder_layers_num", 3)),
        predictor_hidden_dim=int(model_config.get("predictor_hidden_dim", 32)),
        dropout=float(model_config.get("dropout", 0.2)),
    )


def _onlyf_predictions(model, loader, device):
    truths, predictions = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device)
            truth = batch["soh"].to(device)
            prediction = model(features)
            truths.extend(truth.cpu().numpy().reshape(-1).tolist())
            predictions.extend(prediction.cpu().numpy().reshape(-1).tolist())
    return compute_metrics(truths, predictions)


def _load_model(checkpoint_path, config, model_kind, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if model_kind == "raw":
        model = build_raw_soh_model(config["model"]).to(device)
    elif model_kind == "onlyf":
        model = _onlyf_model(config["model"]).to(device)
    else:  # pragma: no cover - internal caller only
        raise ValueError(f"Unknown model kind: {model_kind}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def _format(summary):
    return f"{summary['mean']:.5f} ± {summary['std']:.5f}"


def _evaluate_domain(domain, raw_run_dir, onlyf_run_dir, seeds, device, label_atol, label_rtol):
    raw_reference, _ = _load_seed_artifacts(raw_run_dir, seeds[0])
    onlyf_reference, _ = _load_seed_artifacts(onlyf_run_dir, seeds[0])
    raw_subset, onlyf_subset, matched_keys = _build_matched_test_subsets(
        domain,
        raw_reference,
        onlyf_reference,
        label_atol,
        label_rtol,
    )
    raw_loader = _loader(raw_subset, raw_reference)
    onlyf_loader = _loader(onlyf_subset, onlyf_reference)
    results = {"raw_mamba": {"mape": [], "rmse": []}, "onlyf": {"mape": [], "rmse": []}}

    for seed in seeds:
        raw_config, raw_checkpoint = _load_seed_artifacts(raw_run_dir, seed)
        onlyf_config, onlyf_checkpoint = _load_seed_artifacts(onlyf_run_dir, seed)
        raw_metrics = _raw_predictions(
            _load_model(raw_checkpoint, raw_config, "raw", device), raw_loader, device
        )
        onlyf_metrics = _onlyf_predictions(
            _load_model(onlyf_checkpoint, onlyf_config, "onlyf", device), onlyf_loader, device
        )
        for metric in ("mape", "rmse"):
            results["raw_mamba"][metric].append(float(raw_metrics[metric]))
            results["onlyf"][metric].append(float(onlyf_metrics[metric]))

    return {
        "matched_cycles": len(matched_keys),
        "raw_mamba": {metric: _summary(values) for metric, values in results["raw_mamba"].items()},
        "onlyf": {metric: _summary(values) for metric, values in results["onlyf"].items()},
    }


def main():
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Matched RawMamba evaluation requires a CUDA-visible process.")
    device = torch.device(args.device)
    result = {
        "xjtu": _evaluate_domain(
            "xjtu",
            args.xjtu_raw_run_dir,
            args.xjtu_onlyf_run_dir,
            args.seeds,
            device,
            args.label_atol,
            args.label_rtol,
        ),
        "mit": _evaluate_domain(
            "mit",
            args.mit_raw_run_dir,
            args.mit_onlyf_run_dir,
            args.seeds,
            device,
            args.label_atol,
            args.label_rtol,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)

    print("Matched-cycle E1")
    print(f"XJTU matched cycles: {result['xjtu']['matched_cycles']}")
    print(f"MIT matched cycles: {result['mit']['matched_cycles']}")
    print()
    print("| MAPE / RMSE | XJTU | MIT |")
    print("|---|---|---|")
    for label, key in (("RawMamba", "raw_mamba"), ("PINN4SOH Only-F noLeak", "onlyf")):
        cells = []
        for domain in ("xjtu", "mit"):
            metrics = result[domain][key]
            cells.append(f"{_format(metrics['mape'])} / {_format(metrics['rmse'])}")
        print(f"| {label} | {cells[0]} | {cells[1]} |")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
