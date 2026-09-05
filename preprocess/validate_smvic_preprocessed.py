#!/usr/bin/env python3
"""Validate SMVIC model-ready products and smoke-test both target models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from UnifiedRawSOH.models.paper_backup.model_factory import build_model  # noqa: E402
from UnifiedRawSOH.preprocess.paper_backup.common import FEATURE_NAMES, rich_channel_names  # noqa: E402
from UnifiedRawSOH.preprocess.smvic_common import (  # noqa: E402
    DEFAULT_QUALITY_POLICY,
    FAMILY_SPECS,
    load_quality_policy,
)


DEFAULT_ROOT = REPO_ROOT / "datasets" / "SMVIC_preprocessed_v3_128x128"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--domains", nargs="+", default=["all"])
    parser.add_argument("--quality-policy", type=Path, default=DEFAULT_QUALITY_POLICY)
    parser.add_argument("--skip-model-smoke", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_smoke(cc: np.ndarray, cv: np.ndarray, features: np.ndarray) -> dict:
    count = min(2, len(features))
    mlp = build_model({
        "type": "FinalHI-MLP",
        "input_dim": 24,
        "encoder_hidden_dim": 60,
        "encoder_output_dim": 32,
        "encoder_layers_num": 3,
        "predictor_hidden_dim": 32,
        "dropout": 0.2,
    }).eval()
    bicontext = build_model({
        "type": "FinalBiContextMamba",
        "input_dim": 5,
        "signal_input_dim": 3,
        "d_model": 32,
        "num_layers": 3,
        "d_state": 8,
        "d_conv": 4,
        "expand": 2,
        "dt_rank": "auto",
        "dropout": 0.1,
        "pooling": "last_mean",
        "fusion_dim": 64,
        "head_hidden_dim": 128,
        "use_time_as_input": True,
        "temperature_injection": "input_concat",
        "temperature_features": "delta",
        "use_t0_temperature_meta": True,
        "t0_temperature_meta_dim": 1,
        "time_embedding_time_scale_min": 10.0,
        "bridge_after_layer": 1,
        "bridge_gate_hidden_dim": 4,
        "ordinary_fusion_hidden_dim": 27,
        "backend": "torch_reference",
    }).eval()
    raw_vanilla = build_model({
        "type": "FinalRawVanillaMamba",
        "input_dim": 5,
        "d_model": 52,
        "num_layers": 3,
        "d_state": 8,
        "d_conv": 4,
        "expand": 2,
        "dt_rank": "auto",
        "dropout": 0.1,
        "head_hidden_dim": 128,
        "use_boundary_token": True,
        "backend": "torch_reference",
    }).eval()
    cc_t = torch.from_numpy(np.array(cc[:count], dtype=np.float32, copy=True))
    cv_t = torch.from_numpy(np.array(cv[:count], dtype=np.float32, copy=True))
    mask_cc = torch.ones((count, cc.shape[1]), dtype=torch.bool)
    mask_cv = torch.ones((count, cv.shape[1]), dtype=torch.bool)
    with torch.no_grad():
        mlp_out = mlp(torch.from_numpy(np.array(features[:count], dtype=np.float32, copy=True)))
        sequence_kwargs = {
            "cc_signal": cc_t[:, :, [0, 1, 6]],
            "cv_signal": cv_t[:, :, [0, 1, 6]],
            "cc_mask": mask_cc,
            "cv_mask": mask_cv,
            "cc_time": cc_t[:, :, 2] * 10.0,
            "cv_time": cv_t[:, :, 2] * 10.0,
            "cc_temperature": cc_t[:, :, [3, 4]],
            "cv_temperature": cv_t[:, :, [3, 4]],
            "t0_temperature_norm": cc_t[:, 0, 3].reshape(count, 1),
        }
        joint_sequence = torch.cat((cc_t[:, :, :5], cv_t[:, :, :5]), dim=1)
        joint_mask = torch.ones(
            (count, joint_sequence.shape[1]), dtype=torch.bool
        )
        boundary_index = torch.full((count,), cc_t.shape[1], dtype=torch.long)
        raw_vanilla_out = raw_vanilla(
            joint_sequence,
            joint_mask,
            boundary_index=boundary_index,
        )
        bicontext_out = bicontext(**sequence_kwargs)
    if any(
        output.shape != (count, 1)
        for output in (mlp_out, raw_vanilla_out, bicontext_out)
    ):
        raise ValueError("Unexpected target-model output shape")
    if not all(
        torch.isfinite(output).all()
        for output in (mlp_out, raw_vanilla_out, bicontext_out)
    ):
        raise ValueError("Target-model smoke output is non-finite")
    return {
        "batch_size": count,
        "pinn4soh_like_output_shape": list(mlp_out.shape),
        "raw_vanilla_output_shape": list(raw_vanilla_out.shape),
        "bicontext_output_shape": list(bicontext_out.shape),
        "bicontext_backend": "torch_reference",
    }


def validate_domain(directory: Path, skip_model_smoke: bool, quality_policy) -> dict:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["schema_version"]) != 2:
        raise ValueError(f"Expected schema v2: {manifest_path}")
    if manifest["rich_channel_names"] != list(rich_channel_names(2)):
        raise ValueError(f"Rich-channel mismatch: {manifest_path}")
    if manifest["feature_names"] != list(FEATURE_NAMES):
        raise ValueError(f"Feature schema mismatch: {manifest_path}")
    quality = dict(manifest.get("quality_control", {}))
    if quality.get("policy_id") != quality_policy.policy_id:
        raise ValueError(f"Quality-policy ID mismatch: {manifest_path}")
    if quality.get("sha256") != quality_policy.sha256:
        raise ValueError(f"Quality-policy checksum mismatch: {manifest_path}")
    arrays = {}
    for name, contract in manifest["terminal"]["arrays"].items():
        path = directory / contract["file"]
        if _sha256(path) != contract["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != contract["shape"] or str(values.dtype) != contract["dtype"]:
            raise ValueError(f"Array contract mismatch: {path}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values: {path}")
        arrays[name] = values
    n = int(manifest["terminal"]["records"])
    expected = {
        "cc": (n, int(manifest["resampling"]["cc_length"]), 7),
        "cv": (n, int(manifest["resampling"]["cv_length"]), 7),
        "features": (n, 24),
        "soh": (n, 1),
    }
    for name, shape in expected.items():
        if tuple(arrays[name].shape) != shape:
            raise ValueError(f"Unexpected {name} shape: {arrays[name].shape}, expected={shape}")
    with (directory / manifest["terminal"]["index"]).open(encoding="utf-8", newline="") as handle:
        index = list(csv.DictReader(handle))
    if len(index) != n or [int(row["row"]) for row in index] != list(range(n)):
        raise ValueError(f"Index rows are not contiguous: {directory}")
    keys = [(row["battery_id"], int(row["cycle_id"])) for row in index]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate physical cycle key: {directory}")
    key_set = set(keys)
    bounded_smoke = bool(manifest.get("bounded_smoke_product", False))
    expected_exclusions = [
        item
        for item in quality_policy.excluded_cycles
        if item.battery_id.startswith(f"{manifest['battery_group']}/")
    ]
    leaked = [
        (item.battery_id, item.cycle_id)
        for item in expected_exclusions
        if (item.battery_id, item.cycle_id) in key_set
    ]
    if leaked:
        raise ValueError(f"Curated quality exclusions leaked into arrays: {leaked}")
    classification_path = directory / manifest["audit"]["classification"]
    with classification_path.open(encoding="utf-8", newline="") as handle:
        classification = {
            (row["battery_id"], int(row["cycle_id"])): row
            for row in csv.DictReader(handle)
        }
    # A bounded smoke product sees only the first N cycles of each cell, so
    # later curated exclusions are intentionally absent from its audit.  A
    # formal product must provide evidence for every policy item in the group.
    for item in expected_exclusions:
        row = classification.get((item.battery_id, item.cycle_id))
        if row is None and bounded_smoke:
            continue
        expected_reason = f"quality_exclusion:{item.reason}"
        if row is None or int(row["eligible"]) != 0 or row["reason"] != expected_reason:
            raise ValueError(
                f"Missing audited quality exclusion {item.battery_id}/{item.cycle_id}: "
                f"expected={expected_reason}, actual={row}"
            )
    audited_quality_exclusions = {
        key
        for key, row in classification.items()
        if str(row["reason"]).startswith("quality_exclusion:")
    }
    automatic_leaks = sorted(audited_quality_exclusions & key_set)
    if automatic_leaks:
        raise ValueError(
            f"Audited quality exclusions leaked into model arrays: {automatic_leaks}"
        )
    labels = np.asarray([float(row["soh"]) for row in index], dtype=np.float32).reshape(-1, 1)
    if not np.allclose(labels, arrays["soh"], rtol=1e-6, atol=1e-7):
        raise ValueError(f"Index/array SOH mismatch: {directory}")
    split = json.loads((directory / manifest["split"]).read_text(encoding="utf-8"))
    observed = {row["battery_id"] for row in index}
    protocols = list(split.get("protocols", [split]))
    protocol_tests = {}
    for protocol in protocols:
        protocol_id = str(protocol.get("protocol_id", protocol.get("name", "default")))
        test = set(protocol["test_batteries"])
        if not test or not test <= observed or not observed - test:
            raise ValueError(f"Invalid physical-cell split {protocol_id}: {directory}")
        if int(protocol.get("development_split", {}).get("random_state", -1)) != 420:
            raise ValueError(f"SMVIC train/validation random_state must be 420: {protocol_id}")
        protocol_tests[protocol_id] = sorted(test)
    result = {
        "status": "PASS",
        "domain_id": manifest["domain_id"],
        "records": n,
        "batteries": len(observed),
        "test_protocols": protocol_tests,
        "soh_range": [float(np.min(arrays["soh"])), float(np.max(arrays["soh"]))],
        "curated_quality_exclusions": len(expected_exclusions),
        "audited_quality_exclusions": len(audited_quality_exclusions),
    }
    if not skip_model_smoke:
        result["model_smoke"] = _model_smoke(arrays["cc"], arrays["cv"], arrays["features"])
    return result


def main() -> int:
    args = parse_args()
    quality_policy = load_quality_policy(args.quality_policy)
    known = {spec.domain_id for spec in FAMILY_SPECS.values()}
    domains = sorted(known) if "all" in args.domains else list(dict.fromkeys(args.domains))
    unknown = sorted(set(domains) - known)
    if unknown:
        raise ValueError(f"Unknown SMVIC domains: {unknown}")
    results = {
        domain: validate_domain(
            args.input_root / domain,
            args.skip_model_smoke,
            quality_policy,
        )
        for domain in domains
    }
    print(json.dumps({"status": "PASS", "domains": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
