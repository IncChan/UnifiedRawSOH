#!/usr/bin/env python3
"""Build Paper-Backup v2 mmap products from normalized SMVIC CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from UnifiedRawSOH.preprocess.paper_backup.common import (  # noqa: E402
    FEATURE_NAMES,
    feature_vector,
    materialize_record_tensors,
    preprocessing_policy,
    rich_channel_names,
)
from UnifiedRawSOH.preprocess.smvic_common import (  # noqa: E402
    DEFAULT_QUALITY_POLICY,
    FAMILY_SPECS,
    SMVIC_SCHEMA,
    SOURCE_SCHEMA,
    iter_classified_cycles,
    json_value,
    load_quality_policy,
)


DEFAULT_SOURCE = Path("/data1/chenyanxi/lb_project/datasets/SMVIC/dataset")
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "SMVIC_preprocessed_v3_128x128"
INDEX_COLUMNS = (
    "row", "battery_id", "cycle_id", "condition", "strategy_id", "domain_id",
    "soh", "soh_raw", "cc_raw_points", "cv_raw_points", "raw_point_count",
    "duration_min", "cc_duration_min", "cv_duration_min", "source_file",
    "source_cycle", "source_absolute_start_time", "source_absolute_end_time",
    "source_view",
)
TEST_PROTOCOLS = {
    "Battery01": [("test_cell01", ["Battery01/Cell01"]), ("test_cell02", ["Battery01/Cell02"])],
    "Battery02": [("test_cell01", ["Battery02/Cell01"]), ("test_cell02", ["Battery02/Cell02"])],
    "Battery03": [("test_cell01", ["Battery03/Cell01"]), ("test_cell02", ["Battery03/Cell02"])],
    "Battery04": [("test_cell01", ["Battery04/Cell01"]), ("test_cell02", ["Battery04/Cell02"])],
    "Battery05": [
        ("test_seed420", ["Battery05/Cell01", "Battery05/Cell06"]),
        ("test_seed421", ["Battery05/Cell05", "Battery05/Cell07"]),
    ],
    "Battery06": [("test_cell01", ["Battery06/Cell01"]), ("test_cell02", ["Battery06/Cell02"])],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--groups", nargs="+", choices=(*FAMILY_SPECS, "all"), default=["all"])
    parser.add_argument("--cc-len", type=int, default=128)
    parser.add_argument("--cv-len", type=int, default=128)
    parser.add_argument("--min-phase-points", type=int, default=4)
    parser.add_argument("--quality-policy", type=Path, default=DEFAULT_QUALITY_POLICY)
    parser.add_argument("--max-cycles-per-cell", type=int)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_array(directory: Path, name: str, values: np.ndarray) -> dict[str, Any]:
    path = directory / f"terminal_{name}.npy"
    np.save(path, values, allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": _sha256(path),
    }


def _split_spec(
    group: str,
    domain_id: str,
    protocol_id: str,
    preferred: list[str],
    observed: set[str],
    smoke: bool,
) -> dict[str, Any]:
    missing = sorted(set(preferred) - observed)
    if missing and not smoke:
        raise ValueError(f"Configured test cells were not materialized for {domain_id}/{protocol_id}: {missing}")
    test = sorted(set(preferred) & observed)
    if not test and smoke and observed:
        test = [sorted(observed)[-1]]
    if not test:
        raise ValueError(f"No test cells were materialized for {domain_id}/{protocol_id}: {preferred}")
    development = sorted(observed - set(test))
    if not development:
        raise ValueError(f"No development cells remain for {domain_id}")
    return {
        "name": f"{domain_id}_{protocol_id}",
        "protocol_id": protocol_id,
        "dataset_id": domain_id,
        "status": "smoke_only" if smoke else "recommended",
        "test_batteries": test,
        "development_split": {
            "mode": "mixed_cycle",
            "scope": "single_domain_pool",
            "val_ratio": 0.2,
            "random_state": 420,
            "train_val_battery_overlap_expected": True,
        },
        "counts": {
            "total_batteries": len(observed),
            "development_batteries": len(development),
            "test_batteries": len(test),
        },
        "notes": [
            "Test cells are physically disjoint from development cells.",
            "All train/validation mixed-cycle partitions use random_state=420.",
            *( [f"Bounded smoke product omitted configured cells: {missing}"] if missing else [] ),
        ],
    }


def _split_specs(group: str, domain_id: str, observed: set[str], smoke: bool) -> list[dict[str, Any]]:
    return [
        _split_spec(group, domain_id, protocol_id, cells, observed, smoke)
        for protocol_id, cells in TEST_PROTOCOLS[group]
    ]


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: tuple[str, ...] | list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _empty_state() -> dict[str, Any]:
    return {
        "cc": [],
        "cv": [],
        "features": [],
        "labels": [],
        "index": [],
        "keys": set(),
        "materialize_exclusions": [],
    }


def _append_materialized_record(
    state: dict[str, Any],
    record: Mapping[str, Any],
    spec,
    *,
    cc_len: int,
    cv_len: int,
) -> bool:
    """Immediately reduce one raw terminal record to fixed-size arrays."""

    key = (str(record["battery_id"]), int(record["cycle_id"]))
    if key in state["keys"]:
        raise ValueError(f"Duplicate SMVIC physical cycle key: {key}")
    state["keys"].add(key)
    try:
        cc, cv, stats = materialize_record_tensors(
            record,
            cc_len=cc_len,
            cv_len=cv_len,
            normalization=spec.normalization(),
        )
        features = feature_vector(record)
    except (KeyError, TypeError, ValueError) as exc:
        state["materialize_exclusions"].append({
            "domain_id": spec.domain_id,
            "battery_id": key[0],
            "cycle_id": key[1],
            "eligible": 0,
            "reason": f"materialization:{type(exc).__name__}:{exc}",
        })
        return False
    row = len(state["index"])
    state["cc"].append(cc)
    state["cv"].append(cv)
    state["features"].append(features)
    state["labels"].append(float(record["soh"]))
    state["index"].append({
        "row": row,
        "battery_id": record["battery_id"],
        "cycle_id": int(record["cycle_id"]),
        "condition": record["condition"],
        "strategy_id": record["strategy_id"],
        "domain_id": spec.domain_id,
        "soh": float(record["soh"]),
        "soh_raw": float(record["soh_raw"]),
        **stats,
        "source_file": record["source_file"],
        "source_cycle": record["source_cycle"],
        "source_absolute_start_time": record["source_absolute_start_time"],
        "source_absolute_end_time": record["source_absolute_end_time"],
        "source_view": "terminal",
    })
    return True


def _materialize_domain(
    build_dir: Path,
    spec,
    state: dict[str, Any],
    audits: list[dict[str, Any]],
    *,
    cc_len: int,
    cv_len: int,
    smoke: bool,
    quality_policy,
) -> dict[str, Any]:
    build_dir.mkdir(parents=True)
    (build_dir / "audit").mkdir()
    index: list[dict[str, Any]] = state["index"]
    exclusions = [dict(row) for row in audits if not int(row["eligible"])]
    materialize_exclusions: list[dict[str, Any]] = state["materialize_exclusions"]
    normalization = spec.normalization()
    if not index:
        raise ValueError(f"No usable model-ready cycles for {spec.domain_id}")

    arrays = {
        "cc": np.stack(state["cc"]).astype(np.float32),
        "cv": np.stack(state["cv"]).astype(np.float32),
        "features": np.stack(state["features"]).astype(np.float32),
        "soh": np.asarray(state["labels"], dtype=np.float32).reshape(-1, 1),
    }
    array_contracts = {name: _save_array(build_dir, name, values) for name, values in arrays.items()}
    _write_csv(build_dir / "terminal_index.csv", index, INDEX_COLUMNS)
    all_exclusions = exclusions + materialize_exclusions
    exclusion_fields = sorted({key for row in all_exclusions for key in row}) or ["reason"]
    _write_csv(build_dir / "audit" / "exclusions.csv", all_exclusions, exclusion_fields)
    audit_fields = sorted({key for row in audits for key in row})
    _write_csv(build_dir / "audit" / "cycle_classification.csv", audits, audit_fields)
    observed = {str(row["battery_id"]) for row in index}
    split = {
        "schema_version": "smvic_evaluation_splits_v1",
        "domain_id": spec.domain_id,
        "aggregation": "unweighted mean of the two protocol test metrics",
        "train_val_random_state": 420,
        "protocols": _split_specs(spec.group, spec.domain_id, observed, smoke),
    }
    (build_dir / "splits.json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = {
        "domain_id": spec.domain_id,
        "source_cycles_considered": len(audits),
        "protocol_eligible_cycles": sum(int(row["eligible"]) for row in audits),
        "materialized_cycles": len(index),
        "exclusion_count": len(all_exclusions),
        "exclusion_reasons": dict(sorted(Counter(str(row["reason"]) for row in all_exclusions).items())),
        "battery_counts": dict(sorted(Counter(str(row["battery_id"]) for row in index).items())),
        "soh_range": [float(np.min(arrays["soh"])), float(np.max(arrays["soh"]))],
        "bounded_smoke_product": smoke,
        "quality_control": quality_policy.manifest(),
    }
    (build_dir / "audit" / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "policy_version": preprocessing_policy(2),
        "schema_version": 2,
        "product_schema": SMVIC_SCHEMA,
        "source_schema": SOURCE_SCHEMA,
        "domain_id": spec.domain_id,
        "battery_group": spec.group,
        "source": {
            "kind": "normalized_smvic_cell_csv",
            "soh_label": "cycle_discharge_capacity_Ah / fixed nominal_capacity_Ah",
            "same_cycle_alignment": True,
        },
        "quality_control": quality_policy.manifest(),
        "protocol": json_value(spec.__dict__),
        "resampling": {
            "method": "linear_on_physical_phase_time",
            "cc_length": cc_len,
            "cv_length": cv_len,
            "full_joint_method": None,
            "full_joint_length": 0,
        },
        "normalization": normalization,
        "normalization_clipped": False,
        "rich_channel_names": list(rich_channel_names(2)),
        "feature_names": list(FEATURE_NAMES),
        "feature_extraction": "unresampled_terminal_physical_points",
        "feature_standardization": "not_materialized; fit on train split only",
        "terminal": {
            "index": "terminal_index.csv",
            "records": len(index),
            "arrays": array_contracts,
        },
        "full": None,
        "split": "splits.json",
        "audit": {
            "report": "audit/preprocessing_report.json",
            "exclusions": "audit/exclusions.csv",
            "classification": "audit/cycle_classification.csv",
        },
        "bounded_smoke_product": smoke,
    }
    (build_dir / "manifest.json").write_text(
        json.dumps(json_value(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    if args.cc_len < 2 or args.cv_len < 2 or args.min_phase_points < 2:
        raise ValueError("CC/CV lengths and minimum phase points must be at least two")
    groups = list(FAMILY_SPECS) if "all" in args.groups else list(dict.fromkeys(args.groups))
    quality_policy = load_quality_policy(args.quality_policy)
    selected_domains = {FAMILY_SPECS[group].domain_id for group in groups}
    existing = [args.output_root / domain for domain in selected_domains if (args.output_root / domain).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"SMVIC products already exist; pass --overwrite: {existing}")

    states_by_domain: dict[str, dict[str, Any]] = {
        domain: _empty_state() for domain in selected_domains
    }
    audits_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    processed = 0
    for record, audit in iter_classified_cycles(
        args.source_root,
        groups,
        max_cycles_per_cell=args.max_cycles_per_cell,
        min_phase_points=args.min_phase_points,
        quality_policy=quality_policy,
    ):
        domain = str(audit["domain_id"])
        audits_by_domain[domain].append(audit)
        if record is not None:
            _append_materialized_record(
                states_by_domain[domain],
                record,
                FAMILY_SPECS[str(audit["battery_group"])],
                cc_len=args.cc_len,
                cv_len=args.cv_len,
            )
        processed += 1
        if args.progress_every > 0 and processed % args.progress_every == 0:
            kept = sum(len(state["index"]) for state in states_by_domain.values())
            print(f"[SMVIC preprocess] cycles={processed}, materialized={kept}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".smvic_build_", dir=args.output_root))
    reports = {}
    try:
        for group in groups:
            spec = FAMILY_SPECS[group]
            reports[spec.domain_id] = _materialize_domain(
                temporary_root / spec.domain_id,
                spec,
                states_by_domain[spec.domain_id],
                audits_by_domain[spec.domain_id],
                cc_len=args.cc_len,
                cv_len=args.cv_len,
                smoke=args.max_cycles_per_cell is not None,
                quality_policy=quality_policy,
            )
        for domain in selected_domains:
            destination = args.output_root / domain
            if destination.exists():
                shutil.rmtree(destination)
            (temporary_root / domain).rename(destination)
        temporary_root.rmdir()
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    summary = {
        "product_schema": SMVIC_SCHEMA,
        "output_root": str(args.output_root.resolve()),
        "quality_control": quality_policy.manifest(),
        "reports": reports,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(json_value(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_value(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
