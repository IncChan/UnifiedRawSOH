#!/usr/bin/env python3
"""Build mmap-ready Paper-Backup arrays from canonical terminal records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from UnifiedRawSOH.datasets.paper_backup.full_cccv import (  # noqa: E402
    iter_materialize_full_records,
    iter_match_full_terminal_records,
)
from UnifiedRawSOH.datasets.filters import filter_records_by_invalid_cycles  # noqa: E402
from UnifiedRawSOH.datasets.mit import list_mit_raw_files, read_mit_raw_file  # noqa: E402
from UnifiedRawSOH.datasets.smarthealth import (  # noqa: E402
    list_smarthealth_raw_files,
    read_smarthealth_raw_file,
)
from UnifiedRawSOH.datasets.splits import load_invalid_cycles  # noqa: E402
from UnifiedRawSOH.datasets.xjtu import list_xjtu_csv_files, read_xjtu_file  # noqa: E402
from UnifiedRawSOH.preprocess.paper_backup.common import (  # noqa: E402
    FEATURE_NAMES,
    PAPER_BACKUP_PREPROCESS_POLICY,
    PAPER_BACKUP_PREPROCESS_SCHEMA,
    RICH_CHANNEL_NAMES,
    feature_vector,
    materialize_full_joint_tensor,
    materialize_record_tensors,
    normalization_contract,
    preprocessing_policy,
    rich_channel_names,
)
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402


DEFAULT_CONFIGS = {
    "xjtu": "configs/paper_backup/common/domain_xjtu_sequence.json",
    "mit": "configs/paper_backup/common/domain_mit_sequence.json",
    "smarthealth_lishen40": "configs/paper_backup/common/domain_lishen_sequence.json",
    "smarthealth_catl280": "configs/paper_backup/common/domain_catl_sequence.json",
    "smarthealth_eve280": "configs/paper_backup/common/domain_eve_sequence.json",
}
FULL_SUPPORTED_DOMAINS = {
    "xjtu": "xjtu_mat",
    "mit": "normalized_full_csv",
    "smarthealth_lishen40": "smarthealth_gb18030",
    "smarthealth_catl280": "smarthealth_gb18030",
    "smarthealth_eve280": "smarthealth_gb18030",
}
INDEX_COLUMNS = (
    "row",
    "battery_id",
    "cycle_id",
    "condition",
    "strategy_id",
    "domain_id",
    "soh",
    "soh_raw",
    "cc_raw_points",
    "cv_raw_points",
    "raw_point_count",
    "duration_min",
    "cc_duration_min",
    "cv_duration_min",
    "source_file",
    "source_cycle",
    "source_absolute_start_time",
    "source_absolute_end_time",
    "source_view",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _metadata_row(
    record: Mapping[str, Any], row: int, stats: Mapping[str, float], source_view: str
) -> dict[str, Any]:
    return {
        "row": int(row),
        "battery_id": str(record["battery_id"]),
        "cycle_id": int(record["cycle_id"]),
        "condition": str(record.get("condition", record.get("strategy_id", "unknown"))),
        "strategy_id": str(record.get("strategy_id", record.get("condition", "unknown"))),
        "domain_id": str(record.get("domain_id", record.get("dataset_id", "unknown"))),
        "soh": float(record["soh"]),
        "soh_raw": float(record.get("soh_raw", record["soh"])),
        **{key: float(stats[key]) for key in (
            "cc_raw_points", "cv_raw_points", "raw_point_count", "duration_min",
            "cc_duration_min", "cv_duration_min",
        )},
        "source_file": str(record.get("source_file", "")),
        "source_cycle": str(record.get("source_cycle", "")),
        "source_absolute_start_time": str(record.get("source_absolute_start_time", "")),
        "source_absolute_end_time": str(record.get("source_absolute_end_time", "")),
        "source_view": source_view,
    }


def _materialize_collection(
    records: Iterable[Mapping[str, Any]],
    *,
    cc_len: int,
    cv_len: int,
    normalization: Mapping[str, float],
    include_features: bool,
    source_view: str,
    progress_label: str | None = None,
    progress_every: int = 5000,
    full_joint_len: int = 0,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    cc_rows: list[np.ndarray] = []
    cv_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    labels: list[float] = []
    joint_rows: list[np.ndarray] = []
    boundary_rows: list[int] = []
    metadata: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    keys: set[tuple[str, int]] = set()
    for processed_records, record in enumerate(records, start=1):
        key = (str(record.get("battery_id")), int(record.get("cycle_id", -1)))
        if key in keys:
            raise ValueError(f"Duplicate physical cycle key in {source_view}: {key}")
        keys.add(key)
        try:
            cc, cv, stats = materialize_record_tensors(
                record,
                cc_len=cc_len,
                cv_len=cv_len,
                normalization=normalization,
            )
            features = feature_vector(record) if include_features else None
            if int(full_joint_len) > 0:
                joint, boundary_index = materialize_full_joint_tensor(
                    record,
                    joint_len=int(full_joint_len),
                    normalization=normalization,
                )
            else:
                joint, boundary_index = None, None
        except (KeyError, TypeError, ValueError) as exc:
            exclusions.append(
                {
                    "battery_id": key[0],
                    "cycle_id": key[1],
                    "source_view": source_view,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        row = len(cc_rows)
        cc_rows.append(cc)
        cv_rows.append(cv)
        if features is not None:
            feature_rows.append(features)
        if joint is not None and boundary_index is not None:
            joint_rows.append(joint)
            boundary_rows.append(int(boundary_index))
        labels.append(float(record["soh"]))
        metadata.append(_metadata_row(record, row, stats, source_view))
        if progress_label and processed_records % int(progress_every) == 0:
            print(
                f"[Paper-Backup materialize] {progress_label}: input {processed_records}, "
                f"kept {len(metadata)}, excluded {len(exclusions)}",
                flush=True,
            )
    if not cc_rows:
        raise ValueError(f"No usable {source_view} Paper-Backup cycles")
    arrays = {
        "cc": np.stack(cc_rows).astype(np.float32),
        "cv": np.stack(cv_rows).astype(np.float32),
        "soh": np.asarray(labels, dtype=np.float32).reshape(-1, 1),
    }
    if include_features:
        arrays["features"] = np.stack(feature_rows).astype(np.float32)
    if int(full_joint_len) > 0:
        arrays["joint"] = np.stack(joint_rows).astype(np.float32)
        arrays["boundary_index"] = np.asarray(boundary_rows, dtype=np.int64)
    return arrays, metadata, exclusions


def _write_arrays(directory: Path, prefix: str, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, values in arrays.items():
        path = directory / f"{prefix}_{name}.npy"
        np.save(path, values, allow_pickle=False)
        output[name] = {
            "file": path.name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": _sha256(path),
        }
    return output


def _full_root(domain_id: str, args: argparse.Namespace) -> Path | None:
    if domain_id == "xjtu" and args.xjtu_full_source_root:
        return args.xjtu_full_source_root.resolve()
    if domain_id.startswith("smarthealth_") and args.smarthealth_full_source_root:
        return args.smarthealth_full_source_root.resolve()
    if domain_id == "mit" and args.mit_full_source_root:
        return args.mit_full_source_root.resolve()
    return None


def _resolve_config_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _terminal_file_inventory(
    domain_id: str, config: Mapping[str, Any]
) -> tuple[list[Path], Any]:
    data = dict(config.get("data", {}))
    experiment = dict(config.get("experiment", {}))
    root_value = data.get("terminal_data_root", data.get("data_root"))
    if not root_value:
        raise ValueError(f"No canonical Terminal root configured for {domain_id}")
    root = _resolve_config_path(config, root_value)
    batches = list(experiment.get("batches", []))
    if domain_id == "xjtu":
        files = list_xjtu_csv_files(root) if not batches else sorted(
            {path for batch in batches for path in list_xjtu_csv_files(root, batch=batch)}
        )
        reader = lambda path: read_xjtu_file(
            path,
            nominal_capacity=float(data.get("nominal_capacity", 2.0)),
            label_scale_mode=str(data.get("label_scale_mode", "auto_capacity_to_soh")),
        )
    elif domain_id == "mit":
        files = list_mit_raw_files(root) if not batches else sorted(
            {path for batch in batches for path in list_mit_raw_files(root, batch=batch)}
        )
        reader = lambda path: read_mit_raw_file(
            path,
            nominal_capacity=float(data.get("nominal_capacity", 1.1)),
            label_scale_mode=str(data.get("label_scale_mode", "none")),
        )
    elif domain_id.startswith("smarthealth_"):
        files = list_smarthealth_raw_files(root, domain_id=domain_id)
        reader = lambda path: read_smarthealth_raw_file(
            path,
            label_scale_mode=str(data.get("label_scale_mode", "label_capacity_to_nominal")),
            domain_id=domain_id,
        )
    else:
        raise ValueError(f"Unsupported Paper-Backup preprocessing domain: {domain_id}")
    return files, reader


def _stream_terminal_records(
    domain_id: str,
    config: Mapping[str, Any],
    *,
    max_records: int | None,
) -> tuple[Iterable[dict[str, Any]], dict[str, Any]]:
    files, reader = _terminal_file_inventory(domain_id, config)
    split_value = config.get("data", {}).get("split_file") or config.get("experiment", {}).get("split_file")
    invalid = load_invalid_cycles(_resolve_config_path(config, split_value)) if split_value else {}
    audit = {"files": len(files), "invalid_cycle_filter": Counter(), "streaming": True}

    def generate():
        emitted = 0
        for path in files:
            current = list(reader(path))
            current, invalid_audit = filter_records_by_invalid_cycles(current, invalid)
            audit["invalid_cycle_filter"]["removed_records"] += int(
                invalid_audit.get("removed_records", 0)
            )
            for record in current:
                record["domain_id"] = domain_id
                record["source_view"] = "terminal"
                record["is_full"] = False
                yield record
                emitted += 1
                if max_records is not None and emitted >= int(max_records):
                    return

    return generate(), {
        "domain_id": domain_id,
        "canonical_files": len(files),
        "streaming": "one canonical battery/cell CSV at a time",
        "invalid_cycle_filter": audit["invalid_cycle_filter"],
    }


def build_domain(domain_id: str, args: argparse.Namespace) -> dict[str, Any]:
    config_path = REPO_ROOT / DEFAULT_CONFIGS[domain_id]
    config = load_config(config_path)
    config.setdefault("data", {})["source_mode"] = "legacy_runtime"
    records, source_info = _stream_terminal_records(
        domain_id, config, max_records=args.max_records
    )
    normalization = normalization_contract(config, schema_version=args.schema_version)
    policy_version = preprocessing_policy(args.schema_version)
    channel_names = rich_channel_names(args.schema_version)
    cc_len = int(args.cc_len or config.get("data", {}).get("raw_len_cc", 128))
    cv_len = int(args.cv_len or config.get("data", {}).get("raw_len_cv", 256))

    final_directory = args.output_root.resolve() / domain_id
    build_directory = args.output_root.resolve() / f".{domain_id}.building"
    if final_directory.exists() and not args.overwrite:
        raise FileExistsError(f"Paper-Backup preprocessed domain exists: {final_directory}")
    if build_directory.exists():
        shutil.rmtree(build_directory)
    build_directory.mkdir(parents=True)
    (build_directory / "audit").mkdir()
    (build_directory / "cohorts").mkdir()

    terminal_arrays, terminal_index, exclusions = _materialize_collection(
        records,
        cc_len=cc_len,
        cv_len=cv_len,
        normalization=normalization,
        include_features=True,
        source_view="terminal",
        progress_label=f"{domain_id} terminal",
    )
    terminal_files = _write_arrays(build_directory, "terminal", terminal_arrays)
    del terminal_arrays
    _write_csv(build_directory / "terminal_index.csv", terminal_index)
    print(
        f"[Paper-Backup terminal] {domain_id}: cycles {len(terminal_index)}, "
        f"excluded {sum(row['source_view'] == 'terminal' for row in exclusions)}",
        flush=True,
    )

    full_files = None
    full_index: list[dict[str, Any]] = []
    matching_audit = None
    if domain_id in args.include_full:
        source_root = _full_root(domain_id, args)
        if source_root is None:
            raise ValueError(f"FULL preprocessing requested for {domain_id} without a source root")
        data_config = dict(config.get("data", {}))
        data_config.update(
            {
                "full_data_root": str(source_root),
                "full_source_format": FULL_SUPPORTED_DOMAINS[domain_id],
                "nominal_capacity": float(data_config.get("nominal_capacity", 1.0)),
                "full_workers": int(args.workers),
                "full_progress_label": domain_id,
            }
        )
        terminal_links = [
            {
                "dataset_id": domain_id,
                "domain_id": domain_id,
                "condition": row["condition"],
                "battery_id": row["battery_id"],
                "cycle_id": int(row["cycle_id"]),
                "soh": float(row["soh"]),
                "soh_raw": float(row["soh_raw"]),
                "source_file": row["source_file"],
                "source_cycle": row["source_cycle"],
                "source_absolute_start_time": row["source_absolute_start_time"],
                "source_absolute_end_time": row["source_absolute_end_time"],
            }
            for row in terminal_index
        ]
        full_records = iter_materialize_full_records(
            terminal_links,
            domain_id=domain_id,
            data_config=data_config,
        )
        matched, matching_audit = iter_match_full_terminal_records(terminal_links, full_records)
        full_arrays, full_index, full_exclusions = _materialize_collection(
            matched,
            cc_len=cc_len,
            cv_len=cv_len,
            normalization=normalization,
            include_features=False,
            source_view="full_cccv",
            progress_label=f"{domain_id} full",
            full_joint_len=int(args.full_joint_len),
        )
        exclusions.extend(full_exclusions)
        full_files = _write_arrays(build_directory, "full", full_arrays)
        del full_arrays
        _write_csv(build_directory / "full_index.csv", full_index)
        cohort_path = build_directory / "cohorts" / "full_matched_keys.csv"
        with cohort_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("battery_id", "cycle_id"))
            writer.writeheader()
            for row in full_index:
                writer.writerow({"battery_id": row["battery_id"], "cycle_id": row["cycle_id"]})
        print(
            f"[Paper-Backup full] {domain_id}: cycles {len(full_index)}, "
            f"workers {args.workers}, excluded {len(full_exclusions)}",
            flush=True,
        )

    exclusion_path = build_directory / "audit" / "exclusions.csv"
    with exclusion_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("battery_id", "cycle_id", "source_view", "reason")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(exclusions)
    report = {
        "policy_version": policy_version,
        "schema_version": int(args.schema_version),
        "domain_id": domain_id,
        "terminal_source_records_considered": len(terminal_index)
        + sum(row["source_view"] == "terminal" for row in exclusions),
        "terminal_materialized_records": len(terminal_index),
        "full_materialized_records": len(full_index),
        "full_workers": int(args.workers) if domain_id in args.include_full else None,
        "exclusion_count": len(exclusions),
        "exclusion_reasons": dict(Counter(row["reason"] for row in exclusions)),
        "matching": matching_audit,
    }
    (build_directory / "audit" / "preprocessing_report.json").write_text(
        json.dumps(_json_value(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "policy_version": policy_version,
        "schema_version": int(args.schema_version),
        "domain_id": domain_id,
        "source_config": str(config_path.relative_to(REPO_ROOT)),
        "source": source_info,
        "resampling": {
            "method": "linear_on_physical_phase_time",
            "cc_length": cc_len,
            "cv_length": cv_len,
            "full_joint_method": (
                "linear_on_complete_charge_physical_time"
                if int(args.full_joint_len) > 0
                else None
            ),
            "full_joint_length": int(args.full_joint_len),
        },
        "normalization": normalization,
        "normalization_clipped": False,
        "rich_channel_names": list(channel_names),
        "feature_names": list(FEATURE_NAMES),
        "feature_extraction": "unresampled_terminal_physical_points",
        "feature_standardization": "not_materialized; fit on train split only",
        "terminal": {
            "index": "terminal_index.csv",
            "records": len(terminal_index),
            "arrays": terminal_files,
        },
        "full": None if full_files is None else {
            "index": "full_index.csv",
            "records": len(full_index),
            "arrays": full_files,
            "cohort": "cohorts/full_matched_keys.csv",
            "definition": "complete observed principal charge event split into inferred CC and CV",
            "source_file_workers": int(args.workers),
            "streaming": "bounded ordered source-file results",
        },
        "audit": {
            "report": "audit/preprocessing_report.json",
            "exclusions": "audit/exclusions.csv",
        },
    }
    (build_directory / "manifest.json").write_text(
        json.dumps(_json_value(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if final_directory.exists():
        shutil.rmtree(final_directory)
    build_directory.rename(final_directory)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=(*DEFAULT_CONFIGS, "all"),
        default=["all"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "PaperBackup_preprocessed",
    )
    parser.add_argument("--schema-version", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--include-full",
        nargs="*",
        choices=tuple(FULL_SUPPORTED_DOMAINS),
        default=[],
    )
    parser.add_argument("--xjtu-full-source-root", type=Path)
    parser.add_argument(
        "--mit-full-source-root",
        type=Path,
        help="Normalized MIT full-charge CSV root (physical IDs/global cycle IDs).",
    )
    parser.add_argument("--smarthealth-full-source-root", type=Path)
    parser.add_argument("--cc-len", type=int, default=128)
    parser.add_argument("--cv-len", type=int, default=256)
    parser.add_argument(
        "--full-joint-len",
        type=int,
        default=0,
        help="Also materialize each FULL charge event on one joint time grid.",
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel source-file workers for SmartHealth FULL extraction.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError(f"--workers must be at least 1, got {args.workers}")
    domains = list(DEFAULT_CONFIGS) if "all" in args.domains else list(dict.fromkeys(args.domains))
    invalid_full = sorted(set(args.include_full) - set(domains))
    if invalid_full:
        raise ValueError(f"FULL domains must also be selected in --domains: {invalid_full}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "policy_version": preprocessing_policy(args.schema_version),
        "schema_version": int(args.schema_version),
        "domains": {},
    }
    for domain_id in domains:
        print(f"[Paper-Backup preprocess] {domain_id}", flush=True)
        summary["domains"][domain_id] = build_domain(domain_id, args)
    print(json.dumps(_json_value(summary), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
