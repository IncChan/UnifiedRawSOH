#!/usr/bin/env python3
"""Validate Paper-Backup mmap products without loading vendor source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from UnifiedRawSOH.preprocess.paper_backup.common import (  # noqa: E402
    FEATURE_NAMES,
    PAPER_BACKUP_PREPROCESS_POLICY,
    PAPER_BACKUP_PREPROCESS_SCHEMA,
    RICH_CHANNEL_NAMES,
    preprocessing_policy,
    rich_channel_names,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty Paper-Backup index: {path}")
    keys = [(row["battery_id"], int(row["cycle_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate physical cycle keys in {path}")
    if [int(row["row"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"Non-contiguous array rows in {path}")
    return rows


def _cohort_keys(path: Path) -> set[tuple[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(str(row["battery_id"]), int(row["cycle_id"])) for row in rows}
    if not keys or len(keys) != len(rows):
        raise ValueError(f"Empty or duplicate FULL cohort keys: {path}")
    return keys


def _arrays(
    directory: Path,
    section: dict[str, Any],
    rows: int,
    *,
    require_features: bool,
    allow_full_joint: bool = False,
) -> dict[str, Any]:
    expected_names = {"cc", "cv", "soh"} | ({"features"} if require_features else set())
    optional_names = {"joint", "boundary_index"} if allow_full_joint else set()
    actual_names = set(section["arrays"])
    if not expected_names <= actual_names or actual_names - expected_names not in (
        set(), optional_names
    ):
        raise ValueError(f"Unexpected array inventory: {set(section['arrays'])}")
    if ("joint" in actual_names) != ("boundary_index" in actual_names):
        raise ValueError("FULL joint array and boundary index must be materialized together")
    output = {}
    for name, contract in section["arrays"].items():
        path = directory / contract["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != contract["sha256"]:
            raise ValueError(f"Array checksum mismatch: {path}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != list(contract["shape"]) or str(values.dtype) != contract["dtype"]:
            raise ValueError(f"Array contract mismatch: {path}")
        if values.shape[0] != rows or not np.all(np.isfinite(values)):
            raise ValueError(f"Array row/finite check failed: {path}")
        output[name] = {"shape": list(values.shape), "dtype": str(values.dtype)}
    return output


def validate_domain(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = int(manifest.get("schema_version", 0))
    channel_names = rich_channel_names(schema_version)
    if manifest.get("policy_version") != preprocessing_policy(schema_version):
        raise ValueError(f"Policy mismatch: {manifest_path}")
    if manifest.get("rich_channel_names") != list(channel_names):
        raise ValueError(f"Rich-channel contract mismatch: {manifest_path}")
    if manifest.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError(f"Feature contract mismatch: {manifest_path}")
    terminal_rows = _index(directory / manifest["terminal"]["index"])
    terminal_arrays = _arrays(
        directory, manifest["terminal"], len(terminal_rows), require_features=True
    )
    cc_len = int(manifest["resampling"]["cc_length"])
    cv_len = int(manifest["resampling"]["cv_length"])
    if terminal_arrays["cc"]["shape"][1:] != [cc_len, len(channel_names)]:
        raise ValueError("Terminal CC shape does not match manifest")
    if terminal_arrays["cv"]["shape"][1:] != [cv_len, len(channel_names)]:
        raise ValueError("Terminal CV shape does not match manifest")
    if terminal_arrays["features"]["shape"][1:] != [len(FEATURE_NAMES)]:
        raise ValueError("Terminal feature shape does not match manifest")
    terminal_keys = {(row["battery_id"], int(row["cycle_id"])) for row in terminal_rows}
    full_summary = None
    if manifest.get("full") is not None:
        full_rows = _index(directory / manifest["full"]["index"])
        full_arrays = _arrays(
            directory,
            manifest["full"],
            len(full_rows),
            require_features=False,
            allow_full_joint=True,
        )
        full_keys = {(row["battery_id"], int(row["cycle_id"])) for row in full_rows}
        if not full_keys <= terminal_keys:
            raise ValueError("FULL cohort contains a cycle absent from Terminal")
        cohort_keys = _cohort_keys(directory / manifest["full"]["cohort"])
        if cohort_keys != full_keys:
            raise ValueError("FULL cohort key file does not match full_index.csv")
        joint_length = int(manifest["resampling"].get("full_joint_length", 0) or 0)
        if joint_length > 0:
            if full_arrays.get("joint", {}).get("shape", [None, None])[1:] != [
                joint_length,
                len(channel_names),
            ]:
                raise ValueError("FULL joint shape does not match manifest")
            boundary = np.load(
                directory / manifest["full"]["arrays"]["boundary_index"]["file"],
                mmap_mode="r",
                allow_pickle=False,
            )
            if boundary.ndim != 1 or np.any(boundary <= 0) or np.any(boundary >= joint_length):
                raise ValueError("FULL joint boundary indices are invalid")
        full_summary = {"records": len(full_rows), "arrays": full_arrays}
    return {
        "domain_id": manifest["domain_id"],
        "terminal_records": len(terminal_rows),
        "terminal_arrays": terminal_arrays,
        "full": full_summary,
        "status": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "datasets" / "PaperBackup_preprocessed",
    )
    parser.add_argument("--domains", nargs="*", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    directories = [root / name for name in args.domains] if args.domains else sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").is_file()
    )
    if not directories:
        raise FileNotFoundError(f"No Paper-Backup preprocessed domains under {root}")
    result = {directory.name: validate_domain(directory) for directory in directories}
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
