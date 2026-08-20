#!/usr/bin/env python3
"""Fast, read-only readiness check for an E1 benchmark launcher.

The check intentionally reads only CSV headers and the first data row.  It is
not a data loader, does not resample a sequence, and does not train a model.
Its purpose is to stop a multi-seed launcher before it creates several failed
jobs against a header-only or incompletely copied canonical export.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.domains import (  # noqa: E402
    build_default_domain_registry,
    canonical_domain_id,
)
from UnifiedRawSOH.datasets.mit import (  # noqa: E402
    inspect_mit_raw_inventory,
    parse_mit_file_identity,
    validate_mit_physical_cohort,
)
from UnifiedRawSOH.datasets.smarthealth import list_smarthealth_raw_files  # noqa: E402
from UnifiedRawSOH.datasets.splits import (  # noqa: E402
    load_split_spec,
)
from UnifiedRawSOH.datasets.xjtu import list_xjtu_csv_files  # noqa: E402
from UnifiedRawSOH.models.baselines.pinn4soh_no_leak_onlyf import (  # noqa: E402
    list_feature_csv_files,
)
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


def _resolve_path(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _domain_id(config: dict) -> str:
    experiment = config.get("experiment", {})
    data = config.get("data", {})
    return canonical_domain_id(
        experiment.get(
            "domain_id",
            experiment.get("dataset_id", data.get("domain_id", data.get("dataset", "xjtu"))),
        )
    )


def _data_root(config: dict, domain_id: str) -> Path:
    configured = config.get("data", {}).get("data_root")
    if configured:
        return _resolve_path(configured)
    domain = build_default_domain_registry().get(domain_id)
    root = domain.data_root
    if not root:
        raise RuntimeError(f"Domain {domain_id!r} has no configured data root")
    return _resolve_path(root)


def _header_has_data_row(path: Path) -> bool:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return False
        return next(reader, None) is not None


def _require_nonempty_files(files: list[Path], label: str) -> None:
    empty = [path.name for path in files if not _header_has_data_row(path)]
    if empty:
        preview = ", ".join(empty[:5])
        suffix = " ..." if len(empty) > 5 else ""
        raise RuntimeError(
            f"{label} is not ready: {len(empty)}/{len(files)} CSVs contain only a header "
            f"({preview}{suffix}). Regenerate/copy the canonical product; no legacy or "
            "aligned fallback is used."
        )


def _split_spec_from_config(config: dict) -> dict:
    split_file = (
        config.get("data", {}).get("split_file")
        or config.get("experiment", {}).get("split_file")
    )
    if not split_file:
        raise RuntimeError("E1 config has no JSON split_file for cohort validation")
    return load_split_spec(_resolve_path(split_file))


def _mit_physical_ids(files: list[Path], product_name: str) -> set[str]:
    """Return canonical IDs and reject legacy/source-file MIT exports."""

    physical_ids: list[str] = []
    for path in files:
        _, battery_id, is_physical = parse_mit_file_identity(path)
        if not is_physical:
            raise RuntimeError(
                f"{product_name} includes a legacy source-file CSV ({path.name}); "
                "the Paper E1 MIT protocol requires canonical physical-cell files."
            )
        physical_ids.append(battery_id)
    if len(physical_ids) != len(set(physical_ids)):
        raise RuntimeError(
            f"{product_name} has duplicate physical-cell file identities: {sorted(physical_ids)}"
        )
    return set(physical_ids)


def _validate_mit_e1_cohort(config: dict, files: list[Path], product_name: str) -> dict:
    """Validate the JSON-owned MIT cohort before multi-seed training starts.

    The official E1 MIT configs declare the continuation-aware Paper-124 cohort.
    A separately named debugging/subset config can set
    ``require_full_physical_cohort`` to false and provide its own split JSON;
    this guard must not silently reinterpret the official config as a subset.
    """

    physical_ids = _mit_physical_ids(files, product_name)
    split_spec = _split_spec_from_config(config)
    try:
        return validate_mit_physical_cohort(
            physical_ids,
            split_spec,
            require_full_physical_cohort=bool(
                config.get("data", {}).get("require_full_physical_cohort", False)
            ),
        )
    except ValueError as exc:
        raise RuntimeError(
            f"{product_name} is not compatible with the MIT JSON cohort/test rule: {exc}. "
            "Wait for the paired canonical export to finish or use a separately declared "
            "subset split; do not run the official MIT E1 config on a partial copy."
        ) from exc


def check_raw(config: dict) -> dict:
    domain_id = _domain_id(config)
    root = _data_root(config, domain_id)
    if domain_id == "mit":
        inventory = inspect_mit_raw_inventory(root)
        if inventory["header_only_files"] or inventory["missing_required_headers"]:
            raise RuntimeError(
                "MIT phase-aware raw export is not ready: "
                f"files={inventory['files']}, nonempty={len(inventory['nonempty_files'])}, "
                f"header_only={len(inventory['header_only_files'])}, "
                f"missing_headers={len(inventory['missing_required_headers'])}. "
                "Expected nonempty physical124 CSVs produced by "
                "mit_proposed_phase_aware_cccv_v3 under UnifiedRawSOH/datasets/MIT_raw."
            )
        files = [root / name for name in inventory["nonempty_files"]]
        mit_cohort = _validate_mit_e1_cohort(config, files, "MIT raw product")
    elif domain_id.startswith("smarthealth_"):
        files = list_smarthealth_raw_files(root, domain_id=domain_id)
        if not files:
            raise RuntimeError(f"No canonical SmartHealth raw CSVs found for {domain_id!r} under {root}")
        _require_nonempty_files(files, f"SmartHealth raw {domain_id}")
    elif domain_id == "xjtu":
        files = list_xjtu_csv_files(root)
        _require_nonempty_files(files, "XJTU raw")
    else:
        raise RuntimeError(f"No E1 RawMamba readiness rule for domain {domain_id!r}")
    result = {
        "mode": "raw",
        "domain_id": domain_id,
        "data_root": str(root),
        "files": len(files),
        "status": "ready",
    }
    if domain_id == "mit":
        result.update(
            physical_cells=len(mit_cohort["physical_ids"]),
            test_cells=len(mit_cohort["test_ids"]),
        )
    return result


def check_onlyf(config: dict) -> dict:
    domain_id = _domain_id(config)
    root = _data_root(config, domain_id)
    files = list_feature_csv_files(root, domain_id=domain_id)
    _require_nonempty_files(files, f"Only-F features {domain_id}")
    if domain_id == "mit":
        feature_cohort = _validate_mit_e1_cohort(config, files, "MIT Only-F feature product")
        raw_root = _resolve_path(
            build_default_domain_registry().get(domain_id).data_root or ""
        )
        raw_inventory = inspect_mit_raw_inventory(raw_root)
        if raw_inventory["header_only_files"] or raw_inventory["missing_required_headers"]:
            raise RuntimeError(
                "MIT Only-F requires the paired canonical raw product to be ready as well; "
                "no old MIT feature-only fallback is used."
            )
        raw_files = [raw_root / name for name in raw_inventory["nonempty_files"]]
        raw_cohort = _validate_mit_e1_cohort(config, raw_files, "MIT raw product")
        if feature_cohort["physical_ids"] != raw_cohort["physical_ids"]:
            raise RuntimeError(
                "MIT raw/Only-F physical-cell mismatch: "
                f"raw_only={sorted(raw_cohort['physical_ids'] - feature_cohort['physical_ids'])[:5]}, "
                f"feature_only={sorted(feature_cohort['physical_ids'] - raw_cohort['physical_ids'])[:5]}."
            )
    if domain_id.startswith("smarthealth_"):
        raw_root = _resolve_path(
            build_default_domain_registry().get(domain_id).data_root or ""
        )
        raw_names = {path.name for path in list_smarthealth_raw_files(raw_root, domain_id=domain_id)}
        feature_names = {path.name for path in files}
        if raw_names != feature_names:
            raise RuntimeError(
                f"SmartHealth raw/Only-F feature product mismatch for {domain_id}: "
                f"raw_only={sorted(raw_names - feature_names)[:3]}, "
                f"feature_only={sorted(feature_names - raw_names)[:3]}"
            )
    result = {
        "mode": "onlyf",
        "domain_id": domain_id,
        "data_root": str(root),
        "files": len(files),
        "status": "ready",
    }
    if domain_id == "mit":
        result.update(
            physical_cells=len(feature_cohort["physical_ids"]),
            test_cells=len(feature_cohort["test_ids"]),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Check E1 raw/Only-F dataset readiness without training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "onlyf"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = check_raw(config) if args.mode == "raw" else check_onlyf(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(f"E1 dataset readiness failed: {exc}") from exc
