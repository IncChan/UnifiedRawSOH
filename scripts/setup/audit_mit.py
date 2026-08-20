#!/usr/bin/env python3
"""Audit MIT raw/feature provenance without fabricating a raw sequence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_RAW_COLUMNS = {
    "cycle",
    "SOH",
    "capacity_Ah",
    "segment",
    "relative_time_min",
    "voltage_V",
    "current_A",
    "temperature_C",
    "physical_cell_id",
    "source_batch_date",
    "source_cell",
    "source_cycle",
}


def _headers(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return set(csv.DictReader(handle).fieldnames or ())


def _read_extraction_report(path, cycle_keys=("written_cycles", "written_raw_cycles"), row_keys=("written_rows", "written_raw_rows")):
    if not path.is_file():
        return {"rows": 0, "written_cycles": 0, "written_rows": 0, "source_batches": []}
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    def metric(row, keys):
        for key in keys:
            value = row.get(key, "")
            if value not in ("", None):
                return int(value)
        return 0

    return {
        "rows": len(rows),
        "written_cycles": sum(metric(row, cycle_keys) for row in rows),
        "written_rows": sum(metric(row, row_keys) for row in rows),
        "source_batches": sorted({row.get("batch_file", "") for row in rows if row.get("batch_file")}),
        "empty_cc_cycles": sum(int(row.get("empty_cc_cycles", 0) or 0) for row in rows),
        "empty_cv_cycles": sum(int(row.get("empty_cv_cycles", 0) or 0) for row in rows),
        "failed_cycles": sum(int(row.get("failed_cycles", 0) or 0) for row in rows),
    }


def _repository_dir(path):
    """Accept either the standalone repository or its parent workspace."""

    path = Path(path).resolve()
    if (path / "datasets").is_dir() and (path / "configs").is_dir():
        return path
    candidate = path / "UnifiedRawSOH"
    if (candidate / "datasets").is_dir() and (candidate / "configs").is_dir():
        return candidate
    raise ValueError(f"Cannot locate a UnifiedRawSOH repository below {path}")


def audit(project_root):
    repository_dir = _repository_dir(project_root)
    mit_raw = repository_dir / "datasets/MIT_raw"
    mit_features = repository_dir / "datasets/MIT_features"
    mit_original = repository_dir / "datasets/MIT_original"

    raw_files = sorted(mit_raw.glob("MIT_*_physical-*.csv"))
    feature_files = sorted(path for path in mit_features.glob("*.csv") if not path.name.endswith("_report.csv"))
    original_files = sorted(mit_original.glob("*/*.csv"))
    raw_headers = sorted(set().union(*(_headers(path) for path in raw_files))) if raw_files else []
    feature_headers = sorted(set().union(*(_headers(path) for path in feature_files))) if feature_files else []
    original_headers = sorted(set().union(*(_headers(path) for path in original_files))) if original_files else []
    report_path = mit_raw / "mit_physical_extraction_report.csv"
    raw_report = _read_extraction_report(report_path)
    feature_report = _read_extraction_report(
        report_path,
        cycle_keys=("written_feature_rows",),
        row_keys=("written_feature_rows",),
    )

    return {
        "status": "raw_source_available_adapter_ready" if raw_files else "raw_source_missing",
        "dataset_id": "mit",
        "raw_source": str(mit_raw),
        "raw_files": len(raw_files),
        "raw_columns": raw_headers,
        "raw_required_columns_present": sorted(REQUIRED_RAW_COLUMNS & set(raw_headers)),
        "raw_extraction_report": raw_report,
        "feature_source": str(mit_features),
        "feature_files": len(feature_files),
        "feature_columns": feature_headers,
        "feature_extraction_report": feature_report,
        "repository_root": str(repository_dir),
        "original_source": str(mit_original),
        "original_files": len(original_files),
        "original_columns": original_headers,
        "confirmed_protocol": {
            "phase_rule": "infer actual persistent CC-to-CV taper before windowing",
            "cc_voltage_window_V": [3.45, 3.60],
            "cv_c_rate_window": [0.05, 0.25],
            "cv_sampling_tolerance_C": 0.002,
            "nominal_capacity_Ah": 1.1,
            "soh_definition": "capacity_Ah / 1.1",
        },
        "shared_invalid_cycles": [
            {
                "battery_id": "mit_p015",
                "cycle_id": 39,
                "reason": "source 2017-05-12 cell-019 capacity spike removed by canonical extraction",
            }
        ],
        "generation_chain": [
            "MIT_raw continuation/cell-curation extraction from an authorized MIT batch source",
            "datasets/MIT_raw/MIT_PHYSICAL_PROVENANCE.json",
            "datasets/MIT_raw/mit_physical_extraction_report.csv",
            "datasets/MIT_features/*.csv statistical features",
        ],
        "notes": [
            "The canonical source has 124 physical cells, not 140 source files; continuation provenance is mandatory.",
            "MITRawAdapter validates physical IDs/global cycles, and raw/Only-F can match directly by physical key.",
            "Do not synthesize raw sequences or fallback to historical aligned tables.",
        ],
    }


def main():
    parser = argparse.ArgumentParser("Audit MIT raw and feature provenance")
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.project_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
