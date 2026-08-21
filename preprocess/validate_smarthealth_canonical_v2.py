#!/usr/bin/env python3
"""Validate generated SmartHealth v3 canonical products without source CSVs.

Run after one or more family-specific RAW and FEATURE jobs.  The validator
reads only canonical RAW/feature files, their v3 audit files, and split JSON;
it never revisits the GB18030 SmartHealth source tree.

The filename is retained as a compatibility entry point for the launcher.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

from smarthealth_common import (
    CATL280_CONFIG,
    EVE280_CONFIG,
    FEATURE_COLUMNS,
    FEATURE_PREFIX_COLUMNS,
    LISHEN40_CONFIG,
    POLICY_VERSION,
    SPLIT_STRATEGY_VERSION,
    RAW_COLUMNS,
    DomainConfig,
    raw_audit_paths,
)


CONFIGS = {
    LISHEN40_CONFIG.domain_id: LISHEN40_CONFIG,
    CATL280_CONFIG.domain_id: CATL280_CONFIG,
    EVE280_CONFIG.domain_id: EVE280_CONFIG,
}


class ValidationError(RuntimeError):
    """A generated canonical v3 product violates its auditable contract."""


def default_paths() -> tuple[Path, Path, Path]:
    repository = Path(__file__).resolve().parents[1]
    return (
        repository / "datasets/SmartHealth_raw",
        repository / "datasets/SmartHealth_features",
        repository / "splits/smarthealth",
    )


def finite(row: dict[str, str], name: str, path: Path) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"{path}: invalid {name}={row.get(name)!r}") from exc
    if not math.isfinite(value):
        raise ValidationError(f"{path}: non-finite {name}={row.get(name)!r}")
    return value


def integer(row: dict[str, str], name: str, path: Path) -> int:
    value = finite(row, name, path)
    if not value.is_integer() or value <= 0:
        raise ValidationError(f"{path}: invalid positive integer {name}={value}")
    return int(value)


def source_time(row: dict[str, str], name: str, path: Path) -> datetime:
    text = str(row.get(name, "")).strip().replace("/", "-").replace("T", " ")
    if not text:
        raise ValidationError(f"{path}: missing {name}")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
        for layout in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ):
            try:
                parsed = datetime.strptime(text, layout)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValidationError(f"{path}: invalid {name}={row.get(name)!r}")
    return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


def required_fields(path: Path, actual: Iterable[str] | None, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(actual or ()))
    if missing:
        raise ValidationError(f"{path}: missing required columns {missing}")


def load_manifest(config: DomainConfig, raw_root: Path) -> tuple[dict, Path]:
    raw_directory = raw_root / config.domain_id
    path = raw_directory / "SMARTHEALTH_CANONICAL_MANIFEST.json"
    if not path.is_file():
        raise ValidationError(f"Missing RAW manifest for {config.domain_id}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("domain_id") != config.domain_id:
        raise ValidationError(f"{path}: domain mismatch")
    if manifest.get("strategy_version") != POLICY_VERSION:
        raise ValidationError(f"{path}: strategy version mismatch")
    if manifest.get("split_strategy_version") != SPLIT_STRATEGY_VERSION:
        raise ValidationError(f"{path}: split strategy version mismatch")
    required_fields(path, manifest.get("raw_schema"), RAW_COLUMNS)
    return manifest, raw_directory


def load_exported_provenance(
    config: DomainConfig, raw_root: Path
) -> dict[tuple[str, int], dict[str, str]]:
    paths = raw_audit_paths(raw_root / "audit", config)
    path = paths["cycle_provenance"]
    if not path.is_file():
        raise ValidationError(f"Missing cycle provenance: {path}")
    expected: dict[tuple[str, int], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields(
            path,
            reader.fieldnames,
            {
                "domain_id",
                "logical_sequence_id",
                "cycle",
                "source_cycle",
                "source_absolute_start_time",
                "source_absolute_end_time",
                "selected_candidate",
                "output_status",
                "raw_rows_written",
                "SOH",
                "label_source",
                "temperature_complete",
                "cc_window_complete",
                "cv_window_complete",
                "split_role",
                "split_status",
                "split_strategy_version",
                "strategy_version",
            },
        )
        selected_timeline: dict[str, list[tuple[int, datetime, datetime]]] = {}
        for row in reader:
            if row["domain_id"] != config.domain_id:
                raise ValidationError(f"{path}: row from another domain")
            if row["strategy_version"] != POLICY_VERSION:
                raise ValidationError(f"{path}: strategy version mismatch")
            if row["split_strategy_version"] != SPLIT_STRATEGY_VERSION:
                raise ValidationError(f"{path}: split strategy version mismatch")
            if str(row["selected_candidate"]).lower() not in {"true", "1"}:
                continue
            logical_sequence_id = str(row["logical_sequence_id"])
            cycle = integer(row, "cycle", path)
            integer(row, "source_cycle", path)
            start = source_time(row, "source_absolute_start_time", path)
            end = source_time(row, "source_absolute_end_time", path)
            if end < start:
                raise ValidationError(f"{path}: reversed source time interval for {logical_sequence_id}/{cycle}")
            selected_timeline.setdefault(logical_sequence_id, []).append((cycle, start, end))
            if row["output_status"] != "exported":
                continue
            key = (logical_sequence_id, cycle)
            if key in expected:
                raise ValidationError(f"{path}: duplicate selected/exported provenance {key}")
            for boolean in ("temperature_complete", "cc_window_complete", "cv_window_complete"):
                if str(row[boolean]).lower() not in {"true", "1"}:
                    raise ValidationError(f"{path}: exported cycle lacks {boolean}: {key}")
            expected[key] = row
    for logical_sequence_id, events in selected_timeline.items():
        ordered = sorted(events)
        cycle_ids = [cycle for cycle, _, _ in ordered]
        if cycle_ids != list(range(1, len(cycle_ids) + 1)):
            raise ValidationError(
                f"{path}: selected canonical cycle IDs are not one-based chronology for "
                f"{logical_sequence_id}"
            )
        intervals = [(start, end) for _, start, end in ordered]
        if any(current < previous for previous, current in zip(intervals, intervals[1:])):
            raise ValidationError(
                f"{path}: canonical source-time chronology regresses for {logical_sequence_id}"
            )
    if not expected:
        raise ValidationError(f"{path}: no exported canonical cycles")
    return expected


def validate_raw_domain(
    config: DomainConfig, raw_root: Path, expected: dict[tuple[str, int], dict[str, str]]
) -> dict[tuple[str, int], dict[str, object]]:
    raw_directory = raw_root / config.domain_id
    files = sorted(raw_directory.glob(f"{config.domain_id}__*.csv"))
    if not files:
        raise ValidationError(f"No RAW CSVs under {raw_directory}")
    observed: dict[tuple[str, int], dict[str, object]] = {}
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_fields(path, reader.fieldnames, RAW_COLUMNS)
            active_key: tuple[str, int] | None = None
            active: list[dict[str, str]] = []
            seen_local: set[tuple[str, int]] = set()

            def finalize() -> None:
                nonlocal active_key, active
                if active_key is None:
                    return
                if active_key in seen_local or active_key in observed:
                    raise ValidationError(f"{path}: repeated canonical raw cycle {active_key}")
                if active_key not in expected:
                    raise ValidationError(f"{path}: RAW cycle absent from exported provenance {active_key}")
                segments = [str(item["segment"]).strip().upper() for item in active]
                if not segments or segments[0] != "CC" or "CV" not in segments:
                    raise ValidationError(f"{path}: missing CC/CV pair for {active_key}")
                boundary = segments.index("CV")
                if any(item != "CC" for item in segments[:boundary]) or any(
                    item != "CV" for item in segments[boundary:]
                ):
                    raise ValidationError(f"{path}: non-contiguous CC/CV segments for {active_key}")
                label_source = active[0]["label_source"]
                if label_source not in {"calibration_direct", "calibration_interpolated"}:
                    raise ValidationError(f"{path}: invalid label source for {active_key}: {label_source!r}")
                soh = finite(active[0], "SOH", path)
                lineage = (
                    integer(active[0], "source_cycle", path),
                    source_time(active[0], "source_absolute_start_time", path),
                    source_time(active[0], "source_absolute_end_time", path),
                )
                if lineage[2] < lineage[1]:
                    raise ValidationError(f"{path}: reversed source time interval for {active_key}")
                for row in active:
                    if row["domain_id"] != config.domain_id or row["strategy_version"] != POLICY_VERSION:
                        raise ValidationError(f"{path}: metadata policy/domain mismatch for {active_key}")
                    if row["split_strategy_version"] != SPLIT_STRATEGY_VERSION:
                        raise ValidationError(f"{path}: split strategy mismatch for {active_key}")
                    if row["split_status"] not in {
                        "complete",
                        "manual_confirmation_required",
                    }:
                        raise ValidationError(f"{path}: invalid split status for {active_key}")
                    if row["label_source"] != label_source or not math.isclose(
                        finite(row, "SOH", path), soh, rel_tol=1e-7, abs_tol=1e-8
                    ):
                        raise ValidationError(f"{path}: label varies within {active_key}")
                    row_lineage = (
                        integer(row, "source_cycle", path),
                        source_time(row, "source_absolute_start_time", path),
                        source_time(row, "source_absolute_end_time", path),
                    )
                    if row_lineage != lineage:
                        raise ValidationError(f"{path}: source lineage varies within {active_key}")
                    segment = str(row["segment"]).strip().upper()
                    voltage = finite(row, "voltage_V", path)
                    c_rate = finite(row, "c_rate", path)
                    finite(row, "temperature_C", path)
                    finite(row, "relative_time", path)
                    if segment == "CC" and not (
                        config.cc_voltage_low_v - 1e-9
                        <= voltage
                        <= config.cc_voltage_high_v + 1e-9
                    ):
                        raise ValidationError(f"{path}: CC point outside v3 voltage window for {active_key}")
                    if segment == "CV" and not (
                        config.cv_c_rate_low - config.cv_selection_tolerance_c - 1e-9
                        <= c_rate
                        <= config.cv_c_rate_high + config.cv_selection_tolerance_c + 1e-9
                    ):
                        raise ValidationError(f"{path}: CV point outside v3 C-rate window for {active_key}")
                provenance = expected[active_key]
                if not math.isclose(
                    float(provenance["SOH"]), soh, rel_tol=1e-7, abs_tol=1e-8
                ) or provenance["label_source"] != label_source:
                    raise ValidationError(f"{path}: RAW/provenance label mismatch for {active_key}")
                if integer(provenance, "raw_rows_written", path) != len(active):
                    raise ValidationError(f"{path}: RAW/provenance row-count mismatch for {active_key}")
                provenance_lineage = (
                    integer(provenance, "source_cycle", path),
                    source_time(provenance, "source_absolute_start_time", path),
                    source_time(provenance, "source_absolute_end_time", path),
                )
                if lineage != provenance_lineage:
                    raise ValidationError(f"{path}: RAW/provenance source lineage mismatch for {active_key}")
                observed[active_key] = {
                    "rows": len(active),
                    "soh": soh,
                    "label_source": label_source,
                    "cell": active[0]["cell"],
                    "condition": active[0]["condition"],
                    "split_role": active[0]["split_role"],
                    "split_status": active[0]["split_status"],
                    "split_strategy_version": active[0]["split_strategy_version"],
                    "source_cycle": lineage[0],
                    "source_absolute_start_time": lineage[1],
                    "source_absolute_end_time": lineage[2],
                }
                seen_local.add(active_key)
                active_key = None
                active = []

            for row in reader:
                key = (str(row["logical_sequence_id"]), integer(row, "cycle", path))
                if active_key is None:
                    active_key = key
                elif key != active_key:
                    finalize()
                    active_key = key
                active.append(row)
            finalize()
    if set(observed) != set(expected):
        raise ValidationError(
            f"{config.domain_id}: RAW/provenance mismatch; "
            f"missing={sorted(set(expected) - set(observed))[:8]}, "
            f"unexpected={sorted(set(observed) - set(expected))[:8]}"
        )
    return observed


def validate_feature_domain(
    config: DomainConfig,
    feature_root: Path,
    raw_cycles: dict[tuple[str, int], dict[str, object]],
) -> int:
    directory = feature_root / config.domain_id
    pointer = directory / "SMARTHEALTH_FEATURE_PROVENANCE_POINTER.json"
    if not pointer.is_file():
        raise ValidationError(f"Missing feature pointer: {pointer}")
    with pointer.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("domain_id") != config.domain_id or payload.get("strategy_version") != POLICY_VERSION:
        raise ValidationError(f"{pointer}: incompatible feature pointer")
    if payload.get("split_strategy_version") != SPLIT_STRATEGY_VERSION:
        raise ValidationError(f"{pointer}: incompatible feature split strategy")
    files = sorted(directory.glob(f"{config.domain_id}__*.csv"))
    if not files:
        raise ValidationError(f"No feature CSVs under {directory}")
    found: set[tuple[str, int]] = set()
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_fields(path, reader.fieldnames, [*FEATURE_PREFIX_COLUMNS, *FEATURE_COLUMNS])
            for row in reader:
                key = (str(row["logical_sequence_id"]), integer(row, "cycle", path))
                if key in found:
                    raise ValidationError(f"{path}: duplicate feature cycle {key}")
                raw = raw_cycles.get(key)
                if raw is None:
                    raise ValidationError(f"{path}: feature cycle absent from RAW {key}")
                if row["label_source"] != raw["label_source"] or not math.isclose(
                    finite(row, "SOH", path), float(raw["soh"]), rel_tol=1e-7, abs_tol=1e-8
                ):
                    raise ValidationError(f"{path}: feature/RAW label mismatch for {key}")
                if (
                    row["split_role"] != raw["split_role"]
                    or row["split_status"] != raw["split_status"]
                    or row["split_strategy_version"] != SPLIT_STRATEGY_VERSION
                ):
                    raise ValidationError(f"{path}: feature/RAW split mismatch for {key}")
                feature_lineage = (
                    integer(row, "source_cycle", path),
                    source_time(row, "source_absolute_start_time", path),
                    source_time(row, "source_absolute_end_time", path),
                )
                raw_lineage = (
                    int(raw["source_cycle"]),
                    raw["source_absolute_start_time"],
                    raw["source_absolute_end_time"],
                )
                if feature_lineage != raw_lineage:
                    raise ValidationError(f"{path}: feature/RAW source lineage mismatch for {key}")
                for name in FEATURE_COLUMNS:
                    finite(row, name, path)
                found.add(key)
    if found != set(raw_cycles):
        raise ValidationError(
            f"{config.domain_id}: feature/RAW mismatch; missing={sorted(set(raw_cycles) - found)[:8]}"
        )
    return len(found)


def validate_split_domain(
    config: DomainConfig,
    split_root: Path,
    cells: dict[tuple[str, int], dict[str, object]],
) -> dict[str, object]:
    path = split_root / f"{config.domain_id}_cell_split.json"
    if not path.is_file():
        raise ValidationError(f"Missing split JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    if split.get("domain_id") != config.domain_id or split.get("strategy_version") != POLICY_VERSION:
        raise ValidationError(f"{path}: incompatible split policy")
    if split.get("split_strategy_version") != SPLIT_STRATEGY_VERSION:
        raise ValidationError(f"{path}: incompatible split strategy version")
    protocol = split.get("development_split", {})
    if (
        protocol.get("mode") != "mixed_cycle"
        or protocol.get("scope")
        != "all development logical sequences pooled within the battery family"
        or protocol.get("random_state") != 420
        or float(protocol.get("train_ratio", -1)) != 0.8
        or float(protocol.get("val_ratio", -1)) != 0.2
    ):
        raise ValidationError(f"{path}: incompatible development mixed-cycle protocol")
    by_condition: dict[str, set[str]] = {}
    for (_, _), row in cells.items():
        by_condition.setdefault(str(row["condition"]), set()).add(str(row["cell"]))
    dev = split.get("development_batteries_by_condition", {})
    test = split.get("test_batteries_by_condition", {})
    conditions = split.get("conditions")
    if not isinstance(conditions, dict):
        raise ValidationError(f"{path}: v3 split conditions must be an object")
    manual = split.get("manual_confirmation_conditions", {})
    if not isinstance(manual, dict):
        raise ValidationError(f"{path}: manual-confirmation conditions must be an object")
    manual_seen: dict[str, str] = {}
    for condition, specification in sorted(conditions.items()):
        if not isinstance(specification, dict):
            raise ValidationError(f"{path}: invalid condition record for {condition}")
        logical_sequences = specification.get("logical_sequences")
        if not isinstance(logical_sequences, list):
            raise ValidationError(f"{path}: missing logical-sequence inventory for {condition}")
        listed_cells = {
            str(item.get("logical_sequence_id", ""))
            for item in logical_sequences
            if isinstance(item, dict)
        }
        if len(listed_cells) != len(logical_sequences) or "" in listed_cells:
            raise ValidationError(f"{path}: duplicate/invalid logical sequence in {condition}")
        status = specification.get("status")
        development_cells = set(specification.get("development_cells", []))
        test_cell = specification.get("test_cell")
        if status == "complete":
            if (
                len(listed_cells) != 3
                or int(specification.get("eligible_logical_sequences", -1)) != 3
                or len(development_cells) != 2
                or not isinstance(test_cell, str)
            ):
                raise ValidationError(f"{path}: invalid 2-development/1-test allocation for {condition}")
            test_cells = {test_cell}
            if development_cells | test_cells != listed_cells or development_cells & test_cells:
                raise ValidationError(f"{path}: split inventory mismatch for {condition}")
            if set(dev.get(condition, [])) != development_cells or set(test.get(condition, [])) != test_cells:
                raise ValidationError(f"{path}: split maps mismatch condition inventory for {condition}")
            observed_cells = by_condition.get(str(condition), set())
            if observed_cells != listed_cells:
                raise ValidationError(f"{path}: split cells do not exactly match RAW cells for {condition}")
            for row in cells.values():
                if str(row["condition"]) != str(condition):
                    continue
                expected_role = "development" if str(row["cell"]) in development_cells else "test"
                if row["split_status"] != "complete" or row["split_role"] != expected_role:
                    raise ValidationError(f"{path}: RAW split role mismatch for {condition}")
        elif status == "manual_confirmation_required":
            issue = str(specification.get("manual_confirmation_issue") or "")
            if (
                development_cells
                or test_cell is not None
                or condition in dev
                or condition in test
                or not issue
            ):
                raise ValidationError(f"{path}: invalid manual-confirmation record for {condition}")
            manual_seen[str(condition)] = issue
            for row in cells.values():
                if str(row["condition"]) != str(condition):
                    continue
                if (
                    row["split_status"] != "manual_confirmation_required"
                    or row["split_role"] != "unassigned_manual_confirmation"
                ):
                    raise ValidationError(f"{path}: RAW manual-confirmation role mismatch for {condition}")
        else:
            raise ValidationError(f"{path}: unknown split status for {condition}: {status!r}")
    if set(by_condition) - set(conditions):
        raise ValidationError(f"{path}: RAW contains conditions absent from split JSON")
    if manual_seen != {str(key): str(value) for key, value in manual.items()}:
        raise ValidationError(f"{path}: manual-confirmation summary mismatch")
    expected_status = "manual_confirmation_required" if manual_seen else "complete"
    if split.get("split_status") != expected_status:
        raise ValidationError(f"{path}: global split status mismatch")
    return {
        "split_status": expected_status,
        "manual_confirmation_conditions": manual_seen,
    }


def validate_report_domain(
    config: DomainConfig,
    raw_root: Path,
    raw_cycles: dict[tuple[str, int], dict[str, object]],
    split_summary: dict[str, object],
) -> None:
    path = raw_audit_paths(raw_root / "audit", config)["report"]
    if not path.is_file():
        raise ValidationError(f"Missing preprocessing report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("strategy_version") != POLICY_VERSION or report.get("domain_id") != config.domain_id:
        raise ValidationError(f"{path}: report policy/domain mismatch")
    if int(report.get("final_eligible_cycles", -1)) != len(raw_cycles):
        raise ValidationError(f"{path}: final eligible cycle count mismatch")
    if report.get("split_strategy_version") != SPLIT_STRATEGY_VERSION:
        raise ValidationError(f"{path}: split strategy version mismatch")
    if report.get("split_status") != split_summary["split_status"]:
        raise ValidationError(f"{path}: split status mismatch")


def parse_args() -> argparse.Namespace:
    raw_root, feature_root, split_root = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=raw_root)
    parser.add_argument("--feature-root", type=Path, default=feature_root)
    parser.add_argument("--split-root", type=Path, default=split_root)
    parser.add_argument("--domain", action="append", choices=sorted(CONFIGS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domains = args.domain or [
        domain for domain in sorted(CONFIGS) if (args.raw_root / domain).is_dir()
    ]
    if not domains:
        raise ValidationError("No generated v3 SmartHealth domain directory was found")
    summary: dict[str, object] = {"strategy_version": POLICY_VERSION, "domains": {}}
    for domain in domains:
        config = CONFIGS[domain]
        load_manifest(config, args.raw_root)
        expected = load_exported_provenance(config, args.raw_root)
        raw_cycles = validate_raw_domain(config, args.raw_root, expected)
        features = validate_feature_domain(config, args.feature_root, raw_cycles)
        split_summary = validate_split_domain(config, args.split_root, raw_cycles)
        validate_report_domain(config, args.raw_root, raw_cycles, split_summary)
        summary["domains"][domain] = {
            "raw_cycles": len(raw_cycles),
            "raw_rows": sum(int(row["rows"]) for row in raw_cycles.values()),
            "feature_rows": features,
            "label_sources": dict(Counter(row["label_source"] for row in raw_cycles.values())),
            **split_summary,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"SmartHealth v3 validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
