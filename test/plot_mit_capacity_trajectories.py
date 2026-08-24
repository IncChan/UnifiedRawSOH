#!/usr/bin/env python3
"""Plot MIT physical-cell capacity trajectories against global cycle.

This diagnostic reads the canonical ``datasets/MIT_raw`` physical-cell CSVs,
not the Only-F feature table.  It keeps one capacity_Ah label per global
physical cycle and uses ``mit_p###`` as the cell identity.  The MIT physical
split marks physical IDs whose numeric suffix is divisible by five as test;
the remaining cells are development because train/validation are mixed-cycle
records pooled across those development cells.

The invalid-cycle entries declared by the canonical MIT split are excluded
before plotting, so the known mit_p015/cycle-39 source capacity spike cannot
stretch the axes or appear as a fake degradation event.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capacity_trajectory_plotting import (  # noqa: E402
    CellTrajectory,
    CapacityPoint,
    finite,
    group_by_condition,
    integer_cycle,
    plot_trajectories,
    trajectories_summary,
    write_summary,
)


DEFAULT_RAW_ROOT = REPO_ROOT / "datasets" / "MIT_raw"
DEFAULT_SPLIT_FILE = REPO_ROOT / "splits" / "mit" / "mit_paper_physical124_v2_split.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "mit_capacity_trajectories"
NOMINAL_CAPACITY_AH = 1.1
PHYSICAL_FILE_RE = re.compile(r"MIT_(?P<date>\d{4}-\d{2}-\d{2})_physical-(?P<id>\d+)\.csv$")
REQUIRED_COLUMNS = {"physical_cell_id", "primary_batch_date", "cycle", "SOH", "capacity_Ah"}


def parse_file_identity(path: Path) -> tuple[str, str]:
    match = PHYSICAL_FILE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Cannot parse canonical MIT physical filename: {path.name}")
    return match.group("date"), f"mit_p{int(match.group('id')):03d}"


def list_data_files(raw_root: Path) -> list[Path]:
    files = []
    for path in sorted(raw_root.glob("MIT_*_physical-*.csv")):
        parse_file_identity(path)
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No canonical MIT physical CSV files found under {raw_root}")
    return files


def load_split(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        split = json.load(handle)
    if not isinstance(split, dict):
        raise ValueError(f"MIT split file must contain a JSON object: {path}")
    if split.get("dataset_id") != "mit":
        raise ValueError(f"Unexpected dataset_id in MIT split {path}: {split.get('dataset_id')!r}")
    return split


def invalid_cycles_from_split(split: dict[str, object]) -> set[tuple[str, int]]:
    invalid = set()
    for item in split.get("invalid_cycles", []):
        if not isinstance(item, dict) or "battery_id" not in item or "cycle_id" not in item:
            raise ValueError(f"Invalid MIT split invalid_cycles entry: {item!r}")
        invalid.add((str(item["battery_id"]), int(item["cycle_id"])))
    return invalid


def test_ids_from_split(split: dict[str, object], observed_ids: set[str]) -> set[str]:
    rule = split.get("test_rule") or {}
    if rule.get("type") == "physical_id_modulo":
        modulus = int(rule.get("modulus", 0))
        remainder = int(rule.get("remainder", 0))
        if modulus <= 0:
            raise ValueError(f"MIT split has invalid physical_id modulo rule: {rule!r}")
        pattern = re.compile(r"mit_p(\d+)")
        selected = {
            cell_id
            for cell_id in observed_ids
            if (match := pattern.fullmatch(cell_id)) is not None
            and int(match.group(1)) % modulus == remainder
        }
    else:
        selected = {str(item) for item in split.get("test_batteries", [])} & observed_ids
    if not selected:
        raise ValueError(
            "MIT split selected no observed test physical cells; "
            f"observed={sorted(observed_ids)[:5]}"
        )
    return selected


def load_trajectory(
    path: Path,
    *,
    split_role: str,
    invalid_cycles: set[tuple[str, int]],
) -> tuple[CellTrajectory, list[tuple[str, int]]]:
    filename_condition, filename_cell_id = parse_file_identity(path)
    cycle_capacity: dict[int, float] = {}
    cycle_soh: dict[int, float] = {}
    seen_invalid: set[tuple[str, int]] = set()
    file_physical_ids: set[str] = set()
    file_primary_dates: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        for line_number, row in enumerate(reader, start=2):
            cell_id = str(row.get("physical_cell_id") or "").strip()
            primary_date = str(row.get("primary_batch_date") or "").strip()
            if not cell_id or not primary_date:
                raise ValueError(f"{path}:{line_number}: missing physical identity metadata")
            file_physical_ids.add(cell_id)
            file_primary_dates.add(primary_date)
            cycle = integer_cycle(row.get("cycle"), path=path, line_number=line_number)
            if (filename_cell_id, cycle) in invalid_cycles:
                seen_invalid.add((filename_cell_id, cycle))
                continue
            capacity = finite(
                row.get("capacity_Ah"),
                column="capacity_Ah",
                path=path,
                line_number=line_number,
            )
            soh = finite(row.get("SOH"), column="SOH", path=path, line_number=line_number)
            previous_capacity = cycle_capacity.get(cycle)
            if previous_capacity is not None and not abs(previous_capacity - capacity) <= max(
                1e-7, 1e-5 * max(abs(previous_capacity), abs(capacity), 1.0)
            ):
                raise ValueError(
                    f"{path}:{line_number}: capacity label changes within cycle {cycle}: "
                    f"{previous_capacity} vs {capacity}"
                )
            previous_soh = cycle_soh.get(cycle)
            if previous_soh is not None and not abs(previous_soh - soh) <= max(
                1e-7, 1e-5 * max(abs(previous_soh), abs(soh), 1.0)
            ):
                raise ValueError(
                    f"{path}:{line_number}: SOH label changes within cycle {cycle}: "
                    f"{previous_soh} vs {soh}"
                )
            cycle_capacity[cycle] = capacity
            cycle_soh[cycle] = soh

    if file_physical_ids != {filename_cell_id}:
        raise ValueError(
            f"{path}: filename/physical_cell_id mismatch: "
            f"filename={filename_cell_id}, columns={sorted(file_physical_ids)}"
        )
    if file_primary_dates != {filename_condition}:
        raise ValueError(
            f"{path}: filename/primary_batch_date mismatch: "
            f"filename={filename_condition}, columns={sorted(file_primary_dates)}"
        )
    if not cycle_capacity:
        raise ValueError(f"{path}: no valid capacity-labelled cycles found")

    return (
        CellTrajectory(
            condition=filename_condition,
            cell_id=filename_cell_id,
            split_role=split_role,
            source_file=str(path),
            points=[CapacityPoint(cycle, capacity) for cycle, capacity in cycle_capacity.items()],
            metadata={
                "label_field": "capacity_Ah",
                "primary_batch_date": filename_condition,
            },
        ),
        sorted(seen_invalid),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Canonical MIT_raw root containing physical-cell CSVs.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
        help="Canonical physical124 split JSON used for test cells and invalid cycles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for the MIT figure and summary.json.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Optional subset of primary batch dates, e.g. 2017-05-12 2018-04-12.",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow fewer cells than split.physical_cell_count (useful for local smoke data).",
    )
    parser.add_argument(
        "--show-cell-legend",
        action="store_true",
        help="Add one legend entry per MIT physical cell; omitted by default for readability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    split_file = args.split_file.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Canonical MIT raw root does not exist: {raw_root}")
    if not split_file.is_file():
        raise FileNotFoundError(f"MIT split file does not exist: {split_file}")

    split = load_split(split_file)
    files = list_data_files(raw_root)
    observed_ids = {parse_file_identity(path)[1] for path in files}
    declared_count = split.get("physical_cell_count")
    if not args.allow_subset and declared_count is not None and len(observed_ids) != int(declared_count):
        raise ValueError(
            f"MIT split declares {int(declared_count)} physical cells, but raw root has "
            f"{len(observed_ids)} files; pass --allow-subset only for a deliberate smoke run"
        )
    expected_ids = {f"mit_p{index:03d}" for index in range(1, int(declared_count) + 1)} if declared_count else set()
    if not args.allow_subset and expected_ids and observed_ids != expected_ids:
        raise ValueError(
            "MIT physical-cell inventory does not match split cohort: "
            f"missing={sorted(expected_ids - observed_ids)[:5]}, "
            f"unexpected={sorted(observed_ids - expected_ids)[:5]}"
        )

    test_ids = test_ids_from_split(split, observed_ids)
    invalid_cycles = invalid_cycles_from_split(split)
    selected_conditions = set(args.conditions) if args.conditions else None
    observed_conditions = {parse_file_identity(path)[0] for path in files}
    if selected_conditions is not None:
        unknown = sorted(selected_conditions - observed_conditions)
        if unknown:
            raise ValueError(f"Requested MIT primary batch dates are not present: {unknown}")

    trajectories: list[CellTrajectory] = []
    excluded_invalid: set[tuple[str, int]] = set()
    for path in files:
        condition, cell_id = parse_file_identity(path)
        if selected_conditions is not None and condition not in selected_conditions:
            continue
        role = "test" if cell_id in test_ids else "development"
        trajectory, excluded = load_trajectory(
            path,
            split_role=role,
            invalid_cycles=invalid_cycles,
        )
        trajectories.append(trajectory)
        excluded_invalid.update(excluded)

    grouped = group_by_condition(trajectories)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mit_capacity_vs_cycle.png"
    plot_trajectories(
        "mit",
        grouped,
        output_path,
        nominal_capacity_ah=NOMINAL_CAPACITY_AH,
        x_label="Canonical global physical cycle",
        title_suffix=(
            "Every point is the exported capacity_Ah label; continuation files are "
            "already merged into global physical-cycle order."
        ),
        show_cell_legend=args.show_cell_legend,
    )
    summary = {
        "dataset_id": "mit",
        "raw_root": str(raw_root),
        "split_file": str(split_file),
        "nominal_capacity_Ah": NOMINAL_CAPACITY_AH,
        "label_field": "capacity_Ah",
        "cell_role_policy": "test = canonical physical_id modulo 5; other cells = development",
        "train_validation_note": "MIT development train/validation is mixed-cycle; this figure uses cell-level development/test styles",
        "test_physical_cells": sorted(test_ids),
        "invalid_cycles_declared": [list(item) for item in sorted(invalid_cycles)],
        "invalid_cycles_excluded_from_loaded_points": [list(item) for item in sorted(excluded_invalid)],
        "observed_physical_cell_count": len(observed_ids),
        **trajectories_summary(grouped),
    }
    summary_path = output_dir / "summary.json"
    write_summary(summary_path, summary)
    print(f"[mit] wrote {output_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
