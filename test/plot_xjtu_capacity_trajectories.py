#!/usr/bin/env python3
"""Plot XJTU canonical capacity labels against source cycle.

The canonical XJTU raw product is point-level.  Its ``SOH`` column is the
capacity-like degradation trajectory emitted by the XJTU extractor (Ah), so
this diagnostic reduces each file to one validated capacity label per
``cycle``.  It intentionally does not use the feature table: the feature
table has no cycle identifier and may have different valid-cycle coverage.

The paper split is battery-4/8 as the independent test batteries for every
condition.  Because development train/validation is a mixed-cycle protocol,
the figure labels all non-test batteries as ``development`` rather than
pretending that a whole battery belongs exclusively to train or validation.
"""

from __future__ import annotations

import argparse
import csv
import json
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


DEFAULT_RAW_ROOT = REPO_ROOT / "datasets" / "XJTU_raw"
DEFAULT_SPLIT_FILE = REPO_ROOT / "splits" / "xjtu" / "paper_v1_mixed_split.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "xjtu_capacity_trajectories"
NOMINAL_CAPACITY_AH = 2.0
REQUIRED_COLUMNS = {"cycle", "SOH"}
CONDITION_ORDER = {"2C": 0, "3C": 1, "R2.5": 2, "R3": 3, "RW": 4, "satellite": 5}


def parse_file_identity(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "_battery-" not in stem:
        raise ValueError(f"Cannot parse XJTU battery identity from {path.name}")
    condition, suffix = stem.split("_battery-", maxsplit=1)
    if condition == "Sim_satellite":
        condition = "satellite"
    if not suffix:
        raise ValueError(f"Cannot parse XJTU battery suffix from {path.name}")
    return condition, f"{condition}_battery-{suffix}"


def condition_sort_key(value: str) -> tuple[int, str]:
    return (CONDITION_ORDER.get(value, len(CONDITION_ORDER)), value)


def list_data_files(raw_root: Path) -> list[Path]:
    files = []
    for path in sorted(raw_root.glob("*.csv")):
        if path.name.endswith("_report.csv"):
            continue
        parse_file_identity(path)
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No XJTU battery CSV files found under {raw_root}")
    return files


def load_split(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        split = json.load(handle)
    if not isinstance(split, dict):
        raise ValueError(f"XJTU split file must contain a JSON object: {path}")
    if split.get("dataset_id") != "xjtu":
        raise ValueError(f"Unexpected dataset_id in XJTU split {path}: {split.get('dataset_id')!r}")
    return split


def test_ids_for_condition(
    split: dict[str, object], condition: str, observed: set[str]
) -> set[str]:
    by_condition = split.get("test_batteries_by_condition")
    if isinstance(by_condition, dict):
        declared = {str(item) for item in by_condition.get(condition, [])}
    else:
        declared = {str(item) for item in split.get("test_batteries", [])}
    selected = declared & observed
    if not selected:
        raise ValueError(
            f"XJTU split selected no observed test batteries for condition {condition!r}; "
            f"observed={sorted(observed)}"
        )
    return selected


def load_trajectory(path: Path, *, split_role: str) -> CellTrajectory:
    condition, cell_id = parse_file_identity(path)
    cycle_capacity: dict[int, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        for line_number, row in enumerate(reader, start=2):
            cycle = integer_cycle(row.get("cycle"), path=path, line_number=line_number)
            # XJTU's canonical raw schema keeps this historical column name,
            # but the extractor writes the capacity trajectory in Ah here.
            capacity = finite(row.get("SOH"), column="SOH/capacity_Ah", path=path, line_number=line_number)
            previous = cycle_capacity.get(cycle)
            if previous is not None and not abs(previous - capacity) <= max(
                1e-6, 1e-5 * max(abs(previous), abs(capacity), 1.0)
            ):
                raise ValueError(
                    f"{path}:{line_number}: capacity label changes within cycle {cycle}: "
                    f"{previous} vs {capacity}"
                )
            cycle_capacity[cycle] = capacity
    if not cycle_capacity:
        raise ValueError(f"{path}: no capacity-labelled cycles found")
    return CellTrajectory(
        condition=condition,
        cell_id=cell_id,
        split_role=split_role,
        source_file=str(path),
        points=[CapacityPoint(cycle, capacity) for cycle, capacity in cycle_capacity.items()],
        metadata={"label_field": "SOH (capacity_Ah-like canonical XJTU label)"},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Canonical XJTU_raw root containing point-level battery CSVs.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
        help="JSON split whose battery-4/8 test assignment is applied.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for the XJTU figure and summary.json.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Optional subset of condition labels, e.g. 2C 3C satellite.",
    )
    parser.add_argument(
        "--hide-cell-legend",
        action="store_true",
        help="Use only development/test/nominal legend entries instead of one entry per cell.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    split_file = args.split_file.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Canonical XJTU raw root does not exist: {raw_root}")
    if not split_file.is_file():
        raise FileNotFoundError(f"XJTU split file does not exist: {split_file}")

    split = load_split(split_file)
    files = list_data_files(raw_root)
    identities = [parse_file_identity(path) for path in files]
    observed_by_condition: dict[str, set[str]] = {}
    for condition, cell_id in identities:
        observed_by_condition.setdefault(condition, set()).add(cell_id)

    selected_conditions = set(args.conditions) if args.conditions else set(observed_by_condition)
    unknown = sorted(selected_conditions - set(observed_by_condition))
    if unknown:
        raise ValueError(f"Requested XJTU conditions are not present: {unknown}")

    trajectories: list[CellTrajectory] = []
    test_ids_by_condition: dict[str, list[str]] = {}
    for condition in sorted(observed_by_condition, key=condition_sort_key):
        if condition not in selected_conditions:
            continue
        test_ids = test_ids_for_condition(split, condition, observed_by_condition[condition])
        test_ids_by_condition[condition] = sorted(test_ids)
        for path in files:
            path_condition, cell_id = parse_file_identity(path)
            if path_condition != condition:
                continue
            role = "test" if cell_id in test_ids else "development"
            trajectories.append(load_trajectory(path, split_role=role))

    grouped = group_by_condition(trajectories)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "xjtu_capacity_vs_cycle.png"
    plot_trajectories(
        "xjtu",
        grouped,
        output_path,
        nominal_capacity_ah=NOMINAL_CAPACITY_AH,
        x_label="Canonical source cycle (XJTU raw cycle)",
        title_suffix=(
            "Every point is the exported XJTU capacity-like SOH label; "
            "cycle gaps are preserved."
        ),
        show_cell_legend=not args.hide_cell_legend,
    )
    summary = {
        "dataset_id": "xjtu",
        "raw_root": str(raw_root),
        "split_file": str(split_file),
        "nominal_capacity_Ah": NOMINAL_CAPACITY_AH,
        "label_field": "SOH",
        "label_semantics": "canonical XJTU raw SOH column is capacity in Ah",
        "cell_role_policy": "test batteries come from split JSON; all other batteries are development",
        "test_batteries_by_condition": test_ids_by_condition,
        **trajectories_summary(grouped),
    }
    summary_path = output_dir / "summary.json"
    write_summary(summary_path, summary)
    print(f"[xjtu] wrote {output_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
