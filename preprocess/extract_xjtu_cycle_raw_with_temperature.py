import argparse
import csv
import math
import os
import sys
import types
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

Battery = None


RAW_COLUMNS = [
    "cycle",
    "SOH",
    "segment",
    "cycle_point_index",
    "segment_point_index",
    "relative_time_min",
    "voltage_V",
    "current_A",
    "power_Wh",
    "temperature_C",
]

REPORT_COLUMNS = [
    "batch",
    "file",
    "cycle_life",
    "start_cycle",
    "written_cycles",
    "written_rows",
    "skipped_test_capacity",
    "skipped_nonfinite_rows",
    "empty_segments",
    "failed_cycles",
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "XJTU_raw"


def default_input_root() -> Path | None:
    """Use an explicit source archive; never bake one developer path into Git."""

    value = os.environ.get("XJTU_SOURCE_ROOT")
    return Path(value).expanduser() if value else None


def load_runtime_dependencies() -> None:
    global Battery
    try:
        import matplotlib.pyplot  # noqa: F401
    except ModuleNotFoundError:
        # XJTUBatteryClass imports pyplot, but raw extraction does not use it.
        matplotlib_stub = types.ModuleType("matplotlib")
        pyplot_stub = types.ModuleType("matplotlib.pyplot")
        matplotlib_stub.pyplot = pyplot_stub
        sys.modules.setdefault("matplotlib", matplotlib_stub)
        sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)

    from XJTUBatteryClass import Battery as XJTUBattery

    Battery = XJTUBattery


def parse_float_pair(text: str) -> List[float]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected two comma-separated floats, got: {text}")
    return [float(parts[0]), float(parts[1])]


def is_test_capacity(description: str) -> bool:
    return "test capacity" in str(description).lower()


def batch_name(mat_path: Path) -> str:
    parent = mat_path.parent.name
    if parent.startswith("Batch-"):
        return parent
    return "unknown"


def find_all_mat_files(input_root: Path) -> List[Path]:
    ignored = {"Temperature_Compensation_Data.mat"}
    return sorted(path for path in input_root.rglob("*.mat") if path.name not in ignored)


def resolve_soh_series(battery: Battery) -> np.ndarray:
    # Keep the same per-cycle reference target used by the v1 feature extractor.
    return np.asarray(battery.get_degradation_trajectory(), dtype=float).reshape(-1)


def as_1d_float_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def row_has_nonfinite(row: Dict[str, object]) -> bool:
    for key in ["SOH", "relative_time_min", "voltage_V", "current_A", "power_Wh", "temperature_C"]:
        value = row.get(key, np.nan)
        if not np.isfinite(float(value)):
            return True
    return False


def extract_segment_arrays(
    battery: Battery,
    cycle: int,
    segment: str,
    cc_voltage_range: Sequence[float],
    cv_current_range: Sequence[float],
) -> Dict[str, np.ndarray]:
    variables = ["relative_time_min", "voltage_V", "current_A", "power_Wh", "temperature_C"]

    if segment == "CC":
        return {
            variable: as_1d_float_array(
                battery.get_CC_value(
                    cycle=cycle,
                    variable=variable,
                    voltage_range=list(cc_voltage_range),
                )
            )
            for variable in variables
        }

    if segment == "CV":
        return {
            variable: as_1d_float_array(
                battery.get_CV_value(
                    cycle=cycle,
                    variable=variable,
                    current_range=list(cv_current_range),
                )
            )
            for variable in variables
        }

    raise ValueError(f"unsupported segment: {segment}")


def extract_cycle_raw_rows(
    battery: Battery,
    cycle: int,
    soh_series: np.ndarray,
    cc_voltage_range: Sequence[float],
    cv_current_range: Sequence[float],
) -> tuple[List[Dict[str, object]], int]:
    rows: List[Dict[str, object]] = []
    empty_segments = 0
    cycle_point_index = 0
    soh = float(soh_series[cycle - 1])

    for segment in ["CC", "CV"]:
        arrays = extract_segment_arrays(
            battery=battery,
            cycle=cycle,
            segment=segment,
            cc_voltage_range=cc_voltage_range,
            cv_current_range=cv_current_range,
        )
        lengths = [values.size for values in arrays.values()]
        n = min(lengths) if lengths else 0
        if n == 0:
            empty_segments += 1
            continue

        for segment_point_index in range(n):
            rows.append(
                {
                    "cycle": cycle,
                    "SOH": soh,
                    "segment": segment,
                    "cycle_point_index": cycle_point_index,
                    "segment_point_index": segment_point_index,
                    "relative_time_min": arrays["relative_time_min"][segment_point_index],
                    "voltage_V": arrays["voltage_V"][segment_point_index],
                    "current_A": arrays["current_A"][segment_point_index],
                    "power_Wh": arrays["power_Wh"][segment_point_index],
                    "temperature_C": arrays["temperature_C"][segment_point_index],
                }
            )
            cycle_point_index += 1

    return rows, empty_segments


def write_raw_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in RAW_COLUMNS:
                value = row.get(key, "")
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    out[key] = ""
                else:
                    out[key] = value
            writer.writerow(out)


def write_report(rows: List[Dict[str, object]], report_csv: Path) -> None:
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def process_one_file(args: argparse.Namespace, mat_path: Path) -> Dict[str, object]:
    battery = Battery(str(mat_path))
    soh_series = resolve_soh_series(battery)

    rows: List[Dict[str, object]] = []
    written_cycles = 0
    skipped_test_capacity = 0
    skipped_nonfinite_rows = 0
    empty_segments = 0
    failed_cycles = 0

    for cycle in range(args.start_cycle, int(battery.cycle_life) + 1):
        description = battery.get_one_cycle_description(cycle)
        if args.skip_test_capacity and is_test_capacity(description):
            skipped_test_capacity += 1
            continue

        try:
            cycle_rows, cycle_empty_segments = extract_cycle_raw_rows(
                battery=battery,
                cycle=cycle,
                soh_series=soh_series,
                cc_voltage_range=args.cc_voltage_range,
                cv_current_range=args.cv_current_range,
            )
        except Exception as exc:
            failed_cycles += 1
            if args.strict:
                raise RuntimeError(f"failed extracting {mat_path.name} cycle {cycle}") from exc
            continue

        empty_segments += cycle_empty_segments
        if args.drop_nonfinite_rows:
            finite_rows = []
            for row in cycle_rows:
                if row_has_nonfinite(row):
                    skipped_nonfinite_rows += 1
                else:
                    finite_rows.append(row)
            cycle_rows = finite_rows

        if cycle_rows:
            written_cycles += 1
            rows.extend(cycle_rows)

    output_csv = args.output_dir / f"{mat_path.stem}.csv"
    write_raw_csv(rows, output_csv)

    return {
        "batch": batch_name(mat_path),
        "file": mat_path.name,
        "cycle_life": int(battery.cycle_life),
        "start_cycle": args.start_cycle,
        "written_cycles": written_cycles,
        "written_rows": len(rows),
        "skipped_test_capacity": skipped_test_capacity,
        "skipped_nonfinite_rows": skipped_nonfinite_rows,
        "empty_segments": empty_segments,
        "failed_cycles": failed_cycles,
    }


def process_one_file_worker(args: argparse.Namespace, mat_path: Path) -> Dict[str, object]:
    """Spawn-safe file worker: each process opens and writes one battery only."""

    load_runtime_dependencies()
    return process_one_file(args, mat_path)


def print_summary(report_rows: List[Dict[str, object]]) -> None:
    grouped = defaultdict(Counter)
    for row in report_rows:
        group = grouped[str(row["batch"])]
        for key in [
            "written_cycles",
            "written_rows",
            "skipped_test_capacity",
            "skipped_nonfinite_rows",
            "empty_segments",
            "failed_cycles",
        ]:
            group[key] += int(row[key])
        group["files"] += 1

    print(
        "batch,files,written_cycles,written_rows,skipped_test_capacity,"
        "skipped_nonfinite_rows,empty_segments,failed_cycles"
    )
    for batch in sorted(grouped):
        group = grouped[batch]
        print(
            f"{batch},{group['files']},{group['written_cycles']},{group['written_rows']},"
            f"{group['skipped_test_capacity']},{group['skipped_nonfinite_rows']},"
            f"{group['empty_segments']},{group['failed_cycles']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract raw end-of-charge CC/CV signal points from XJTU .mat files, "
            "one CSV per battery."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--start-cycle", type=int, default=2)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing existing CSV products in --output-dir",
    )
    parser.add_argument("--skip-test-capacity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-nonfinite-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cc-voltage-range", type=parse_float_pair, default=[4.0, 4.195])
    parser.add_argument("--cv-current-range", type=parse_float_pair, default=[0.5, 0.1])
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
        help="independent .mat-file workers; use 1 for serial debugging",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.input_root is None:
        parser.error("--input-root is required (or set XJTU_SOURCE_ROOT)")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.report_csv is None:
        args.report_csv = args.output_dir / "extraction_report.csv"
    else:
        args.report_csv = args.report_csv.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()

    if not args.input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {args.input_root}")

    mat_files = find_all_mat_files(args.input_root)
    if not mat_files:
        raise FileNotFoundError(f"no .mat files found under: {args.input_root}")
    existing = list(args.output_dir.glob("*.csv")) if args.output_dir.is_dir() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} already contains CSV products; choose a new output directory "
            "or pass --overwrite deliberately"
        )

    print(f"Found {len(mat_files)} battery .mat files under {args.input_root}")
    args.workers = min(int(args.workers), len(mat_files))
    if args.workers == 1:
        load_runtime_dependencies()
        report_rows = [process_one_file(args, mat_path) for mat_path in mat_files]
    else:
        print(f"execution=parallel; workers={args.workers}; partition=one .mat per worker")
        completed = {}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one_file_worker, args, mat_path): mat_path
                for mat_path in mat_files
            }
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
        # The aggregate report is independent of completion order.
        report_rows = [completed[mat_path] for mat_path in mat_files]

    for row in report_rows:
        print(
            f"OK: {row['file']} -> {args.output_dir / (Path(str(row['file'])).stem + '.csv')} "
            f"(cycles={row['written_cycles']}, rows={row['written_rows']}, "
            f"skipped_test_capacity={row['skipped_test_capacity']}, "
            f"skipped_nonfinite_rows={row['skipped_nonfinite_rows']}, "
            f"empty_segments={row['empty_segments']}, failed={row['failed_cycles']})"
        )

    write_report(report_rows, args.report_csv)
    print_summary(report_rows)
    print(f"wrote CSVs: {args.output_dir}")
    print(f"wrote report: {args.report_csv}")


if __name__ == "__main__":
    main()
