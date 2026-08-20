import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

entropy = None
kurtosis = None
linregress = None
skew = None
Battery = None

FEATURE_COLUMNS = [
    "voltage mean",
    "voltage std",
    "voltage kurtosis",
    "voltage skewness",
    "CC Q",
    "CC charge time",
    "voltage slope",
    "voltage entropy",
    "T_CC_mean",
    "T_CC_max",
    "T_CC_delta",
    "T_CC_slope",
    "current mean",
    "current std",
    "current kurtosis",
    "current skewness",
    "CV Q",
    "CV charge time",
    "current slope",
    "current entropy",
    "T_CV_mean",
    "T_CV_max",
    "T_CV_delta",
    "T_CV_slope",
    "capacity",
]

REPORT_COLUMNS = [
    "batch",
    "file",
    "cycle_life",
    "start_cycle",
    "written_rows",
    "skipped_test_capacity",
    "skipped_nonfinite",
    "failed_cycles",
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "XJTU_features"


def default_input_root() -> Path | None:
    """Use an explicit source archive; never bake one developer path into Git."""

    value = os.environ.get("XJTU_SOURCE_ROOT")
    return Path(value).expanduser() if value else None


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else np.nan


def safe_std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=0)) if x.size > 0 else np.nan


def safe_max(x: np.ndarray) -> float:
    return float(np.max(x)) if x.size > 0 else np.nan


def safe_kurtosis(x: np.ndarray) -> float:
    if x.size < 4 or kurtosis is None:
        return np.nan
    return float(kurtosis(x))


def safe_skewness(x: np.ndarray) -> float:
    if x.size < 3 or skew is None:
        return np.nan
    return float(skew(x))


def safe_entropy(x: np.ndarray, bins: int | None = None) -> float:
    """Match the original XJTU data extractor's histogram entropy."""
    if entropy is None:
        return np.nan
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    if np.isclose(np.max(x), np.min(x)):
        return 0.0
    hist_bins = bins or x.size
    hist, _ = np.histogram(x, bins=hist_bins)
    if np.sum(hist) == 0:
        return np.nan
    return float(entropy(hist))


def safe_slope(time_x: np.ndarray, y: np.ndarray) -> float:
    """Match the original XJTU data extractor's linregress slope."""
    if linregress is None:
        return np.nan
    x = np.asarray(time_x, dtype=float).reshape(-1)
    z = np.asarray(y, dtype=float).reshape(-1)

    n = min(x.size, z.size)
    if n < 2:
        return np.nan

    x = x[:n]
    z = z[:n]
    mask = np.isfinite(x) & np.isfinite(z)
    x = x[mask]
    z = z[mask]

    if x.size < 2 or np.isclose(np.std(x), 0.0):
        return np.nan
    return float(linregress(x, z).slope)


def delta_quantity(x: np.ndarray) -> float:
    return float(x[-1] - x[0]) if x.size >= 2 else np.nan


def delta_time(t: np.ndarray) -> float:
    return float(t[-1] - t[0]) if t.size >= 2 else np.nan


def row_has_nonfinite(row: Dict[str, float]) -> bool:
    for key in FEATURE_COLUMNS:
        value = row.get(key, np.nan)
        if not np.isfinite(float(value)):
            return True
    return False


def is_test_capacity(description: str) -> bool:
    return "test capacity" in str(description).lower()


def resolve_capacity_series(battery: Battery) -> np.ndarray:
    # For partial-discharge batches, XJTUBatteryClass interpolates the
    # degradation trajectory from test-capacity cycles. We still skip those
    # test-capacity cycles as model inputs below.
    return np.asarray(battery.get_degradation_trajectory(), dtype=float).reshape(-1)


def load_runtime_dependencies() -> None:
    global Battery, entropy, kurtosis, linregress, skew
    from scipy.stats import entropy as scipy_entropy
    from scipy.stats import kurtosis as scipy_kurtosis
    from scipy.stats import linregress as scipy_linregress
    from scipy.stats import skew as scipy_skew
    from XJTUBatteryClass import Battery as XJTUBattery

    entropy = scipy_entropy
    kurtosis = scipy_kurtosis
    linregress = scipy_linregress
    skew = scipy_skew
    Battery = XJTUBattery


def extract_cycle_features(
    battery: Battery,
    cycle: int,
    capacity_series: np.ndarray,
    cc_voltage_range: Sequence[float],
    cv_current_range: Sequence[float],
    cv_slope_current_range: Sequence[float],
) -> Dict[str, float]:
    cc_v = np.asarray(
        battery.get_CC_value(cycle=cycle, variable="voltage_V", voltage_range=list(cc_voltage_range))
    )
    cc_t = (
        np.asarray(
            battery.get_CC_value(cycle=cycle, variable="relative_time_min", voltage_range=list(cc_voltage_range))
        )
        * 60
    )
    cc_q = np.asarray(
        battery.get_CC_value(cycle=cycle, variable="capacity_Ah", voltage_range=list(cc_voltage_range))
    )
    cc_temp = np.asarray(
        battery.get_CC_value(cycle=cycle, variable="temperature_C", voltage_range=list(cc_voltage_range))
    )

    cv_i = np.asarray(
        battery.get_CV_value(cycle=cycle, variable="current_A", current_range=list(cv_current_range))
    )
    cv_t = (
        np.asarray(
            battery.get_CV_value(cycle=cycle, variable="relative_time_min", current_range=list(cv_current_range))
        )
        * 60
    )
    cv_q = np.asarray(
        battery.get_CV_value(cycle=cycle, variable="capacity_Ah", current_range=list(cv_current_range))
    )
    cv_temp = np.asarray(
        battery.get_CV_value(cycle=cycle, variable="temperature_C", current_range=list(cv_current_range))
    )

    cv_i_slope = np.asarray(
        battery.get_CV_value(cycle=cycle, variable="current_A", current_range=list(cv_slope_current_range))
    )
    cv_t_slope = (
        np.asarray(
            battery.get_CV_value(
                cycle=cycle,
                variable="relative_time_min",
                current_range=list(cv_slope_current_range),
            )
        )
        * 60
    )

    return {
        "voltage mean": safe_mean(cc_v),
        "voltage std": safe_std(cc_v),
        "voltage kurtosis": safe_kurtosis(cc_v),
        "voltage skewness": safe_skewness(cc_v),
        "CC Q": delta_quantity(cc_q),
        "CC charge time": delta_time(cc_t),
        "voltage slope": safe_slope(cc_t, cc_v),
        "voltage entropy": safe_entropy(cc_v),
        "T_CC_mean": safe_mean(cc_temp),
        "T_CC_max": safe_max(cc_temp),
        "T_CC_delta": delta_quantity(cc_temp),
        "T_CC_slope": safe_slope(cc_t, cc_temp),
        "current mean": safe_mean(cv_i),
        "current std": safe_std(cv_i),
        "current kurtosis": safe_kurtosis(cv_i),
        "current skewness": safe_skewness(cv_i),
        "CV Q": delta_quantity(cv_q),
        "CV charge time": delta_time(cv_t),
        "current slope": safe_slope(cv_t_slope, cv_i_slope),
        "current entropy": safe_entropy(cv_i),
        "T_CV_mean": safe_mean(cv_temp),
        "T_CV_max": safe_max(cv_temp),
        "T_CV_delta": delta_quantity(cv_temp),
        "T_CV_slope": safe_slope(cv_t, cv_temp),
        "capacity": float(capacity_series[cycle - 1]),
    }


def write_csv(rows: List[Dict[str, float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in FEATURE_COLUMNS:
                value = row.get(key, np.nan)
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


def find_all_mat_files(input_root: Path) -> List[Path]:
    ignored = {"Temperature_Compensation_Data.mat"}
    return sorted(path for path in input_root.rglob("*.mat") if path.name not in ignored)


def batch_name(input_path: Path) -> str:
    parent = input_path.parent.name
    if parent.startswith("Batch-"):
        return parent
    if parent:
        return parent
    return "unknown"


def process_one_file(args: argparse.Namespace, input_path: Path) -> Dict[str, object]:
    battery = Battery(str(input_path))
    capacity_series = resolve_capacity_series(battery)

    rows: List[Dict[str, float]] = []
    skipped_test_capacity = 0
    skipped_nonfinite = 0
    failed_cycles = 0

    for cycle in range(args.start_cycle, int(battery.cycle_life) + 1):
        description = battery.get_one_cycle_description(cycle)
        if args.skip_test_capacity and is_test_capacity(description):
            skipped_test_capacity += 1
            continue

        try:
            row = extract_cycle_features(
                battery=battery,
                cycle=cycle,
                capacity_series=capacity_series,
                cc_voltage_range=args.cc_voltage_range,
                cv_current_range=args.cv_current_range,
                cv_slope_current_range=args.cv_slope_current_range,
            )
        except Exception as exc:
            failed_cycles += 1
            if args.strict:
                raise RuntimeError(f"failed extracting {input_path.name} cycle {cycle}") from exc
            continue

        if args.drop_nonfinite_rows and row_has_nonfinite(row):
            skipped_nonfinite += 1
            continue
        rows.append(row)

    if args.drop_leading_rows > 0:
        rows = rows[args.drop_leading_rows :]

    output_csv = args.output_dir / f"{input_path.stem}.csv"
    write_csv(rows, output_csv)

    return {
        "batch": batch_name(input_path),
        "file": input_path.name,
        "cycle_life": int(battery.cycle_life),
        "start_cycle": args.start_cycle,
        "written_rows": len(rows),
        "skipped_test_capacity": skipped_test_capacity,
        "skipped_nonfinite": skipped_nonfinite,
        "failed_cycles": failed_cycles,
    }


def process_one_file_worker(args: argparse.Namespace, input_path: Path) -> Dict[str, object]:
    """Spawn-safe file worker: each process owns one source/output pair."""

    load_runtime_dependencies()
    return process_one_file(args, input_path)


def parse_float_pair(text: str) -> List[float]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected two comma-separated floats, got: {text}")
    return [float(parts[0]), float(parts[1])]


def print_summary(report_rows: List[Dict[str, object]]) -> None:
    grouped = defaultdict(Counter)
    for row in report_rows:
        group = grouped[str(row["batch"])]
        for key in ["written_rows", "skipped_test_capacity", "skipped_nonfinite", "failed_cycles"]:
            group[key] += int(row[key])
        group["files"] += 1

    print("batch,files,written_rows,skipped_test_capacity,skipped_nonfinite,failed_cycles")
    for batch in sorted(grouped):
        group = grouped[batch]
        print(
            f"{batch},{group['files']},{group['written_rows']},"
            f"{group['skipped_test_capacity']},{group['skipped_nonfinite']},{group['failed_cycles']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract end-of-charge XJTU features with voltage/current/temperature "
            "from raw .mat files, excluding test-capacity cycles from PINN4SOH inputs."
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
    parser.add_argument("--drop-leading-rows", type=int, default=0)
    parser.add_argument("--cc-voltage-range", type=parse_float_pair, default=[4.0, 4.195])
    parser.add_argument("--cv-current-range", type=parse_float_pair, default=[0.5, 0.1])
    parser.add_argument("--cv-slope-current-range", type=parse_float_pair, default=[0.5, 0.4])
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
        report_rows = [completed[mat_path] for mat_path in mat_files]

    for row in report_rows:
        print(
            f"OK: {row['file']} -> {args.output_dir / (Path(str(row['file'])).stem + '.csv')} "
            f"(rows={row['written_rows']}, skipped_test_capacity={row['skipped_test_capacity']}, "
            f"skipped_nonfinite={row['skipped_nonfinite']}, failed={row['failed_cycles']})"
        )

    write_report(report_rows, args.report_csv)
    print_summary(report_rows)
    print(f"wrote CSVs: {args.output_dir}")
    print(f"wrote report: {args.report_csv}")


if __name__ == "__main__":
    main()
