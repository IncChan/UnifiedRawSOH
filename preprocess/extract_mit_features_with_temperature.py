import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import h5py
import numpy as np
from scipy.stats import entropy, kurtosis, linregress, skew

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


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
    "batch_file",
    "cell",
    "policy",
    "cycle_life",
    "start_cycle",
    "written_rows",
    "skipped_nonfinite",
    "failed_cycles",
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "MIT_features"


def default_input_root() -> Path | None:
    value = os.environ.get("MIT_SOURCE_ROOT")
    return Path(value).expanduser() if value else None


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else np.nan


def safe_std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=0)) if x.size > 0 else np.nan


def safe_max(x: np.ndarray) -> float:
    return float(np.max(x)) if x.size > 0 else np.nan


def safe_kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return np.nan
    return float(kurtosis(x))


def safe_skewness(x: np.ndarray) -> float:
    if x.size < 3:
        return np.nan
    return float(skew(x))


def safe_entropy(x: np.ndarray, bins: Optional[int] = None) -> float:
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


def decode_h5_string(dataset) -> str:
    raw = dataset[()]
    if hasattr(raw, "tobytes"):
        return raw.tobytes()[::2].decode(errors="ignore").strip("\x00")
    return str(raw)


class MITBatch:
    def __init__(self, path: Path):
        self.path = path
        self.file = h5py.File(path, "r")
        self.batch = self.file["batch"]
        self.num_cells = int(self.batch["summary"].shape[0])

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _ref_array(self, ref) -> np.ndarray:
        return np.asarray(self.file[ref][()]).reshape(-1)

    def _ref_scalar(self, ref) -> float:
        values = self._ref_array(ref)
        if values.size == 0:
            return np.nan
        return float(values[0])

    def _summary_group(self, cell_index: int):
        return self.file[self.batch["summary"][cell_index, 0]]

    def _cycles_group(self, cell_index: int):
        return self.file[self.batch["cycles"][cell_index, 0]]

    def policy(self, cell_index: int) -> str:
        return decode_h5_string(self.file[self.batch["policy_readable"][cell_index, 0]])

    def capacity_series(self, cell_index: int) -> np.ndarray:
        summary = self._summary_group(cell_index)
        values = summary["QDischarge"][0, 1:]
        if values.dtype == object:
            return np.asarray([self._ref_scalar(ref) for ref in values], dtype=float)
        return np.asarray(values, dtype=float).reshape(-1)

    def cycle_life(self, cell_index: int) -> int:
        return int(self.capacity_series(cell_index).size)

    def cycle_data(self, cell_index: int, cycle: int) -> Dict[str, np.ndarray]:
        cycles = self._cycles_group(cell_index)
        if cycle < 1 or cycle >= cycles["I"].shape[0]:
            raise ValueError(f"cycle should be in [1,{cycles['I'].shape[0] - 1}]")

        return {
            "current_A": self._ref_array(cycles["I"][cycle, 0]),
            "voltage_V": self._ref_array(cycles["V"][cycle, 0]),
            "capacity_Ah": self._ref_array(cycles["Qc"][cycle, 0]),
            "discharge_capacity_Ah": self._ref_array(cycles["Qd"][cycle, 0]),
            "time_min": self._ref_array(cycles["t"][cycle, 0]),
            "temperature_C": self._ref_array(cycles["T"][cycle, 0]),
        }

    def charge_stage(self, cell_index: int, cycle: int) -> Dict[str, np.ndarray]:
        data = self.cycle_data(cell_index, cycle)
        current = data["current_A"]
        charge_indices = np.where(current > -1e-1)[0]
        if charge_indices.size == 0:
            return slice_data(data, charge_indices)

        breaks = np.where(np.diff(charge_indices) != 1)[0]
        if breaks.size > 0:
            charge_indices = charge_indices[: breaks[0] + 1]
        return slice_data(data, charge_indices)

    def cccv_stage(self, cell_index: int, cycle: int) -> Dict[str, np.ndarray]:
        data = self.charge_stage(cell_index, cycle)
        q = data["capacity_Ah"]
        current = data["current_A"]
        if q.size == 0:
            return data

        indices = np.where((q >= 0.79 * np.max(q)) & (current > 0.01))[0]
        if indices.size == 0:
            return slice_data(data, indices)

        breaks = np.where(np.diff(indices) != 1)[0]
        if breaks.size > 0:
            indices = indices[breaks[0] + 1 :]
        return slice_data(data, indices)

    def cc_stage(self, cell_index: int, cycle: int, voltage_range: Sequence[float]) -> Dict[str, np.ndarray]:
        data = self.cccv_stage(cell_index, cycle)
        voltage = data["voltage_V"]
        indices = np.where((voltage > voltage_range[0]) & (voltage < voltage_range[1]))[0]
        return slice_data(data, indices)

    def cv_stage(self, cell_index: int, cycle: int, current_range: Optional[Sequence[float]]) -> Dict[str, np.ndarray]:
        data = self.cccv_stage(cell_index, cycle)
        voltage = data["voltage_V"]
        indices = np.where(voltage > 3.595)[0]
        if current_range is not None:
            current = data["current_A"]
            indices = indices[
                (current[indices] > min(current_range)) & (current[indices] < max(current_range))
            ]
        return slice_data(data, indices)


def slice_data(data: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value)[indices] for key, value in data.items()}


def _window_features(
    cc: Mapping[str, np.ndarray],
    cv: Mapping[str, np.ndarray],
    capacity_ah: float,
    *,
    cv_slope: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, float]:
    """Compute the validated Only-F statistics from already selected points.

    The function intentionally contains no phase detection or window
    selection.  It is shared by the legacy standalone extractor and the
    canonical physical-cell export, whose inputs have already been selected
    by the raw phase-aware contract.
    """

    if cv_slope is None:
        cv_slope = cv
    cc_t = np.asarray(cc["time_min"], dtype=float) * 60
    cv_t = np.asarray(cv["time_min"], dtype=float) * 60
    cv_t_slope = np.asarray(cv_slope["time_min"], dtype=float) * 60

    return {
        "voltage mean": safe_mean(np.asarray(cc["voltage_V"], dtype=float)),
        "voltage std": safe_std(np.asarray(cc["voltage_V"], dtype=float)),
        "voltage kurtosis": safe_kurtosis(np.asarray(cc["voltage_V"], dtype=float)),
        "voltage skewness": safe_skewness(np.asarray(cc["voltage_V"], dtype=float)),
        "CC Q": delta_quantity(np.asarray(cc["capacity_Ah"], dtype=float)),
        "CC charge time": delta_time(cc_t),
        "voltage slope": safe_slope(cc_t, np.asarray(cc["voltage_V"], dtype=float)),
        "voltage entropy": safe_entropy(np.asarray(cc["voltage_V"], dtype=float)),
        "T_CC_mean": safe_mean(np.asarray(cc["temperature_C"], dtype=float)),
        "T_CC_max": safe_max(np.asarray(cc["temperature_C"], dtype=float)),
        "T_CC_delta": delta_quantity(np.asarray(cc["temperature_C"], dtype=float)),
        "T_CC_slope": safe_slope(cc_t, np.asarray(cc["temperature_C"], dtype=float)),
        "current mean": safe_mean(np.asarray(cv["current_A"], dtype=float)),
        "current std": safe_std(np.asarray(cv["current_A"], dtype=float)),
        "current kurtosis": safe_kurtosis(np.asarray(cv["current_A"], dtype=float)),
        "current skewness": safe_skewness(np.asarray(cv["current_A"], dtype=float)),
        "CV Q": delta_quantity(np.asarray(cv["capacity_Ah"], dtype=float)),
        "CV charge time": delta_time(cv_t),
        "current slope": safe_slope(
            cv_t_slope, np.asarray(cv_slope["current_A"], dtype=float)
        ),
        "current entropy": safe_entropy(np.asarray(cv["current_A"], dtype=float)),
        "T_CV_mean": safe_mean(np.asarray(cv["temperature_C"], dtype=float)),
        "T_CV_max": safe_max(np.asarray(cv["temperature_C"], dtype=float)),
        "T_CV_delta": delta_quantity(np.asarray(cv["temperature_C"], dtype=float)),
        "T_CV_slope": safe_slope(cv_t, np.asarray(cv["temperature_C"], dtype=float)),
        "capacity": float(capacity_ah),
    }


def _selected_raw_segment(
    rows: Sequence[Mapping[str, object]], segment: str
) -> Dict[str, np.ndarray]:
    """Materialize one already-selected raw phase in point order."""

    selected = sorted(
        (row for row in rows if str(row["segment"]).strip() == segment),
        key=lambda row: int(float(row["segment_point_index"])),
    )
    if not selected:
        raise ValueError(f"canonical MIT cycle has no selected {segment} points")
    raw_to_feature_key = {
        "relative_time_min": "time_min",
        "voltage_V": "voltage_V",
        "current_A": "current_A",
        "charge_capacity_Ah": "capacity_Ah",
        "temperature_C": "temperature_C",
    }
    return {
        feature_key: np.asarray([float(row[raw_key]) for row in selected], dtype=float)
        for raw_key, feature_key in raw_to_feature_key.items()
    }


def extract_phase_aware_features_from_raw_rows(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, float]:
    """Build Only-F statistics from exactly the canonical raw CC/CV points.

    ``rows`` must describe one physical cycle from ``datasets/MIT_raw`` after
    its actual CC/CV phase has been inferred and the paper windows selected:
    CC 3.45--3.60 V and CV 0.25C--0.05C.  This helper never re-applies a
    broader historical voltage/current window.
    """

    if not rows:
        raise ValueError("cannot derive MIT features from an empty raw cycle")
    return _window_features(
        _selected_raw_segment(rows, "CC"),
        _selected_raw_segment(rows, "CV"),
        capacity_ah=float(rows[0]["capacity_Ah"]),
    )


def extract_cycle_features(
    batch: MITBatch,
    cell_index: int,
    cycle: int,
    capacity_series: np.ndarray,
    cc_voltage_range: Sequence[float],
    cv_current_range: Sequence[float],
    cv_slope_current_range: Sequence[float],
) -> Dict[str, float]:
    """Legacy direct-HDF5 feature extraction for non-canonical products.

    The Paper-v1 physical MIT product uses
    :func:`extract_phase_aware_features_from_raw_rows` instead, so its
    statistics inherit the canonical raw model's phase-aware selected points.
    """

    cc = batch.cc_stage(cell_index, cycle, cc_voltage_range)
    cv = batch.cv_stage(cell_index, cycle, cv_current_range)
    cv_slope = batch.cv_stage(cell_index, cycle, cv_slope_current_range)
    return _window_features(
        cc,
        cv,
        capacity_ah=float(capacity_series[cycle - 1]),
        cv_slope=cv_slope,
    )


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


def find_all_batch_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*batchdata*_struct*.mat"))


def output_name(batch_path: Path, cell_index: int) -> str:
    return f"MIT_{batch_path.stem}_cell-{cell_index + 1:03d}.csv"


def selected_cells(args: argparse.Namespace, batch: MITBatch):
    if args.cells:
        return [int(item) - 1 for item in args.cells]
    return range(batch.num_cells)


def process_one_cell(args: argparse.Namespace, batch_path: Path, batch: MITBatch, cell_index: int) -> Dict[str, object]:
    capacity_series = batch.capacity_series(cell_index)
    cycle_life = int(capacity_series.size)
    rows: List[Dict[str, float]] = []
    skipped_nonfinite = 0
    failed_cycles = 0

    for cycle in range(args.start_cycle, cycle_life + 1):
        try:
            row = extract_cycle_features(
                batch=batch,
                cell_index=cell_index,
                cycle=cycle,
                capacity_series=capacity_series,
                cc_voltage_range=args.cc_voltage_range,
                cv_current_range=args.cv_current_range,
                cv_slope_current_range=args.cv_slope_current_range,
            )
        except Exception as exc:
            failed_cycles += 1
            if args.strict:
                raise RuntimeError(
                    f"failed extracting {batch_path.name} cell {cell_index + 1} cycle {cycle}"
                ) from exc
            continue

        if args.drop_nonfinite_rows and row_has_nonfinite(row):
            skipped_nonfinite += 1
            continue
        rows.append(row)

    if args.drop_leading_rows > 0:
        rows = rows[args.drop_leading_rows :]

    output_csv = args.output_dir / output_name(batch_path, cell_index)
    write_csv(rows, output_csv)

    return {
        "batch_file": batch_path.name,
        "cell": cell_index + 1,
        "policy": batch.policy(cell_index),
        "cycle_life": cycle_life,
        "start_cycle": args.start_cycle,
        "written_rows": len(rows),
        "skipped_nonfinite": skipped_nonfinite,
        "failed_cycles": failed_cycles,
    }


def parse_float_pair(text: str) -> List[float]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected two comma-separated floats, got: {text}")
    return [float(parts[0]), float(parts[1])]


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def print_summary(report_rows: List[Dict[str, object]]) -> None:
    grouped = defaultdict(Counter)
    for row in report_rows:
        group = grouped[str(row["batch_file"])]
        for key in ["written_rows", "skipped_nonfinite", "failed_cycles"]:
            group[key] += int(row[key])
        group["cells"] += 1

    print("batch_file,cells,written_rows,skipped_nonfinite,failed_cycles")
    for batch_file in sorted(grouped):
        group = grouped[batch_file]
        print(
            f"{batch_file},{group['cells']},{group['written_rows']},"
            f"{group['skipped_nonfinite']},{group['failed_cycles']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract end-of-charge MIT/A123 features with voltage/current/temperature."
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--start-cycle", type=int, default=2)
    parser.add_argument("--drop-nonfinite-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-leading-rows", type=int, default=0)
    parser.add_argument("--cc-voltage-range", type=parse_float_pair, default=[3.4, 3.595])
    parser.add_argument("--cv-current-range", type=parse_float_pair, default=[0.5, 0.1])
    parser.add_argument("--cv-slope-current-range", type=parse_float_pair, default=[0.5, 0.1])
    parser.add_argument("--cells", type=parse_int_list, default=None, help="1-based cell list, e.g. 1,2,8")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.input_root is None:
        parser.error("--input-root is required (or set MIT_SOURCE_ROOT)")
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.report_csv is None:
        args.report_csv = args.output_dir / "mit_extraction_report.csv"
    else:
        args.report_csv = args.report_csv.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {args.input_root}")

    batch_files = find_all_batch_files(args.input_root)
    if args.max_batches is not None:
        batch_files = batch_files[: args.max_batches]
    if not batch_files:
        raise FileNotFoundError(f"no MIT batch .mat files found under: {args.input_root}")

    report_rows = []
    print(f"Found {len(batch_files)} MIT batch files under {args.input_root}")
    for batch_path in batch_files:
        with MITBatch(batch_path) as batch:
            for cell_index in selected_cells(args, batch):
                row = process_one_cell(args, batch_path, batch, cell_index)
                report_rows.append(row)
                print(
                    f"OK: {batch_path.name} cell {cell_index + 1} -> "
                    f"{args.output_dir / output_name(batch_path, cell_index)} "
                    f"(rows={row['written_rows']}, skipped_nonfinite={row['skipped_nonfinite']}, "
                    f"failed={row['failed_cycles']})"
                )

    write_report(report_rows, args.report_csv)
    print_summary(report_rows)
    print(f"wrote CSVs: {args.output_dir}")
    print(f"wrote report: {args.report_csv}")


if __name__ == "__main__":
    main()
