#!/usr/bin/env python3
"""Export MIT paper124 full charging events with canonical physical identities.

This is deliberately a separate, auditable product from ``datasets/MIT_raw``:
the latter contains only the terminal CC/CV windows used by E1, whereas this
export retains the complete principal charge event required by E2 FULL.
"""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_ROOT = REPO_ROOT / "preprocess"
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

from extract_mit_cycle_raw_with_temperature import MITRawBatch  # noqa: E402
from extract_mit_physical_with_temperature import (  # noqa: E402
    _batch_paths_by_date,
    _segment_inventory,
)
from mit_physical_provenance import (  # noqa: E402
    build_physical_cells,
    source_cycle_is_known_invalid,
)


COLUMNS = (
    "battery_id", "cycle_id", "condition", "segment",
    "relative_time_min", "voltage_V", "current_A", "temperature_C", "soh",
    "source_batch_date", "source_cell", "source_cycle",
)


def _boundary(current: np.ndarray, voltage: np.ndarray, capacity: np.ndarray) -> int | None:
    """Find the first sustained current taper near the charge-voltage ceiling."""

    if current.size < 16:
        return None
    magnitude = np.abs(current.astype(float))
    reference = float(np.quantile(magnitude[: max(8, int(np.ceil(0.25 * len(magnitude))))], 0.9))
    ceiling = float(np.max(voltage))
    if not np.isfinite(reference) or reference <= 0:
        return None
    for index in range(4, len(magnitude) - 4):
        if voltage[index] >= ceiling - 0.02 and np.all(magnitude[index : index + 5] <= 0.99 * reference):
            return index
    threshold = 0.79 * float(np.max(capacity))
    candidates = np.flatnonzero((capacity >= threshold) & (magnitude < 0.99 * reference))
    return int(candidates[0]) if candidates.size else None


def _charge_event(batch: MITRawBatch, cell_index: int, source_cycle: int):
    data, mismatch = batch.cycle_data(cell_index, source_cycle)
    if mismatch:
        return None
    indices = batch.charge_indices(data)
    if indices.size < 16:
        return None
    # Retain the complete principal event but remove trailing zero-current rest.
    positive = np.flatnonzero(data["current_A"][indices] > 0.01)
    if positive.size:
        indices = indices[: int(positive[-1]) + 1]
    values = {key: np.asarray(value[indices], dtype=float) for key, value in data.items()}
    boundary = _boundary(values["current_A"], values["voltage_V"], values["charge_capacity_Ah"])
    if boundary is None or boundary < 4 or boundary > len(indices) - 5:
        return None
    time = values["time_min"] - values["time_min"][0]
    if np.any(~np.isfinite(time)) or np.any(np.diff(time) < 0):
        return None
    for key in ("voltage_V", "current_A", "temperature_C"):
        if np.any(~np.isfinite(values[key])):
            return None
    return time, values, boundary


def export(input_root: Path, output_root: Path, *, overwrite: bool, max_cells: int | None) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_root.glob("MIT_*_full_cccv.csv"))
    if existing and not overwrite:
        raise FileExistsError(f"{output_root} already contains {len(existing)} MIT FULL files; pass --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()

    cells = build_physical_cells("paper124")
    if max_cells is not None:
        cells = cells[:max_cells]
    paths = _batch_paths_by_date(input_root)
    written_cycles = 0
    skipped_cycles = 0
    with ExitStack() as stack:
        batches = {date: stack.enter_context(MITRawBatch(path)) for date, path in paths.items()}
        for cell_number, physical in enumerate(cells, start=1):
            segments, capacities, _, _, _ = _segment_inventory(physical, batches, strict=False)
            output_path = output_root / f"MIT_{physical.primary_batch_date}_physical-{physical.physical_index:03d}_full_cccv.csv"
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                writer.writeheader()
                for info, source in zip(segments, physical.source_segments):
                    batch = batches[source.batch_date]
                    global_start = int(info["physical_cycle_start"])
                    for source_cycle in range(1, int(info["used_cycle_count"]) + 1):
                        global_cycle = global_start + source_cycle - 1
                        # Match canonical MIT terminal extraction exactly.
                        if global_cycle < 2 or source_cycle_is_known_invalid(source.batch_date, source.cell, source_cycle):
                            continue
                        event = _charge_event(batch, source.cell - 1, source_cycle)
                        if event is None:
                            skipped_cycles += 1
                            continue
                        time, values, boundary = event
                        soh = float(capacities[global_cycle - 1]) / 1.1
                        for index in range(len(time)):
                            writer.writerow(
                                {
                                    "battery_id": physical.physical_cell_id,
                                    "cycle_id": global_cycle,
                                    "condition": physical.primary_batch_date,
                                    "segment": "CC" if index < boundary else "CV",
                                    "relative_time_min": float(time[index]),
                                    "voltage_V": float(values["voltage_V"][index]),
                                    "current_A": float(values["current_A"][index]),
                                    "temperature_C": float(values["temperature_C"][index]),
                                    "soh": soh,
                                    "source_batch_date": source.batch_date,
                                    "source_cell": source.cell,
                                    "source_cycle": source_cycle,
                                }
                            )
                        written_cycles += 1
            print(f"[MIT FULL {cell_number}/{len(cells)}] {physical.physical_cell_id}", flush=True)
    print(f"MIT FULL export complete: cycles={written_cycles}, rejected={skipped_cycles}, root={output_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args()
    export(args.input_root.resolve(), args.output_root.resolve(), overwrite=args.overwrite, max_cells=args.max_cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
