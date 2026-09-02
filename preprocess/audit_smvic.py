#!/usr/bin/env python3
"""Stream SMVIC CSVs and audit model-eligible aging cycles."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from UnifiedRawSOH.preprocess.smvic_common import (  # noqa: E402
    FAMILY_SPECS,
    SOURCE_SCHEMA,
    iter_classified_cycles,
    json_value,
)


DEFAULT_SOURCE = Path("/data1/chenyanxi/lb_project/datasets/SMVIC/dataset")
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "SMVIC_preprocess_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--groups", nargs="+", choices=(*FAMILY_SPECS, "all"), default=["all"])
    parser.add_argument("--max-cycles-per-cell", type=int)
    parser.add_argument("--min-phase-points", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = list(FAMILY_SPECS) if "all" in args.groups else list(dict.fromkeys(args.groups))
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    reasons: dict[str, Counter] = defaultdict(Counter)
    cell_counts: dict[str, Counter] = defaultdict(Counter)
    processed = 0
    for _, audit in iter_classified_cycles(
        args.source_root,
        groups,
        max_cycles_per_cell=args.max_cycles_per_cell,
        min_phase_points=args.min_phase_points,
    ):
        rows.append(audit)
        processed += 1
        domain = str(audit["domain_id"])
        reasons[domain][str(audit["reason"])] += 1
        cell_counts[domain][str(audit["battery_id"])] += int(audit["eligible"])
        if args.progress_every > 0 and processed % args.progress_every == 0:
            print(f"[SMVIC audit] cycles={processed}", flush=True)

    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_root / "cycle_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_schema": SOURCE_SCHEMA,
        "source_root": str(args.source_root.resolve()),
        "groups": groups,
        "bounded_smoke_audit": args.max_cycles_per_cell is not None,
        "cycles_considered": processed,
        "domains": {
            domain: {
                "group": spec.group,
                "eligible_cycles": int(sum(cell_counts[domain].values())),
                "eligible_cycles_by_cell": dict(sorted(cell_counts[domain].items())),
                "reason_counts": dict(sorted(reasons[domain].items())),
                "protocol": json_value(spec.__dict__),
                "normalization": spec.normalization(),
            }
            for domain, spec in ((FAMILY_SPECS[group].domain_id, FAMILY_SPECS[group]) for group in groups)
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(json_value(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_value(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
