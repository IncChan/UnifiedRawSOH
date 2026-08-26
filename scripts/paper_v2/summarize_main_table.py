#!/usr/bin/env python3
"""Build the four-row Paper-v2 main table from complete seed summaries only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DOMAINS = ("xjtu", "mit", "smarthealth_lishen40", "smarthealth_catl280", "smarthealth_eve280")
DOMAIN_LABELS = {
    "xjtu": "XJTU",
    "mit": "MIT",
    "smarthealth_lishen40": "LISHEN",
    "smarthealth_catl280": "CATL",
    "smarthealth_eve280": "EVE",
}
STRATEGIES = (
    ("e1_feature", "Single-domain Feature MLP"),
    ("e1_raw", "Single-domain RawMamba"),
    ("e2_full", "Full-domain RawMamba"),
    ("e3_lodo", "LODO Zero-cell RawMamba"),
)
METRICS = ("mape", "rmse")


class MainTableError(RuntimeError):
    """Raised when the requested main table is not fully reproducible."""


def _finite(value, context):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MainTableError(f"{context} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise MainTableError(f"{context} is not finite: {value!r}")
    return result


def _kind_from_batch(batch_root: Path, paper_root: Path) -> tuple[str, str] | None:
    try:
        parts = batch_root.relative_to(paper_root).parts
    except ValueError:
        return None
    if len(parts) < 4:
        return None
    experiment, model, data_id = parts[0], parts[1], parts[2]
    if experiment == "e1_single_domain" and model == "FeatureMLP-BOL":
        return "e1_feature", data_id
    if experiment == "e1_single_domain" and model == "RawMamba-noCycleAux":
        return "e1_raw", data_id
    if experiment == "e2_full_domain" and model == "RawMamba-noCycleAux" and data_id == "full_domain":
        return "e2_full", "__all__"
    if experiment == "e3_lodo_zero_cell" and model == "RawMamba-noCycleAux":
        prefix = "lodo_zero_cell_to_"
        if data_id.startswith(prefix):
            return "e3_lodo", data_id[len(prefix):]
    return None


def _summary_for_batch(batch_root: Path, expected_seeds: list[int], expected_domains: list[str]) -> dict:
    path = batch_root / "summary_mean_std.json"
    if not path.is_file():
        raise MainTableError(f"Complete batch summary is missing: {path}")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MainTableError(f"Invalid summary JSON: {path}") from exc
    if summary.get("complete") is not True:
        raise MainTableError(f"Summary is not marked complete: {path}")
    if summary.get("label_rule") != "bol_peak_mean_top5_first100_v1":
        raise MainTableError(f"Unexpected label rule in {path}: {summary.get('label_rule')!r}")
    observed_seeds = {int(seed) for seed in summary.get("expected_seeds", [])}
    if observed_seeds != set(expected_seeds):
        raise MainTableError(
            f"Summary seed set mismatch in {path}: observed={sorted(observed_seeds)}, "
            f"expected={expected_seeds}"
        )
    observed_domains = {str(domain) for domain in summary.get("expected_domains", [])}
    if not set(expected_domains).issubset(observed_domains):
        raise MainTableError(
            f"Summary domain set mismatch in {path}: observed={sorted(observed_domains)}, "
            f"expected at least={expected_domains}"
        )
    return summary


def _discover_complete_batches(paper_root: Path) -> dict[tuple[str, str], list[Path]]:
    if not paper_root.is_dir():
        raise MainTableError(f"Paper-v2 output root does not exist: {paper_root}")
    found: dict[tuple[str, str], list[Path]] = {}
    for metric_path in paper_root.rglob("metrics_by_domain.csv"):
        seed_dir = metric_path.parent
        if not seed_dir.name.startswith("seed_"):
            continue
        batch_root = seed_dir.parent
        if "Paper-v1" in batch_root.parts:
            continue
        key = _kind_from_batch(batch_root, paper_root)
        if key is None or not (batch_root / "summary_mean_std.json").is_file():
            continue
        found.setdefault(key, []).append(batch_root)
    return found


def _value(summary: dict, domain: str, metric: str, batch_root: Path) -> tuple[float, float]:
    blocks = summary.get("per_domain", summary.get("metrics", {}))
    block = blocks.get(domain)
    if not isinstance(block, dict):
        raise MainTableError(f"{batch_root} has no per-domain summary for {domain}")
    metric_block = block.get(metric)
    if isinstance(metric_block, dict):
        mean = _finite(metric_block.get("mean"), f"{batch_root}/{domain}/{metric}.mean")
        std = _finite(metric_block.get("std"), f"{batch_root}/{domain}/{metric}.std")
    else:
        mean = _finite(metric_block, f"{batch_root}/{domain}/{metric}")
        std = 0.0
    return mean, std


def _choose_batch(candidates: list[Path], expected_seeds: list[int], expected_domains: list[str]) -> tuple[Path, dict]:
    valid = []
    for batch_root in candidates:
        try:
            summary = _summary_for_batch(batch_root, expected_seeds, expected_domains)
        except MainTableError:
            continue
        valid.append((batch_root, summary))
    if not valid:
        raise MainTableError(
            f"No complete valid batch among: {[str(path) for path in candidates]}"
        )
    # A rerun may leave multiple complete runtimes.  Select the newest complete
    # summary deterministically; incomplete candidates are never accepted.
    return max(valid, key=lambda item: (item[0].stat().st_mtime, str(item[0])))


def _format_cell(summary: dict, domain: str, batch_root: Path) -> str:
    mape = _value(summary, domain, "mape", batch_root)
    rmse = _value(summary, domain, "rmse", batch_root)
    return f"MAPE {mape[0]:.6f} \u00b1 {mape[1]:.6f}; RMSE {rmse[0]:.6f} \u00b1 {rmse[1]:.6f}"


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_main_table(
    output_root: str | Path,
    expected_seeds: list[int] | tuple[int, ...] = (42, 52, 62),
) -> dict:
    output_root = Path(output_root).resolve()
    paper_root = output_root / "Paper-v2"
    seeds = [int(seed) for seed in expected_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise MainTableError(f"expected_seeds must be unique and non-empty: {seeds}")
    candidates = _discover_complete_batches(paper_root)
    cells: dict[str, dict[str, str]] = {kind: {} for kind, _ in STRATEGIES}
    provenance = {}
    for kind, _label in STRATEGIES:
        for domain in DOMAINS:
            key = (kind, "__all__") if kind == "e2_full" else (kind, domain)
            paths = candidates.get(key, [])
            if not paths:
                raise MainTableError(
                    f"Missing complete result for strategy={kind}, domain={domain}"
                )
            expected_for_batch = list(DOMAINS) if kind == "e2_full" else [domain]
            batch_root, summary = _choose_batch(paths, seeds, expected_for_batch)
            cells[kind][domain] = _format_cell(summary, domain, batch_root)
            provenance[f"{kind}:{domain}"] = str(batch_root)

    header = ["Cross-Domain", *(DOMAIN_LABELS[domain] for domain in DOMAINS)]
    csv_rows = [
        [label, *(cells[kind][domain] for domain in DOMAINS)]
        for kind, label in STRATEGIES
    ]
    csv_lines = []
    csv_lines.append(",".join(header))
    for row in csv_rows:
        csv_lines.append(",".join('"' + str(value).replace('"', '""') + '"' for value in row))
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    md_lines.extend("| " + " | ".join(row) + " |" for row in csv_rows)
    paper_root.mkdir(parents=True, exist_ok=True)
    _write_atomic(paper_root / "main_table.csv", "\n".join(csv_lines) + "\n")
    _write_atomic(paper_root / "main_table.md", "\n".join(md_lines) + "\n")
    payload = {
        "complete": True,
        "label_rule": "bol_peak_mean_top5_first100_v1",
        "expected_seeds": seeds,
        "domains": list(DOMAINS),
        "strategies": [kind for kind, _ in STRATEGIES],
        "provenance": provenance,
    }
    _write_atomic(
        paper_root / "main_table.json",
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--expected_seeds", nargs="+", type=int, default=[42, 52, 62])
    args = parser.parse_args(argv)
    try:
        payload = build_main_table(args.output_root, args.expected_seeds)
    except MainTableError as exc:
        parser.error(str(exc))
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
