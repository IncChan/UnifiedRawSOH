"""Compare completed Paper-v1 E2 diagnostics without rerunning their methods."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from UnifiedRawSOH.utils.config import save_json


DEFAULT_BASELINE = "UnifiedRawSOH/outputs/Paper-v1/v1_diagnostics/e2_full_d"
DEFAULT_ABLATION = (
    "UnifiedRawSOH/outputs/Paper-v1/v1_diagnostics/e2_full_d_no_cycle_aux"
)


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _summary(values):
    values = [float(value) for value in values]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "values": values,
    }


def _comparison(baseline_values, ablation_values):
    baseline = _summary(baseline_values)
    ablation = _summary(ablation_values)
    return {
        "baseline": baseline,
        "no_cycle_aux": ablation,
        "delta_mean_no_cycle_aux_minus_baseline": (
            ablation["mean"] - baseline["mean"]
        ),
    }


def _pair_key(pair):
    return str(
        pair.get("pair_id")
        or f'{pair["domain_a"]}__vs__{pair["domain_b"]}'
    )


def _completed_pairs(report):
    return {
        _pair_key(pair): pair
        for pair in report.get("pairs", [])
        if pair.get("status", "completed") == "completed"
    }


def _read_seed(root, seed):
    seed_root = Path(root) / f"seed_{int(seed)}"
    required = {
        "probe": seed_root / "representation_pairwise_probe.json",
        "gradient": seed_root / "gradient_conflict.json",
        "calibration": seed_root / "residual_calibration.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing completed diagnostic artifacts:\n  " + "\n  ".join(missing)
        )
    return {name: _load(path) for name, path in required.items()}


def _compare_pair_metric(baseline_reports, ablation_reports, section, metric):
    baseline_pairs = [_completed_pairs(report[section]) for report in baseline_reports]
    ablation_pairs = [_completed_pairs(report[section]) for report in ablation_reports]
    common = set.intersection(*(set(value) for value in baseline_pairs + ablation_pairs))
    result = {}
    for pair_id in sorted(common):
        first = baseline_pairs[0][pair_id]
        result[pair_id] = {
            "domain_a": str(first["domain_a"]),
            "domain_b": str(first["domain_b"]),
            metric: _comparison(
                [pairs[pair_id][metric] for pairs in baseline_pairs],
                [pairs[pair_id][metric] for pairs in ablation_pairs],
            ),
        }
    return result


def compare_diagnostic_roots(baseline_root, ablation_root, seeds=(42, 52, 62)):
    """Build the fixed FULL-D versus no-cycle-aux comparison report."""

    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("seeds cannot be empty")
    baseline_reports = [_read_seed(baseline_root, seed) for seed in seeds]
    ablation_reports = [_read_seed(ablation_root, seed) for seed in seeds]

    scalar_metrics = {
        "pairwise_domain_probe_accuracy": ("probe", "accuracy"),
        "pairwise_domain_probe_macro_f1": ("probe", "macro_f1"),
        "gradient_negative_pair_fraction": ("gradient", "negative_pair_fraction"),
        "residual_before_domain_macro_rmse": (
            "calibration",
            "before_domain_macro_rmse",
        ),
        "residual_after_domain_macro_rmse": (
            "calibration",
            "after_domain_macro_rmse",
        ),
        "residual_domain_macro_rmse_change": (
            "calibration",
            "domain_macro_rmse_change",
        ),
    }
    overall = {
        name: _comparison(
            [report[section][metric] for report in baseline_reports],
            [report[section][metric] for report in ablation_reports],
        )
        for name, (section, metric) in scalar_metrics.items()
    }

    probe_by_pair = _compare_pair_metric(
        baseline_reports, ablation_reports, "probe", "accuracy"
    )
    probe_f1_by_pair = _compare_pair_metric(
        baseline_reports, ablation_reports, "probe", "macro_f1"
    )
    for pair_id, values in probe_by_pair.items():
        values["macro_f1"] = probe_f1_by_pair[pair_id]["macro_f1"]

    gradient_by_pair = _compare_pair_metric(
        baseline_reports, ablation_reports, "gradient", "cosine"
    )

    baseline_domains = [
        report["calibration"]["per_domain"] for report in baseline_reports
    ]
    ablation_domains = [
        report["calibration"]["per_domain"] for report in ablation_reports
    ]
    common_domains = set.intersection(
        *(set(value) for value in baseline_domains + ablation_domains)
    )
    residual_by_domain = {
        domain: {
            "rmse_change": _comparison(
                [value[domain]["rmse_change"] for value in baseline_domains],
                [value[domain]["rmse_change"] for value in ablation_domains],
            )
        }
        for domain in sorted(common_domains)
    }

    return {
        "definition": (
            "same three-seed diagnostics; deltas are no-cycle-aux minus baseline"
        ),
        "baseline_root": str(Path(baseline_root).resolve()),
        "no_cycle_aux_root": str(Path(ablation_root).resolve()),
        "seeds": seeds,
        "overall": overall,
        "pairwise_domain_probe_by_pair": probe_by_pair,
        "gradient_cosine_by_pair": gradient_by_pair,
        "residual_calibration_by_domain": residual_by_domain,
    }


def _write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(report, output_root):
    """Write one JSON report plus focused, paper-friendly CSV tables."""

    output_root = Path(output_root)
    save_json(output_root / "comparison_vs_e2_full_d.json", report)

    overall_rows = []
    for metric, values in report["overall"].items():
        overall_rows.append(
            {
                "metric": metric,
                "baseline_mean": values["baseline"]["mean"],
                "baseline_std": values["baseline"]["std"],
                "no_cycle_aux_mean": values["no_cycle_aux"]["mean"],
                "no_cycle_aux_std": values["no_cycle_aux"]["std"],
                "delta_mean": values["delta_mean_no_cycle_aux_minus_baseline"],
            }
        )
    _write_rows(output_root / "comparison_overall.csv", overall_rows)

    for section, filename in (
        ("pairwise_domain_probe_by_pair", "comparison_pairwise_probe_by_pair.csv"),
        ("gradient_cosine_by_pair", "comparison_gradient_cosine_by_pair.csv"),
    ):
        rows = []
        for pair_id, values in report[section].items():
            for metric, comparison in values.items():
                if metric in {"domain_a", "domain_b"}:
                    continue
                rows.append(
                    {
                        "pair_id": pair_id,
                        "domain_a": values["domain_a"],
                        "domain_b": values["domain_b"],
                        "metric": metric,
                        "baseline_mean": comparison["baseline"]["mean"],
                        "baseline_std": comparison["baseline"]["std"],
                        "no_cycle_aux_mean": comparison["no_cycle_aux"]["mean"],
                        "no_cycle_aux_std": comparison["no_cycle_aux"]["std"],
                        "delta_mean": comparison[
                            "delta_mean_no_cycle_aux_minus_baseline"
                        ],
                    }
                )
        _write_rows(output_root / filename, rows)

    residual_rows = []
    for domain, values in report["residual_calibration_by_domain"].items():
        comparison = values["rmse_change"]
        residual_rows.append(
            {
                "domain": domain,
                "baseline_mean_rmse_change": comparison["baseline"]["mean"],
                "baseline_std_rmse_change": comparison["baseline"]["std"],
                "no_cycle_aux_mean_rmse_change": comparison["no_cycle_aux"]["mean"],
                "no_cycle_aux_std_rmse_change": comparison["no_cycle_aux"]["std"],
                "delta_mean_rmse_change": comparison[
                    "delta_mean_no_cycle_aux_minus_baseline"
                ],
            }
        )
    _write_rows(
        output_root / "comparison_residual_calibration_by_domain.csv",
        residual_rows,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        "Compare E2-FULL-D baseline and w/o cycle auxiliary diagnostics"
    )
    parser.add_argument("--baseline_root", default=DEFAULT_BASELINE)
    parser.add_argument("--ablation_root", default=DEFAULT_ABLATION)
    parser.add_argument("--output_root", default=DEFAULT_ABLATION)
    parser.add_argument("--seed", type=int, action="append", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    report = compare_diagnostic_roots(
        args.baseline_root,
        args.ablation_root,
        seeds=args.seed or (42, 52, 62),
    )
    write_comparison(report, args.output_root)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
