"""Deterministic contracts for the Paper-v1 E2 diagnostic module."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.evaluation.paper_v1.compare_diagnostics import (  # noqa: E402
    compare_diagnostic_roots,
    write_comparison,
)
from UnifiedRawSOH.evaluation.paper_v1.domain_diagnostics import (  # noqa: E402
    _run_diagnostic_safely,
    aggregate_from_config,
    battery_group_split,
    fit_affine_calibration,
    gradient_cosine_report,
    matched_health_indices,
    run_pairwise_representation_probe,
    run_strict_representation_probe,
)
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


DIAGNOSTIC_ROOT = PROJECT_ROOT / "UnifiedRawSOH/configs/paper_v1/diagnostics"


class V1DiagnosticsTest(unittest.TestCase):
    def test_parallel_launcher_assigns_gpus_by_seed(self):
        launcher = (
            PROJECT_ROOT
            / "UnifiedRawSOH/scripts/paper_v1/diagnostics/run_e2_diagnostics.sh"
        )
        environment = dict(os.environ)
        environment.update(
            {
                "DRY_RUN": "1",
                "GPU_IDS": "6 7",
                "MAX_PARALLEL": "3",
                "SEEDS": "42 52 62",
            }
        )
        result = subprocess.run(
            ["bash", str(launcher)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        expected = {42: "6", 52: "7", 62: "6"}
        for seed, gpu in expected.items():
            self.assertIn(f"[seed worker] seed={seed}; gpu={gpu}", result.stdout)
            self.assertIn(f"e2_full_b; seed={seed}; gpu={gpu}", result.stdout)
            self.assertIn(f"e2_full_d; seed={seed}; gpu={gpu}", result.stdout)

    def test_no_cycle_aux_launcher_schedules_only_full_d_three_seeds(self):
        launcher = (
            PROJECT_ROOT
            / "UnifiedRawSOH/scripts/paper_v1/diagnostics"
            / "run_e2_full_d_no_cycle_aux_diagnostics.sh"
        )
        environment = dict(os.environ)
        environment.update(
            {
                "DRY_RUN": "1",
                "GPU_IDS": "4 5 6",
                "MAX_PARALLEL": "3",
            }
        )
        result = subprocess.run(
            ["bash", str(launcher)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        expected = {42: "4", 52: "5", 62: "6"}
        for seed, gpu in expected.items():
            self.assertIn(
                f"e2_full_d_no_cycle_aux; seed={seed}; gpu={gpu}",
                result.stdout,
            )
        self.assertIn(
            "[aggregate] e2_full_d_no_cycle_aux; seeds=42 52 62",
            result.stdout,
        )
        self.assertNotIn("[worker] e2_full_b;", result.stdout)
        self.assertNotIn("[worker] e2_full_d; seed=", result.stdout)

    def test_single_device_can_schedule_three_seed_models(self):
        launcher = (
            PROJECT_ROOT
            / "UnifiedRawSOH/scripts/paper_v1/diagnostics/run_e2_diagnostics.sh"
        )
        environment = dict(os.environ)
        environment.pop("GPU_IDS", None)
        environment.update(
            {
                "DRY_RUN": "1",
                "DIAGNOSTICS": "e2_full_b",
                "GPU_ID": "7",
                "MAX_PARALLEL": "3",
                "SEEDS": "42 52 62",
            }
        )
        result = subprocess.run(
            ["bash", str(launcher)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("max parallel seeds: 3", result.stdout)
        for seed in (42, 52, 62):
            self.assertIn(f"e2_full_b; seed={seed}; gpu=7", result.stdout)
        self.assertIn("[aggregate] e2_full_b; seeds=42 52 62", result.stdout)

    def test_seed42_worker_can_aggregate_all_existing_seeds(self):
        launcher = (
            PROJECT_ROOT
            / "UnifiedRawSOH/scripts/paper_v1/diagnostics/run_e2_diagnostics.sh"
        )
        environment = dict(os.environ)
        environment.update(
            {
                "DRY_RUN": "1",
                "GPU_IDS": "6 7",
                "MAX_PARALLEL": "2",
                "SEEDS": "42",
                "SUMMARY_SEEDS": "42 52 62",
            }
        )
        result = subprocess.run(
            ["bash", str(launcher)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("worker seeds: 42", result.stdout)
        self.assertIn("summary seeds: 42 52 62", result.stdout)
        self.assertNotIn("[worker] e2_full_b; seed=52", result.stdout)
        self.assertIn("[aggregate] e2_full_b; seeds=42 52 62", result.stdout)

    def test_battery_group_split_has_no_battery_leakage(self):
        domains = np.asarray(["a"] * 6 + ["b"] * 6)
        batteries = np.asarray(
            ["a1", "a1", "a2", "a2", "a3", "a3", "b1", "b1", "b2", "b2", "b3", "b3"]
        )
        train, test, assignment = battery_group_split(domains, batteries, 0.34, seed=7)
        self.assertTrue(np.all(train ^ test))
        for domain in ("a", "b"):
            self.assertFalse(
                set(assignment[domain]["train_batteries"])
                & set(assignment[domain]["test_batteries"])
            )
            self.assertTrue(np.any(train & (domains == domain)))
            self.assertTrue(np.any(test & (domains == domain)))

    def test_health_matching_equalizes_domains_inside_each_bin(self):
        truths = np.asarray([0.805, 0.812, 0.818, 0.806, 0.811, 0.905, 0.912, 0.906, 0.914])
        domains = np.asarray(["a", "a", "a", "b", "b", "a", "a", "b", "b"])
        selected, _ = matched_health_indices(
            truths,
            domains,
            bin_width=0.1,
            seed=11,
            require_all_domains=True,
        )
        bins = np.floor(truths[selected] / 0.1).astype(int)
        counts = defaultdict(Counter)
        for bin_id, domain in zip(bins, domains[selected]):
            counts[int(bin_id)][str(domain)] += 1
        self.assertEqual(set(counts), {8, 9})
        for values in counts.values():
            self.assertEqual(values["a"], values["b"])

    def test_pairwise_probe_survives_missing_all_domain_overlap(self):
        supports = {
            "a": (0.81, 0.91),
            "b": (0.91, 1.01),
            "c": (0.81, 1.01),
        }
        features = []
        truths = []
        domains = []
        batteries = []
        for domain_index, (domain, values) in enumerate(supports.items()):
            for battery_index in range(4):
                for truth in values:
                    feature = np.zeros(4, dtype=np.float32)
                    feature[domain_index] = 1.0
                    feature[-1] = truth
                    features.append(feature)
                    truths.append(truth)
                    domains.append(domain)
                    batteries.append(f"{domain}{battery_index}")
        data = {
            "features": np.asarray(features),
            "truth": np.asarray(truths),
            "domain": np.asarray(domains),
            "battery": np.asarray(batteries),
        }
        config = {
            "soh_bin_width": 0.1,
            "soh_bin_origin": 0.0,
            "max_per_domain_bin": 16,
            "pairwise_split_search_attempts": 4,
            "require_all_domains": True,
            "test_battery_fraction": 0.25,
            "probe_device": "cpu",
            "epochs": 30,
            "learning_rate": 0.1,
            "weight_decay": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "selected no samples"):
                run_strict_representation_probe(data, config, 42, directory)
            result = run_pairwise_representation_probe(data, config, 42, directory)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["n_pairs_total"], 3)
        self.assertEqual(result["n_pairs_completed"], 3)
        self.assertGreater(result["accuracy"], result["chance_accuracy"])

    def test_unavailable_diagnostic_does_not_block_following_call(self):
        unavailable, status = _run_diagnostic_safely(
            "synthetic_failure",
            lambda: (_ for _ in ()).throw(ValueError("no common support")),
        )
        completed, completed_status = _run_diagnostic_safely(
            "synthetic_success", lambda: {"value": 1}
        )
        self.assertIsNone(unavailable)
        self.assertEqual(status["status"], "unavailable")
        self.assertEqual(completed, {"value": 1})
        self.assertEqual(completed_status["status"], "completed")

    def test_affine_calibration_recovers_known_domain_bias(self):
        prediction = np.asarray([0.1, 0.3, 0.5, 0.8, 1.0])
        truth = 1.25 * prediction - 0.07
        scale, bias = fit_affine_calibration(prediction, truth, ridge=0.0)
        self.assertAlmostEqual(scale, 1.25, places=10)
        self.assertAlmostEqual(bias, -0.07, places=10)

    def test_gradient_report_marks_negative_pairs(self):
        report = gradient_cosine_report(
            {
                "a": np.asarray([1.0, 0.0]),
                "b": np.asarray([-1.0, 0.0]),
                "c": np.asarray([0.0, 1.0]),
            }
        )
        self.assertEqual(report["n_pairs"], 3)
        self.assertEqual(report["negative_pair_count"], 1)
        self.assertAlmostEqual(report["negative_pair_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(report["mean_pairwise_cosine"], -1.0 / 3.0)

    def test_configs_are_validation_only_and_write_outside_source_runs(self):
        configs = [
            load_config(DIAGNOSTIC_ROOT / "e2_full_b.json"),
            load_config(DIAGNOSTIC_ROOT / "e2_full_d.json"),
            load_config(DIAGNOSTIC_ROOT / "e2_full_d_no_cycle_aux.json"),
        ]
        self.assertEqual(
            len({config["diagnostic"]["name"] for config in configs}),
            len(configs),
        )
        for config in configs:
            diagnostic = config["diagnostic"]
            self.assertEqual(config["status"], "runnable")
            self.assertEqual(diagnostic["split"], "val")
            self.assertNotEqual(diagnostic["run_root"], diagnostic["output_root"])
            self.assertIn("outputs/Paper-v1/v1_diagnostics/", diagnostic["output_root"])
            self.assertEqual(diagnostic["seeds"], [42, 52, 62])
            self.assertIn("representation_probe", diagnostic)
            self.assertEqual(
                diagnostic["representation_probe"]["pairwise_split_search_attempts"], 64
            )
            self.assertIn("residual_calibration", diagnostic)
            self.assertIn("gradient_conflict", diagnostic)

        full_d = configs[1]["diagnostic"]
        no_cycle_aux = configs[2]["diagnostic"]
        for key in (
            "seeds",
            "split",
            "device",
            "max_samples_per_domain",
            "save_features",
            "representation_probe",
            "residual_calibration",
            "gradient_conflict",
        ):
            self.assertEqual(no_cycle_aux[key], full_d[key], key)
        self.assertIn("RawMamba-noCycleAux", no_cycle_aux["run_root"])
        self.assertIn("runtime_260825-022226", no_cycle_aux["run_root"])
        self.assertTrue(
            no_cycle_aux["output_root"].endswith("e2_full_d_no_cycle_aux")
        )

    def test_aggregate_only_combines_completed_seed_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "diagnostics"
            source.mkdir()
            config = {
                "status": "runnable",
                "diagnostic": {
                    "name": "synthetic",
                    "split": "val",
                    "run_root": str(source),
                    "output_root": str(output),
                },
            }
            for seed, accuracy in ((42, 0.4), (52, 0.6)):
                seed_output = output / f"seed_{seed}"
                seed_output.mkdir(parents=True)
                report = {
                    "representation_probe": {"accuracy": accuracy, "macro_f1": accuracy},
                    "residual_calibration": {
                        "before_domain_macro_rmse": 0.02,
                        "after_domain_macro_rmse": 0.01,
                        "domain_macro_rmse_change": -0.01,
                    },
                    "gradient_conflict": {
                        "negative_pair_fraction": 0.25,
                        "mean_pairwise_cosine": 0.1,
                    },
                }
                (seed_output / "diagnostic_report.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
            summary = aggregate_from_config(
                config, repo_root=root, seed_override=[42, 52]
            )
            summary_exists = (output / "diagnostic_summary.json").is_file()
        self.assertAlmostEqual(summary["aggregate"]["domain_probe_accuracy"]["mean"], 0.5)
        self.assertEqual(summary["seeds"], [42, 52])
        self.assertTrue(summary_exists)

    def test_aggregate_upgrades_legacy_probe_from_feature_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "diagnostics"
            seed_output = output / "seed_42"
            source.mkdir()
            seed_output.mkdir(parents=True)
            features = []
            truths = []
            domains = []
            batteries = []
            for domain_index, domain in enumerate(("a", "b")):
                for battery_index in range(4):
                    feature = np.zeros(2, dtype=np.float32)
                    feature[domain_index] = 1.0
                    features.append(feature)
                    truths.append(0.95)
                    domains.append(domain)
                    batteries.append(f"{domain}{battery_index}")
            np.savez_compressed(
                seed_output / "validation_features.npz",
                features=np.asarray(features),
                truth=np.asarray(truths),
                domain=np.asarray(domains),
                battery=np.asarray(batteries),
            )
            legacy = {
                "representation_probe": {
                    "definition": "linear domain probe on SOH-bin-matched z_health",
                    "accuracy": 0.75,
                    "macro_f1": 0.74,
                }
            }
            (seed_output / "diagnostic_report.json").write_text(
                json.dumps(legacy), encoding="utf-8"
            )
            config = {
                "status": "runnable",
                "diagnostic": {
                    "name": "synthetic",
                    "split": "val",
                    "run_root": str(source),
                    "output_root": str(output),
                    "representation_probe": {
                        "soh_bin_width": 0.02,
                        "pairwise_split_search_attempts": 2,
                        "test_battery_fraction": 0.25,
                        "probe_device": "cpu",
                        "epochs": 20,
                        "learning_rate": 0.1,
                        "weight_decay": 0.0,
                    },
                },
            }
            summary = aggregate_from_config(
                config, repo_root=root, seed_override=[42], skip_gradients=True
            )
            upgraded = json.loads(
                (seed_output / "diagnostic_report.json").read_text(encoding="utf-8")
            )
            strict_file_exists = (
                seed_output / "representation_strict_probe.json"
            ).is_file()
        self.assertIn("pairwise", upgraded["representation_probe"]["definition"])
        self.assertEqual(upgraded["representation_strict_probe"]["accuracy"], 0.75)
        self.assertEqual(summary["aggregate"]["strict_domain_probe_accuracy"]["mean"], 0.75)
        self.assertTrue(strict_file_exists)

    def test_full_d_no_cycle_aux_comparison_reports_pair_level_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            ablation = root / "ablation"
            for run_root, offset in ((baseline, 0.0), (ablation, 0.2)):
                seed_root = run_root / "seed_42"
                seed_root.mkdir(parents=True)
                artifacts = {
                    "representation_pairwise_probe.json": {
                        "accuracy": 0.6 + offset,
                        "macro_f1": 0.5 + offset,
                        "pairs": [
                            {
                                "pair_id": "a__vs__b",
                                "domain_a": "a",
                                "domain_b": "b",
                                "accuracy": 0.6 + offset,
                                "macro_f1": 0.5 + offset,
                                "status": "completed",
                            }
                        ],
                    },
                    "gradient_conflict.json": {
                        "negative_pair_fraction": 0.3 + offset,
                        "pairs": [
                            {
                                "domain_a": "a",
                                "domain_b": "b",
                                "cosine": -0.3 + offset,
                            }
                        ],
                    },
                    "residual_calibration.json": {
                        "before_domain_macro_rmse": 0.02 + offset,
                        "after_domain_macro_rmse": 0.01 + offset,
                        "domain_macro_rmse_change": -0.01 + offset,
                        "per_domain": {
                            "a": {"rmse_change": -0.01 + offset},
                            "b": {"rmse_change": -0.02 + offset},
                        },
                    },
                }
                for filename, payload in artifacts.items():
                    (seed_root / filename).write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
            report = compare_diagnostic_roots(
                baseline, ablation, seeds=[42]
            )
            write_comparison(report, ablation)
            json_exists = (ablation / "comparison_vs_e2_full_d.json").is_file()
            pair_csv_exists = (
                ablation / "comparison_gradient_cosine_by_pair.csv"
            ).is_file()

        self.assertAlmostEqual(
            report["overall"]["gradient_negative_pair_fraction"][
                "delta_mean_no_cycle_aux_minus_baseline"
            ],
            0.2,
        )
        self.assertAlmostEqual(
            report["gradient_cosine_by_pair"]["a__vs__b"]["cosine"][
                "delta_mean_no_cycle_aux_minus_baseline"
            ],
            0.2,
        )
        self.assertTrue(json_exists)
        self.assertTrue(pair_csv_exists)


if __name__ == "__main__":
    unittest.main()
