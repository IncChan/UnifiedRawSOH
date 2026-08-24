"""Deterministic contracts for the Paper-v1 E2 diagnostic module."""

from __future__ import annotations

from collections import Counter, defaultdict
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.evaluation.paper_v1.domain_diagnostics import (  # noqa: E402
    battery_group_split,
    fit_affine_calibration,
    gradient_cosine_report,
    matched_health_indices,
)
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


DIAGNOSTIC_ROOT = PROJECT_ROOT / "UnifiedRawSOH/configs/paper_v1/diagnostics"


class V1DiagnosticsTest(unittest.TestCase):
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
        ]
        self.assertNotEqual(configs[0]["diagnostic"]["name"], configs[1]["diagnostic"]["name"])
        for config in configs:
            diagnostic = config["diagnostic"]
            self.assertEqual(config["status"], "runnable")
            self.assertEqual(diagnostic["split"], "val")
            self.assertNotEqual(diagnostic["run_root"], diagnostic["output_root"])
            self.assertIn("outputs/Paper-v1/v1_diagnostics/", diagnostic["output_root"])
            self.assertEqual(diagnostic["seeds"], [42, 52, 62])
            self.assertIn("representation_probe", diagnostic)
            self.assertIn("residual_calibration", diagnostic)
            self.assertIn("gradient_conflict", diagnostic)


if __name__ == "__main__":
    unittest.main()
