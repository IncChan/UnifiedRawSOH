"""Regression tests for E2 per-domain result reporting."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.trainers.c5b_trainer import (  # noqa: E402
    _domain_metrics_with_sample_counts,
    build_test_metrics_by_domain,
)


SUMMARIZER = PROJECT_ROOT / "UnifiedRawSOH" / "scripts" / "summarize_batch_runs.py"


def _metrics(rmse):
    mse = rmse * rmse
    return {
        "mae": rmse / 2.0,
        "mape": rmse / 3.0,
        "mse": mse,
        "rmse": rmse,
    }


class E2ResultReportingTest(unittest.TestCase):
    def test_domain_report_uses_e1_aligned_test_loss_fields(self):
        per_domain = _domain_metrics_with_sample_counts(
            truths=[1.0, 0.8, 0.9],
            predictions=[0.9, 0.7, 0.95],
            domains=["xjtu", "xjtu", "mit"],
        )
        report = build_test_metrics_by_domain({"per_domain": per_domain})

        self.assertEqual(report["domains"]["xjtu"]["n_samples"], 2)
        self.assertEqual(report["domains"]["mit"]["n_samples"], 1)
        for values in report["domains"].values():
            self.assertAlmostEqual(values["loss"], values["mse"])
            self.assertAlmostEqual(values["soh_loss"], values["mse"])

    def test_batch_summary_writes_one_mean_std_result_per_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            (batch_root / "run_manifest.json").write_text(
                json.dumps({"experiment": {"domain_ids": ["xjtu", "mit"]}}),
                encoding="utf-8",
            )
            for seed, xjtu_rmse, mit_rmse in (
                (42, 0.10, 0.20),
                (52, 0.30, 0.40),
            ):
                run_dir = batch_root / f"seed_{seed}"
                run_dir.mkdir()
                payload = {
                    "mae": 0.01,
                    "mape": 0.02,
                    "mse": 0.0001,
                    "rmse": 0.01,
                    "loss": 0.0001,
                    "per_domain": {
                        "xjtu": {**_metrics(xjtu_rmse), "n_samples": 12},
                        "mit": {**_metrics(mit_rmse), "n_samples": 8},
                    },
                }
                (run_dir / "test_metrics.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

            subprocess.run(
                [
                    sys.executable,
                    str(SUMMARIZER),
                    "--batch_root",
                    str(batch_root),
                    "--expected_seeds",
                    "42",
                    "52",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            per_domain_path = batch_root / "summary_per_domain_mean_std.json"
            summary = json.loads(per_domain_path.read_text(encoding="utf-8"))
            xjtu = summary["summary"]["xjtu"]
            mit = summary["summary"]["mit"]
            self.assertEqual(xjtu["seed_count"], 2)
            self.assertEqual(xjtu["test_sample_counts"]["value"], 12)
            self.assertTrue(xjtu["test_sample_counts"]["consistent"])
            self.assertTrue(math.isclose(xjtu["metrics"]["rmse"]["mean"], 0.20))
            self.assertTrue(math.isclose(mit["metrics"]["rmse"]["mean"], 0.30))
            self.assertTrue(math.isclose(mit["metrics"]["loss"]["mean"], 0.10))

            total = json.loads((batch_root / "summary_mean_std.json").read_text(encoding="utf-8"))
            self.assertIn("per_domain_summary", total)
            self.assertTrue((batch_root / "summary_per_domain_mean_std.csv").is_file())


if __name__ == "__main__":
    unittest.main()

