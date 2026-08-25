"""Contracts for runnable Paper-v1 no-cycle five-fold LODO."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.loaders import build_lodo_loaders  # noqa: E402
from UnifiedRawSOH.trainers.reusability import parse_reusability_protocol  # noqa: E402
from UnifiedRawSOH.utils.config import load_config  # noqa: E402


CONFIG_ROOT = (
    PROJECT_ROOT
    / "UnifiedRawSOH/configs/paper_v1/e3_cross_domain_reusability"
    / "leave_one_domain_out"
)
E2_CONFIG = (
    PROJECT_ROOT
    / "UnifiedRawSOH/configs/paper_v1/e2_unified_multidomain/unified"
    / "public_all_domains_domain_balanced_no_cycle_aux.json"
)
LAUNCHER = (
    PROJECT_ROOT
    / "UnifiedRawSOH/scripts/paper_v1/e3_cross_domain_reusability"
    / "run_lodo_no_cycle_aux.sh"
)
SUMMARIZER = PROJECT_ROOT / "UnifiedRawSOH/scripts/summarize_batch_runs.py"
DOMAINS = (
    "xjtu",
    "mit",
    "smarthealth_lishen40",
    "smarthealth_catl280",
    "smarthealth_eve280",
)


class E3LodoNoCycleAuxTest(unittest.TestCase):
    def test_all_folds_inherit_no_cycle_model_and_original_splits(self):
        e2 = load_config(E2_CONFIG)
        for target in DOMAINS:
            config = load_config(
                CONFIG_ROOT / f"lodo_no_cycle_aux_{target}.json"
            )
            protocol = parse_reusability_protocol(config)
            self.assertEqual(config["status"], "runnable")
            self.assertEqual(config["experiment"]["loader"], "leave_one_domain_out")
            self.assertEqual(protocol["target_domain_ids"], [target])
            self.assertEqual(
                set(protocol["source_domain_ids"]),
                set(DOMAINS) - {target},
            )
            self.assertEqual(config["experiment"]["domain_ids"], list(DOMAINS))
            self.assertEqual(
                config["data"]["split_files"],
                e2["data"]["split_files"],
            )
            self.assertEqual(
                config["data"]["balance_mode"],
                "domain_battery_hierarchical",
            )
            self.assertFalse(config["model"]["use_cycle_prediction"])
            self.assertFalse(config["model"]["use_predicted_cycle_for_soh"])
            self.assertEqual(config["train"]["lambda_cycle"], 0.0)
            self.assertEqual(config["train"]["cycle_loss_mode"], "disabled")

    def test_loader_routes_only_source_train_val_and_target_test(self):
        config = load_config(
            CONFIG_ROOT / "lodo_no_cycle_aux_xjtu.json"
        )

        def fake_build(
            domain_config,
            repo_root,
            seed,
            domain_id,
            data_root,
        ):
            datasets = {
                split: [
                    {
                        "domain_id": domain_id,
                        "dataset_id": domain_id,
                        "battery_id": f"{domain_id}_{split}",
                        "split_marker": split,
                    }
                ]
                for split in ("train", "val", "test")
            }
            return datasets, {"domain_id": domain_id}

        with patch(
            "UnifiedRawSOH.datasets.loaders._build_raw_domain",
            side_effect=fake_build,
        ):
            loaders, split_info = build_lodo_loaders(
                config, PROJECT_ROOT, seed=42
            )

        train_items = [
            loaders["train"].dataset[index]
            for index in range(len(loaders["train"].dataset))
        ]
        val_items = [
            loaders["val"].dataset[index]
            for index in range(len(loaders["val"].dataset))
        ]
        test_items = [
            loaders["test"].dataset[index]
            for index in range(len(loaders["test"].dataset))
        ]
        self.assertEqual(
            {item["domain_id"] for item in train_items},
            set(DOMAINS) - {"xjtu"},
        )
        self.assertEqual(
            {item["domain_id"] for item in val_items},
            set(DOMAINS) - {"xjtu"},
        )
        self.assertEqual({item["domain_id"] for item in test_items}, {"xjtu"})
        self.assertEqual({item["split_marker"] for item in train_items}, {"train"})
        self.assertEqual({item["split_marker"] for item in val_items}, {"val"})
        self.assertEqual({item["split_marker"] for item in test_items}, {"test"})
        self.assertEqual(split_info["target_domain_id"], "xjtu")

    def test_launcher_selects_exactly_one_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment.update(
                {
                    "DRY_RUN": "1",
                    "LEFT_OUT_DOMAIN": "eve280",
                    "SEEDS": "42",
                    "GPU_IDS": "4",
                    "MAX_PARALLEL": "1",
                    "OUTPUT_ROOT": directory,
                    "PYTHON_BIN": sys.executable,
                }
            )
            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn(
            "[LODO fold] left_out=smarthealth_eve280",
            result.stdout,
        )
        self.assertIn("lodo_no_cycle_aux_smarthealth_eve280.json", result.stdout)
        self.assertNotIn("[LODO fold] left_out=xjtu", result.stdout)
        self.assertIn("[launch] module=UnifiedRawSOH.main seed=42 gpu=4", result.stdout)

    def test_all_launcher_assigns_one_fold_per_gpu_with_per_gpu_limit(self):
        expected = {
            "xjtu": "0",
            "mit": "1",
            "smarthealth_lishen40": "2",
            "smarthealth_catl280": "3",
            "smarthealth_eve280": "4",
        }
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment.update(
                {
                    "DRY_RUN": "1",
                    "LEFT_OUT_DOMAIN": "all",
                    "SEEDS": "42 52 62",
                    "GPU_IDS": "0 1 2 3 4",
                    "MAX_PARALLEL": "3",
                    "OUTPUT_ROOT": directory,
                    "PYTHON_BIN": sys.executable,
                }
            )
            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("max parallel processes per GPU: 3", result.stdout)
        self.assertIn("maximum aggregate processes: 15", result.stdout)
        for fold, gpu in expected.items():
            self.assertIn(
                f"[schedule] gpu={gpu}; fold={fold}; "
                "seeds=42 52 62; max_parallel=3",
                result.stdout,
            )
            self.assertIn(
                f"[LODO fold] left_out={fold}; gpu={gpu};",
                result.stdout,
            )
            for seed in (42, 52, 62):
                self.assertIn(
                    f"[launch] module=UnifiedRawSOH.main seed={seed} gpu={gpu}",
                    result.stdout,
                )

    def test_lodo_summary_requires_only_target_test_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            manifest = {
                "experiment": {
                    "loader": "leave_one_domain_out",
                    "domain_ids": list(DOMAINS),
                    "source_domain_ids": [value for value in DOMAINS if value != "mit"],
                    "target_domain_id": "mit",
                }
            }
            (batch_root / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            run_dir = batch_root / "seed_42"
            run_dir.mkdir()
            metrics = {
                "mae": 0.01,
                "mape": 0.01,
                "mse": 0.0004,
                "rmse": 0.02,
                "loss": 0.0004,
                "per_domain": {
                    "mit": {
                        "mae": 0.01,
                        "mape": 0.01,
                        "mse": 0.0004,
                        "rmse": 0.02,
                        "n_samples": 10,
                    }
                },
            }
            (run_dir / "test_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SUMMARIZER),
                    "--batch_root",
                    str(batch_root),
                    "--expected_seeds",
                    "42",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(
                (batch_root / "summary_per_domain_mean_std.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(summary["expected_domain_ids"], ["mit"])
        self.assertEqual(set(summary["summary"]), {"mit"})


if __name__ == "__main__":
    unittest.main()
