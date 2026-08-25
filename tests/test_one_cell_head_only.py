"""Contracts for Paper-v1 group-wise one-cell head-only adaptation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from UnifiedRawSOH.datasets.one_cell import (
    select_support_cell,
    stratified_support_split,
)
from UnifiedRawSOH.evaluation.paper_v1.summarize_one_cell import summarize_runtime
from UnifiedRawSOH.models.raw_soh_model import build_raw_soh_model
from UnifiedRawSOH.trainers.one_cell_head_only import (
    freeze_head_only,
    load_strict_lodo_model,
    run_job,
)
from UnifiedRawSOH.trainers.one_cell_launcher import (
    support_choices_for_checkpoint,
)
from UnifiedRawSOH.utils.config import load_config, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = (
    PROJECT_ROOT
    / "UnifiedRawSOH/configs/paper_v1/e3_cross_domain_reusability"
    / "one_cell_head_only"
)
E2_NO_CYCLE = (
    PROJECT_ROOT
    / "UnifiedRawSOH/configs/paper_v1/e2_unified_multidomain/unified"
    / "public_all_domains_domain_balanced_no_cycle_aux.json"
)
LAUNCHER = (
    PROJECT_ROOT
    / "UnifiedRawSOH/scripts/paper_v1/e3_cross_domain_reusability"
    / "run_one_cell_head_only.sh"
)


class _SupportDataset:
    def __init__(self, sohs):
        self.sohs = list(sohs)

    def __len__(self):
        return len(self.sohs)

    def __getitem__(self, index):
        return {
            "soh": torch.tensor([self.sohs[index]], dtype=torch.float32),
            "battery_id": "cell_a",
            "condition": "group_a",
            "cycle_id": index,
        }


def _raw_sample(index, group="g1", cell="test_1"):
    generator = torch.Generator().manual_seed(100 + index)
    return {
        "cc_signal": torch.randn(128, 2, generator=generator),
        "cv_signal": torch.randn(256, 2, generator=generator),
        "cc_mask": torch.ones(128),
        "cv_mask": torch.ones(256),
        "cc_time": torch.linspace(0, 10, 128),
        "cv_time": torch.linspace(10, 20, 256),
        "cc_temperature": torch.zeros(128, 2),
        "cv_temperature": torch.zeros(256, 2),
        "t0_temperature_norm": torch.zeros(1),
        "soh": torch.tensor([0.9 - 0.01 * index]),
        "battery_id": cell,
        "condition": group,
        "cycle_id": index,
    }


class OneCellSelectionTest(unittest.TestCase):
    def test_seed_rotation_is_stable_and_avoids_repeats_when_possible(self):
        config = {
            "one_cell": {
                "target_domain_id": "xjtu",
                "support_selection_mode": "stable_seed_rotation",
                "support_seeds": [42, 52, 62],
            }
        }
        inventory = {
            "development_cells_by_group": {
                "2C": ["cell_1", "cell_2", "cell_3", "cell_4"]
            },
            "all_test_cells": ["cell_test"],
        }
        first = [
            select_support_cell(config, inventory, "2C", seed)["support_cell"]
            for seed in (42, 52, 62)
        ]
        second = [
            select_support_cell(config, inventory, "2C", seed)["support_cell"]
            for seed in (42, 52, 62)
        ]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 3)

    def test_smarthealth_ab_follows_split_order(self):
        config = {
            "one_cell": {
                "target_domain_id": "smarthealth_eve280",
                "support_selection_mode": "ordered_ab",
            }
        }
        inventory = {
            "development_cells_by_group": {
                "0.5C-100%DOD": ["first_cell", "second_cell"]
            },
            "all_test_cells": ["test_cell"],
        }
        self.assertEqual(
            select_support_cell(
                config, inventory, "0.5C-100%DOD", "A"
            )["support_cell"],
            "first_cell",
        )
        self.assertEqual(
            select_support_cell(
                config, inventory, "0.5C-100%DOD", "B"
            )["support_cell"],
            "second_cell",
        )

    def test_support_validation_split_is_deterministic_and_nonempty(self):
        dataset = _SupportDataset(
            [0.99, 0.97, 0.95, 0.93, 0.90, 0.87, 0.84, 0.80, 0.76, 0.72]
        )
        first = stratified_support_split(
            dataset, range(len(dataset)), 0.2, seed=42
        )
        second = stratified_support_split(
            dataset, range(len(dataset)), 0.2, seed=42
        )
        self.assertEqual(first, second)
        self.assertTrue(first[0])
        self.assertTrue(first[1])
        self.assertFalse(set(first[0]) & set(first[1]))


class OneCellConfigTest(unittest.TestCase):
    def test_five_configs_define_57_support_choices_and_no_cycle_model(self):
        expected = {
            "one_cell_xjtu.json": 18,
            "one_cell_mit.json": 9,
            "one_cell_smarthealth_lishen40.json": 18,
            "one_cell_smarthealth_catl280.json": 6,
            "one_cell_smarthealth_eve280.json": 6,
        }
        total = 0
        paired_total = 0
        for filename, count in expected.items():
            config = load_config(CONFIG_ROOT / filename)
            one_cell = config["one_cell"]
            jobs = len(one_cell["support_groups"]) * len(
                one_cell["support_choices"]
            )
            self.assertEqual(jobs, count)
            self.assertFalse(config["model"]["use_cycle_prediction"])
            self.assertFalse(config["model"]["use_predicted_cycle_for_soh"])
            self.assertEqual(config["train"]["lambda_cycle"], 0.0)
            total += jobs
            paired_total += len(one_cell["support_groups"]) * sum(
                len(support_choices_for_checkpoint(config, seed))
                for seed in (42, 52, 62)
            )
        self.assertEqual(total, 57)
        self.assertEqual(paired_total, 117)

    def test_launcher_has_top_level_settings_and_needs_no_cli_arguments(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        for name in (
            "TARGET_DOMAINS",
            "GPU_IDS",
            "JOBS_PER_GPU",
            "PAIRED_SEEDS",
            "CHECKPOINT_ROOT_XJTU",
            "CHECKPOINT_ROOT_MIT",
            "CHECKPOINT_ROOT_LISHEN40",
            "CHECKPOINT_ROOT_CATL280",
            "CHECKPOINT_ROOT_EVE280",
            "DRY_RUN",
            "RESUME",
        ):
            self.assertIn(f"{name}=", text)
        self.assertNotIn("--checkpoint-", text)


class OneCellTrainingTest(unittest.TestCase):
    def _checkpoint(self, directory, target="xjtu", cycle=False):
        config = load_config(E2_NO_CYCLE)
        config["experiment"]["source_domain_ids"] = [
            value
            for value in config["experiment"]["domain_ids"]
            if value != target
        ]
        config["experiment"]["target_domain_id"] = target
        if cycle:
            config["model"]["use_cycle_prediction"] = True
            config["model"]["use_predicted_cycle_for_soh"] = True
            config["train"]["lambda_cycle"] = 0.0035
        model = build_raw_soh_model(
            config["model"], backend_override="torch_reference"
        )
        path = Path(directory) / "best.pt"
        torch.save({"model": model.state_dict(), "config": config}, path)
        return path

    def test_checkpoint_load_is_strict_no_cycle_and_target_unseen(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(CONFIG_ROOT / "one_cell_xjtu.json")
            path = self._checkpoint(directory)
            model, _, manifest = load_strict_lodo_model(
                config, path, "xjtu", backend_override="torch_reference"
            )
            trainable = freeze_head_only(model)
        self.assertTrue(manifest["strict_load"])
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("head.") for name in trainable))
        self.assertIsNone(model.cycle_head)
        self.assertIsNone(model.cycle_adapter)

    def test_cycle_aux_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(CONFIG_ROOT / "one_cell_xjtu.json")
            path = self._checkpoint(directory, cycle=True)
            with self.assertRaisesRegex(ValueError, "not a no-cycle"):
                load_strict_lodo_model(
                    config, path, "xjtu", backend_override="torch_reference"
                )

    def test_checkpoint_that_trained_on_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(CONFIG_ROOT / "one_cell_xjtu.json")
            path = self._checkpoint(directory)
            payload = torch.load(path, map_location="cpu")
            payload["config"]["experiment"]["source_domain_ids"].append("xjtu")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "trained on target"):
                load_strict_lodo_model(
                    config, path, "xjtu", backend_override="torch_reference"
                )

    def test_cpu_reference_smoke_completes_one_full_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(CONFIG_ROOT / "one_cell_xjtu.json")
            config["one_cell"]["support_groups"] = ["g1", "g2"]
            config["one_cell"]["epochs"] = 2
            config["one_cell"]["patience"] = 1
            config["one_cell"]["batch_size"] = 2
            config_path = root / "config.json"
            save_json(config_path, config)
            checkpoint = self._checkpoint(root)

            train = DataLoader([_raw_sample(0), _raw_sample(1)], batch_size=2)
            val = DataLoader([_raw_sample(2), _raw_sample(3)], batch_size=2)
            test = DataLoader(
                [
                    _raw_sample(4, "g1", "test_1"),
                    _raw_sample(5, "g1", "test_1"),
                    _raw_sample(6, "g2", "test_2"),
                    _raw_sample(7, "g2", "test_2"),
                ],
                batch_size=2,
            )
            selection = {
                "support_group": "g1",
                "support_choice": "42",
                "support_cell": "support_cell",
            }
            split = {"all_test_sample_count": 4}
            inventory = {"all_test_sample_count": 4}
            output = root / "job"
            from UnifiedRawSOH.trainers.one_cell_head_only import file_sha256
            job = {
                "config_path": str(config_path),
                "checkpoint_path": str(checkpoint),
                "checkpoint_seed": 42,
                "checkpoint_sha256": file_sha256(checkpoint),
                "support_group": "g1",
                "support_choice": "42",
                "support_cell": "support_cell",
                "output_dir": str(output),
            }
            with patch(
                "UnifiedRawSOH.trainers.one_cell_head_only.build_one_cell_loaders",
                return_value=(
                    {"train": train, "val": val, "test": test},
                    selection,
                    split,
                    inventory,
                ),
            ):
                status = run_job(
                    job,
                    backend_override="torch_reference",
                    device_override="cpu",
                )
            metrics = json.loads(
                (output / "metrics_overall.json").read_text(encoding="utf-8")
            )
        self.assertEqual(status["status"], "completed")
        self.assertEqual(
            metrics["encoder_sha256_before"],
            metrics["encoder_sha256_after"],
        )
        self.assertTrue(
            all(
                name.startswith("head.")
                for name in metrics["changed_parameter_names"]
            )
        )


class OneCellSummaryTest(unittest.TestCase):
    def test_summary_builds_support_to_test_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for support in ("g1", "g2"):
                for choice, offset in (("A", 0.0), ("B", 0.01)):
                    output = root / "jobs" / support / choice
                    output.mkdir(parents=True)
                    job = {
                        "job_id": f"target::{support}::{choice}",
                        "target_domain": "target",
                        "checkpoint_seed": 42 if choice == "A" else 52,
                        "support_group": support,
                        "support_choice": choice,
                        "support_cell": f"{support}_{choice}",
                        "output_dir": str(output),
                    }
                    jobs.append(job)
                    save_json(output / "status.json", {"status": "completed"})
                    save_json(
                        output / "metrics_overall.json",
                        {"mape": 0.1 + offset, "rmse": 0.2 + offset},
                    )
                    rows = [
                        {
                            "test_group": test,
                            "mape": 0.1 + offset,
                            "rmse": 0.2 + offset,
                        }
                        for test in ("g1", "g2")
                    ]
                    with (output / "metrics_by_test_group.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as handle:
                        writer = csv.DictWriter(
                            handle, fieldnames=list(rows[0])
                        )
                        writer.writeheader()
                        writer.writerows(rows)
            save_json(root / "job_manifest.json", {"jobs": jobs})
            result = summarize_runtime(root)
            matrix_exists = (
                root / "summary/target/rmse_support_to_test_group.csv"
            ).is_file()
            seed_summary_exists = (
                root / "summary/summary_by_checkpoint_seed.csv"
            ).is_file()
        self.assertEqual(result["status"], "completed")
        self.assertTrue(matrix_exists)
        self.assertTrue(seed_summary_exists)

    def test_incomplete_jobs_do_not_create_transfer_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job"
            output.mkdir()
            save_json(output / "status.json", {"status": "failed"})
            save_json(
                root / "job_manifest.json",
                {
                    "jobs": [
                        {
                            "job_id": "target::g1::A",
                            "target_domain": "target",
                            "checkpoint_seed": 42,
                            "support_group": "g1",
                            "support_choice": "A",
                            "support_cell": "cell_a",
                            "output_dir": str(output),
                        }
                    ]
                },
            )
            result = summarize_runtime(root)
            matrix_exists = (
                root / "summary/target/rmse_support_to_test_group.csv"
            ).exists()
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(matrix_exists)




if __name__ == "__main__":
    unittest.main()
