"""Paper-v2 BOL labels, configuration, routing, and launcher contracts."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT / "UnifiedRawSOH"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.loaders import _make_balanced_sampler, build_lodo_loaders
from UnifiedRawSOH.datasets.soh_labels import (
    BOL_LABEL_MODE,
    BOL_REFERENCE_SOURCE,
    BOL_RULE_VERSION,
    BOLReferenceError,
    apply_bol_relative_soh,
    build_bol_reference,
    build_bol_references,
    frozen_smarthealth_bol_references,
)
from UnifiedRawSOH.preprocess.smarthealth_bol import (
    build_frozen_smarthealth_bol_reference,
)
from UnifiedRawSOH.models.c5b_model import build_c5b_model
from UnifiedRawSOH.models.baselines.pinn4soh_no_leak_onlyf import (
    StatFeatureDataset,
    build_feature_lodo_loaders,
)
from UnifiedRawSOH.utils.config import load_config


DOMAINS = (
    "xjtu",
    "mit",
    "smarthealth_lishen40",
    "smarthealth_catl280",
    "smarthealth_eve280",
)
V2_CONFIG_ROOT = REPO_ROOT / "configs" / "paper_v2"
V2_RAW_CONFIG = V2_CONFIG_ROOT / "e1_single_domain" / "raw_mamba" / "xjtu.json"
V1_CONFIG = (
    REPO_ROOT
    / "configs"
    / "paper_v1"
    / "e1_raw_soh_learning"
    / "benchmark"
    / "raw_mamba_xjtu.json"
)
LAUNCHER = REPO_ROOT / "scripts" / "paper_v2" / "run_bol_soh_retraining.sh"


def xjtu_records(capacities, cell="cell-a"):
    return [
        {
            "domain_id": "xjtu",
            "battery_id": cell,
            "cycle": index + 1,
            "SOH": float(capacity),
            "soh": float(capacity) / 2.0,
        }
        for index, capacity in enumerate(capacities)
    ]


def smarthealth_records(capacities, sources, cell="smart-a"):
    return [
        {
            "domain_id": "smarthealth_lishen40",
            "battery_id": cell,
            "cycle": index + 1,
            "label_capacity_Ah": float(capacity),
            "label_source": source,
        }
        for index, (capacity, source) in enumerate(zip(capacities, sources))
    ]


class _MetadataDataset(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class PaperV2BOLTest(unittest.TestCase):
    def test_top5_comes_from_first_100_and_is_numeric(self):
        capacities = [1.0 + 0.001 * index for index in range(100)]
        reference = build_bol_reference(xjtu_records(capacities + [9.0]))
        expected = sum(capacities[-5:]) / 5.0
        self.assertEqual(reference["rule_version"], BOL_RULE_VERSION)
        self.assertAlmostEqual(reference["Q_ref"], expected, places=12)
        self.assertEqual(sorted(reference["selected_cycle_ids"]), [96, 97, 98, 99, 100])
        self.assertEqual(reference["candidate_count"], 100)

    def test_observations_after_reference_window_do_not_change_qref(self):
        capacities = [1.0 + 0.001 * index for index in range(100)]
        first = build_bol_reference(xjtu_records(capacities))
        changed = build_bol_reference(xjtu_records(capacities + [100.0, 200.0]))
        self.assertEqual(first["Q_ref"], changed["Q_ref"])
        self.assertEqual(first["selected_cycle_ids"], changed["selected_cycle_ids"])

    def test_each_physical_cell_gets_its_own_reference(self):
        records = xjtu_records([1.0 + 0.001 * index for index in range(10)], "cell-a")
        records += xjtu_records([2.0 + 0.001 * index for index in range(10)], "cell-b")
        references = build_bol_references(records)
        self.assertEqual(set(references), {"cell-a", "cell-b"})
        self.assertNotEqual(references["cell-a"]["Q_ref"], references["cell-b"]["Q_ref"])

    def test_smarthealth_interpolation_cannot_define_peak(self):
        capacities = [40, 39, 38, 37, 36] + [100.0] * 95
        sources = ["calibration_direct"] * 5 + ["calibration_interpolated"] * 95
        reference = build_bol_reference(smarthealth_records(capacities, sources))
        self.assertAlmostEqual(reference["Q_ref"], 38.0)
        self.assertEqual(reference["selected_cycle_ids"], [1, 2, 3, 4, 5])
        status = reference["smarthealth_direct_interpolated_status"]
        self.assertEqual(status["direct_count_in_reference_window"], 5)
        self.assertEqual(status["interpolated_count_in_reference_window"], 95)

    def test_smarthealth_expands_only_until_fifth_direct(self):
        capacities = [40, 39, 38] + [80.0] * 97 + [37, 36, 1000]
        sources = (
            ["calibration_direct"] * 3
            + ["calibration_interpolated"] * 97
            + ["calibration_direct"] * 2
            + ["calibration_direct"]
        )
        reference = build_bol_reference(smarthealth_records(capacities, sources))
        self.assertAlmostEqual(reference["Q_ref"], 38.0)
        self.assertEqual(reference["reference_window_end_cycle"], 102)
        self.assertEqual(reference["selected_cycle_ids"], [1, 2, 3, 101, 102])
        self.assertEqual(reference["candidate_count"], 5)

    def test_smarthealth_expands_after_mad_rejects_one_of_first_five(self):
        capacities = [42.0, 39.1, 39.0] + [80.0] * 97 + [38.9, 38.8, 38.7]
        sources = (
            ["calibration_direct"] * 3
            + ["calibration_interpolated"] * 97
            + ["calibration_direct"] * 3
        )
        reference = build_bol_reference(smarthealth_records(capacities, sources))
        self.assertAlmostEqual(reference["Q_ref"], (39.1 + 39.0 + 38.9 + 38.8 + 38.7) / 5.0)
        self.assertEqual(reference["candidate_count"], 6)
        self.assertEqual(reference["valid_candidate_count_after_outlier_filter"], 5)
        self.assertEqual(reference["reference_window_end_cycle"], 103)
        self.assertTrue(reference["reference_window_expanded_after_mad"])
        self.assertEqual([item["cycle_id"] for item in reference["rejected_outliers"]], [1])

    def test_mad_filter_failure_mentions_cell_id(self):
        with self.assertRaisesRegex(BOLReferenceError, "cell-outlier"):
            build_bol_reference(xjtu_records([1, 1, 1, 1, 100], "cell-outlier"))

    def test_soh_bol_is_not_clipped_and_source_soh_is_preserved(self):
        records = xjtu_records([1.0] * 5 + [2.0])
        labeled = apply_bol_relative_soh(records, build_bol_reference(records))
        self.assertAlmostEqual(labeled[-1]["soh_bol"], 2.0)
        self.assertEqual(labeled[-1]["soh_label_mode"], BOL_LABEL_MODE)
        self.assertEqual(records[-1]["soh"], 1.0)
        self.assertNotIn("soh_bol", records[-1])

    def test_raw_and_feature_label_paths_are_exactly_equal(self):
        capacities = [1.0 + 0.01 * index for index in range(8)]
        raw = xjtu_records(capacities, "same-cell")
        feature = xjtu_records(capacities, "same-cell")
        reference = build_bol_reference((row for row in raw))
        raw_labeled = apply_bol_relative_soh((row for row in raw), reference)
        feature_labeled = apply_bol_relative_soh(feature, reference)
        self.assertEqual(
            {row["cycle"]: row["soh_bol"] for row in raw_labeled},
            {row["cycle"]: row["soh_bol"] for row in feature_labeled},
        )

    def test_preprocessing_reference_keeps_model_ineligible_direct_calibrations(self):
        provenance = [
            {
                "cycle_id": cycle,
                "capacity_Ah": capacity,
                "calibration_direct": True,
                "model_eligible": cycle >= 3,
                "label_source": "calibration_direct" if cycle >= 3 else "",
            }
            for cycle, capacity in enumerate([40.0, 39.0, 38.0, 37.0, 36.0], start=1)
        ]
        reference = build_frozen_smarthealth_bol_reference(
            provenance,
            domain_id="smarthealth_lishen40",
            cell_id="smart-a",
        )
        self.assertAlmostEqual(reference["Q_ref"], 38.0)
        self.assertEqual(reference["selected_cycle_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(
            reference["source_model_ineligible_direct_calibration_count"], 2
        )
        self.assertEqual(reference["reference_source"], BOL_REFERENCE_SOURCE)

    def test_smarthealth_raw_and_feature_consume_same_frozen_reference(self):
        def records():
            return [
                {
                    "domain_id": "smarthealth_lishen40",
                    "battery_id": "same-smart-cell",
                    "cycle": cycle,
                    "label_capacity_Ah": capacity,
                    "label_source": "calibration_direct",
                    "bol_q_ref_Ah": 40.0,
                    "bol_q_ref_rule": BOL_RULE_VERSION,
                    "bol_q_ref_source": BOL_REFERENCE_SOURCE,
                }
                for cycle, capacity in enumerate([40.0, 39.0, 38.0], start=1)
            ]

        raw_reference = frozen_smarthealth_bol_references(records())
        feature_reference = frozen_smarthealth_bol_references(records())
        raw_labeled = apply_bol_relative_soh(records(), raw_reference)
        feature_labeled = apply_bol_relative_soh(records(), feature_reference)
        self.assertEqual(raw_reference, feature_reference)
        self.assertEqual(
            [row["soh_bol"] for row in raw_labeled],
            [row["soh_bol"] for row in feature_labeled],
        )

    def test_smarthealth_frozen_reference_is_fail_fast(self):
        missing = smarthealth_records(
            [40.0, 39.0, 38.0, 37.0, 36.0], ["calibration_direct"] * 5
        )
        with self.assertRaisesRegex(BOLReferenceError, "regenerate canonical RAW"):
            frozen_smarthealth_bol_references(missing)

        inconsistent = []
        for index, row in enumerate(missing):
            inconsistent.append(
                {
                    **row,
                    "bol_q_ref_Ah": 40.0 if index == 0 else 39.0,
                    "bol_q_ref_rule": BOL_RULE_VERSION,
                    "bol_q_ref_source": BOL_REFERENCE_SOURCE,
                }
            )
        with self.assertRaisesRegex(BOLReferenceError, "inconsistent"):
            frozen_smarthealth_bol_references(inconsistent)

    def test_v1_config_remains_rated_and_outside_v2_namespace(self):
        v1 = load_config(V1_CONFIG)
        self.assertNotEqual(v1.get("data", {}).get("label_mode"), BOL_LABEL_MODE)
        self.assertEqual(v1["output"]["paper_version"], "Paper-v1")
        v2 = load_config(V2_RAW_CONFIG)
        self.assertEqual(v2["data"]["label_mode"], BOL_LABEL_MODE)
        self.assertEqual(v2["output"]["paper_version"], "Paper-v2")

    def test_no_cycle_raw_model_has_no_cycle_head_or_adapter(self):
        config = load_config(V2_RAW_CONFIG)
        model = build_c5b_model(config["model"], backend_override="torch_reference")
        self.assertIsNone(model.cycle_head)
        self.assertIsNone(model.cycle_adapter)
        self.assertFalse(model.use_cycle_prediction)
        self.assertFalse(model.use_predicted_cycle_for_soh)

    def test_lodo_returns_only_source_train_val_and_target_test(self):
        config = load_config(
            V2_CONFIG_ROOT / "e3_lodo_zero_cell" / "lodo_xjtu.json"
        )

        def fake_build(domain_config, repo_root, seed, domain_id, data_root):
            datasets = {
                split: [
                    {
                        "domain_id": domain_id,
                        "dataset_id": domain_id,
                        "battery_id": f"{domain_id}-{split}",
                        "split_marker": split,
                        "soh": 1.0,
                    }
                ]
                for split in ("train", "val", "test")
            }
            return datasets, {"domain_id": domain_id}

        with patch(
            "UnifiedRawSOH.datasets.loaders._build_raw_domain",
            side_effect=fake_build,
        ):
            loaders, split_info = build_lodo_loaders(config, PROJECT_ROOT, seed=42)
        train = [loaders["train"].dataset[index] for index in range(len(loaders["train"].dataset))]
        val = [loaders["val"].dataset[index] for index in range(len(loaders["val"].dataset))]
        test = [loaders["test"].dataset[index] for index in range(len(loaders["test"].dataset))]
        sources = set(DOMAINS) - {"xjtu"}
        self.assertEqual({row["domain_id"] for row in train}, sources)
        self.assertEqual({row["domain_id"] for row in val}, sources)
        self.assertEqual({row["domain_id"] for row in test}, {"xjtu"})
        self.assertTrue(all(row["split_marker"] == "train" for row in train))
        self.assertTrue(all(row["split_marker"] == "val" for row in val))
        self.assertTrue(all(row["split_marker"] == "test" for row in test))
        self.assertEqual(split_info["target_domain_id"], "xjtu")

    def test_feature_lodo_interface_has_zero_target_train_val_emission(self):
        config = load_config(
            V2_CONFIG_ROOT
            / "e3_lodo_zero_cell"
            / "feature_mlp"
            / "lodo_xjtu.json"
        )
        feature_names = np.zeros(24, dtype=np.float32)

        def fake_feature(domain_config, repo_root, seed):
            domain = domain_config["experiment"]["domain_id"]
            rows = [
                {
                    "features": feature_names.copy(),
                    "soh": np.asarray([1.0], dtype=np.float32),
                    "battery_id": f"{domain}-cell",
                    "cycle_id": 1,
                    "condition": "condition",
                    "soh_label_mode": BOL_LABEL_MODE,
                }
            ]
            dataset = StatFeatureDataset(rows, domain_id=domain)
            loader = DataLoader(dataset, batch_size=1)
            return {"train": loader, "val": loader, "test": loader}, {"domain_id": domain, "normalization": {"min": [0.0] * 24, "max": [1.0] * 24, "eps": 1e-8, "feature_names": []}}

        with patch(
            "UnifiedRawSOH.models.baselines.pinn4soh_no_leak_onlyf.build_feature_loaders",
            side_effect=fake_feature,
        ):
            loaders, info = build_feature_lodo_loaders(config, PROJECT_ROOT, seed=42)
        self.assertEqual(info["target_domain_id"], "xjtu")
        self.assertEqual(info["sample_counts"]["train"], 4)
        self.assertEqual(info["sample_counts"]["val"], 4)
        self.assertEqual(info["sample_counts"]["test"], 1)
        self.assertEqual(info["target_train_validation_samples_not_emitted"], {"train": 1, "val": 1})
        self.assertEqual(
            {row["domain_id"] for row in loaders["test"].dataset},
            {"xjtu"},
        )

    def test_full_domain_sampler_equalizes_domain_mass_then_battery_mass(self):
        rows = [
            {"domain_id": "xjtu", "battery_id": "a"},
            {"domain_id": "xjtu", "battery_id": "a"},
            {"domain_id": "xjtu", "battery_id": "b"},
            {"domain_id": "xjtu", "battery_id": "b"},
            {"domain_id": "mit", "battery_id": "m"},
            {"domain_id": "mit", "battery_id": "m"},
            {"domain_id": "mit", "battery_id": "m"},
            {"domain_id": "mit", "battery_id": "m"},
        ]
        sampler = _make_balanced_sampler(_MetadataDataset(rows), "domain_battery_hierarchical")
        weights = sampler.weights.tolist()
        self.assertAlmostEqual(sum(weights[:4]), sum(weights[4:]), places=12)
        self.assertAlmostEqual(weights[0], weights[2], places=12)
        self.assertAlmostEqual(weights[4], weights[7], places=12)

    def test_all_v2_configs_fix_bol_and_no_cycle_contract(self):
        configs = [
            path for path in V2_CONFIG_ROOT.rglob("*.json")
            if load_config(path).get("status") == "runnable"
        ]
        self.assertGreaterEqual(len(configs), 15)
        for path in configs:
            with self.subTest(path=path):
                config = load_config(path)
                self.assertEqual(config["output"]["paper_version"], "Paper-v2")
                self.assertEqual(config["experiment"]["label_mode"], BOL_LABEL_MODE)
                if config["model"].get("type") == "PaperRawSOHModel":
                    self.assertFalse(config["model"]["use_cycle_prediction"])
                    self.assertFalse(config["model"]["use_predicted_cycle_for_soh"])
                    self.assertEqual(config["train"]["lambda_cycle"], 0.0)

    def test_launcher_dry_run_lists_dynamic_gpu_jobs_and_writes_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment.update(
                {
                    "STAGE": "e1_raw",
                    "TARGET_DOMAINS": "xjtu",
                    "SEEDS": "42 52 62",
                    "GPU_IDS": "0 1",
                    "JOBS_PER_GPU": "2",
                    "DRY_RUN": "1",
                    "RESUME": "1",
                    "OUTPUT_ROOT": directory,
                    "PYTHON_BIN": sys.executable,
                }
            )
            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=REPO_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("maximum aggregate processes: 4", result.stdout)
            self.assertIn("stage=e1_raw; domain/fold=xjtu; seed=42; GPU=dynamic", result.stdout)
            self.assertIn("stage=e1_raw; domain/fold=xjtu; seed=52; GPU=dynamic", result.stdout)
            self.assertIn("stage=e1_raw; domain/fold=xjtu; seed=62; GPU=dynamic", result.stdout)
            self.assertIn("dry run complete", result.stdout)
            self.assertFalse((Path(directory) / "Paper-v2").exists())

    def test_summary_refuses_missing_seed_before_writing(self):
        from UnifiedRawSOH.scripts.paper_v2.summarize_bol_soh import (
            IncompleteBatchError,
            summarize_batch,
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(IncompleteBatchError):
                summarize_batch(directory, expected_seeds=[42, 52], expected_domains=["xjtu"])
            self.assertFalse((Path(directory) / "summary_mean_std.json").exists())


def _write_smoke_seed(seed_dir, domain="xjtu"):
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "completed.status").write_text("completed\n", encoding="utf-8")
    (seed_dir / "test_metrics.json").write_text(json.dumps({"mape": 0.1, "rmse": 0.2}), encoding="utf-8")
    fields = ["domain_id", "group_id", "cell_id", "aggregation", "n_samples", "n_cells", "n_groups", "mae", "mape", "mse", "rmse"]
    for filename in ("metrics_by_cell.csv", "metrics_by_group.csv", "metrics_by_domain.csv"):
        with (seed_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "domain_id": domain,
                "group_id": "condition",
                "cell_id": "cell",
                "aggregation": "group_macro",
                "n_samples": 1,
                "n_cells": 1,
                "n_groups": 1,
                "mae": 0.1,
                "mape": 0.1,
                "mse": 0.04,
                "rmse": 0.2,
            })


if __name__ == "__main__":
    unittest.main()
