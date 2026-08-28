from __future__ import annotations

import copy
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.paper_backup.full_cccv import (  # noqa: E402
    FullSourceUnavailable,
    match_full_terminal_records,
    materialize_full_records,
    validate_full_record,
)
from UnifiedRawSOH.datasets.paper_backup.sequence_views import (  # noqa: E402
    SequenceViewDataset,
    build_strategy_sampler,
)
from UnifiedRawSOH.datasets.paper_backup.strategy_pooling import pooled_strategy_splits  # noqa: E402
from UnifiedRawSOH.evaluation.paper_backup.aggregation import metrics_from_rows  # noqa: E402
from UnifiedRawSOH.evaluation.paper_backup.comparisons import paired_comparison  # noqa: E402
from UnifiedRawSOH.models.paper_backup.model_factory import build_model  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_contract import validate_config  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.trainer import run_epoch  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.trainer import train_from_config  # noqa: E402


def synthetic_record(battery_id: str, cycle_id: int, condition: str = "2C", soh: float = 1.0) -> dict:
    return {
        "dataset_id": "synthetic",
        "domain_id": "xjtu",
        "condition": condition,
        "strategy_id": condition,
        "battery_id": battery_id,
        "cycle_id": cycle_id,
        "segment": np.asarray(["CC"] * 4 + ["CV"] * 4, dtype=object),
        "time": np.arange(8, dtype=np.float32),
        "voltage": np.asarray([4.0, 4.05, 4.12, 4.195, 4.195, 4.198, 4.199, 4.19975], dtype=np.float32),
        "current": np.asarray([4.0, 4.0, 4.0, 4.0, 0.5, 0.35, 0.2, 0.1], dtype=np.float32),
        "temperature": np.asarray([25.0, 25.1, 25.2, 25.3, 25.4, 25.5, 25.6, 25.7], dtype=np.float32),
        "soh": float(soh),
        "soh_raw": float(soh),
    }


def sequence_config(view_id: str = "terminal_phase") -> dict:
    config = load_config(REPO_ROOT / "configs/paper_backup/e1_main_estimation/ours/xjtu.json")
    config["data"].update({"input_view": view_id, "raw_len_cc": 8, "raw_len_cv": 8})
    config["train"].update({"batch_size": 2, "device": "cpu"})
    return config


class PaperBackupContractTest(unittest.TestCase):
    def test_e1_matrix_is_restricted_to_requested_three_methods(self):
        paths = sorted((REPO_ROOT / "configs/paper_backup/e1_main_estimation").rglob("*.json"))
        self.assertEqual(len(paths), 15)
        configs = [load_config(path) for path in paths]
        self.assertEqual({config["model"]["type"] for config in configs}, {"HI-MLP", "Transformer", "Ours"})
        self.assertFalse(any(config["model"]["type"] in {"RawCNN", "LSTM", "VanillaMamba"} for config in configs))
        for config in configs:
            validate_config(config, REPO_ROOT, check_files=True)

    def test_full_config_rejects_terminal_source_and_non_backup_output(self):
        config = load_config(REPO_ROOT / "configs/paper_backup/e2_charging_information/full_vanilla/xjtu.json")
        broken_source = copy.deepcopy(config)
        broken_source["data"]["full_source_kind"] = "canonical_terminal"
        with self.assertRaises(ValueError):
            validate_config(broken_source, REPO_ROOT, check_files=True)
        broken_output = copy.deepcopy(config)
        broken_output["output"]["root"] = "outputs/Paper-v1"
        with self.assertRaises(ValueError):
            validate_config(broken_output, REPO_ROOT, check_files=True)

    def test_full_record_cannot_be_faked_by_terminal_record(self):
        terminal = synthetic_record("battery-a", 1)
        with self.assertRaises(FullSourceUnavailable):
            validate_full_record(terminal)
        full = copy.deepcopy(terminal)
        full.update({"source_view": "full_cccv", "is_full": True})
        validate_full_record(full)
        matched, audit = match_full_terminal_records([terminal], [full])
        self.assertEqual(len(matched), 1)
        self.assertEqual(audit["pair_key"], "(physical battery_id, cycle_id)")

    def test_explicit_normalized_full_csv_is_materialized_and_linkable(self):
        terminal = synthetic_record("battery-a", 1)
        rows = [
            "battery_id,cycle_id,condition,segment,relative_time_min,voltage_V,current_A,temperature_C,soh",
            "battery-a,1,2C,CC,0,3.60,4.0,25.0,1.0",
            "battery-a,1,2C,CC,1,4.00,4.0,25.0,1.0",
            "battery-a,1,2C,CV,2,4.20,0.5,25.0,1.0",
            "battery-a,1,2C,CV,3,4.20,0.1,25.0,1.0",
        ]
        with tempfile.TemporaryDirectory(prefix="paper_backup_full_csv_") as temporary:
            full_root = Path(temporary)
            (full_root / "full.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            records = materialize_full_records(
                [terminal],
                domain_id="synthetic",
                data_config={"full_data_root": str(full_root), "full_source_format": "csv"},
            )
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["is_full"])
        matched, _ = match_full_terminal_records([terminal], records)
        self.assertEqual((matched[0]["battery_id"], matched[0]["cycle_id"]), ("battery-a", 1))

    def test_views_emit_only_tensor_inputs_and_metadata_stays_outside_forward(self):
        records = [synthetic_record(f"battery-{index}", index, condition="2C", soh=1.0 - index * 0.01) for index in range(3)]
        for view_id in ("terminal_joint", "terminal_cc", "terminal_cv", "terminal_phase"):
            dataset = SequenceViewDataset(records, sequence_config(view_id), "train", view_id)
            item = dataset[0]
            self.assertIn("soh", item)
            self.assertIn("battery_id", item)
            if view_id == "terminal_phase":
                self.assertIn("cc_signal", item)
                self.assertNotIn("sequence", item)
            else:
                self.assertIn("sequence", item)
                self.assertIn("mask", item)
            self.assertEqual(float(item["raw_point_count"].item()), 8.0 if view_id in {"terminal_joint", "terminal_phase"} else 4.0)
            tensor_keys = {key for key, value in item.items() if isinstance(value, torch.Tensor)}
            self.assertNotIn("battery_id", tensor_keys)
            self.assertNotIn("strategy_id", tensor_keys)

    def test_ours_has_no_lifetime_head_or_public_lifetime_argument(self):
        config = load_config(REPO_ROOT / "configs/paper_backup/e1_main_estimation/ours/xjtu.json")
        model = build_model(config["model"], backend_override="torch_reference")
        self.assertIsNone(model.cycle_head)
        self.assertIsNone(model.cycle_adapter)
        self.assertNotIn("cycle_life_norm", inspect.signature(model.forward).parameters)
        self.assertFalse(model.use_cycle_prediction)
        self.assertFalse(model.use_predicted_cycle_for_soh)

    def test_hierarchical_strategy_sampler_has_declared_policy(self):
        records = [
            synthetic_record("battery-a", cycle, condition="2C") for cycle in range(3)
        ] + [
            synthetic_record("battery-b", cycle, condition="3C") for cycle in range(3)
        ] + [
            synthetic_record("battery-c", cycle, condition="3C") for cycle in range(3)
        ]
        dataset = SequenceViewDataset(records, sequence_config(), "train", "terminal_phase")
        sampler, audit = build_strategy_sampler(dataset, seed=7)
        self.assertEqual(audit["policy"], "strategy -> battery -> cycle hierarchical equal-mass")
        self.assertFalse(audit["strategy_id_in_model_input"])
        self.assertEqual(len(list(sampler)), len(dataset))

    def test_pooled_e3_test_cohort_is_the_exact_strategy_union(self):
        strategy_splits = {
            "2C": {
                "train": [synthetic_record("2C-dev", 1, "2C")],
                "val": [synthetic_record("2C-dev", 2, "2C")],
                "test": [synthetic_record("2C-test", 1, "2C")],
            },
            "3C": {
                "train": [synthetic_record("3C-dev", 1, "3C")],
                "val": [synthetic_record("3C-dev", 2, "3C")],
                "test": [synthetic_record("3C-test", 1, "3C")],
            },
        }
        pooled, audit = pooled_strategy_splits(strategy_splits)
        self.assertTrue(audit["test_cohort_exact_union"])
        self.assertEqual({item["battery_id"] for item in pooled["test"]}, {"2C-test", "3C-test"})

    def test_aggregations_and_paired_comparison_use_physical_cycle_keys(self):
        left = [
            {"battery_id": "a", "cycle_id": 1, "strategy_id": "2C", "y_true": 1.0, "y_pred": 0.9},
            {"battery_id": "a", "cycle_id": 2, "strategy_id": "2C", "y_true": 0.8, "y_pred": 0.7},
        ]
        right = [
            {"battery_id": "a", "cycle_id": 1, "strategy_id": "2C", "y_true": 1.0, "y_pred": 1.0},
            {"battery_id": "a", "cycle_id": 2, "strategy_id": "2C", "y_true": 0.8, "y_pred": 0.8},
        ]
        metrics = metrics_from_rows(left)
        self.assertEqual(metrics["n_batteries"], 1)
        self.assertEqual(metrics["battery_macro"]["n_batteries"], 1)
        paired = paired_comparison(left, right)
        self.assertEqual(paired["common_cycle_count"], 2)
        self.assertEqual(paired["pair_key"], "(battery_id, cycle_id)")

    def test_sequence_and_phase_models_complete_one_cpu_step(self):
        records = [synthetic_record(f"battery-{index}", index, soh=1.0 - index * 0.01) for index in range(4)]
        for model_type, view_id in (("Transformer", "terminal_joint"), ("VanillaMamba", "terminal_joint"), ("RawCNN", "terminal_joint"), ("LSTM", "terminal_joint"), ("Ours", "terminal_phase")):
            if model_type == "Ours":
                config = sequence_config(view_id)
            elif model_type == "Transformer":
                config = load_config(REPO_ROOT / "configs/paper_backup/e1_main_estimation/transformer/xjtu.json")
                config["data"].update({"raw_len_cc": 8, "raw_len_cv": 8})
            else:
                config = load_config(REPO_ROOT / "configs/paper_backup/e2_charging_information/terminal_vanilla/xjtu.json")
                config["data"].update({"raw_len_cc": 8, "raw_len_cv": 8})
                config["model"] = {"type": model_type, "input_dim": 5, "d_model": 16, "dropout": 0.0, "head_hidden_dim": 16}
                if model_type == "LSTM":
                    config["model"]["num_layers"] = 1
                if model_type == "VanillaMamba":
                    config["model"].update({"d_state": 4, "d_conv": 2, "expand": 2})
            dataset = SequenceViewDataset(records, config, "train", view_id)
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            model = build_model(config["model"], backend_override="torch_reference")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            metrics, rows = run_epoch(model_type, model, loader, torch.device("cpu"), optimizer=optimizer)
            self.assertEqual(len(rows), len(dataset))
            self.assertTrue(np.isfinite(metrics["loss"]))

    def test_trainer_writes_non_overwriting_checkpoint_bundle(self):
        records = [synthetic_record(f"battery-{index}", index, soh=1.0 - index * 0.01) for index in range(4)]
        config = sequence_config("terminal_phase")
        config["train"].update({"epochs": 1, "patience": 1})
        dataset = SequenceViewDataset(records, config, "train", "terminal_phase")
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        loader_info = {"loader_type": "synthetic", "metadata_in_forward": False}
        with tempfile.TemporaryDirectory(prefix="paper_backup_trainer_") as temporary:
            output_root = Path(temporary) / "Paper-Backup"
            with mock.patch(
                "UnifiedRawSOH.trainers.paper_backup.trainer._build_loaders",
                return_value=({"train": loader, "val": loader, "test": loader}, loader_info),
            ):
                result = train_from_config(
                    config,
                    REPO_ROOT,
                    device="cpu",
                    backend="torch_reference",
                    output_root=output_root,
                    run_time="unit",
                )
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "best.pt").is_file())
            self.assertTrue((run_dir / "predictions.json").is_file())
            checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
            self.assertIn("model", checkpoint)
            reloaded = build_model(config["model"], backend_override="torch_reference")
            reloaded.load_state_dict(checkpoint["model"], strict=True)


if __name__ == "__main__":
    unittest.main()
