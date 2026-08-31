from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from UnifiedRawSOH.models.paper_backup.model_factory import build_model
from UnifiedRawSOH.preprocess.paper_backup.common import (
    materialize_full_joint_tensor,
)
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config


def summary_module():
    path = REPO_ROOT / "scripts/paper_backup/summarize_results.py"
    spec = importlib.util.spec_from_file_location("e2_final_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class E2Final256BudgetTest(unittest.TestCase):
    def test_full_joint_resampling_uses_one_grid_and_records_boundary(self):
        record = {
            "segment": np.asarray(["CC"] * 4 + ["CV"] * 3),
            "time": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "voltage": np.asarray([3.4, 3.6, 3.8, 4.0, 4.2, 4.2, 4.2]),
            "current": np.asarray([2.0, 2.0, 2.0, 2.0, 1.5, 1.0, 0.5]),
            "temperature": np.asarray([25.0, 25.1, 25.2, 25.3, 25.4, 25.5, 25.6]),
        }
        normalization = {
            "voltage_low": 4.0,
            "voltage_high": 4.2,
            "current_scale": 2.0,
            "cc_voltage_low": 4.0,
            "cc_voltage_high": 4.2,
            "cv_current_low": 0.1,
            "cv_current_high": 0.5,
            "temp_room": 25.0,
            "temp_abs_scale": 20.0,
            "temp_delta_scale": 10.0,
            "time_scale_min": 10.0,
            "schema_version": 2,
            "current_mode": "nominal_c_rate",
            "nominal_capacity_ah": 2.0,
        }
        joint, boundary = materialize_full_joint_tensor(
            record, joint_len=256, normalization=normalization
        )
        self.assertEqual(joint.shape, (256, 7))
        self.assertTrue(0 < boundary < 256)
        self.assertAlmostEqual(float(joint[0, 1]), 1.0)
        self.assertTrue(np.all(np.isfinite(joint)))

    def test_boundary_token_vanilla_accepts_variable_boundaries(self):
        config = load_config(
            REPO_ROOT
            / "configs/paper_backup/e2_final_256budget/full_vanilla_256/xjtu.json"
        )
        model = build_model(config["model"], backend_override="torch_reference")
        prediction = model(
            torch.randn(2, 256, 5),
            torch.ones(2, 256, dtype=torch.bool),
            boundary_index=torch.as_tensor([96, 192]),
        )
        self.assertEqual(tuple(prediction.shape), (2, 1))

    def test_single_phase_ours_does_not_consume_removed_phase(self):
        common = {
            "cc_signal": torch.randn(2, 128, 3),
            "cv_signal": torch.randn(2, 128, 3),
            "cc_mask": torch.ones(2, 128, dtype=torch.bool),
            "cv_mask": torch.ones(2, 128, dtype=torch.bool),
            "cc_time": torch.linspace(0, 10, 128).repeat(2, 1),
            "cv_time": torch.linspace(10, 20, 128).repeat(2, 1),
            "cc_temperature": torch.randn(2, 128, 2),
            "cv_temperature": torch.randn(2, 128, 2),
            "t0_temperature_norm": torch.randn(2, 1),
        }
        for variant, removed in (("ours_cc_only_128", "cv"), ("ours_cv_only_128", "cc")):
            config = load_config(
                REPO_ROOT
                / f"configs/paper_backup/e2_final_256budget/{variant}/xjtu.json"
            )
            model = build_model(config["model"], backend_override="torch_reference").eval()
            first = model.forward_with_aux(**common)["soh_pred"]
            changed = dict(common)
            changed[f"{removed}_signal"] = torch.randn(2, 128, 3) * 1000
            changed[f"{removed}_time"] = torch.randn(2, 128) * 1000
            changed[f"{removed}_temperature"] = torch.randn(2, 128, 2) * 1000
            second = model.forward_with_aux(**changed)["soh_pred"]
            self.assertTrue(torch.equal(first, second))

    def test_formal_matrix_is_five_models_three_families_ten_seeds(self):
        module = summary_module()
        seeds = (42, 52, 62, 72, 82, 92, 102, 112, 122, 123)
        expected, paths = module.expected_jobs("e2_final_256budget", seeds)
        self.assertEqual(len(paths), 15)
        self.assertEqual(len(expected), 150)
        self.assertEqual({key[0] for key in expected}, {"e2_final_256budget"})

    def test_paired_summary_rejects_or_accepts_exact_cycle_coverage(self):
        module = summary_module()
        models = {
            "Full-VanillaMamba-Matched-256": "full",
            "Terminal-VanillaMamba-Matched-SEP-128x128": "terminal",
            "Ours-CC-Only-FullVI-128": "cc",
            "Ours-CV-Only-FullVI-128": "cv",
            "Ours-FullVI-PointBridge-128x128": "ours",
        }
        temporary = Path("/tmp/e2_final_summary_test")
        temporary.mkdir(parents=True, exist_ok=True)
        expected, selected = {}, {}
        for index, (model, data_id) in enumerate(models.items()):
            predictions = [
                {
                    "battery_id": "battery-1",
                    "cycle_id": cycle,
                    "strategy_id": "s",
                    "y_true": 1.0,
                    "y_pred": 1.0 - 0.01 * (index + cycle),
                }
                for cycle in (1, 2)
            ]
            path = temporary / f"{data_id}.json"
            path.write_text(json.dumps(predictions), encoding="utf-8")
            key = ("e2_final_256budget", model, data_id, 42)
            expected[key] = {
                "family": "xjtu",
                "seed": 42,
                "model_id": model,
            }
            selected[key] = {"predictions_path": str(path)}
        rows, summaries = module.e2_final_paired_rows(expected, selected)
        self.assertEqual(len(rows), 10)
        self.assertEqual(len(summaries), 10)


if __name__ == "__main__":
    unittest.main()
