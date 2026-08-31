from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from UnifiedRawSOH.models.paper_backup.model_factory import build_model  # noqa: E402
from UnifiedRawSOH.preprocess.paper_backup.common import (  # noqa: E402
    materialize_record_tensors,
    normalization_contract,
)
from UnifiedRawSOH.trainers.paper_backup.config_contract import validate_config  # noqa: E402
from UnifiedRawSOH.trainers.paper_backup.config_loader import load_config  # noqa: E402


def _record() -> dict:
    return {
        "battery_id": "xjtu-synthetic",
        "cycle_id": 1,
        "segment": np.asarray(["CC"] * 4 + ["CV"] * 4, dtype=object),
        "time": np.arange(8, dtype=np.float32),
        "voltage": np.asarray([4.0, 4.05, 4.12, 4.195, 4.195, 4.198, 4.199, 4.19975]),
        "current": np.asarray([4.0, 4.0, 4.0, 4.0, 0.5, 0.35, 0.2, 0.1]),
        "temperature": np.linspace(25.0, 25.7, 8),
    }


class CRateV2ContractTest(unittest.TestCase):
    def test_current_is_pointwise_nominal_c_rate_without_second_scaling(self):
        config = load_config(
            REPO_ROOT / "configs/paper_backup/e1_shared_crate_fullvi/ours_fullvi/xjtu.json"
        )
        normalization = normalization_contract(config, schema_version=2)
        cc, cv, _ = materialize_record_tensors(
            _record(), cc_len=4, cv_len=4, normalization=normalization
        )
        np.testing.assert_allclose(cc[:, 1], 2.0, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(cv[:, 1], [0.25, 0.175, 0.1, 0.05], rtol=0.0, atol=1e-7)
        self.assertEqual(normalization["nominal_capacity_ah"], 2.0)
        self.assertEqual(normalization["current_mode"], "nominal_c_rate")

    def test_fullvi_extends_the_same_ours_backbone_by_exactly_64_parameters(self):
        dominant = load_config(
            REPO_ROOT / "configs/paper_backup/e1_shared_crate_fullvi/ours_dominant/xjtu.json"
        )
        fullvi = load_config(
            REPO_ROOT / "configs/paper_backup/e1_shared_crate_fullvi/ours_fullvi/xjtu.json"
        )
        validate_config(dominant, REPO_ROOT, check_files=False)
        validate_config(fullvi, REPO_ROOT, check_files=False)
        left = build_model(dominant["model"], backend_override="torch_reference")
        right = build_model(fullvi["model"], backend_override="torch_reference")
        left_count = sum(parameter.numel() for parameter in left.parameters())
        right_count = sum(parameter.numel() for parameter in right.parameters())
        self.assertEqual(right_count - left_count, 64)
        output = right(
            cc_signal=torch.zeros(2, 8, 3),
            cv_signal=torch.zeros(2, 8, 3),
            cc_time=torch.zeros(2, 8),
            cv_time=torch.ones(2, 8),
            cc_temperature=torch.zeros(2, 8, 2),
            cv_temperature=torch.zeros(2, 8, 2),
            t0_temperature_norm=torch.zeros(2, 1),
        )
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_new_suite_has_an_isolated_experiment_and_output_namespace(self):
        paths = sorted(
            (REPO_ROOT / "configs/paper_backup/e1_shared_crate_fullvi").rglob("*.json")
        )
        self.assertEqual(len(paths), 20)
        for path in paths:
            config = load_config(path)
            self.assertEqual(config["output"]["experiment_id"], "e1_shared_crate_fullvi")
            self.assertEqual(config["data"]["source_mode"], "preprocessed_v2")
            self.assertEqual(config["data"]["preprocessed_schema_version"], 2)
            self.assertIn("Paper-Backup/CRateV2", config["output"]["root"])

    def test_128x128_suite_has_six_controlled_models(self):
        paths = sorted(
            (REPO_ROOT / "configs/paper_backup/e1_shared_crate_128x128").rglob("*.json")
        )
        self.assertEqual(len(paths), 30)
        model_ids = set()
        for path in paths:
            config = load_config(path)
            validate_config(config, REPO_ROOT, check_files=False)
            self.assertEqual(config["output"]["experiment_id"], "e1_shared_crate_128x128")
            self.assertEqual(config["data"]["source_mode"], "preprocessed_v2")
            self.assertEqual(config["data"]["preprocessed_schema_version"], 2)
            self.assertEqual(config["data"]["raw_len_cc"], 128)
            self.assertEqual(config["data"]["raw_len_cv"], 128)
            self.assertEqual(
                config["data"]["preprocessed_data_root"],
                "datasets/PaperBackup_preprocessed_v2_128x128",
            )
            self.assertEqual(
                config["output"]["root"],
                "outputs/Paper-Backup/CRateV2-128x128",
            )
            model_ids.add(config["output"]["model_id"])
        self.assertEqual(
            model_ids,
            {
                "Ours-Dominant-SharedCRate-128x128",
                "Ours-FullVI-SharedCRate-128x128",
                "Ours-GatedFullVI-SharedCRate-128x128",
                "Ours-FullVI-PointBridge-SharedCRate-128x128",
                "Smaller-Transformer-SharedCRate-128x128",
                "Transformer-SharedCRate-128x128",
            },
        )

    def test_core3_128x128_suite_reuses_the_original_training_protocol(self):
        paths = sorted(
            (REPO_ROOT / "configs/paper_backup/e1_core3_128x128").rglob("*.json")
        )
        self.assertEqual(len(paths), 15)
        model_types = set()
        for path in paths:
            config = load_config(path)
            validate_config(config, REPO_ROOT, check_files=False)
            self.assertEqual(config["output"]["experiment_id"], "e1_shared_crate_128x128")
            self.assertEqual(config["data"]["raw_len_cc"], 128)
            self.assertEqual(config["data"]["raw_len_cv"], 128)
            self.assertEqual(
                config["data"]["preprocessed_data_root"],
                "datasets/PaperBackup_preprocessed_v2_128x128",
            )
            train = config["train"]
            self.assertEqual(train["learning_rate"], 1e-3)
            self.assertEqual(train["patience"], 20)
            self.assertEqual(train["epochs"], 400)
            self.assertNotIn("scheduler", train)
            self.assertNotIn("gradient_accumulation_steps", train)
            model_types.add(config["model"]["type"])
        self.assertEqual(model_types, {"HI-MLP", "Transformer", "Ours"})

    def test_128x128_transformer_controls_use_joint_256_point_input(self):
        smaller_config = load_config(
            REPO_ROOT
            / "configs/paper_backup/e1_shared_crate_128x128/smaller_transformer/xjtu.json"
        )
        transformer_config = load_config(
            REPO_ROOT
            / "configs/paper_backup/e1_shared_crate_128x128/transformer/xjtu.json"
        )
        for config in (smaller_config, transformer_config):
            validate_config(config, REPO_ROOT, check_files=False)
            self.assertEqual(config["data"]["input_view"], "terminal_joint")
            self.assertEqual(config["data"]["raw_len_cc"], 128)
            self.assertEqual(config["data"]["raw_len_cv"], 128)
            model = build_model(config["model"])
            output = model(
                sequence=torch.zeros(2, 256, 5),
                mask=torch.ones(2, 256, dtype=torch.bool),
            )
            self.assertEqual(tuple(output.shape), (2, 1))
        smaller = build_model(smaller_config["model"])
        transformer = build_model(transformer_config["model"])
        self.assertEqual(sum(p.numel() for p in smaller.parameters()), 78097)
        self.assertEqual(sum(p.numel() for p in transformer.parameters()), 125889)

    def test_pointbridge_starts_as_fullvi_and_learns_a_gate_per_cv_point(self):
        fullvi_config = load_config(
            REPO_ROOT / "configs/paper_backup/e1_shared_crate_128x128/ours_fullvi/xjtu.json"
        )
        point_config = load_config(
            REPO_ROOT / "configs/paper_backup/e1_shared_crate_128x128/ours_pointbridge/xjtu.json"
        )
        validate_config(point_config, REPO_ROOT, check_files=False)
        torch.manual_seed(2026)
        fullvi = build_model(fullvi_config["model"], backend_override="torch_reference")
        torch.manual_seed(2026)
        pointbridge = build_model(point_config["model"], backend_override="torch_reference")
        fullvi_count = sum(parameter.numel() for parameter in fullvi.parameters())
        point_count = sum(parameter.numel() for parameter in pointbridge.parameters())
        self.assertEqual(point_count - fullvi_count, 97)

        point_state = pointbridge.state_dict()
        for name, value in fullvi.state_dict().items():
            self.assertIn(name, point_state)
            torch.testing.assert_close(value, point_state[name], rtol=0.0, atol=0.0)

        batch, length = 2, 8
        tau = torch.linspace(0.0, 1.0, length).repeat(batch, 1)
        cc_signal = torch.stack(
            (tau * 2.0 - 1.0, torch.full_like(tau, 2.0), tau), dim=-1
        )
        cv_signal = torch.stack(
            (torch.ones_like(tau), 0.25 - 0.2 * tau, tau), dim=-1
        )
        inputs = {
            "cc_signal": cc_signal,
            "cv_signal": cv_signal,
            "cc_time": 5.0 * tau,
            "cv_time": 15.0 * tau,
            "cc_temperature": torch.zeros(batch, length, 2),
            "cv_temperature": torch.zeros(batch, length, 2),
            "t0_temperature_norm": torch.zeros(batch, 1),
            "cc_mask": torch.ones(batch, length, dtype=torch.bool),
            "cv_mask": torch.ones(batch, length, dtype=torch.bool),
        }
        fullvi.eval()
        pointbridge.eval()
        with torch.no_grad():
            expected = fullvi(**inputs)
            result = pointbridge(**inputs)
            aux = pointbridge.forward_with_aux(**inputs)
        torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            aux["cc_to_cv_point_gate"], torch.ones(batch, length, 1)
        )

        pointbridge.train()
        pointbridge.zero_grad(set_to_none=True)
        pointbridge(**inputs).sum().backward()
        self.assertGreater(float(pointbridge.cc_to_cv_bridge.weight.grad.abs().sum()), 0.0)
        with torch.no_grad():
            pointbridge.cc_to_cv_bridge.weight.fill_(0.01)
        pointbridge.zero_grad(set_to_none=True)
        pointbridge(**inputs).sum().backward()
        self.assertGreater(float(pointbridge.cc_to_cv_point_gate.weight.grad.abs().sum()), 0.0)

    def test_gated_fullvi_starts_as_dominant_and_secondary_path_gets_gradient(self):
        dominant_config = load_config(
            REPO_ROOT / "configs/paper_backup/e1_shared_crate_128x128/ours_dominant/xjtu.json"
        )
        gated_config = load_config(
            REPO_ROOT / "configs/paper_backup/e1_shared_crate_128x128/ours_gated/xjtu.json"
        )
        validate_config(gated_config, REPO_ROOT, check_files=False)
        torch.manual_seed(2026)
        dominant = build_model(dominant_config["model"], backend_override="torch_reference")
        torch.manual_seed(2026)
        gated = build_model(gated_config["model"], backend_override="torch_reference")
        dominant_count = sum(parameter.numel() for parameter in dominant.parameters())
        gated_count = sum(parameter.numel() for parameter in gated.parameters())
        self.assertEqual(gated_count - dominant_count, 114)

        dominant_state = dominant.state_dict()
        gated_state = gated.state_dict()
        for name, value in dominant_state.items():
            self.assertIn(name, gated_state)
            torch.testing.assert_close(value, gated_state[name], rtol=0.0, atol=0.0)

        batch, length = 2, 8
        voltage_cc = torch.linspace(-1.0, 1.0, length).repeat(batch, 1)
        current_cv = torch.linspace(0.25, 0.05, length).repeat(batch, 1)
        tau = torch.linspace(0.0, 1.0, length).repeat(batch, 1)
        cc_full = torch.stack((voltage_cc, torch.full_like(tau, 2.0), tau), dim=-1)
        cv_full = torch.stack((torch.ones_like(tau), current_cv, tau), dim=-1)
        common = {
            "cc_time": torch.linspace(0.0, 5.0, length).repeat(batch, 1),
            "cv_time": torch.linspace(0.0, 15.0, length).repeat(batch, 1),
            "cc_temperature": torch.zeros(batch, length, 2),
            "cv_temperature": torch.zeros(batch, length, 2),
            "t0_temperature_norm": torch.zeros(batch, 1),
            "cc_mask": torch.ones(batch, length, dtype=torch.bool),
            "cv_mask": torch.ones(batch, length, dtype=torch.bool),
        }
        dominant.eval()
        gated.eval()
        with torch.no_grad():
            expected = dominant(
                cc_signal=cc_full[..., [0, 2]],
                cv_signal=cv_full[..., [1, 2]],
                **common,
            )
            result = gated(cc_signal=cc_full, cv_signal=cv_full, **common)
            changed = gated(
                cc_signal=cc_full + torch.tensor([0.0, 5.0, 0.0]),
                cv_signal=cv_full + torch.tensor([5.0, 0.0, 0.0]),
                **common,
            )
            aux = gated.forward_with_aux(cc_signal=cc_full, cv_signal=cv_full, **common)
        torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(changed, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            aux["cc_secondary_gate"], torch.full((batch, 1, 1), 0.5)
        )
        torch.testing.assert_close(
            aux["cv_secondary_gate"], torch.full((batch, 1, 1), 0.5)
        )

        gated.train()
        gated.zero_grad(set_to_none=True)
        gated(cc_signal=cc_full, cv_signal=cv_full, **common).sum().backward()
        self.assertGreater(
            float(gated.cc_branch.input_encoder.secondary_proj.weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(gated.cv_branch.input_encoder.secondary_proj.weight.grad.abs().sum()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
