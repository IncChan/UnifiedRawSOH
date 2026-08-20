"""Contract checks independent of the CUDA Mamba extension."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.models.c5b_model import PaperRawSOHModel, StandardSingleCycleMamba  # noqa: E402


class C5BContractTest(unittest.TestCase):
    def test_reference_forward_shapes_and_zero_init_paths(self):
        torch.manual_seed(7)
        model = StandardSingleCycleMamba(backend="torch_reference")
        self.assertEqual(model.fusion_dim, 128)
        self.assertEqual(tuple(model.cc_to_cv_bridge.weight.shape), (32, 64))
        self.assertEqual(tuple(model.cycle_adapter.weight.shape), (128, 1))
        self.assertTrue(torch.equal(model.cc_to_cv_bridge.weight, torch.zeros_like(model.cc_to_cv_bridge.weight)))
        self.assertTrue(torch.equal(model.cycle_adapter.weight, torch.zeros_like(model.cycle_adapter.weight)))
        batch = 2
        aux = model.forward_with_aux(
            cc_signal=torch.randn(batch, 128, 2),
            cv_signal=torch.randn(batch, 256, 2),
            cc_mask=torch.ones(batch, 128),
            cv_mask=torch.ones(batch, 256),
            cc_time=torch.linspace(0, 10, 128).repeat(batch, 1),
            cv_time=torch.linspace(0, 20, 256).repeat(batch, 1),
            cc_temperature=torch.randn(batch, 128, 2),
            cv_temperature=torch.randn(batch, 256, 2),
            t0_temperature_norm=torch.zeros(batch, 1),
        )
        self.assertEqual(tuple(aux["signal_feature"].shape), (batch, 128))
        self.assertEqual(tuple(aux["z_health"].shape), (batch, 128))
        encoded = model.encode(
            cc_signal=torch.randn(batch, 128, 2),
            cv_signal=torch.randn(batch, 256, 2),
            cc_mask=torch.ones(batch, 128),
            cv_mask=torch.ones(batch, 256),
            cc_time=torch.linspace(0, 10, 128).repeat(batch, 1),
            cv_time=torch.linspace(0, 20, 256).repeat(batch, 1),
            cc_temperature=torch.randn(batch, 128, 2),
            cv_temperature=torch.randn(batch, 256, 2),
        )
        self.assertEqual(tuple(encoded.shape), (batch, 128))
        self.assertEqual(tuple(aux["soh_pred"].shape), (batch, 1))
        self.assertEqual(tuple(aux["cycle_life_hat"].shape), (batch, 1))
        self.assertTrue(torch.all(aux["cycle_life_hat"].abs() <= 1.0))

    def test_temperature_ablation_token_modes_do_not_change_default_contract(self):
        batch = 2
        vi_only = PaperRawSOHModel(
            input_dim=3,
            temperature_injection="none",
            temperature_features="none",
            use_t0_temperature_meta=False,
            t0_temperature_meta_dim=0,
            backend="torch_reference",
        )
        vi_aux = vi_only.forward_with_aux(
            cc_signal=torch.randn(batch, 8, 2),
            cv_signal=torch.randn(batch, 12, 2),
            cc_time=torch.linspace(0, 10, 8).repeat(batch, 1),
            cv_time=torch.linspace(0, 20, 12).repeat(batch, 1),
        )
        self.assertEqual(tuple(vi_aux["soh_pred"].shape), (batch, 1))
        self.assertEqual(tuple(vi_aux["z_health"].shape), (batch, 128))

        t0_only = PaperRawSOHModel(
            input_dim=3,
            temperature_injection="none",
            temperature_features="none",
            use_t0_temperature_meta=True,
            t0_temperature_meta_dim=1,
            backend="torch_reference",
        )
        t0_aux = t0_only.forward_with_aux(
            cc_signal=torch.randn(batch, 8, 2),
            cv_signal=torch.randn(batch, 12, 2),
            cc_time=torch.linspace(0, 10, 8).repeat(batch, 1),
            cv_time=torch.linspace(0, 20, 12).repeat(batch, 1),
            t0_temperature_norm=torch.zeros(batch, 1),
        )
        self.assertEqual(tuple(t0_aux["soh_pred"].shape), (batch, 1))


if __name__ == "__main__":
    unittest.main()
