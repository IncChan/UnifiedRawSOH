from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.trainers.paper_v2.mldg import first_order_mldg_step  # noqa: E402


class ToyPaperModel(nn.Module):
    """A tiny current-cycle-only model for the numerical MLDG contract."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward_with_aux(self, cc_signal, cv_signal, **kwargs):
        del cv_signal, kwargs
        return {"soh_pred": self.weight * cc_signal, "balance_loss": None}

    def forward(self, cc_signal, cv_signal):
        del cv_signal
        return self.weight * cc_signal


def batch(x: float, y: float) -> dict[str, torch.Tensor]:
    return {
        "cc_signal": torch.tensor([[x]], dtype=torch.float32),
        "cv_signal": torch.tensor([[0.0]], dtype=torch.float32),
        "soh": torch.tensor([[y]], dtype=torch.float32),
    }


class FirstOrderMLDGTest(unittest.TestCase):
    def test_inner_fast_weight_changes_and_outer_gradient_is_finite(self):
        model = ToyPaperModel()
        result = first_order_mldg_step(
            model,
            batch(2.0, 4.0),
            batch(1.0, 0.0),
            inner_learning_rate=0.1,
            beta=1.0,
        )
        self.assertTrue(result["fast_parameters_changed"])
        self.assertIsNotNone(model.weight.grad)
        self.assertTrue(torch.isfinite(model.weight.grad).all())
        self.assertGreater(abs(float(model.weight.grad.item())), 0.0)

    def test_beta_zero_matches_erm_gradient_path(self):
        left = ToyPaperModel()
        right = ToyPaperModel()
        meta = batch(2.0, 4.0)
        target = batch(1.0, 0.0)
        result = first_order_mldg_step(
            left, meta, target, inner_learning_rate=0.1, beta=0.0
        )
        expected_loss = nn.functional.mse_loss(
            right(cc_signal=meta["cc_signal"], cv_signal=meta["cv_signal"]),
            meta["soh"],
        )
        expected_loss.backward()
        self.assertAlmostEqual(float(left.weight.grad.item()), float(right.weight.grad.item()), places=6)
        self.assertAlmostEqual(result["pseudo_target_loss"], 1.0, places=6)

    def test_lambda_balance_zero_does_not_add_balance_term(self):
        model = ToyPaperModel()
        result = first_order_mldg_step(
            model,
            batch(1.0, 1.0),
            batch(1.0, 1.0),
            lambda_balance=0.0,
        )
        self.assertEqual(result["balance_loss"], 0.0)
        self.assertEqual(result["lambda_balance"], 0.0)


if __name__ == "__main__":
    unittest.main()
