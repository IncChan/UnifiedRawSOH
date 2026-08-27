from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.models.c5b_model import build_c5b_model  # noqa: E402


class V1RegressionTest(unittest.TestCase):
    def test_v1_state_dict_shape_contract_survives_v2_import(self):
        v1_config = {
            "backend": "torch_reference",
            "input_dim": 4,
            "d_model": 32,
            "num_layers": 3,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
            "dropout": 0.1,
            "pooling": "last_mean",
            "fusion_type": "concat",
            "fusion_phase_dim": 64,
            "head_hidden_dim": 128,
            "use_time_as_input": True,
            "temperature_injection": "input_concat",
            "temperature_features": "delta",
            "use_t0_temperature_meta": True,
            "t0_temperature_meta_dim": 1,
            "use_cc_to_cv_bridge": True,
            "cc_to_cv_bridge_type": "zero_init_linear",
            "cc_to_cv_bridge_input_dim": 64,
            "cc_to_cv_bridge_output_dim": 32,
            "use_cycle_prediction": True,
            "cycle_target": "cycle_life_norm",
            "cycle_head_hidden_dim": 64,
            "cycle_output_activation": "sigmoid",
            "use_predicted_cycle_for_soh": True,
            "detach_predicted_cycle_for_soh": False,
        }
        torch.manual_seed(101)
        before = build_c5b_model(v1_config)
        expected = {key: tuple(value.shape) for key, value in before.state_dict().items()}

        # Importing and constructing the independent V2 namespace must not
        # mutate the V1 class, defaults, or state-dict naming scheme.
        from UnifiedRawSOH.models.paper_v2.raw_mamba_moe import build_paper_v2_model

        build_paper_v2_model(
            {
                **v1_config,
                "variant": "base",
                "use_cycle_prediction": False,
                "use_predicted_cycle_for_soh": False,
            }
        )
        torch.manual_seed(101)
        after = build_c5b_model(v1_config)
        self.assertEqual(
            expected,
            {key: tuple(value.shape) for key, value in after.state_dict().items()},
        )
        self.assertEqual(set(before.state_dict()), set(after.state_dict()))
        before.load_state_dict(after.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
