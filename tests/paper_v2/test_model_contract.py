from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.models.paper_v2.dense_adapter import (  # noqa: E402
    choose_parameter_matched_dense_bottleneck,
)
from UnifiedRawSOH.models.paper_v2.raw_mamba_moe import build_paper_v2_model  # noqa: E402


def model_config(variant: str) -> dict:
    config = {
        "type": "PaperRawSOHModel",
        "variant": variant,
        "backend": "torch_reference",
        "input_dim": 4,
        "d_model": 32,
        "num_layers": 1,
        "d_state": 4,
        "d_conv": 2,
        "expand": 2,
        "dropout": 0.0,
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
        "use_cycle_prediction": False,
        "use_predicted_cycle_for_soh": False,
        "detach_predicted_cycle_for_soh": False,
        "time_embedding_time_scale_min": 10.0,
    }
    if variant == "dense_adapter":
        config.update({"adapter_bottleneck_dim": 136, "adapter_init": "zero_output"})
    if variant == "residual_moe":
        config.update(
            {
                "num_experts": 8,
                "top_k": 2,
                "expert_bottleneck_dim": 16,
                "expert_init": "zero_output",
                "router_input": "z_base",
            }
        )
    return config


def inputs(batch_size: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    return {
        "cc_signal": torch.randn(batch_size, 5, 2, generator=generator),
        "cv_signal": torch.randn(batch_size, 6, 2, generator=generator),
        "cc_mask": torch.ones(batch_size, 5),
        "cv_mask": torch.ones(batch_size, 6),
        "cc_time": torch.rand(batch_size, 5, generator=generator),
        "cv_time": torch.rand(batch_size, 6, generator=generator),
        "cc_temperature": torch.randn(batch_size, 5, 2, generator=generator),
        "cv_temperature": torch.randn(batch_size, 6, 2, generator=generator),
        "t0_temperature_norm": torch.randn(batch_size, 1, generator=generator),
    }


class ModelContractTest(unittest.TestCase):
    def test_zero_initialized_moe_and_dense_are_identity(self):
        torch.manual_seed(7)
        base = build_paper_v2_model(model_config("base"))
        moe = build_paper_v2_model(model_config("residual_moe"))
        dense = build_paper_v2_model(model_config("dense_adapter"))
        moe.base_model.load_state_dict(base.base_model.state_dict())
        dense.base_model.load_state_dict(base.base_model.state_dict())
        batch = inputs()
        base.eval()
        moe.eval()
        dense.eval()
        with torch.no_grad():
            base_out = base.forward_with_aux(**batch)
            moe_out = moe.forward_with_aux(**batch)
            dense_out = dense.forward_with_aux(**batch)
        self.assertTrue(torch.equal(base_out["z_base"], moe_out["z_out"]))
        self.assertTrue(torch.equal(base_out["z_base"], dense_out["z_out"]))
        self.assertTrue(torch.equal(base_out["soh_pred"], moe_out["soh_pred"]))
        self.assertTrue(torch.equal(base_out["soh_pred"], dense_out["soh_pred"]))

    def test_topk_shapes_weights_and_balance_backward(self):
        model = build_paper_v2_model(model_config("residual_moe"))
        batch = inputs()
        output = model.forward_with_aux(**batch)
        self.assertEqual(tuple(output["router_logits"].shape), (3, 8))
        self.assertEqual(tuple(output["topk_indices"].shape), (3, 2))
        self.assertEqual(tuple(output["topk_weights"].shape), (3, 2))
        self.assertTrue(torch.allclose(output["topk_weights"].sum(dim=-1), torch.ones(3)))
        self.assertTrue(torch.isfinite(output["balance_loss"]))
        target = torch.zeros_like(output["soh_pred"])
        loss = torch.nn.functional.mse_loss(output["soh_pred"], target) + output["balance_loss"]
        loss.backward()
        self.assertIsNotNone(model.adapter.router.weight.grad)
        self.assertTrue(torch.isfinite(model.adapter.router.weight.grad).all())
        selected = set(int(value) for value in output["topk_indices"].detach().reshape(-1))
        for index in selected:
            gradient = model.adapter.experts[index].up.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_router_api_has_no_metadata_arguments(self):
        model = build_paper_v2_model(model_config("residual_moe"))
        names = set(inspect.signature(model.forward_with_aux).parameters)
        self.assertNotIn("domain_id", names)
        self.assertNotIn("strategy_group", names)
        self.assertNotIn("cycle_id", names)
        with self.assertRaises(TypeError):
            model.forward_with_aux(**inputs(), domain_id="forbidden")

    def test_all_variants_cpu_forward_backward_and_strict_reload(self):
        for variant in ("base", "dense_adapter", "residual_moe"):
            with self.subTest(variant=variant):
                model = build_paper_v2_model(model_config(variant))
                output = model.forward_with_aux(**inputs())
                loss = output["soh_pred"].square().mean() + 0.01 * output["balance_loss"]
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                reloaded = build_paper_v2_model(model_config(variant))
                reloaded.load_state_dict(model.state_dict(), strict=True)

    def test_dense_parameter_match_is_within_five_percent(self):
        match = choose_parameter_matched_dense_bottleneck(
            128, num_experts=8, top_k=2, expert_bottleneck_dim=16
        )
        self.assertEqual(match["dense_bottleneck_dim"], 136)
        self.assertLessEqual(match["relative_error"], 0.05)
        model = build_paper_v2_model(model_config("dense_adapter"))
        summary = model.parameter_summary()["parameter_match"]
        self.assertEqual(summary["target_moe_parameters"], 34952)
        self.assertEqual(summary["dense_parameters"], 35080)
        self.assertLessEqual(summary["relative_error"], 0.05)

    def test_unknown_variant_does_not_fallback_to_base(self):
        with self.assertRaisesRegex(ValueError, "explicit model.variant"):
            build_paper_v2_model({**model_config("base"), "variant": ""})


if __name__ == "__main__":
    unittest.main()
