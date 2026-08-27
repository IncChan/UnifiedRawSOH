#!/usr/bin/env python3
"""Bounded CPU wiring smoke for all V2 raw variants and the hierarchy APIs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from UnifiedRawSOH.datasets.paper_v2.episodic_sampler import SourceEpisodeBuilder  # noqa: E402
from UnifiedRawSOH.datasets.paper_v2.hierarchical_sampler import HierarchicalReplacementSampler  # noqa: E402
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


def main() -> int:
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(42)
    inputs = {
        "cc_signal": torch.randn(2, 5, 2, generator=generator),
        "cv_signal": torch.randn(2, 6, 2, generator=generator),
        "cc_mask": torch.ones(2, 5),
        "cv_mask": torch.ones(2, 6),
        "cc_time": torch.rand(2, 5, generator=generator),
        "cv_time": torch.rand(2, 6, generator=generator),
        "cc_temperature": torch.randn(2, 5, 2, generator=generator),
        "cv_temperature": torch.randn(2, 6, 2, generator=generator),
        "t0_temperature_norm": torch.randn(2, 1, generator=generator),
    }
    for variant in ("base", "dense_adapter", "residual_moe"):
        model = build_paper_v2_model(model_config(variant))
        output = model.forward_with_aux(**inputs)
        loss = output["soh_pred"].square().mean() + 0.01 * output["balance_loss"]
        loss.backward()
        assert torch.isfinite(loss).item()

    rows = [
        {
            "domain_id": domain,
            "condition": strategy,
            "battery_id": f"{domain}-{cell}",
            "cycle_id": cycle,
        }
        for domain in ("d1", "d2", "d3", "d4")
        for strategy in ("s1", "s2")
        for cell in ("c1", "c2")
        for cycle in (1, 2)
    ]
    sampler = HierarchicalReplacementSampler(rows, num_samples=32, seed=42)
    assert list(sampler) == list(sampler)
    episode = SourceEpisodeBuilder(rows, ["d1", "d2", "d3", "d4"], seed=42).sample_episode("strategy")
    meta_cells = {(rows[i]["domain_id"], rows[i]["battery_id"]) for i in episode.meta_train_indices}
    target_cells = {(rows[i]["domain_id"], rows[i]["battery_id"]) for i in episode.pseudo_target_indices}
    assert not meta_cells & target_cells
    print("Paper-v2 CPU wiring smoke: PASS (3 model variants, sampler, strategy episode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
