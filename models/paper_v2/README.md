# Paper-v2 models

`PaperV2RawMambaModel` delegates `encode_base()` to the existing
`PaperRawSOHModel.encode()` and does not alter the V1 model or checkpoint
keys. Its public path is:

```text
encode_base(current-cycle CC/CV/time/temperature) -> z_base
compose(z_base) -> z_out, routing_aux
predict_from_composed_feature(z_out, T0) -> soh_pred
```

The router receives `z_base` only. Domain, strategy, battery, cycle index,
lifetime, Q_ref, and labels are never arguments to the model or router.

Residual MoE uses `num_experts=8`, `top_k=2`, and a bottleneck of 16 by
default:

```text
z_out = z_base + sum_{k in TopK} alpha_k(z_base) E_k(z_base)
```

Top-k weights are re-normalized with a softmax over selected logits. The
balance loss is exactly
`E * sum(mean(router_probabilities) * mean(hard_topk_load))`; the hard load is
an audit statistic and the importance term keeps the loss differentiable for
batch size one. Expert up-projections are zero initialized, so a fresh MoE is
numerically identical to Base.

| Component | Status | Config | Command | Output | Tests | Last verified | Limitations |
|---|---|---|---|---|---|---|---|
| Base wrapper | smoke-tested | E2/E3 `base` | `python scripts/paper_v2/train.py --help` | `outputs/Paper-v2/...` | model contract | 2026-08-27 | uses the configured raw backend |
| Dense residual adapter | smoke-tested | E2/E3 `dense_adapter` | E2 launcher with `MODEL_VARIANTS=dense_adapter` | `RawMambaV2-DenseAdapter-ERM` | parameter/forward contract | 2026-08-27 | width 136 is the nearest integer match for the default MoE |
| Residual MoE | smoke-tested | E2/E3 `moe_erm` | E2 launcher with `MODEL_VARIANTS=moe_erm` | `RawMambaV2-ResidualMoE-ERM` | routing/backward contract | 2026-08-27 | formal results not run |

For the default `z_base=128`, MoE adds 34,952 trainable parameters and the
Dense width-136 control adds 35,080, a 0.366% relative difference. Exact
counts are written to `parameter_summary.json`.
