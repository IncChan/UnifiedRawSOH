# Paper-v2 configurations

This tree is versioned separately from Paper-v1. Every resolved config uses
output.paper_version = "Paper-v2" and its runtime namespace is below
outputs/Paper-v2/.

common/bol_soh_base.json fixes the shared BOL label rule
bol_peak_mean_top5_first100_v1, no-cycle RawMamba auxiliary settings, fixed
physical-window normalization, optimizer/scheduler defaults, and hierarchical
metric reporting. The E1, E2, and E3 configs only select domain composition,
data roots/splits, and output identity.

The optional Feature MLP LODO implementation is in
e3_lodo_zero_cell/feature_mlp/. The base.json is an interface template;
the five lodo_*.json files are runnable fold configs.

P0–P2 V2-native raw configurations are kept beside those compatibility
configs:

| Matrix | Status | Paths |
|---|---|---|
| E2 Base-ERM | runnable | `e2_full_domain/base/config.json` |
| E2 Dense-Adapter-ERM | runnable | `e2_full_domain/dense_adapter/config.json` |
| E2 Residual-MoE-ERM | runnable | `e2_full_domain/moe_erm/config.json` |
| E3 Base/Dense/MoE-ERM/MoE-DG | runnable | `e3_lodo_zero_cell/{base_erm,dense_adapter_erm,moe_erm,moe_dg}/lodo_*.json` |

The older `e2_full_domain/raw_mamba_domain_balanced.json` and
`e3_lodo_zero_cell/lodo_*.json` remain in place as existing Base compatibility
configs. V2-native configs require explicit model and trainer variants and
never silently fall back to Base.
