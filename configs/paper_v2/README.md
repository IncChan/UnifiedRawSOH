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
