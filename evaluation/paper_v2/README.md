# Paper-v2 evaluation

The inherited hierarchical metric builder reduces predictions in the order
physical cell → strategy/group → domain → overall domain macro. MoE runs add
`routing_summary.json`, containing expert load, importance, entropy, and top-k
usage globally and by domain/strategy/cell. These are diagnostics only; no
held-out target statistic is fitted or used for selection.

| Component | Status | Config | Command | Output | Tests | Last verified | Limitations |
|---|---|---|---|---|---|---|---|
| Hierarchical metric tables | runnable | all Paper-v2 runnable configs | training entry point | `metrics_by_cell.csv`, `metrics_by_group.csv`, `metrics_by_domain.csv` | existing Paper-v2 metrics tests | 2026-08-27 | formal result aggregation not run |
| MoE routing audit | smoke-tested | MoE E2/E3 configs | CPU/reference smoke | `routing_summary.json` | model/routing contract tests | 2026-08-27 | diagnostics do not create routing priors |
