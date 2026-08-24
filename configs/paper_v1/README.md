# Paper-v1 configurations

- `common/` contains the inherited RawMamba base contract.
- `e1_raw_soh_learning/` contains single-domain benchmarks and ablations.
- `e2_unified_multidomain/` contains battery-group and hierarchical
  domain-balanced shared-model experiments.
- `e3_cross_domain_reusability/` contains the planned V1 transfer interfaces.
- `e4_industrial_external/` contains the V1 enterprise protocol interface.
- `diagnostics/` contains validation-only analysis of the frozen E2-FULL
  checkpoints for representation dominance, residual calibration, and gradient
  conflict.
- `conditional/` records diagnosis-triggered extensions that were not part of
  the V1 model.

Formal runs continue to write under `outputs/Paper-v1/`.
