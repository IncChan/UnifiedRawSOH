# V1 E2 diagnostics

These configs analyze the frozen E2-FULL checkpoints on their validation
splits. They never train or rewrite a source checkpoint.

The three diagnostics are:

1. `representation_probe`: a battery-disjoint linear domain probe on
   SOH-bin-matched `z_health`, plus a PCA projection for visualization.
2. `residual_calibration`: per-domain affine calibration fitted and evaluated
   on disjoint validation batteries, with residuals reported by SOH bin.
3. `gradient_conflict`: pairwise cosine similarity between equal-budget,
   per-domain SOH-loss gradients on the shared CC/CV encoder.

All formal outputs are kept under `outputs/Paper-v1/v1_diagnostics/`, separate
from the source E2 runtime. These are diagnostic evidence for deciding V2
architecture changes, not replacements for the original V1 metrics.

`e2_full_d_no_cycle_aux.json` applies the exact E2-FULL-D diagnostic
parameters above to the frozen E2-FULL-D w/o cycle auxiliary runtime
`runtime_260825-022226` for seeds 42, 52, and 62. Its output is isolated under
`outputs/Paper-v1/v1_diagnostics/e2_full_d_no_cycle_aux/`.
