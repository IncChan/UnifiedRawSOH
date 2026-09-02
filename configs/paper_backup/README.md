# Paper-Backup configurations

All JSON files in this directory are isolated from `paper_v1` and `paper_v2`.
They use `Paper-Backup` output roots. The main suites use the SOH-only contract:

- `train.lambda_cycle = 0`;
- no cycle/lifetime target or predicted lifetime coordinate;
- battery, strategy, domain and cycle IDs remain provenance metadata;
- all production jobs use the offline `paper_backup_preprocessed_v1` product;
  raw models read its Terminal arrays unless the config selects `full_cccv`.

The explicitly isolated `e1_bicontext_cycle_mtl_5seed` suite is the only
exception. It keeps SOH as the primary target and adds a training-only
cycle-order auxiliary loss on the shared representation. It retains
`train.lambda_cycle = 0`, does not use a lifetime/EOL target, and never passes
cycle metadata or a cycle prediction into the SOH forward path.

The SOH-only `e1_bicontext_adaptive_fusion_5seed` suite keeps the plain
BiContext Mamba backbone and fusion residual unchanged, and adds only a
zero-initialized per-sample CC/CV weighting gate.

The final E1 Feature MLP reuses only the PINN4SOH F-only sinusoidal
encoder/predictor structure. It consumes the 24 offline terminal statistics
from the same complete cycle cohort as the raw models. The archived 3-sigma
and adjacent-x1 filters are disabled by the checked `sample_filter_mode=none`
contract; feature mean/std are fitted on the training split only.

The current E1 matrix is deliberately limited to the requested three methods:
`HI-MLP`, `Transformer`, and `Ours`, independently for XJTU, MIT, LISHEN,
CATL, and EVE. E2 contains the five charging-view jobs for XJTU, LISHEN, and
CATL. E3 contains strategy-specific and family-pooled Ours jobs for XJTU and
LISHEN.

The E2 full configs consume the independently materialized FULL arrays. All E2
views are restricted to the same `full_matched` physical-cycle cohort. Run the
Paper-Backup FULL preprocessing and validator before launching E2.
