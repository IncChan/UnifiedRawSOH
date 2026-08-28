# Paper-Backup implementation status

Updated 2026-08-27.

## Implemented

- isolated `models/paper_backup`, `datasets/paper_backup`,
  `trainers/paper_backup`, and `evaluation/paper_backup` namespaces;
- E1 configs for only `HI-MLP`, `Transformer`, and `Ours` across five families;
- E2 Full/Terminal Vanilla, terminal CC-only, terminal CV-only, and terminal
  Ours interfaces for XJTU/LISHEN/CATL;
- E3 XJTU six-strategy and LISHEN nine-condition specific/pooled interfaces;
- SOH-only training, battery/strategy macro metrics, paired comparison helpers,
  and non-overwriting seed output directories;
- full-source adapter contracts for explicit normalized CSV, XJTU MATLAB, and
  source-linked SmartHealth records;
- CPU synthetic tests for model, view, split-provenance, sampler, and metric
  invariants.

## Data status

The tracked canonical products are terminal-only. E2 full configs therefore
remain `blocked_by_data`; a real source root and successful physical
`(battery_id, cycle_id)` matching audit are required before they become
runnable. The machine-local source inventory and exact terminal policies are
recorded in [DATA_AND_IMPLEMENTATION_AUDIT.md](DATA_AND_IMPLEMENTATION_AUDIT.md).

## Not run in this development pass

No formal multi-seed training was launched. The official CUDA Mamba backend was
not substituted by the CPU reference. Existing Paper-v1/Paper-v2 configs,
launchers, and default scientific protocols were not changed.
