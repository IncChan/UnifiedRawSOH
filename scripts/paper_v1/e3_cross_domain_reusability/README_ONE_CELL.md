# E3 one-cell head-only adaptation

Edit the `USER CONFIGURATION` block at the top of
`run_one_cell_head_only.sh`, then launch without arguments:

    bash scripts/paper_v1/e3_cross_domain_reusability/run_one_cell_head_only.sh

Required settings are the target list, GPU pool, per-GPU slot count, support
seeds, and one LODO no-cycle checkpoint for every selected target. With
`TARGET_DOMAINS="all"`, all five checkpoint paths must be valid.

The default protocol creates 57 independent jobs:

- XJTU: 6 support groups × seeds 42/52/62 = 18;
- MIT: 3 batches × seeds 42/52/62 = 9;
- LISHEN40: 9 groups × A/B = 18;
- CATL280: 3 groups × A/B = 6;
- EVE280: 3 groups × A/B = 6.

Each job reloads its original target-unseen LODO checkpoint, selects one
development physical cell, uses all valid cycles from that cell with an
SOH-stratified 80/20 support train/validation split, trains only `model.head`,
and tests every fixed target test cell. Encoder parameters are hashed before
and after fitting.

`JOBS_PER_GPU` is an independent limit for every GPU. Jobs are assigned
round-robin and each subprocess receives one physical GPU through
`CUDA_VISIBLE_DEVICES`; model code uses `cuda:0` inside that process.

Set `DRY_RUN=1` in the script to create manifests and print every selected
support cell, checkpoint, GPU, and output path without fitting. Set
`RESUME=1` and fill `RUN_TIME` with an existing runtime name to skip only
jobs that have both completed status and final metrics.

Outputs are stored under:

    outputs/Paper-v1/e3_cross_domain_reusability/
      RawMamba-noCycleAux-OneCellHeadOnly/runtime_*/

A complete runtime contains checkpoint and job manifests, one isolated
directory and log per job, and support-to-test-group MAPE/RMSE matrices under
`summary/<target>/`. If any job fails, the runtime and summary are marked
incomplete and the launcher returns a nonzero status.
