# E3 one-cell head-only adaptation

Edit the `USER CONFIGURATION` block at the top of
`run_one_cell_head_only.sh`, then launch without arguments:

    bash scripts/paper_v1/e3_cross_domain_reusability/run_one_cell_head_only.sh

Set `PAIRED_SEEDS` and one LODO runtime root for every selected target. Each
runtime root must directly contain `seed_42/best.pt`, `seed_52/best.pt`, and
`seed_62/best.pt` (or the seeds listed in `PAIRED_SEEDS`). With
`TARGET_DOMAINS="all"`, all five roots must be valid.

XJTU and MIT use strict matched-seed pairing: checkpoint seed 42 only uses
support seed 42, 52 only uses 52, and 62 only uses 62. There is no 3 x 3
checkpoint/support cross product. SmartHealth support choices are physical-cell
labels A/B rather than seeds, so A and B are both evaluated for each checkpoint
seed.

The default protocol creates 117 independent jobs:

- XJTU: 3 checkpoint seeds x 6 support groups x 1 matched support seed = 18;
- MIT: 3 checkpoint seeds x 3 batches x 1 matched support seed = 9;
- LISHEN40: 3 checkpoint seeds x 9 groups x A/B = 54;
- CATL280: 3 checkpoint seeds x 3 groups x A/B = 18;
- EVE280: 3 checkpoint seeds x 3 groups x A/B = 18.

Each job reloads its target-unseen LODO checkpoint, selects one development
physical cell, uses all valid cycles from that cell with an SOH-stratified
80/20 support train/validation split, trains only `model.head`, and tests every
fixed target test cell. Encoder parameters are hashed before and after fitting.

`JOBS_PER_GPU` is an independent limit for every GPU. Jobs are assigned
round-robin and each subprocess receives one physical GPU through
`CUDA_VISIBLE_DEVICES`; model code uses `cuda:0` inside that process.

Set `DRY_RUN=1` in the script to validate every checkpoint, create manifests,
and print every checkpoint seed, selected support cell, GPU, and output path
without fitting. Set `RESUME=1` and fill `RUN_TIME` with an existing runtime
name to skip only jobs that have both completed status and final metrics.

Outputs are stored under:

    outputs/Paper-v1/e3_cross_domain_reusability/
      RawMamba-noCycleAux-OneCellHeadOnly/runtime_*/

Atomic job directories include `checkpoint_seed_<seed>`. A complete runtime
contains checkpoint and job manifests, one isolated directory and log per job,
support-to-test-group MAPE/RMSE matrices under `summary/<target>/`, and
`summary_by_checkpoint_seed.csv`. If any job fails, the runtime and summary
are marked incomplete and the launcher returns a nonzero status.
