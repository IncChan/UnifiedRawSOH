# E3 cross-domain scripts

`run_lodo_no_cycle_aux.sh` runs the Paper-v1 no-cycle five-fold LODO
protocol. `MAX_PARALLEL` is the maximum number of training processes allowed
on each GPU, not a global process limit.

For five GPUs with one fold per GPU and all three seeds concurrent on each
GPU:

    LEFT_OUT_DOMAIN=all GPU_IDS="0 1 2 3 4" MAX_PARALLEL=3 \
      bash scripts/paper_v1/e3_cross_domain_reusability/run_lodo_no_cycle_aux.sh

This assigns XJTU, MIT, LISHEN40, CATL280, and EVE280 to GPUs 0 through 4,
respectively. Each GPU runs seeds 42, 52, and 62 for its assigned fold. Set
`MAX_PARALLEL=1` to run those three seeds sequentially on each GPU.

When fewer than five GPUs are provided, folds are assigned round-robin and
each GPU processes its assigned folds sequentially. For example,
`GPU_IDS="2 4"` assigns XJTU/LISHEN40/EVE280 to GPU 2 and MIT/CATL280 to GPU
4. The per-GPU process count never exceeds `MAX_PARALLEL`.

For one fold, only the first configured GPU is used:

    LEFT_OUT_DOMAIN=xjtu GPU_IDS="0" MAX_PARALLEL=3 \
      bash scripts/paper_v1/e3_cross_domain_reusability/run_lodo_no_cycle_aux.sh

Valid `LEFT_OUT_DOMAIN` values are `xjtu`, `mit`,
`smarthealth_lishen40`, `smarthealth_catl280`, and
`smarthealth_eve280`. The short aliases `lishen40`, `catl280`, and
`eve280` are accepted. Use `DRY_RUN=1` to print the complete
GPU-to-fold-to-seed schedule without starting training.

Each fold automatically writes three-seed `summary_mean_std.*` and
`summary_per_domain_mean_std.*` files under:

    outputs/Paper-v1/e3_cross_domain_reusability/
      RawMamba-noCycleAux-LODO/lodo_no_cycle_aux_to_<target>/runtime_*/

The LODO command never trains on the left-out domain and never evaluates on
the source-domain test partitions. Adaptation and cross-dataset-holdout remain
separate, non-runnable interfaces.

## One-cell head-only adaptation

The group-wise one-reference-cell experiment has its own argument-free
launcher. Edit the settings and five checkpoint paths inside
`run_one_cell_head_only.sh`, then run:

    bash scripts/paper_v1/e3_cross_domain_reusability/run_one_cell_head_only.sh

See `README_ONE_CELL.md` for the 57-job protocol, per-GPU slots, resume
contract, and output matrices.
