# E3 cross-domain scripts

`run_lodo_no_cycle_aux.sh` runs the runnable Paper-v1 no-cycle five-fold LODO
protocol. Select one held-out domain with `LEFT_OUT_DOMAIN`:

    LEFT_OUT_DOMAIN=xjtu GPU_IDS="0 1 2" MAX_PARALLEL=3 \
      bash scripts/paper_v1/e3_cross_domain_reusability/run_lodo_no_cycle_aux.sh

Valid values are `xjtu`, `mit`, `smarthealth_lishen40`,
`smarthealth_catl280`, and `smarthealth_eve280`. The short SmartHealth
aliases `lishen40`, `catl280`, and `eve280` are accepted. Use
`LEFT_OUT_DOMAIN=all` to run all five folds sequentially; within each fold,
seeds 42, 52, and 62 run according to `GPU_IDS` and `MAX_PARALLEL`.

Each fold automatically writes three-seed `summary_mean_std.*` and
`summary_per_domain_mean_std.*` files under:

    outputs/Paper-v1/e3_cross_domain_reusability/
      RawMamba-noCycleAux-LODO/lodo_no_cycle_aux_to_<target>/runtime_*/

The LODO command never trains on the left-out domain and never evaluates on
the source-domain test partitions. Adaptation and cross-dataset-holdout remain
separate, non-runnable interfaces.
