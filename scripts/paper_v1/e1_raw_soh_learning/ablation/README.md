# E1 ablation launcher

run_raw_mamba_ablation.sh selects one implemented control using its ABLATION
variable:

    vi_only
    delta_t
    t0
    independent_cc_cv
    no_degradation_aux

It shares the same multi-seed launcher, split, normalization, and model
selection protocol as E1 RawMamba. The proposed default remains under the E1
benchmark launcher.

`run_raw_ours_no_cycle_aux.sh` is the five-domain RawOurs w/o cycle auxiliary
launcher. Edit its `DOMAIN`, `SEEDS`, `GPU_IDS`, and `MAX_PARALLEL`
lines directly. It selects the matching domain config, runs the canonical E1
raw-data readiness guard, and delegates multi-seed execution to
`scripts/run_seed_batch.sh`. Its safe default is one process on one GPU.
