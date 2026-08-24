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
