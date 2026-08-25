# E1 launchers

E1 has benchmark and ablation launchers. They keep fixed settings inside the
script so ordinary use does not require a long command line.

    benchmark/run_raw_mamba_benchmark.sh
    benchmark/run_onlyf_benchmark.sh
    ablation/run_raw_mamba_ablation.sh
    ablation/run_raw_ours_no_cycle_aux.sh

For a benchmark script, edit the single DOMAIN line (`xjtu`, `mit`,
`smarthealth_lishen40`, `smarthealth_catl280`, or `smarthealth_eve280`), then set GPU_IDS,
SEEDS, and MAX_PARALLEL if needed. Each seed receives a separate checkpoint;
the batch directory receives one mean/std summary:

    outputs/Paper-v1/e1_raw_soh_learning/<model>/<domain>/runtime_<time>/

The matched-cycle script is evaluation only. It loads the four explicitly
configured historical checkpoint directories, intersects physical cycle IDs,
and never trains or reselects a checkpoint:

    benchmark/run_matched_cycle_eval.sh

Its result stays at outputs/e1_matched_cycle/result.json.
