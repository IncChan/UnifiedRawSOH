# V1 diagnostic launcher

Run both frozen E2-FULL variants with:

    bash scripts/paper_v1/diagnostics/run_e2_diagnostics.sh

The launcher treats each seed as one parallel worker. With seeds 42, 52, and
62, `MAX_PARALLEL=3` starts exactly three seed processes together. Inside each
seed process, the selected diagnostics (B and D by default) run sequentially,
so B/D do not create extra parallel processes.

To put all three seed processes on GPU 7:

    GPU_ID=7 MAX_PARALLEL=3 bash scripts/paper_v1/diagnostics/run_e2_diagnostics.sh

For CPU execution, the same setting starts three seed processes on the shared
CPU. Limit threads per process to avoid oversubscription:

    DEVICE_OVERRIDE=cpu BACKEND_OVERRIDE=torch_reference MAX_PARALLEL=3 CPU_THREADS_PER_JOB=4 bash scripts/paper_v1/diagnostics/run_e2_diagnostics.sh

GPU_IDS assigns GPUs round-robin by seed. For `GPU_IDS="6 7"`, seed 42 uses
GPU 6, seed 52 uses GPU 7, and seed 62 uses GPU 6. Use three IDs if each seed
should have its own GPU:

    GPU_IDS="6 7 8" MAX_PARALLEL=3 bash scripts/paper_v1/diagnostics/run_e2_diagnostics.sh

GPU_ID remains the single-GPU compatibility option. If neither GPU_IDS nor
GPU_ID is set, the launcher defaults to GPU IDs 6 and 7.
DIAGNOSTICS="e2_full_d" selects only D. Other useful overrides are SEEDS,
SUMMARY_SEEDS, MAX_SAMPLES_PER_DOMAIN, and SKIP_GRADIENTS=1. Use DRY_RUN=1 to
inspect seed assignments without loading a model. Shared summaries are written
after every selected seed process succeeds.

## Representation protocols and partial reruns

The primary representation result is now the macro average of battery-disjoint
binary probes for every domain pair. Each pair is matched only inside its own
SOH overlap. The original strict five-domain common-bin probe is preserved as
`representation_strict_probe.json`; when no common bin exists it is marked
`unavailable` without blocking residual calibration or gradient conflict.

To rerun only the missing B/D seed 42 process while rebuilding summaries from
all three seeds, use:

    GPU_ID=6 MAX_PARALLEL=1 SEEDS=42 SUMMARY_SEEDS="42 52 62" bash scripts/paper_v1/diagnostics/run_e2_diagnostics.sh > v1_diagnostics_seed42.log 2>&1 &
    disown

During aggregation, legacy seed 52/62 strict-probe reports are retained and the
new pairwise probes are computed from their existing `validation_features.npz`
caches. Their models are not loaded again.

For a clean formal run with all three seed processes active together:

    GPU_IDS="6 7 8" MAX_PARALLEL=3 bash scripts/paper_v1/diagnostics/run_e2_diagnostics.sh > v1_diagnostics.log 2>&1 &
    disown

Use `GPU_ID=7 MAX_PARALLEL=3` if all three seed processes should share GPU 7,
or `GPU_IDS="6 7" MAX_PARALLEL=3` if seeds 42 and 62 may share GPU 6. Ensure
each GPU has enough memory for its concurrent models. CPU/reference execution
is intended for structural smoke tests rather than formal reported diagnostics.
