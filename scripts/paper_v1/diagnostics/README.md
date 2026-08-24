# V1 diagnostic launcher

Run both frozen E2-FULL variants with:

    bash scripts/paper_v1/diagnostics/run_e2_diagnostics.sh

The default is the formal CUDA/Mamba path for seeds 42, 52, and 62. Useful
overrides are GPU_ID, SEEDS, DEVICE_OVERRIDE, BACKEND_OVERRIDE,
MAX_SAMPLES_PER_DOMAIN, and SKIP_GRADIENTS=1. CPU/reference overrides are for
structural smoke tests only.
