# E2 unified launchers

- `run_public_xjtu_mit.sh` runs E2-Pilot on XJTU+MIT.
- `run_public_all_domains.sh` runs E2-Full on XJTU, MIT, LISHEN40, CATL280,
  and EVE280.

Both launchers resolve the current Conda Python through
`scripts/resolve_python_bin.sh`, check the canonical raw product for every
participating E1 domain, require the official Mamba backend by default, and
then call the common multi-seed launcher. The safe default is one seed at a
time on GPU 0. Set `GPU_IDS="0 1 2" MAX_PARALLEL=3` to use three distinct GPUs
concurrently.
