# Paper-Backup launchers

The launchers default to `DRY_RUN=1`, use the local Python resolver supplied by
the caller, and validate every config without creating training output.

```bash
DRY_RUN=1 PYTHON_BIN=/path/to/python bash scripts/paper_backup/run_e1.sh
DRY_RUN=1 PYTHON_BIN=/path/to/python bash scripts/paper_backup/run_e2.sh
DRY_RUN=1 PYTHON_BIN=/path/to/python bash scripts/paper_backup/run_e3.sh
```

For a bounded CPU reference run, select one config directly:

```bash
PYTHON_BIN=/path/to/python python scripts/paper_backup/run_experiment.py \
  --config configs/paper_backup/e1_main_estimation/ours/xjtu.json \
  --device_override cpu --backend_override torch_reference \
  --epochs 1 --patience 1 --debug_num_samples 2 \
  --output_root /tmp/paper_backup_smoke/Paper-Backup \
  --run_time smoke
```

Formal Mamba runs should leave `backend_override` unset. Full E2 jobs remain
blocked until their real full source is configured and audited.

The model-only smoke (no dataset load and no output write) is:

```bash
/path/to/python scripts/paper_backup/smoke_test.py
```

After that audit, set the same source root in `data.full_data_root` for each
`full_vanilla` config and in `data.matched_full_data_root` for each terminal
E2 config. The terminal loader then evaluates the exact full/terminal matched
physical-cycle cohort.

Completed runs can be summarized with:

```bash
python scripts/paper_backup/summarize_results.py \
  --root outputs/Paper-Backup
```
