# Portable final E1/E2 runbook

The portable launcher starts with vendor source data and finishes with the
five-seed E1/E2 summary tables. It always uses `python` from the currently
activated Conda environment; it does not use a machine-specific Python path
from `preprocess/paths.env`.

## First run on a new server

```bash
git pull
conda activate YOUR_ENVIRONMENT

mkdir -p outputs/Paper-Backup
nohup bash scripts/paper_backup/run_final_interaction_5seed_portable.sh \
  --xjtu-source "/path/to/XJTU battery dataset" \
  --mit-source "/path/to/A123 Dataset" \
  --smarthealth-source "/path/to/SmartHealth" \
  --gpus "0 1 2 3" \
  --max-parallel 1 \
  --workers 8 \
  > outputs/Paper-Backup/final_interaction_portable.log 2>&1 &
```

The script performs these stages sequentially:

1. canonical XJTU, MIT physical124 and SmartHealth raw/feature extraction;
2. five-domain E1 schema-v2 128+128 preprocessing;
3. MIT complete-charge export and five-domain E2 FULL-matched preprocessing;
4. E1 training and summary;
5. E2 training and summary.

The formal seed set is `42 52 62 72 82`, maximum epoch count is 600, and
early-stopping patience is 30. `MAX_PARALLEL=1` is the safe per-GPU default; increase
it only after checking GPU memory.

## Resume and individual stages

Generated data are reused when all expected products exist. Partial products
fail closed rather than being silently mixed with new output.

```bash
# Preprocess only.
bash scripts/paper_backup/run_final_interaction_5seed_portable.sh \
  --stage preprocess \
  --xjtu-source "/path/to/XJTU battery dataset" \
  --mit-source "/path/to/A123 Dataset" \
  --smarthealth-source "/path/to/SmartHealth"

# Train E1 and E2 from existing preprocessed data.
bash scripts/paper_backup/run_final_interaction_5seed_portable.sh \
  --stage train --gpus "0 1 2 3" --max-parallel 1

# Re-run summaries only.
bash scripts/paper_backup/run_final_interaction_5seed_portable.sh --stage summary
```

Use `--canonical-mode rebuild` or `--paper-mode rebuild` only when the
corresponding generated products should deliberately be overwritten. Use
`--canonical-mode skip` or `--paper-mode skip` when externally prepared data
have already been placed in the expected repository data directories.

Outputs remain isolated under:

- `outputs/Paper-Backup/E1-Final-Interaction-5Seed/`
- `outputs/Paper-Backup/E2-Final-Interaction-5Seed/`

The final Markdown tables are written below each output root's `summaries/`
directory as `*_macro_table.md`.
