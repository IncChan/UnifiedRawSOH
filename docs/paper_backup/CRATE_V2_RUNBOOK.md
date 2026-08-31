# Paper-Backup C-rate v2 runbook

This suite is intentionally isolated from the historical Paper-Backup runs.

| Contract | Historical suite | C-rate v2 suite |
|---|---|---|
| preprocessing schema | `paper_backup_preprocessed_v1` | `paper_backup_preprocessed_crate_v2` |
| data root | `datasets/PaperBackup_preprocessed` | `datasets/PaperBackup_preprocessed_v2` |
| experiment ID | `e1_main_estimation` | `e1_shared_crate_fullvi` |
| result root | `outputs/Paper-Backup` | `outputs/Paper-Backup/CRateV2` |
| summary selector | `e1` | `e1_crate` |

The v2 voltage channel uses the fixed family Terminal voltage window. The v2
current channel is the measured pointwise charging-current magnitude divided
by the fixed family nominal capacity:

```text
current_c_rate(t) = abs(current_A(t)) / nominal_capacity_Ah
```

No cycle capacity, BOL measured capacity, SOH label, train statistic, second
min-max transform, or clipping is used. CC and CV use the same voltage and
current formulas. Development-validation splits and checkpoint selection are
unchanged.

## 1. Build the v2 Terminal product

From the repository root, create the log directory first:

```bash
mkdir -p outputs/Paper-Backup/CRateV2
```

Run preprocessing in the background:

```bash
nohup env \
  PAPER_BACKUP_SCHEMA_VERSION=2 \
  PAPER_BACKUP_PREPROCESSED_ROOT="$PWD/datasets/PaperBackup_preprocessed_v2" \
  PAPER_BACKUP_WORKERS=4 \
  PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  bash preprocess/paper_backup/run_preprocess.sh terminal \
  > outputs/Paper-Backup/CRateV2/preprocess_terminal.log 2>&1 &
```

This command does not overwrite v1. If an incomplete v2 product must be
rebuilt, explicitly add `OVERWRITE=1`; do not point the v2 command at the v1
directory.

Monitor it with:

```bash
tail -f outputs/Paper-Backup/CRateV2/preprocess_terminal.log
```

After completion, validate checksums, shapes, identities and finite values:

```bash
PAPER_BACKUP_SCHEMA_VERSION=2 \
PAPER_BACKUP_PREPROCESSED_ROOT="$PWD/datasets/PaperBackup_preprocessed_v2" \
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
bash preprocess/paper_backup/run_preprocess.sh validate
```

The first controlled experiment remains CC=128/CV=256. A future 128/128
product must use another root, for example
`datasets/PaperBackup_preprocessed_v2_cc128_cv128`; it must not overwrite this
product.

## 2. Validate the new experiment matrix

The v2 launcher defaults to all four controlled raw models:

```text
ours_dominant
ours_fullvi
smaller_transformer
transformer
```

Run a no-training preflight:

```bash
DRY_RUN=1 CHECK_DATA=1 MODELS=all \
GPU_IDS="0 1" SEEDS="42 52 62" \
bash scripts/paper_backup/run_e1_crate.sh
```

`Ours-Dominant-SharedCRate` isolates normalization from the legacy Ours input.
`Ours-FullVI-SharedCRate` then adds CC current and CV voltage. Both Transformer
variants consume the identical v2 V/C-rate arrays.

## 3. Run training in the background

```bash
nohup env \
  MODELS=all \
  GPU_IDS="0 1" \
  JOBS_PER_GPU=3 \
  SEEDS="42 52 62" \
  EPOCHS=400 \
  PATIENCE=20 \
  BATCH_SIZE=128 \
  NUM_WORKERS=1 \
  RUN_TIME=e1_crate_v2_128x256 \
  bash scripts/paper_backup/run_e1_crate.sh \
  > outputs/Paper-Backup/CRateV2/e1_crate_v2_128x256.launcher.log 2>&1 &
```

Select models manually when needed, for example:

```bash
MODELS="ours_dominant ours_fullvi" bash scripts/paper_backup/run_e1_crate.sh
```

Launcher logs live below:

```text
outputs/Paper-Backup/CRateV2/_launcher_logs/e1_shared_crate_fullvi/
```

Training runs live below:

```text
outputs/Paper-Backup/CRateV2/e1_shared_crate_fullvi/
```

## 4. Summarize only the v2 runs

```bash
/home/chenyanxi/.conda/envs/pinn/bin/python \
  scripts/paper_backup/summarize_results.py \
  --experiment e1_crate \
  --seeds "42 52 62" \
  --root outputs/Paper-Backup/CRateV2 \
  --output-dir outputs/Paper-Backup/CRateV2/summaries
```

The formal files are:

```text
outputs/Paper-Backup/CRateV2/summaries/e1_crate_metrics_per_seed.csv
outputs/Paper-Backup/CRateV2/summaries/e1_crate_metrics_mean_std.csv
outputs/Paper-Backup/CRateV2/summaries/e1_crate_status.json
```

The summarizer refuses to write a formal metrics table until all 20 model ×
family tasks exist for every requested seed. It chooses only runs whose
experiment ID is `e1_shared_crate_fullvi`.

## 5. Historical results remain separate

Do not move, rename, or delete historical runs. They keep their original
experiment ID and remain under `outputs/Paper-Backup/e1_main_estimation`.
Regenerate only the historical E1 summary with:

```bash
/home/chenyanxi/.conda/envs/pinn/bin/python \
  scripts/paper_backup/summarize_results.py \
  --experiment e1 \
  --seeds "42 52 62" \
  --root outputs/Paper-Backup \
  --output-dir outputs/Paper-Backup/summaries
```

The v2 summary does not import historical checkpoints or metrics. The old
PINN4SOH-like/HI-MLP result is therefore retained as an unchanged statistical-
feature reference, but is not silently copied into the controlled raw-model
v2 summary. If a publication table combines it with v2 later, that merge must
be explicit and label it as an unchanged feature baseline.

Avoid `--experiment all` when preparing either isolated table. Use `e1` for
historical results and `e1_crate` for the controlled C-rate experiment.
