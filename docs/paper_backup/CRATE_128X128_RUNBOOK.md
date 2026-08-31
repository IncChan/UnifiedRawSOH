# Paper-Backup C-rate 128x128 runbook

This is a strict sequence-length ablation of the existing C-rate v2 suite. It
keeps nominal-capacity C-rate normalization, fixed voltage normalization, the
model backbone, optimizer, static cell splits, development validation,
checkpoint selection, seeds and test aggregation unchanged.

| Contract | C-rate v2 control | 128x128 ablation |
|---|---|---|
| CC/CV length | 128 / 256 | 128 / 128 |
| data root | `datasets/PaperBackup_preprocessed_v2` | `datasets/PaperBackup_preprocessed_v2_128x128` |
| experiment ID | `e1_shared_crate_fullvi` | `e1_shared_crate_128x128` |
| result root | `outputs/Paper-Backup/CRateV2` | `outputs/Paper-Backup/CRateV2-128x128` |
| summary selector | `e1_crate` | `e1_crate_128x128` |

The preprocessing wrapper reopens the canonical XJTU/MIT/SmartHealth Terminal
records and interpolates each physical phase directly onto its new uniform
physical-time grid. It never downsamples the existing 256-point `.npy` file.
The resulting manifest records both lengths, normalization contract, source
inventory, array shapes and SHA-256 hashes.

## Train only the two 128x128 Transformer controls

Both controls read the same schema-v2 arrays, static splits and 128-point CC +
128-point CV sampling product as Ours.  Their `terminal_joint` view concatenates
the two phases into one 256-token, five-channel sequence.  The standard model
has 125,889 parameters; Smaller Transformer has 78,097 parameters and is the
parameter-matched control for the 78,466-parameter PointBridge model.  No
preprocessing or development-validation change is involved.

The existing 60 Ours jobs remain valid.  Launch only the 30 new jobs
(2 models x 5 datasets x 3 seeds) in the background with:

```bash
mkdir -p outputs/Paper-Backup/CRateV2-128x128

nohup env \
  PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  MODELS="smaller_transformer transformer" \
  GPU_IDS="0 1" \
  JOBS_PER_GPU=3 \
  SEEDS="42 52 62" \
  EPOCHS=400 \
  PATIENCE=20 \
  BATCH_SIZE=128 \
  NUM_WORKERS=1 \
  RUN_TIME=e1_crate_v2_128x128_transformers \
  bash scripts/paper_backup/run_e1_crate_128x128_pipeline.sh train \
  > outputs/Paper-Backup/CRateV2-128x128/transformers_train.log 2>&1 &
```

The `train` stage validates the existing manifests and arrays before launching
any process.  The explicit `MODELS` value is important: omitting it means
`all` and would also schedule the 60 already-completed jobs.

Monitor it with:

```bash
tail -f outputs/Paper-Backup/CRateV2-128x128/transformers_train.log
find outputs/Paper-Backup/CRateV2-128x128/_launcher_logs \
  \( -path '*smaller_transformer*' -o -path '*transformer__*' \) \
  -name '*.log' -type f
```

After all 30 jobs finish, combine them with the existing 60 jobs:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
bash scripts/paper_backup/run_e1_crate_128x128_pipeline.sh summary
```

The paper-facing table is:

```text
outputs/Paper-Backup/CRateV2-128x128/summaries/e1_crate_128x128_metrics_mean_std.csv
```

It contains pooled-cycle and battery-macro MAPE/RMSE for all four Ours variants
and both Transformer controls.  The summarizer writes no new formal table until
all 90 expected jobs (6 models x 5 datasets x 3 seeds) are complete.  Missing
jobs are listed in `e1_crate_128x128_status.json` and are never converted to
zero.

## One background command for a fresh full run

Create the log directory once, then detach the whole preprocess -> validate ->
train -> summary pipeline:

```bash
mkdir -p outputs/Paper-Backup/CRateV2-128x128

nohup env \
  PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  GPU_IDS="0 1" \
  JOBS_PER_GPU=3 \
  SEEDS="42 52 62" \
  EPOCHS=400 \
  PATIENCE=20 \
  BATCH_SIZE=128 \
  NUM_WORKERS=1 \
  RUN_TIME=e1_crate_v2_128x128 \
  bash scripts/paper_backup/run_e1_crate_128x128_pipeline.sh all \
  > outputs/Paper-Backup/CRateV2-128x128/pipeline.log 2>&1 &
```

Monitor the top-level pipeline and individual jobs with:

```bash
tail -f outputs/Paper-Backup/CRateV2-128x128/pipeline.log
find outputs/Paper-Backup/CRateV2-128x128/_launcher_logs -name '*.log' -type f
```

`all` expects a fresh 128x128 data root and refuses to overwrite an existing
product. This is intentional. Use the individual stages below when the data
has already been generated.

## Individual stages

Preprocess and validate only:

```bash
nohup env \
  PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  bash scripts/paper_backup/run_e1_crate_128x128_pipeline.sh preprocess \
  > outputs/Paper-Backup/CRateV2-128x128/preprocess.log 2>&1 &
```

Preflight without training:

```bash
DRY_RUN=1 CHECK_DATA=1 \
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
bash scripts/paper_backup/run_e1_crate_128x128.sh
```

Train all six models on all five datasets and all three seeds after preprocessing:

```bash
nohup env \
  PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  GPU_IDS="0 1" JOBS_PER_GPU=3 SEEDS="42 52 62" \
  RUN_TIME=e1_crate_v2_128x128 \
  bash scripts/paper_backup/run_e1_crate_128x128_pipeline.sh train \
  > outputs/Paper-Backup/CRateV2-128x128/train.log 2>&1 &
```

The launcher defaults to all six controlled models. A bounded subset can be
selected with `MODELS="smaller_transformer transformer"`,
`MODELS=ours_pointbridge`, `MODELS=ours_gated`, `MODELS=ours_fullvi`, or
`MODELS=ours_dominant`.

Summarize completed formal runs:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
bash scripts/paper_backup/run_e1_crate_128x128_pipeline.sh summary
```

The paper-facing table is:

```text
outputs/Paper-Backup/CRateV2-128x128/summaries/e1_crate_128x128_metrics_mean_std.csv
```

It contains pooled-cycle and battery-macro MAPE/RMSE. The summarizer writes no
formal table until all 90 expected jobs (6 models x 5 datasets x 3 seeds) are
complete. Missing jobs are listed in `e1_crate_128x128_status.json` and are
never converted to zero.

## Result isolation

The old E1 results, 128x256 C-rate v2 results and new 128x128 results retain
different experiment IDs, model IDs, data IDs, roots and summary filenames.
Do not copy checkpoints or summary files between these roots. A later paper
comparison should explicitly join the three mean/std CSV files rather than
asking one summary selector to mix experiment contracts.
