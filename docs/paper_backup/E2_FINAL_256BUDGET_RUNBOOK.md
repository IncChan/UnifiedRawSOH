# Paper-Backup final E2: equal 256-point budget

This experiment is isolated from the historical `e2_charging_information`
matrix.  It evaluates five models on the FULL-matched XJTU, LISHEN and CATL
test cohorts with ten fixed seeds.

## Scientific contract

| model | physical input | model tokens | role |
|---|---|---:|---|
| Full Vanilla Mamba | complete principal charge event, jointly resampled to 256 points | 256 + one boundary token | full-trajectory reference |
| Terminal Vanilla Mamba | terminal CC 128 + terminal CV 128 | 256 + one boundary token | joint terminal control |
| Ours CC-only | terminal CC; all CV V/I/time/temperature channels fixed to zero | two 128-point branches | input ablation |
| Ours CV-only | terminal CV; all CC V/I/time/temperature channels fixed to zero | two 128-point branches | input ablation |
| Ours PointBridge | terminal CC 128 + terminal CV 128 | two 128-point branches | final method |

CC-only and CV-only keep the complete PointBridge architecture and prediction
head.  They therefore have the same 78,466 registered and trainable parameters
as final Ours; only cycle-specific information from the removed phase is
absent.  Their purpose is to test whether using both terminal phases is
justified, not to provide generic single-stream Mamba baselines.

Both Vanilla views use the same `d_model=50`, three-layer Mamba and learned
CC-to-CV boundary token.  They have 78,799 parameters, a +333 (+0.424%)
difference from Ours.  The match is selected analytically before training and
is checked by the pipeline.  No validation performance is used to select the
Vanilla size.

The FULL reference is an equal-sampling-budget reference, not a theoretical
upper bound over all possible full-trajectory estimators.

## Isolated paths

```text
data:    datasets/PaperBackup_preprocessed_v2_e2_final
configs: configs/paper_backup/e2_final_256budget
results: outputs/Paper-Backup/E2-Final-256Budget
```

Historical E2 configs, runs and summaries are not selected by the final E2
summarizer.

## One background pipeline

The `all` stage expects the new preprocessing root not to exist.  Create only
the log directory, then launch preprocessing, validation, all 150 jobs and the
formal summary:

```bash
mkdir -p outputs/Paper-Backup/E2-Final-256Budget

nohup env \
  PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  PAPER_BACKUP_WORKERS=8 \
  GPU_IDS="0 1" \
  JOBS_PER_GPU=3 \
  NUM_WORKERS=4 \
  SEEDS="42 52 62 72 82 92 102 112 122 123" \
  RUN_TIME=e2_final_256budget_seed10 \
  bash scripts/paper_backup/run_e2_final_256budget_pipeline.sh all \
  > outputs/Paper-Backup/E2-Final-256Budget/pipeline.log 2>&1 &
```

Monitor the pipeline and child jobs with:

```bash
tail -f outputs/Paper-Backup/E2-Final-256Budget/pipeline.log

find outputs/Paper-Backup/E2-Final-256Budget/_launcher_logs \
  -name '*.log' -type f
```

## Separate stages

Preprocess the three paired products:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
PAPER_BACKUP_WORKERS=8 \
bash scripts/paper_backup/run_e2_final_256budget_pipeline.sh preprocess
```

Validate data, configs, parameter counts and the 150-task launch matrix:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
bash scripts/paper_backup/run_e2_final_256budget_pipeline.sh validate
```

Train after preprocessing is complete:

```bash
nohup env \
  PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  GPU_IDS="0 1" JOBS_PER_GPU=3 NUM_WORKERS=4 \
  RUN_TIME=e2_final_256budget_seed10 \
  bash scripts/paper_backup/run_e2_final_256budget_pipeline.sh train \
  > outputs/Paper-Backup/E2-Final-256Budget/train.log 2>&1 &
```

The launcher defaults to all five models.  A failed or bounded subset can be
rerun with, for example:

```bash
MODELS="ours_cc_only_128 ours_cv_only_128" \
SEEDS="42 52" \
bash scripts/paper_backup/run_e2_final_256budget.sh
```

Summarize all ten formal seeds:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
bash scripts/paper_backup/run_e2_final_256budget_pipeline.sh summary
```

## Formal outputs

```text
outputs/Paper-Backup/E2-Final-256Budget/summaries/e2_final_256budget_status.json
outputs/Paper-Backup/E2-Final-256Budget/summaries/e2_final_256budget_metrics_per_seed.csv
outputs/Paper-Backup/E2-Final-256Budget/summaries/e2_final_256budget_metrics_mean_std.csv
outputs/Paper-Backup/E2-Final-256Budget/summaries/e2_final_256budget_paired_gaps_per_seed.csv
outputs/Paper-Backup/E2-Final-256Budget/summaries/e2_final_256budget_paired_gaps_mean_std.csv
```

The paired tables verify identical `(battery_id, cycle_id, y_true)` coverage
before they are written.  Their signed comparisons are:

```text
Terminal Vanilla - Full Vanilla       observation-window gap
Ours - Terminal Vanilla               phase-aware method gain
Ours - CC-only                        value of adding CV
Ours - CV-only                        value of adding CC
Ours - Full Vanilla                   terminal-system vs full reference
```

For every row, a negative delta means the left-hand model has lower error.
