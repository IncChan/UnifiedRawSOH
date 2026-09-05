# SMVIC curated model-ready preprocessing

This pipeline converts the normalized enterprise SMVIC cell CSVs into one
audited product shared by the Paper-Backup PINN4SOH-like MLP, Raw Vanilla
Mamba, and Ours BiContext Mamba. Generated arrays are stored below `datasets/`,
like the public-data products, but are excluded from Git.

## What is repaired

The source capacity and SOH values are never overwritten. The policy in
`preprocess/smvic_quality_policy.json` quarantines cycles with traceable
measurement/protocol failures, and records each rejected cycle and its reason
in the output audit files.

Two quality checks are applied before feature/window extraction:

1. Curated exclusions for 11 confirmed abnormal cycles: long interruptions,
   corrupted/truncated terminal charge, or an isolated capacity-label
   dip/rebound.
2. Dataset-wide continuity checks: reject a cycle when source record time goes
   backwards, or when adjacent records have a gap greater than 3600 seconds.

The continuity threshold and curated evidence are versioned and checksummed in
every output manifest. To change the policy, edit the JSON policy explicitly
and rebuild; do not silently patch labels in generated arrays.

The full audit rejects 41 cycles for quality reasons (11 curated and 30 found
by the generic continuity checks) and yields the following model-eligible
aging cycles:

| Source group | Model domain | Eligible | Quality-rejected | SOH range | Per-cell eligible count |
|---|---|---:|---:|---:|---|
| Battery01 | `smvic_e72_69ah` | 998 | 2 | 0.9692–1.0418 | 499 / 499 |
| Battery02 | `smvic_s5e891_51ah` | 994 | 6 | 0.9846–1.0025 | 497 / 497 |
| Battery03 | `smvic_type1_18ah` | 1751 | 5 | 0.6162–0.9388 | 887 / 864 |
| Battery04 | `smvic_type2_150ah_t40` | 672 | 12 | 0.7682–1.0015 | 338 / 334 |
| Battery05 | `smvic_type3_108ah` | 1694 | 14 | 0.8862–0.9763 | 226 / 221 / 225 / 224 / 218 / 141 / 220 / 219 |
| Battery06 | `smvic_type4_11ah` | 934 | 2 | 0.8655–0.9368 | 467 / 467 |

These counts also include the pre-existing protocol rules. For example,
Battery04 keeps only its approximately 1C, 35--50 °C aging cycles; Battery05
excludes cycle 0, low-rate capacity checks, and DCR cycles.

## Label and input contract

The target remains the current experiment definition:

```text
SOH = cycle_discharge_capacity_Ah / fixed nominal_capacity_Ah
```

The builder recomputes this ratio and requires exact agreement with the source
`SOH_nominal`. It applies no BOL re-reference, smoothing, interpolation,
monotonic correction, or `[0, 1]` clipping. The current cycle's charge input is
paired with the same cycle's subsequent discharge-capacity label.

For each eligible cycle, preprocessing finds the principal contiguous CC→CV
charge event. The CC terminal window is the last 0.20 V below charge cutoff
(0.25 V for Battery04/05); the CV window is near cutoff and between 0.04C and
0.30C. Each physical-time window is independently resampled to 128 points.
Both models therefore use exactly the same retained cycles and labels:

- Raw Vanilla Mamba concatenates the same CC/CV points into one 256-point
  sequence with a boundary token; Ours consumes them as two `[128, 7]` phase
  arrays.
- PINN4SOH-like MLP consumes the 24 statistical features computed from the
  same unresampled CC/CV points. Feature standardization is fitted only on the
  training partition.

## Run preprocessing

Run all commands from the repository root:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python
SOURCE_ROOT=/data1/chenyanxi/lb_project/datasets/SMVIC/dataset
QUALITY_POLICY=$PWD/preprocess/smvic_quality_policy.json
AUDIT_ROOT=$PWD/datasets/SMVIC_preprocess_audit_v3
OUTPUT_ROOT=$PWD/datasets/SMVIC_preprocessed_v3_128x128
```

First perform a cheap bounded build:

```bash
"$PYTHON_BIN" preprocess/build_smvic_preprocessed.py \
  --source-root "$SOURCE_ROOT" \
  --output-root /tmp/smvic_model_ready_v3_smoke \
  --quality-policy "$QUALITY_POLICY" \
  --groups all \
  --max-cycles-per-cell 5 \
  --overwrite

"$PYTHON_BIN" preprocess/validate_smvic_preprocessed.py \
  --input-root /tmp/smvic_model_ready_v3_smoke \
  --quality-policy "$QUALITY_POLICY" \
  --domains all
```

Then generate the full audit and final product:

```bash
"$PYTHON_BIN" preprocess/audit_smvic.py \
  --source-root "$SOURCE_ROOT" \
  --output-root "$AUDIT_ROOT" \
  --quality-policy "$QUALITY_POLICY" \
  --groups all

"$PYTHON_BIN" preprocess/build_smvic_preprocessed.py \
  --source-root "$SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --quality-policy "$QUALITY_POLICY" \
  --groups all \
  --cc-len 128 \
  --cv-len 128

"$PYTHON_BIN" preprocess/validate_smvic_preprocessed.py \
  --input-root "$OUTPUT_ROOT" \
  --quality-policy "$QUALITY_POLICY" \
  --domains all
```

Existing domain products are protected. Use `--overwrite` on the build command
only when deliberately regenerating v3. The full product contains arrays,
`terminal_index.csv`, `splits.json`, a checksummed manifest, and complete
classification/exclusion audits for every domain.

## Evaluation split

All train/validation partitions are fixed with `random_state=420`.

- The five two-cell groups run two folds: train/develop on Cell01 and test on
  Cell02, then swap the cells. Report their unweighted mean.
- Battery05 runs two fixed protocols. Seed 420 selects Cell01/Cell06 as test;
  seed 421 selects Cell05/Cell07. Each protocol trains/develops on the other
  six cells, and the two test results are averaged.

The model optimization seed is independently fixed to 42 for this quick
comparison.

## Re-run Ours and PINN4SOH-like experiments

First validate the complete 36-task matrix without starting training:

```bash
DRY_RUN=1 GPU_IDS="0,1" \
  bash scripts/paper_backup/run_smvic_one_seed.sh all
```

Run one process per GPU:

```bash
nohup env GPU_IDS="0,1" MAX_PARALLEL=1 \
  bash scripts/paper_backup/run_smvic_one_seed.sh all \
  > smvic_curated_seed42.log 2>&1 &
```

Or allow at most two simultaneous processes on each GPU (four in total for
two GPUs):

```bash
nohup env GPU_IDS="0,1" MAX_PARALLEL=2 \
  bash scripts/paper_backup/run_smvic_one_seed.sh all \
  > smvic_curated_seed42.log 2>&1 &
```

`MAX_PARALLEL` is a per-GPU limit. Useful overrides are:

```text
SEED=42
MODELS="hi_mlp raw_vanilla bicontext"
DOMAINS="smvic_type3_108ah smvic_type4_11ah"
EPOCHS=600
PATIENCE=30
BATCH_SIZE=128
NUM_WORKERS=4
OUTPUT_ROOT=/path/to/results
```

The default results are written to
`outputs/Paper-Backup/E4-SMVIC-Curated-OneSeed`. The `all` stage trains and
then creates CSV, JSON, and Markdown summaries. To rebuild only the summary:

```bash
bash scripts/paper_backup/run_smvic_one_seed.sh summary
```

Summary files are under
`outputs/Paper-Backup/E4-SMVIC-Curated-OneSeed/summaries/seed_42/`.

If the PINN4SOH-like and BiContext runs already exist, train only the added
single-stream Raw Vanilla baseline and then combine all three models in one
summary:

```bash
nohup env GPU_IDS="4" MAX_PARALLEL=5 SEED=42 \
  MODELS="raw_vanilla" \
  bash scripts/paper_backup/run_smvic_one_seed.sh all \
  > smvic_curated_raw_mamba_seed42.log 2>&1 &
```

The result discovery is recursive and selects the newest completed run for
each model/domain/protocol key, so the new summary reuses the existing 24
PINN4SOH-like/BiContext results and adds the 12 Raw Vanilla results without
retraining the first two models.
