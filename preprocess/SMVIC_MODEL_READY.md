# SMVIC model-ready preprocessing

This pipeline converts the external normalized SMVIC cell CSVs into one
audited product shared by the Paper-Backup PINN4SOH-like MLP and Ours
BiContext Mamba. Generated arrays live under the repository `datasets/`
namespace like the public products, but remain Git-ignored.

## Label contract

The target follows the current experiment convention exactly:

```text
SOH = cycle_discharge_capacity_Ah / fixed nominal_capacity_Ah
```

The builder recomputes that value and requires it to match the source
`SOH_nominal`.  It does not apply BOL re-referencing, smoothing, monotonic
repair, or clipping to `[0, 1]`.  Current-cycle charge input is paired with the
same cycle's subsequent discharge-capacity label.

## Environment

Run commands from the `UnifiedRawSOH` repository root.  The examples use the
project Python environment:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python
SOURCE_ROOT=/data1/chenyanxi/lb_project/datasets/SMVIC/dataset
AUDIT_ROOT=$PWD/datasets/SMVIC_preprocess_audit
OUTPUT_ROOT=$PWD/datasets/SMVIC_preprocessed_v2_128x128
```

## 1. Protocol audit

First run a bounded check:

```bash
"$PYTHON_BIN" preprocess/audit_smvic.py \
  --source-root "$SOURCE_ROOT" \
  --output-root /tmp/smvic_audit_smoke \
  --groups all \
  --max-cycles-per-cell 5
```

Then scan the complete source:

```bash
"$PYTHON_BIN" preprocess/audit_smvic.py \
  --source-root "$SOURCE_ROOT" \
  --output-root "$AUDIT_ROOT" \
  --groups all
```

Review `summary.json` and `cycle_audit.csv`.  Battery04 is accepted only when
its main charge/discharge rates match the roughly 1C aging protocol and its
charge temperature is 35--50 degrees C.  Battery05 cycle 0, low-rate capacity
steps, and DCR cycles are excluded by the protocol classifier.

## 2. Build model-ready arrays

Run a disposable bounded product before the formal build:

```bash
"$PYTHON_BIN" preprocess/build_smvic_preprocessed.py \
  --source-root "$SOURCE_ROOT" \
  --output-root /tmp/smvic_model_ready_smoke \
  --groups all \
  --max-cycles-per-cell 5 \
  --overwrite
```

Build all six families at 128 CC and 128 CV points:

```bash
"$PYTHON_BIN" preprocess/build_smvic_preprocessed.py \
  --source-root "$SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --groups all \
  --cc-len 128 \
  --cv-len 128
```

Existing domain products are protected.  Pass `--overwrite` only when an
intentional regeneration is required.  To process selected groups, replace
`--groups all` with one or more of `Battery01` through `Battery06`.

Each domain directory contains `terminal_cc.npy`, `terminal_cv.npy`,
`terminal_features.npy`, `terminal_soh.npy`, `terminal_index.csv`, a manifest,
a physical-cell split, and complete exclusion/classification audits.

## 3. Validate and smoke-test both models

```bash
"$PYTHON_BIN" preprocess/validate_smvic_preprocessed.py \
  --input-root "$OUTPUT_ROOT" \
  --domains all
```

Validation checks checksums, array shapes, finite values, SOH/index agreement,
unique physical-cycle keys, physical-cell holdout membership, and forward
passes through both target models.  The BiContext smoke test uses the
device-independent `torch_reference` backend; formal training still uses the
configured `mamba_ssm.Mamba` backend.

## 4. Training examples

The checked-in configs point at the default external output root above.

PINN4SOH-like MLP on Battery05/type3:

```bash
"$PYTHON_BIN" scripts/paper_backup/train.py \
  --config configs/paper_backup/e4_industrial_external/smvic/hi_mlp/smvic_type3_108ah.json \
  --seed 42
```

Ours BiContext on the same physical cycles and split:

```bash
"$PYTHON_BIN" scripts/paper_backup/train.py \
  --config configs/paper_backup/e4_industrial_external/smvic/bicontext/smvic_type3_108ah.json \
  --seed 42
```

Replace the final config name with any of:

- `smvic_e72_69ah.json`
- `smvic_s5e891_51ah.json`
- `smvic_type1_18ah.json`
- `smvic_type2_150ah_t40.json`
- `smvic_type3_108ah.json`
- `smvic_type4_11ah.json`

The five two-cell families use symmetric physical holdout: one run trains on
Cell01 and tests Cell02, and the second swaps those roles. Battery05 uses two
deterministic two-cell test selections: Python `random.sample` seed 420 selects
Cells 01/06 and seed 421 selects Cells 05/07; the other six cells form the
development cohort. Every development cohort uses mixed-cycle train/validation
with `random_state=420`.

## 5. One-seed multi-GPU comparison

The comparison launcher runs the complete 24-task matrix: six SMVIC domains,
two physical test protocols per domain, and two models. The model-training seed
is fixed to 42 by default. Each reported domain/model result is the unweighted
mean of its two test evaluations; the two detailed results are also retained.

Validate all configs and data without training:

```bash
DRY_RUN=1 GPU_IDS="0,1" \
  bash scripts/paper_backup/run_smvic_one_seed.sh all
```

Run two GPUs with one training process per GPU:

```bash
GPU_IDS="0,1" MAX_PARALLEL=1 \
  bash scripts/paper_backup/run_smvic_one_seed.sh all
```

Run two concurrent processes per GPU. With two GPUs, aggregate concurrency is
automatically four:

```bash
GPU_IDS="0,1" MAX_PARALLEL=2 \
  bash scripts/paper_backup/run_smvic_one_seed.sh all
```

Useful overrides include:

```text
SEED=42
MODELS="hi_mlp bicontext"
DOMAINS="smvic_type3_108ah smvic_type4_11ah"
EPOCHS=600
PATIENCE=30
BATCH_SIZE=128
NUM_WORKERS=4
OUTPUT_ROOT=/path/to/results
```

For a short pipeline check rather than a meaningful performance result:

```bash
GPU_IDS="0,1" EPOCHS=2 PATIENCE=1 \
  bash scripts/paper_backup/run_smvic_one_seed.sh train
```

Do not report the two-epoch output as model performance. The default 600/30
setting follows the current final experiment training budget.

After training, the `all` stage automatically writes CSV, JSON, and Markdown
comparisons. To regenerate only the summary:

```bash
bash scripts/paper_backup/run_smvic_one_seed.sh summary
```

The default summary is stored under:

```text
outputs/Paper-Backup/E4-SMVIC-OneSeed/summaries/seed_42/
├── smvic_one_seed_details.csv
├── smvic_one_seed_metrics.csv
├── smvic_one_seed_metrics.json
└── smvic_one_seed_metrics.md
```
