# Paper-Backup core-three 10-seed runbook

This view reuses the existing `e1_shared_crate_128x128` experiment, data,
model IDs and original training protocol. It does not introduce a scheduler,
gradient accumulation or a new development-validation split.

The formal seed set is:

```text
42 52 62 72 82 92 102 112 122 123
```

The three models are PINN4SOH-like MLP, parameter-matched Smaller Transformer,
and Ours FullVI PointBridge. Existing Ours/Transformer results for 42/52/62
are reused. The pipeline trains the MLP on all ten seeds and the two sequence
models on the seven additional seeds, for 120 new jobs and 150 summarized
jobs in total.

Run the missing jobs and write the final summary in one background process:

```bash
mkdir -p outputs/Paper-Backup/CRateV2-128x128

nohup env \
  PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  GPU_IDS="0 1" \
  JOBS_PER_GPU=3 \
  RUN_TIME=e1_core3_seed10_128x128 \
  bash scripts/paper_backup/run_e1_core3_seed10_128x128_pipeline.sh all \
  > outputs/Paper-Backup/CRateV2-128x128/core3_seed10_pipeline.log 2>&1 &
```

Monitor with:

```bash
tail -f outputs/Paper-Backup/CRateV2-128x128/core3_seed10_pipeline.log
```

Generate the summary separately after training:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
bash scripts/paper_backup/run_e1_core3_seed10_128x128_pipeline.sh summary
```

The two main tables are written to:

```text
outputs/Paper-Backup/CRateV2-128x128/summaries/e1_core3_128x128_metrics_per_seed.csv
outputs/Paper-Backup/CRateV2-128x128/summaries/e1_core3_128x128_metrics_mean_std.csv
```
