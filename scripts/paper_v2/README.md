# Paper-v2 BOL retraining

"run_bol_soh_retraining.sh" is the only Paper-v2 training launcher. It
defaults to DRY_RUN=1, so previewing the job matrix never starts training and
does not create output files. Set DRY_RUN=0 only when the formal retraining
is intended.

The launcher schedules at most GPU_COUNT * JOBS_PER_GPU child processes. Each
child receives one CUDA_VISIBLE_DEVICES value and uses cuda:0 internally.

Stage commands:

~~~bash
# Single-domain Feature MLP, all five domains
STAGE=e1_feature TARGET_DOMAINS=all DRY_RUN=0 \
  bash scripts/paper_v2/run_bol_soh_retraining.sh

# Single-domain no-cycle RawMamba, all five domains
STAGE=e1_raw TARGET_DOMAINS=all DRY_RUN=0 \
  bash scripts/paper_v2/run_bol_soh_retraining.sh

# Full-domain no-cycle RawMamba with domain-balanced hierarchical sampling
STAGE=e2_full TARGET_DOMAINS=all DRY_RUN=0 \
  bash scripts/paper_v2/run_bol_soh_retraining.sh

# Five-fold zero-cell LODO no-cycle RawMamba
STAGE=e3_lodo TARGET_DOMAINS=all DRY_RUN=0 \
  bash scripts/paper_v2/run_bol_soh_retraining.sh
~~~

All stages and all three seeds:

~~~bash
STAGE=all TARGET_DOMAINS=all SEEDS="42 52 62" \
  GPU_IDS="0 1 2" JOBS_PER_GPU=2 DRY_RUN=0 RESUME=1 \
  bash scripts/paper_v2/run_bol_soh_retraining.sh
~~~

Useful scheduling variants:

~~~bash
# One GPU, three independent seed processes
STAGE=e1_raw TARGET_DOMAINS=xjtu SEEDS="42 52 62" \
  GPU_IDS=0 JOBS_PER_GPU=3 DRY_RUN=0 \
  bash scripts/paper_v2/run_bol_soh_retraining.sh

# Three GPUs, at most two processes per GPU
STAGE=all TARGET_DOMAINS=all GPU_IDS="0 1 2" JOBS_PER_GPU=2 \
  SEEDS="42 52 62" DRY_RUN=0 \
  bash scripts/paper_v2/run_bol_soh_retraining.sh

# Safe matrix preview
STAGE=all TARGET_DOMAINS=all SEEDS="42 52 62" \
  GPU_IDS="0 1 2" JOBS_PER_GPU=2 DRY_RUN=1 \
  bash scripts/paper_v2/run_bol_soh_retraining.sh
~~~

The optional Feature MLP LODO interface has five runnable fold configs under
configs/paper_v2/e3_lodo_zero_cell/feature_mlp/. It can be invoked directly
with UnifiedRawSOH.main_baseline; the loader concatenates only source
train/validation data and exposes only target test data.

Each completed seed is written below
outputs/Paper-v2/<experiment>/<model>/<data>/runtime_*/seed_<seed>/.
The seed contains test_metrics.json, the three hierarchical metric CSVs,
split_info.json, and a checkpoint. A batch summary is generated only after
all expected seeds and domains are complete. The final table is
outputs/Paper-v2/main_table.csv and main_table.md. No code path in this
launcher writes outputs/Paper-v1.

Bounded smoke test:

~~~bash
PYTHONPATH=.. /data/chenyanxi/miniconda3/envs/pinn/bin/python \
  tests/paper_v2_smoke_test.py
~~~
