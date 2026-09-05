# Paper-Backup launchers

E1 and E2 are formal dynamic launchers and default to `DRY_RUN=0`; E3 retains
the validation-first default `DRY_RUN=1`. All launchers use the local Python
resolver supplied by the caller. Build and validate
`datasets/PaperBackup_preprocessed` first; training never opens a vendor source
or performs sequence interpolation.

The isolated C-rate/FullVI suite uses `run_e1_crate.sh`, schema v2 data and a
separate summary selector. See
[`CRATE_V2_RUNBOOK.md`](../../docs/paper_backup/CRATE_V2_RUNBOOK.md).
The fixed-length CC=128/CV=128 ablation has its own data, experiment and result
namespaces. See
[`CRATE_128X128_RUNBOOK.md`](../../docs/paper_backup/CRATE_128X128_RUNBOOK.md).
The three-model shared stable-training rerun has a separate 45-job matrix and
result namespace. See
[`E1_STABLE_128X128_RUNBOOK.md`](../../docs/paper_backup/E1_STABLE_128X128_RUNBOOK.md).
The final five-seed interaction-fusion E1/E2 suites can be reproduced from
vendor sources on a new server with one active-Conda launcher. See
[`FINAL_INTERACTION_PORTABLE_RUNBOOK.md`](../../docs/paper_backup/FINAL_INTERACTION_PORTABLE_RUNBOOK.md).

The E1 `hi_mlp` group is reported as **Feature MLP (PINN4SOH F-only
structure)**. Its 24 statistics and SOH labels come from every record in the
same offline Terminal product used by the raw models: no 3-sigma cleaning,
adjacent-x1 selection, or other model-specific cycle filter is applied. The
archived run/task ID remains `Final-PINN4SOH-like-MLP` so completed five-seed
runs remain reusable; the summary layer emits the corrected descriptive model
ID. To rebuild tables without retraining, run:

```bash
PYTHON_BIN=/path/to/python \
  bash scripts/paper_backup/run_e1_final_interaction_5seed_pipeline.sh summary
PYTHON_BIN=/path/to/python \
  bash scripts/paper_backup/run_e1_bicontext_5seed_pipeline.sh summary
```

The independent BiContext E1 supplement reuses the same schema-v2 128+128
data, five seeds, 600-epoch cap, patience 30 and batch size 128. It trains only
the five dataset-specific BiContext configurations (25 jobs) and does not
overwrite the archived final E1 matrix:

```bash
bash scripts/paper_backup/run_e1_bicontext_5seed_pipeline.sh validate
GPU_IDS="0 1" MAX_PARALLEL=3 \
  bash scripts/paper_backup/run_e1_bicontext_5seed_pipeline.sh train
bash scripts/paper_backup/run_e1_bicontext_5seed_pipeline.sh summary
```

The optional `diagnostics` stage performs best-checkpoint inference and writes
the two directional gate/contribution distributions. It requires a visible
formal Mamba GPU and defaults to `cuda:0`.

The Late Latent-Token BiContext supplement is a function-preserving extension
of Raw Dual Vanilla: the complete baseline is inherited unchanged, cross is
fixed between Mamba layers 2 and 3, each phase has four learned latent tokens,
and only two four-head token-to-latent attentions are added. Both attention
output projections start at exactly zero. The isolated 25-job matrix uses the
same data, split, optimizer, scheduler, five seeds, epoch cap, patience and
batch size as formal E1.

```bash
bash scripts/paper_backup/run_e1_late_latent_token_bicontext_5seed_pipeline.sh validate
GPU_IDS="0 1" MAX_PARALLEL=3 \
  bash scripts/paper_backup/run_e1_late_latent_token_bicontext_5seed_pipeline.sh train
bash scripts/paper_backup/run_e1_late_latent_token_bicontext_5seed_pipeline.sh summary
DEVICE_OVERRIDE=cuda:0 \
  bash scripts/paper_backup/run_e1_late_latent_token_bicontext_5seed_pipeline.sh diagnostics
```

The summary stage checks identical per-seed test coverage and writes a
three-model table for Raw Dual Vanilla, current Mean-BiContext and Late
Latent-Token BiContext. Results are isolated under
`outputs/Paper-Backup/E1-Late-LatentToken-BiContext-5Seed`.

The stability follow-up keeps K=4, four heads and cross after layer 2 fixed,
but applies independent CC/CV Pre-Norm and bounded ReZero scales
`0.1*tanh(alpha)`. Both alphas start at zero, while MHA uses ordinary
initialization, so the initial function still equals Raw Dual exactly. The
formal matrix uses seeds 42, 52, 62, 72 and 82:

```bash
bash scripts/paper_backup/run_e1_late_latent_token_bicontext_rezero_5seed_pipeline.sh validate
GPU_IDS="0 1" MAX_PARALLEL=3 \
  bash scripts/paper_backup/run_e1_late_latent_token_bicontext_rezero_5seed_pipeline.sh train
bash scripts/paper_backup/run_e1_late_latent_token_bicontext_rezero_5seed_pipeline.sh summary
DEVICE_OVERRIDE=cuda:0 \
  bash scripts/paper_backup/run_e1_late_latent_token_bicontext_rezero_5seed_pipeline.sh diagnostics
```

Its outputs are isolated under
`outputs/Paper-Backup/E1-Late-LatentToken-BiContext-ReZero-5Seed`. The completed
two-seed ReZero screening output remains archived under its original `2Seed`
namespace, and the original five-seed late-token checkpoints retain the
zero-output-projection behavior.

The isolated Cycle MTL supplement keeps the complete BiContext inference path
unchanged and adds a 65-parameter linear cycle-order head during training only.
Its target is the within-battery chronological rank transformed by `log1p` and
scaled with the training split maximum; it does not divide by a battery's final
cycle. The SOH head never receives the real or predicted cycle value.

```bash
bash scripts/paper_backup/run_e1_bicontext_cycle_mtl_5seed_pipeline.sh validate
GPU_IDS="0 1" MAX_PARALLEL=3 \
  bash scripts/paper_backup/run_e1_bicontext_cycle_mtl_5seed_pipeline.sh train
bash scripts/paper_backup/run_e1_bicontext_cycle_mtl_5seed_pipeline.sh summary
```

Results are isolated under
`outputs/Paper-Backup/E1-BiContext-CycleMTL-5Seed`. The summary stage also
writes paired per-seed deltas against the archived plain BiContext run when
`outputs/Paper-Backup/E1-BiContext-5Seed/summaries` is available.

The BiContext Adaptive Fusion supplement retains the complete bidirectional
bridge and the 27-unit ordinary fusion residual. It adds a 521-parameter
sample-adaptive CC/CV gate whose zero-initialized output starts from the exact
0.5/0.5 BiContext mixture. Its 25 formal jobs and paired BiContext comparison
are run with:

```bash
bash scripts/paper_backup/run_e1_bicontext_adaptive_fusion_5seed_pipeline.sh validate
GPU_IDS="0 1" MAX_PARALLEL=3 \
  bash scripts/paper_backup/run_e1_bicontext_adaptive_fusion_5seed_pipeline.sh train
bash scripts/paper_backup/run_e1_bicontext_adaptive_fusion_5seed_pipeline.sh summary
```

Results are isolated under
`outputs/Paper-Backup/E1-BiContext-AdaptiveFusion-5Seed`.

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

Formal Mamba runs should leave `backend_override` unset. Full E2 jobs read the
offline FULL arrays and all E2 views use the same FULL-matched cohort.

E1 and E2 use the same dynamic GPU-slot scheduler. Their default matrices have
seeds `42 52 62`, and every GPU has three concurrent slots. Each config/seed
pair is an independent queue item: as soon as any slot finishes, that same GPU
starts the next queued item without waiting for the other seeds or GPUs. GPU
IDs may be comma- or space-separated; inside each child process the selected
physical GPU is remapped to `cuda:0`. Both launchers explicitly default to 400
epochs, early-stop patience 20, batch size 128 and one persistent DataLoader
worker per loader. Pinned memory and a prefetch factor of two are enabled by
the common configs. Environment variables can override the launch values:

`MODELS` selects config groups and defaults to `all`. Values may be separated
by spaces or commas. E1 accepts `hi_mlp`, `ours`, and `transformer`; E2 accepts
`full_vanilla`, `terminal_cc_only`, `terminal_cv_only`, `terminal_ours`, and
`terminal_vanilla`.

For `run_e1_crate_128x128.sh`, the available groups are `ours_dominant`,
`ours_fullvi`, `ours_gated`, `ours_pointbridge`, `smaller_transformer`, and
`transformer`.  Use `MODELS="smaller_transformer transformer"` to run only the
two 128x128 Transformer controls without rescheduling the completed Ours jobs.

```bash
RUN_TIME=e1_250ep_bs128 \
MODELS="ours transformer" \
GPU_IDS="0 1" MAX_PARALLEL=3 SEEDS="42 52 62" \
EPOCHS=250 PATIENCE=40 BATCH_SIZE=128 NUM_WORKERS=1 \
DRY_RUN=0 DEVICE_OVERRIDE=cuda:0 \
PYTHON_BIN=/path/to/python \
  bash scripts/paper_backup/run_e1.sh
```

The corresponding E2 launch is:

```bash
MODELS="full_vanilla terminal_ours terminal_vanilla" \
GPU_IDS="0 1" MAX_PARALLEL=3 SEEDS="42 52 62" \
EPOCHS=400 PATIENCE=20 BATCH_SIZE=128 NUM_WORKERS=1 \
CHECK_DATA=1 DRY_RUN=0 DEVICE_OVERRIDE=cuda:0 \
PYTHON_BIN=/path/to/python \
  bash scripts/paper_backup/run_e2.sh
```

For a detached run, redirect the launcher output in addition to the per-job
logs already written below `_launcher_logs`:

```bash
nohup env MODELS="ours transformer" GPU_IDS="0 1" MAX_PARALLEL=3 \
  SEEDS="42 52 62" RUN_TIME=e1_timefix_matched \
  PYTHON_BIN=/path/to/python \
  bash scripts/paper_backup/run_e1.sh \
  > outputs/Paper-Backup/e1_timefix_matched.nohup.log 2>&1 &
```

The aggregate concurrency in this example is six. Set `MAX_PARALLEL=1` when
memory is tight. `MAX_PARALLEL` is the maximum number of child processes on
each GPU, so total concurrency is GPU count multiplied by `MAX_PARALLEL`.
Launcher logs are written below
`outputs/Paper-Backup/_launcher_logs/<experiment>/runtime_<RUN_TIME>/`.
Every training epoch writes one line with train loss, validation loss,
monitored battery-macro RMSE, best status and remaining early-stop patience,
so a running job can be inspected with `tail -f` on its seed log.

The model-only smoke (no dataset load and no output write) is:

```bash
/path/to/python scripts/paper_backup/smoke_test.py
```

The preprocessing and training order is:

```bash
OVERWRITE=1 bash preprocess/paper_backup/run_preprocess.sh all
bash preprocess/paper_backup/run_preprocess.sh validate

DRY_RUN=0 DEVICE_OVERRIDE=cuda BACKEND_OVERRIDE=mamba_ssm.Mamba \
  bash scripts/paper_backup/run_e1.sh

DRY_RUN=0 DEVICE_OVERRIDE=cuda BACKEND_OVERRIDE=mamba_ssm.Mamba \
  bash scripts/paper_backup/run_e2.sh
```

E3 uses the same offline Terminal arrays:

```bash
DRY_RUN=0 DEVICE_OVERRIDE=cuda BACKEND_OVERRIDE=mamba_ssm.Mamba \
  bash scripts/paper_backup/run_e3.sh
```

Completed runs can be summarized with:

```bash
# E1 is the default; runtime directory names do not need to be supplied.
python scripts/paper_backup/summarize_results.py

python scripts/paper_backup/summarize_results.py --experiment e2
python scripts/paper_backup/summarize_results.py --experiment e3
```

The expected seed set defaults to `42 52 62` and can be changed explicitly:

```bash
python scripts/paper_backup/summarize_results.py \
  --experiment e1 --seeds "42 52 62"
```

For each model/data/seed task, the summarizer ignores bounded debug runs and
selects the newest completed formal result across all `runtime_*` directories.
Duplicate choices are recorded in
`outputs/Paper-Backup/summaries/<experiment>_status.json`. A missing job is
never treated as zero: formal CSV/JSON tables are written only after the
config-owned experiment matrix is complete. In particular, E3 reports
`waiting_for_strategy_specific_runs` until every strategy-specific and pooled
job is present.

The main output for paper tables is
`<experiment>_metrics_mean_std.csv`; `<experiment>_metrics_per_seed.csv`
retains the individual seed values. MAPE is reported in percent and RMSE in
SOH percentage points. Both pooled-cycle and battery-macro forms are included.
