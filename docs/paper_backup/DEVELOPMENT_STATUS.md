# Paper-Backup development status

Last verified: 2026-08-27. No formal performance result is stored here.

| component | status | config | command | output | tests | last verified | limitation |
|---|---|---|---|---|---|---|---|
| Config contract / inheritance | unit-tested, smoke-tested | `configs/paper_backup/common/` | `scripts/paper_backup/run_e*.sh` with `DRY_RUN=1` | none in dry-run | 47 configs validated | 2026-08-27 | full source paths are user-local |
| E1 HI-MLP / Transformer / Ours | implemented, unit-tested, smoke-tested | `configs/paper_backup/e1_main_estimation/` | `run_e1.sh` | Paper-Backup seed layout | CPU synthetic forward/backward | 2026-08-27 | no formal multi-seed training |
| E2 terminal views | implemented, unit-tested, smoke-tested | `configs/paper_backup/e2_charging_information/terminal_*` | `run_e2.sh` | Paper-Backup seed layout | paired-record contract tests | 2026-08-27 | set `matched_full_data_root` for a full/terminal common cohort |
| E2 full Vanilla | implemented, blocked_by_data | `configs/paper_backup/e2_charging_information/full_vanilla/` | `run_e2.sh` | no run until unblocked | full/terminal invariant tests | 2026-08-27 | configure real point-level full source and matching audit |
| E3 XJTU / LISHEN | implemented, unit-tested, smoke-tested | `configs/paper_backup/e3_strategy_pooling/` | `run_e3.sh` | Paper-Backup seed layout | sampler and pooled-cohort tests | 2026-08-27 | no cross-domain pooling |
| Evaluation / summaries | implemented | `evaluation/paper_backup/`, `scripts/paper_backup/summarize_results.py` | `summarize_results.py` | JSON summary on request | aggregation/paired tests | 2026-08-27 | no result files before training |

The E1 matrix is intentionally only the three methods requested for this
development pass. RawCNN/LSTM controls are reusable code paths but have no E1
configuration. The official CUDA Mamba path and long training jobs were not
started.
