# Paper-v2 P0–P2 状态

| Component | Status | Config | Command | Output | Tests | Last verified | Limitations |
|---|---|---|---|---|---|---|---|
| P0 config contract / isolated entry point | smoke-tested | `configs/paper_v2/common/bol_soh_base.json` | `python scripts/paper_v2/train.py --config <config> --validate_only` | none in validation mode | `tests/paper_v2/test_config_contract.py` | 2026-08-27 | real training still requires declared data and backend |
| Residual MoE | smoke-tested | `configs/paper_v2/common/moe_base.json` | `python scripts/paper_v2/train.py --config <moe-config> --backend_override torch_reference --device_override cpu` | `outputs/Paper-v2/...` | `test_model_contract.py` | 2026-08-27 | no formal result was run |
| Dense Adapter control | smoke-tested | `configs/paper_v2/e2_full_domain/dense_adapter/config.json` | `bash scripts/paper_v2/run_e2_seen_domain.sh` | `outputs/Paper-v2/e2_full_domain/...` | `test_model_contract.py` | 2026-08-27 | integer-width parameter match is reported, not retuned |
| Hierarchical sampler | smoke-tested | E2/E3 `data.sampler` | covered by CPU unit tests | `sampling_audit.json` | `test_hierarchical_sampler.py` | 2026-08-27 | replacement sampler is train-only |
| Source pseudo-LODO / first-order MLDG | smoke-tested | `configs/paper_v2/common/dg_base.json` and E3 DG folds | `bash scripts/paper_v2/run_e3_zero_cell.sh` | `episode_audit.json` | `test_episodic_dg.py`, `test_lodo_leakage.py` | 2026-08-27 | one inner step; no P3 adaptation |
| E2 seen-domain matrix | runnable | `configs/paper_v2/e2_full_domain/*/config.json` | `DRY_RUN=1 bash scripts/paper_v2/run_e2_seen_domain.sh` | Paper-v2 namespace only | config/shell smoke | 2026-08-27 | formal GPU jobs not started |
| E3 five-fold zero-cell matrix | runnable | `configs/paper_v2/e3_lodo_zero_cell/{base_erm,dense_adapter_erm,moe_erm,moe_dg}/lodo_*.json` | `DRY_RUN=1 bash scripts/paper_v2/run_e3_zero_cell.sh` | Paper-v2 namespace only | config/leakage smoke | 2026-08-27 | formal GPU jobs not started |
| P3 one-trajectory adaptation | planned | — | — | — | — | 2026-08-27 | intentionally out of scope |
| P4 enterprise experiment | planned | — | — | — | — | 2026-08-27 | intentionally out of scope |

The scientific path is kept separable: Base-ERM, Dense-Adapter-ERM,
Residual-MoE-ERM, and Residual-MoE-DG have different model/trainer/output IDs.
The existing `run_bol_soh_retraining.sh` remains the compatibility launcher for
the already implemented Paper-v2 BOL baseline; the new launchers call the
independent `scripts/paper_v2/train.py` entry point.

No formal multi-seed GPU training or paper result claim is included in this
P0–P2 implementation.
