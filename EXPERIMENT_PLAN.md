# Paper-v1 experiment plan

This plan follows one paper story: Raw → Unified → Reusable. The experimental
unit is a battery domain, and each config records its domain IDs, data roots,
split JSON, normalization, and output namespace explicitly.

| Group | Scientific question | Config location | Current status |
|---|---|---|---|
| E1 Raw SOH Learning | Can raw CC/CV/T learn SOH within a domain? Is the proposed phase-specific design needed? | configs/paper_v1/e1_raw_soh_learning | XJTU and SmartHealth RawMamba/Only-F runnable; MIT is enabled once its phase-aware local physical124 export is nonempty |
| E2 Unified Multi-domain Learning | Can one model serve multiple battery domains? | configs/paper_v1/e2_unified_multidomain | E2-FULL-B/D checkpoints available; frozen-model diagnostics runnable under configs/paper_v1/diagnostics |
| E3 Cross-domain Reusability | Does a pretraining-unseen domain benefit zero-shot or with the same few-shot/few-cell budget? | configs/paper_v1/e3_cross_domain_reusability | protocol/config validation ready; transfer trainer not yet implemented |
| E4 Industrial External Validation | Can public unified pretraining transfer to enterprise families? | configs/paper_v1/e4_industrial_external | interface only; requires real enterprise data |

Contrastive learning is not a fifth group. It is a conditional extension under
configs/paper_v1/conditional/health_aware_contrastive, triggered only by an E2
diagnosis.

## E1 benchmark and ablation

The current E1 public benchmark uses xjtu, mit, and each SmartHealth family as
individual domains:

    RawMamba(domain) versus PINN4SOH-noLeak-OnlyF(domain)

The paired baseline stays a statistical-feature reference. It does not use
cycle as model input and does not share data-processing code with RawMamba.

Implemented controls reuse the main raw pipeline:

    V/I
    V/I + ΔT
    V/I + T0
    V/I + T0 + ΔT (proposed)
    independent CC/CV phases
    phase-specific CC/CV + bridge (proposed)
    SOH-only
    SOH + degradation auxiliary (default)

True joint CC/CV, LSTM, GRU, TCN, and true Vanilla Mamba controls are planned,
not represented by mislabeled placeholder implementations.

## Protocol choices

E1 retains the verified mixed C5B protocol: fixed independent test
batteries/cells; JSON-owned mixed development train/validation cycles; no
strict train/validation battery-separation claim. RawMamba and Only-F select
checkpoints by condition-macro validation RMSE so the comparison uses the same
model-selection principle.

E2 keeps each domain's split and physical normalization but samples with
domain/battery balancing and selects checkpoints by domain-macro validation
RMSE. Frozen E2-FULL-B/D checkpoints are diagnosed on validation data only:
SOH-matched battery-disjoint domain probing, battery-disjoint affine residual
calibration, and equal-budget shared-encoder gradient cosine conflict.

E3 stores source-domain IDs, target-domain IDs, target-budget unit, and the
mandatory scratch comparator in config. It is not allowed to silently infer
these from a loader or use target test data when selecting a checkpoint.

## Runnable checks

The following only validate wiring. They are not formal paper training:

    bash UnifiedRawSOH/scripts/setup/copy_datasets.sh
    python -m unittest discover -s UnifiedRawSOH/tests -p 'test_*.py'
    python -m UnifiedRawSOH.main --config UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_xjtu.json --backend_override torch_reference --device_override cpu --epochs 1 --patience 1 --debug_num_samples 1 --run_time smoke
    python -m UnifiedRawSOH.main --config UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_mit.json --backend_override torch_reference --device_override cpu --epochs 1 --patience 1 --debug_num_samples 1 --run_time smoke

For intentional multi-seed CUDA training, edit fixed settings at the top of
the corresponding shell script:

    scripts/paper_v1/e1_raw_soh_learning/benchmark/run_raw_mamba_benchmark.sh
    scripts/paper_v1/e1_raw_soh_learning/benchmark/run_onlyf_benchmark.sh
    scripts/paper_v1/e1_raw_soh_learning/ablation/run_raw_mamba_ablation.sh
    scripts/paper_v1/e2_unified_multidomain/run_public_xjtu_mit.sh
    scripts/paper_v1/diagnostics/run_e2_diagnostics.sh

The diagnostic launcher reads frozen checkpoints and never launches training or rewrites a source runtime.
