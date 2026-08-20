# UnifiedRawSOH — Paper-v1

This is the standalone paper-facing codebase for Raw → Unified → Reusable SOH
learning. It is designed to live directly as a repository named
`UnifiedRawSOH`; active training and evaluation do not require the historical
SC_TempMamba_v2 or PINN4SOH repositories beside it.

The inference contract is intentionally narrow: current-cycle terminal raw
CC/CV signal, time, and temperature only. A full-life degradation coordinate
may supervise encoder learning during training, but it is never an inference
input.

## Layout

    configs/
        e1_raw_soh_learning/          within-domain benchmark + ablations
        e2_unified_multidomain/       Separate versus Unified
        e3_cross_domain_reusability/  zero/few-shot and few-cell protocols
        e4_industrial_external/       real enterprise validation interface
        conditional/                  diagnosis-triggered contrastive extension
    datasets/                          adapters, domain registry, manifests
    splits/                            data/split provenance
    models/                            phase-specific raw model and Only-F baseline
    trainers/                          E1/E2 training and E3 protocol validation
    evaluation/                        metrics and matched-cycle evaluation
    scripts/                           fixed-setting multi-seed launchers

Outputs are not moved when code paths are reorganized. Historical E1 runtime
directories remain readable; newly launched runs use:

    outputs/Paper-v1/<experiment-group>/<model>/<domain-composition>/runtime_<time>/

## Current public domains

| Stable ID | Alias | Status |
|---|---|---|
| xjtu | A | raw + Only-F runnable |
| mit | B | phase-aware physical124 protocol; launcher requires a nonempty local raw/feature export |
| smarthealth_lishen40 | C1 | canonical CC/CV/SOH raw + Only-F runnable |
| smarthealth_catl280 | C2 | canonical CC/CV/SOH raw + Only-F runnable |
| smarthealth_eve280 | C3 | canonical CC/CV/SOH raw + Only-F runnable |

C-rate and DOD are conditions inside a SmartHealth battery domain rather than
additional domains.

## Quick checks

From the repository root:

    bash scripts/setup/copy_datasets.sh
    python -m unittest discover -s tests -p 'test_*.py'
    python main.py --config configs/e1_raw_soh_learning/benchmark/raw_mamba_xjtu.json --backend_override torch_reference --device_override cpu --epochs 1 --patience 1 --debug_num_samples 1 --run_time smoke
    python main.py --config configs/e1_raw_soh_learning/benchmark/raw_mamba_mit.json --backend_override torch_reference --device_override cpu --epochs 1 --patience 1 --debug_num_samples 1 --run_time smoke

The torch_reference backend is only for a CPU structural smoke. Formal
RawMamba runs require CUDA plus the official mamba-ssm backend.

## Multi-seed launchers

No long command is required. Edit the fixed variables near the top of each
script, then invoke it directly:

    bash scripts/e1_raw_soh_learning/benchmark/run_raw_mamba_benchmark.sh
    bash scripts/e1_raw_soh_learning/benchmark/run_onlyf_benchmark.sh
    bash scripts/e1_raw_soh_learning/ablation/run_raw_mamba_ablation.sh
    bash scripts/e2_unified_multidomain/run_public_xjtu_mit.sh

The two E1 benchmark scripts use one `DOMAIN=` line. Valid choices are
`xjtu`, `mit`, `smarthealth_lishen40`, `smarthealth_catl280`, and
`smarthealth_eve280`. They launch the configured seeds across GPU_IDS, keep
seed checkpoints separate, and write a mean/std batch summary. A lightweight
readiness check runs before any seed process is created, so a header-only or
incomplete local product fails clearly instead of silently falling back.

The existing fairness check never retrains:

    bash scripts/e1_raw_soh_learning/benchmark/run_matched_cycle_eval.sh

Its four historical checkpoint directories are explicitly visible at the top
of the script so a later formal E1 run can replace them deliberately.

## Data and provenance

Paper-local data names are explicit:

    datasets/XJTU_raw       datasets/XJTU_features
    datasets/MIT_raw        datasets/MIT_features
    datasets/SmartHealth_raw  datasets/SmartHealth_features

Data and runtime outputs are deliberately Git-ignored: a local clone needs
the dataset products provisioned under the paths above, but GitHub contains
only code, configs, split definitions, and documentation. Do not force-add
raw/feature CSVs, checkpoints, per-cycle predictions, or historical runtime
directories. Read datasets/MANIFEST.md for the raw-cycle contract,
normalization, physical identities, and split provenance. SmartHealth's
generated canonical products retain their immutable GB18030 source audit; no
launcher reparses or synthesizes raw source data.

The repository does not include a license for third-party datasets. Verify
each source's redistribution terms before publishing any data-derived artifact
outside this local project.

See PAPER_STORY.md for the scientific logic and EXPERIMENT_PLAN.md for the
actual runnable/planned boundary.
