# UnifiedRawSOH

This repository keeps paper-specific experiment configurations and launchers
under explicit version namespaces. The implemented Raw → Unified → Reusable
SOH pipeline is currently Paper-v1; Paper-v2 is reserved for diagnosed
cross-domain improvements. Active training and evaluation do not require the
historical SC_TempMamba_v2 or PINN4SOH repositories beside it.

The inference contract is intentionally narrow: current-cycle terminal raw
CC/CV signal, time, and temperature only. A full-life degradation coordinate
may supervise encoder learning during training, but it is never an inference
input.

## Layout

    configs/
        paper_v1/                     existing V1 experiment configs
            e1_raw_soh_learning/      within-domain benchmark + ablations
            e2_unified_multidomain/   Separate versus Unified
            e3_cross_domain_reusability/
            e4_industrial_external/
            diagnostics/              frozen E2 representation/residual/gradient analysis
        paper_v2/                     reserved V2 configuration namespace
    datasets/                          adapters, domain registry, manifests
    preprocess/                        source-to-canonical-product pipelines
    splits/                            data/split provenance
    models/                            phase-specific raw model and Only-F baseline
    trainers/                          E1/E2 training and E3 protocol validation
    evaluation/                        metrics and matched-cycle evaluation
    scripts/paper_v1/                  V1 experiment launchers
    scripts/paper_v2/                  reserved V2 launcher namespace
    scripts/setup/                     shared data readiness tools
    results/                           curated, Git-tracked paper summaries

Outputs are not moved when code paths are reorganized. Historical E1 runtime
directories remain readable; newly launched runs use:

    outputs/Paper-v1/<experiment-group>/<model>/<domain-composition>/runtime_<time>/

## Current public domains

| Stable ID | Alias | Status |
|---|---|---|
| xjtu | A | raw + Only-F runnable |
| mit | B | phase-aware physical124 protocol; launcher requires a nonempty local raw/feature export |
| smarthealth_lishen40 | C1 | canonical CC/CV/calibration-capacity raw + Only-F runnable |
| smarthealth_catl280 | C2 | canonical CC/CV/calibration-capacity raw + Only-F runnable |
| smarthealth_eve280 | C3 | canonical CC/CV/calibration-capacity raw + Only-F runnable |

C-rate and DOD are conditions inside a SmartHealth battery domain rather than
additional domains.

## Quick checks

From the repository root:

    bash scripts/setup/copy_datasets.sh
    python -m unittest discover -s tests -p 'test_*.py'
    python main.py --config configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_xjtu.json --backend_override torch_reference --device_override cpu --epochs 1 --patience 1 --debug_num_samples 1 --run_time smoke
    python main.py --config configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_mit.json --backend_override torch_reference --device_override cpu --epochs 1 --patience 1 --debug_num_samples 1 --run_time smoke

The torch_reference backend is only for a CPU structural smoke. Formal
RawMamba runs require CUDA plus the official mamba-ssm backend.

## Multi-seed launchers

No long command is required. Edit the fixed variables near the top of each
script, then invoke it directly:

    bash scripts/paper_v1/e1_raw_soh_learning/benchmark/run_raw_mamba_benchmark.sh
    bash scripts/paper_v1/e1_raw_soh_learning/benchmark/run_onlyf_benchmark.sh
    bash scripts/paper_v1/e1_raw_soh_learning/ablation/run_raw_mamba_ablation.sh
    bash scripts/paper_v1/e2_unified_multidomain/run_public_xjtu_mit.sh
    bash scripts/paper_v1/diagnostics/run_e2_diagnostics.sh

The two E1 benchmark scripts use one `DOMAIN=` line. Valid choices are
`xjtu`, `mit`, `smarthealth_lishen40`, `smarthealth_catl280`, and
`smarthealth_eve280`. They launch the configured seeds across GPU_IDS, keep
seed checkpoints separate, and write a mean/std batch summary. A lightweight
readiness check runs before any seed process is created, so a header-only or
incomplete local product fails clearly instead of silently falling back.

The existing fairness check never retrains:

    bash scripts/paper_v1/e1_raw_soh_learning/benchmark/run_matched_cycle_eval.sh

Its four historical checkpoint directories are explicitly visible at the top
of the script so a later formal E1 run can replace them deliberately.

## Data and provenance

Paper-local data names are explicit:

    datasets/XJTU_raw       datasets/XJTU_features
    datasets/MIT_raw        datasets/MIT_features
    datasets/SmartHealth_raw  datasets/SmartHealth_features

All raw/feature data products are deliberately Git-ignored. A local clone
needs the products provisioned under the paths above, but GitHub contains only
code, configs, split definitions, documentation, and deliberate paper-level
summaries under `results/`. Full runtime directories remain under the ignored
`outputs/` tree: do not force-add raw/feature CSVs, checkpoints, per-cycle
predictions, or historical run directories. Read datasets/MANIFEST.md for the
raw-cycle contract, normalization, physical identities, and split provenance.
SmartHealth's generated canonical products retain their immutable GB18030
source audit; no launcher reparses or synthesizes raw source data.

## Rebuilding local dataset products

The tracked `preprocess/` directory contains the real XJTU, MIT physical124,
and SmartHealth v3 source-to-product implementations. Copy
`preprocess/paths.env.example` to `preprocess/paths.env`, set the three source
roots, then use the one launcher:

    bash preprocess/run_preprocess.sh xjtu all --workers 4
    bash preprocess/run_preprocess.sh mit all --workers 4
    bash preprocess/run_preprocess.sh smarthealth all --workers 8

See `preprocess/README.md` before replacing an existing product or changing a
worker count. Full MIT raw extraction requires an environment containing the
packages listed in `preprocess/requirements.txt`.

The repository does not include a license for third-party datasets. Verify
each source's redistribution terms before publishing any data-derived artifact
outside this local project.

See PAPER_STORY.md for the scientific logic and EXPERIMENT_PLAN.md for the
actual runnable/planned boundary.
