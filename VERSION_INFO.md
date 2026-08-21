# UnifiedRawSOH version information

## Paper-v1 — Raw → Unified → Reusable

Paper-v1 keeps the verified C5B model and restructures the paper code around
battery domains and four experiment groups.

### Preserved and compatible

- phase-specific CC/CV Mamba, zero-initialized CC→CV bridge, T0 + ΔT,
  pooled z_health, SOH head, and encode() interface;
- current C5B degradation auxiliary target, predicted-cycle injection, and
  no-detach semantics as the default E1 model;
- XJTU raw and feature adapters;
- continuation-aware MIT 124 physical-cell identity, split provenance, and
  invalid-cycle guard;
- PINN4SOH-noLeak-OnlyF baseline;
- matched-cycle evaluation and existing runtime/checkpoint readability;
- multi-seed GPU launcher and summary utility.

The default C5B state-dict shape remains unchanged. Optional E1 temperature
ablations alter only their own input projection/head shape and therefore use
their own checkpoints.

### Domain abstraction

The registry separates adapter/source identity from the paper-level domain ID.
It records source, manufacturer, battery model, chemistry, nominal capacity,
voltage range, operating conditions, data root, adapter, split, normalization,
and availability. This metadata drives composition, balancing, and aggregation
without entering model inference inputs.

Current stable IDs:

    xjtu, mit,
    smarthealth_lishen40, smarthealth_catl280, smarthealth_eve280

### Data status

- XJTU: raw terminal source and paired statistical-feature source runnable.
- MIT: the phase-aware continuation-aware physical124 protocol uses
  `mit_proposed_phase_aware_cccv_v3`: infer phase before selecting CC
  `3.45–3.60 V` and nominal-C-rate CV `0.25C–0.05C` (±0.002C sampling
  tolerance, so sampled coverage is accepted at 0.248C–0.052C). Its paired
  Only-F table retains the validated 24-statistic definition but is rebuilt
  from the same accepted raw CC/CV rows. The launcher
  physical-cell extractor accepts `--workers N`; workers use isolated spawned
  HDF5 readers and aggregate provenance remains physical-index ordered.
  rejects a header-only/incomplete Paper-local export rather than reverting to
  an old raw or aligned table. Its Paper split is 24 fixed test cells and 100
  development cells with mixed development cycles.
- SmartHealth: v2 family-specific RAW and feature products are generated. They
  record inferred CC/CV, complete temperature, calibration-only SOH,
  logical-sequence provenance, and
  `smarthealth_condition_cell_split_2development_1test_v3`: two development
  sequences plus one held-out test sequence per valid condition. Inventories
  other than exactly three eligible logical sequences are emitted as explicit
  manual-confirmation issues and cannot be used for training; validation stays
  mixed-cycle (seed 420, 80/20) inside the pooled development cells. RAW scan
  and export are process-parallel but merge/output order is worker-count
  invariant.
- Enterprise: interface only until real data and provenance are supplied.

### Experiment status

E1 contains both benchmarks and ablations. E2 has a runnable XJTU+MIT unified
loader/config; the final public A+B+C1+C2+C3 composition still awaits an
intentional multi-domain experiment rather than a data-interface change. E3 stores validated leave-one-domain-out and
adaptation protocol definitions, but has no transfer trainer or claimed
result. E4 is an enterprise interface. Contrastive learning is conditional,
not an independent paper experiment.

The main protocol uses mixed development train/validation cycles with
independent test batteries/cells. It does not claim strict train/validation
battery separation. Inference never consumes future-life information.

### Layout migration

| Previous Paper-v0/Paper-v1 location | Current Paper-v1 location |
|---|---|
| e1_single_domain | e1_raw_soh_learning/benchmark |
| e2_ablation | e1_raw_soh_learning/ablation |
| e3_unified | e2_unified_multidomain |
| e4_zero_shot and e5_few_shot | e3_cross_domain_reusability |
| e6_external | e4_industrial_external |
| e7_contrastive | conditional/health_aware_contrastive |

Historical runtime output directories are deliberately not moved. The
matched-cycle script retains explicit historical checkpoint paths for that
reason.
