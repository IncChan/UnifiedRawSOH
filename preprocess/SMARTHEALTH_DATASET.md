# SmartHealth canonical preprocessing

## Source layout

The downloaded source root is the local directory assigned to
`SMARTHEALTH_SOURCE_ROOT` in `preprocess/paths.env`.

It has three manufacturer/model families:

| Source directory | UnifiedRawSOH domain | Nominal family |
| --- | --- | --- |
| LISHEN | smarthealth_lishen40 (C1) | 40 Ah LFP |
| CATL | smarthealth_catl280 (C2) | 280 Ah LFP |
| EVE | smarthealth_eve280 (C3) | 280 Ah LFP |

The rate and DOD directories are source operating conditions inside the
manufacturer domain.  They are not separate domains:

* LISHEN: 0.3C, 1C, 2C crossed with 20%, 60%, 100% DOD;
* CATL: 0.5C crossed with 20%, 60%, 100% DOD;
* EVE: 0.5C crossed with 20%, 60%, 100% DOD.

Each GB18030 CSV is a numbered chunk and can contain several source cycles.
One filename series can span many chunks, and adjacent chunks can repeat
source-cycle numbers.  The source columns include cycle, step number/type,
time, current, voltage, charge/discharge capacity, and usually temp1_1.
EVE's first chunk for each logical series lacks temp1_1.

## Canonical v2 policy

The v2 implementation has one RAW entry point and one FEATURE entry point per
battery family; no command-line entry point can co-process all three families.
`smarthealth_common.py` supplies only shared parsing and audit rules.

The source's combined `恒流恒压充电` event is handled in two separate steps:

1. infer the CC→CV boundary as the first persistent 1% current taper occurring
   within 0.02 V of the selected charge event's voltage maximum;
2. select model points only from the inferred partitions: CC `3.45–3.58 V`
   and CV `0.25C–0.05C`, where `C=abs(current_A)/nominal_capacity_Ah`.

`3.58 V` is the v2 SmartHealth CC upper bound. A representative read-only
audit found stable CC coverage through 3.58 V across LISHEN/CATL/EVE and their
sampled DOD/C-rate conditions; 3.60 V was often already the taper transition
or was not stably reached before it. The nominal 3.65 V cutoff is not used as
the CC endpoint.

Each logical sequence is exactly `domain + source serial + C-rate + DOD`.
Numeric `-1/-2/...` filename suffixes are chunks of that sequence. Duplicate
source cycles are selected by boundary success, complete selected-point
temperature, selected CC/CV point coverage, raw point count, source-row count,
and earlier chunk. No C-rate/DOD sequences are merged.

Temperature must be finite at every selected CC/CV point. EVE chunks without
`temp1_1` are retained in audit and can lose to a complete duplicate, but are
never imputed or exported when no valid duplicate exists.

SOH is calibration-based: the principal discharge capacity of reliable,
periodic full-capacity calibrations forms `Q_ref=median(first three)`. Direct
calibration labels use `Q_cal/Q_ref`; normal cycles are linearly interpolated
only between bracketing reliable calibrations. There is no leading/trailing
extrapolation, nominal-40/280-Ah SOH normalization, RUL, or EOL in v2.

For every condition, exactly three eligible logical sequences are required.
`smarthealth_condition_cell_split_2development_1test_v3` sorts the stable
SHA256 ranks ascending, uses ranks 0/1 for development, and keeps rank 2 as
the held-out test cell. All family development cycles are pooled for the
seed-420, 80/20 mixed-cycle train/validation split. If a condition has any
other usable inventory, the split JSON and provenance retain an explicit
manual-confirmation issue instead of silently changing the rule; training
refuses such a split.

## Outputs

RAW output is family-namespaced under:

    UnifiedRawSOH/datasets/SmartHealth_raw/<domain>/

Each row is an original selected source point (never preprocessing-resampled)
and carries cell, source serial, logical sequence, condition, cycle, CC/CV
segment, nominal C-rate, temperature, calibration SOH/label source, and source
chunk/cycle provenance. Audits are independently named under:

    UnifiedRawSOH/datasets/SmartHealth_raw/audit/

for example `SMARTHEALTH_LISHEN40_CYCLE_PROVENANCE.csv` and
`SMARTHEALTH_LISHEN40_PREPROCESSING_REPORT.json`.

FEATURE output is under:

    UnifiedRawSOH/datasets/SmartHealth_features/<domain>/

and is derived only from its canonical RAW/provenance. It writes the existing
16 electrical and 8 temperature statistics plus cycle/SOH/cell/source metadata
and calibration-labelled capacity; it never reopens a source CSV. Splits are
emitted as one v3 split-schema JSON per family in `UnifiedRawSOH/splits/smarthealth/`.

## Run

After configuring `preprocess/paths.env`, run RAW first, then the matching
FEATURE job through the standalone launcher:

```bash
bash preprocess/run_preprocess.sh smarthealth_lishen40 raw --workers 8
bash preprocess/run_preprocess.sh smarthealth_catl280 raw --workers 8
bash preprocess/run_preprocess.sh smarthealth_eve280 raw --workers 8

bash preprocess/run_preprocess.sh smarthealth_lishen40 features
bash preprocess/run_preprocess.sh smarthealth_catl280 features
bash preprocess/run_preprocess.sh smarthealth_eve280 features
```

RAW scanning and RAW export use a process pool by default, capped at eight
workers. Scans are partitioned by independent source CSV; export is partitioned
by logical sequence so every worker owns a distinct canonical output file.
The audit merge order, raw CSV row order, and split definition are invariant to
the worker count; the report additionally records the worker count used.
Override the default for available CPU/RAM, for example:

```bash
bash preprocess/run_preprocess.sh smarthealth_eve280 raw --workers 8
```

Use `--workers 1` only for serial debugging. FEATURE extraction remains a
separate RAW-derived step and does not reopen source CSVs.

All wrappers accept their entry point's arguments, including `--overwrite` for
their own family products only. After all desired families are generated,
validate without re-reading source CSVs:

```bash
bash preprocess/run_preprocess.sh smarthealth_validate validate
```
