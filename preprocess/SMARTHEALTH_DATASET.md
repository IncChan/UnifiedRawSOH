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

## Canonical v7 policy

The v7 implementation has one RAW entry point and one FEATURE entry point per
battery family. The shared launcher can run all three sequentially, but never
mixes families in one preprocessing job. `smarthealth_common.py` supplies only
shared parsing and audit rules.

The source's combined `恒流恒压充电` event is handled in two separate steps:

1. infer the CC→CV boundary with the MIT-style short-tail rule: at least 8 CC
   points and 8 CV points, with the first 1% current taper persisting for 5
   points within 0.02 V of the selected charge event's voltage maximum. There
   is no EVE/DOD-specific 30-point gate;
2. select model points only from the inferred partitions: CC `3.45–3.58 V`
   and CV `0.25C–0.05C`, where `C=abs(current_A)/nominal_capacity_Ah`. If two
   adjacent source samples bracket one of these four fixed endpoints, v7
   inserts exactly one linearly interpolated endpoint. This prevents discrete
   sampling (for example, EVE `3.5699→3.5810 V`) from being mistaken for a
   physically incomplete window, without moving the CC endpoint to 3.60 V.
   For
   `100%DOD`, the selected CC points must cover both voltage endpoints. For
   partial-DOD conditions, the source charge can legitimately begin above
   3.45 V, so lower-endpoint coverage is recorded but is not required; the
   selected CC trace must still reach 3.58 V and contain the minimum point
   count. The audit distinguishes physical `cc_window_complete` from policy
   `cc_window_accepted`.

`3.58 V` remains the v7 SmartHealth CC upper bound. A representative read-only
audit found stable CC coverage through 3.58 V across LISHEN/CATL/EVE and their
sampled DOD/C-rate conditions; 3.60 V was often already the taper transition
or was not stably reached before it. The nominal 3.65 V cutoff is not used as
the CC endpoint.

Each logical sequence is exactly `domain + source serial + C-rate + DOD`.
Numeric `-1/-2/...` filename suffixes are chunks of that sequence. The source
`循环号` is local to a chunk and can reset or overlap, so it is retained only as
provenance. A source event is identified by `logical_sequence_id + 绝对时间`
start/end interval; canonical `cycle` is a one-based chronological index over
all source events in that logical sequence. Only exact time-interval duplicates
are selected by boundary success, complete selected-point temperature, selected
CC/CV point coverage, raw point count, source-row count, and earlier chunk. No
C-rate/DOD sequences are merged.

Temperature must be finite at every selected CC/CV point. EVE chunks without
`temp1_1` are retained in audit and can lose to a complete duplicate, but are
never imputed or exported when no valid duplicate exists.

Capacity labels are calibration-based. In partial-DOD conditions, reliable
periodic full-capacity cycles are `calibration_anchor_only`: their measured
capacity supplies an interpolation anchor, but their charge curve is never a
model input and no RAW/FEATURE row is exported for them. Normal partial-DOD
cycles are labelled only by linear interpolation between bracketing anchors.
In 100%DOD conditions there is no separate calibration-only cycle type, so
ordinary cycles remain model inputs with direct discharge-capacity labels.
Calibration eligibility is independent of model CC/CV eligibility. `Q_ref=median(first three)` remains audit provenance, but the
Paper experiment target is `label_capacity_Ah / fixed nominal capacity`
(40 Ah for LISHEN; 280 Ah for CATL/EVE), matching XJTU/MIT. There is no
leading/trailing extrapolation, RUL, or EOL in v7.

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

Each row is either an original selected source point or one of at most four
fixed-window endpoints linearly interpolated from adjacent source rows. This
is endpoint clipping, not uniform preprocessing resampling. `point_origin`,
`window_endpoint`, and the left/right source-row indices make every inserted
point auditable. Rows also carry cell, source serial, logical sequence,
condition, canonical cycle, CC/CV segment, nominal C-rate, temperature,
calibration-derived capacity/label source, `model_input_role`, plus source
chunk/local-cycle/absolute-time provenance. Cycle provenance reports separate
source/interpolated counts, and the preprocessing report counts cycles using
interpolated CC or CV endpoints. Audits are
independently named under:

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

After configuring `preprocess/paths.env`, rebuild RAW and the matching FEATURE
products through the standalone launcher. `--overwrite` replaces only the
selected family under the existing configured output locations; it does not
touch the source directory or sibling families:

```bash
bash preprocess/run_preprocess.sh smarthealth all --workers 8 --overwrite
bash preprocess/run_preprocess.sh smarthealth_validate validate
```

Because v7 changes phase detection and anchor/input semantics, existing v6
RAW, FEATURE, provenance, and split products are intentionally considered
stale. Rebuild all three SmartHealth families before a joint Paper V2 run. For
an EVE-only audit, the narrower command is:

```bash
bash preprocess/run_preprocess.sh smarthealth_eve280 all --workers 8 --overwrite
```

After regeneration, compare `cc_endpoint_interpolated_cycles`,
`cv_endpoint_interpolated_cycles`, `interpolated_window_endpoint_points`, and
the exclusion-reason counts in
`SMARTHEALTH_EVE280_PREPROCESSING_REPORT.json`. The expected changes are that
short but persistent EVE taper tails are no longer lost to a 30-point gate,
fixed endpoints remain bracket-interpolated, and partial-DOD full-capacity
calibrations appear only as anchors in provenance. Missing temperature,
genuinely short traces, incomplete windows, and extrapolation limits remain
excluded.

The frozen v6 detector is under
`preprocess/legacy/smarthealth_v6_boundary_first/` for ablation and
reproduction. Canonical launchers never import it.

RAW scanning and RAW export use a process pool by default, capped at eight
workers. Scans are partitioned by independent source CSV; export is partitioned
by logical sequence so every worker owns a distinct canonical output file.
The audit merge order, canonical time identity, and split definition are
invariant to worker count; the report additionally records the worker count
used. Model adapters sort the canonical cycle IDs before use, so source chunk
file order never becomes a degradation-time order.
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
