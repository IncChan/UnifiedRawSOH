# UnifiedRawSOH data and domain manifest

Last audited: 2026-08-20. Data files are Git-ignored. This manifest records
their canonical source, domain identity, preprocessing boundary, and split
provenance rather than embedding data in the repository.

## Common contract

All runnable raw adapters must emit one physical-cycle record with:

    dataset_id, domain_id, condition, battery_id, cycle_id,
    CC/CV segment rows, relative time, voltage, current, temperature,
    SOH, source file, and provenance identity

The shared preprocessing owns resampling and fixed physical normalization.
The model sees current-cycle tensors only. Domain metadata is for composition,
normalization, balancing, split selection, and aggregation; it is not a model
input.

## Regenerating local products

The source-to-product implementations live in `preprocess/`, not in the
training adapters. Copy `preprocess/paths.env.example` to the ignored local
`preprocess/paths.env`, set source/output locations, then use
`bash preprocess/run_preprocess.sh <domain> <stage> --workers N`. The full
source contracts, overwrite policy, and source-specific parallelism are in
`preprocess/README.md`.

## XJTU — domain xjtu (A)

| Field | Value |
|---|---|
| Canonical local raw location | datasets/XJTU_raw |
| Canonical local feature location | datasets/XJTU_features |
| Raw inventory | 55 battery CSVs: 2C=8, 3C=15, R2.5=8, R3=8, RW=8, satellite=8 |
| Raw schema | point-level cycle, segment, relative time, voltage, current, temperature, SOH |
| Raw preprocessing | CC voltage resampled to 128; CV current resampled to 256; tau/time, ΔT and T0 |
| Normalization | fixed physical XJTU window |
| Split | splits/xjtu/paper_v1_mixed_split.json |

The split holds battery-4 and battery-8 out per condition. Development cycles
use the mixed protocol owned by JSON. XJTU_raw_t_v1_aligned is not part of the
Paper path. The raw and feature tables need not have identical valid-cycle
coverage, so the raw model does not use feature rows as a hidden filter.

Only-F uses the independent feature source: 16 electrical statistics plus 8
temperature statistics, source-native all-column 3-sigma cleaning,
adjacent-x1 selection, and train/validation-fitted min-max. Its cycle index is
a source-row identifier, not a model input.

## MIT — domain mit (B)

| Field | Value |
|---|---|
| Canonical Paper raw location | datasets/MIT_raw |
| Canonical Paper feature location | datasets/MIT_features |
| Cohort | 124 continuation-aware physical cells |
| Identity | mit_p### plus one-based global physical cycle |
| Raw schema | point-level CC/CV rows with physical/source provenance |
| Proposed raw CC/CV convention | phase-aware inferred CC 3.45–3.60 V; inferred CV nominal 0.25C–0.05C |
| CV normalization | abs(current_A) / fixed 1.1 Ah nominal capacity |
| SOH label | capacity_Ah / 1.1 |
| Split | splits/mit/mit_paper_physical124_v2_split.json |

The original MIT continuation relation is resolved before the Paper cohort is
formed. The canonical split applies physical_id modulo 5, yielding 24 test
cells and 100 development cells. Development cycles use seed 420 and a 20%
mixed validation split over the pooled development cells. The explicit guard removes
mit_p015 / cycle 39. Historical 140-source-file products and aligned
intermediates are not Paper inputs.

Raw and feature products share the physical ID and global-cycle identity, so
matched-cycle evaluation matches physical cycles rather than DataLoader index
or CSV row.

The proposed raw phase policy is versioned separately in regenerated MIT raw
provenance. Its expected rows use `mit_proposed_phase_aware_cccv_v3`, infer the
actual CC/CV phase first, retain CC `3.45–3.60 V`, and retain CV
`abs(current_A)/1.1 Ah` at `0.25C→0.05C` with ±0.002C sampling tolerance
(endpoint coverage `0.248C→0.052C`). The
launcher rejects header-only/incomplete exports before it starts multi-seed
training; it does not use a legacy v1 source as a fallback. The historical
handcrafted MIT statistic definitions are retained, but the canonical Only-F
feature table is regenerated directly from the same accepted raw CC/CV points
as RawMamba.

## SmartHealth — domains C1/C2/C3

The external SmartHealth source has 1,128 GB18030 CSV chunks and 45
filename-level logical series. The current code contract is
`smarthealth_cccv_calibration_v5`; any older raw/feature/split products
must be regenerated before SmartHealth training. The source and generated
products are local-only; this repository records their contract but does not
redistribute them.

| Domain | Source files / series | Temperature-header audit | Canonical status |
|---|---:|---|---|
| smarthealth_lishen40 (C1) | 629 / 27 | `temp1_1` in audited headers | v5 regenerated locally |
| smarthealth_catl280 (C2) | 200 / 9 | `temp1_1` in audited headers | v5 regeneration pending |
| smarthealth_eve280 (C3) | 299 / 9 | 9 chunks lack `temp1_1` | v5 regeneration pending; missing-T candidates remain excluded |

Canonical raw files are namespaced under
`datasets/SmartHealth_raw/<domain>`; matching features are under
`datasets/SmartHealth_features/<domain>`; explicit v3 family splits are in
`splits/smarthealth`. Each raw cycle preserves source-file/chunk/local-cycle,
`绝对时间` start/end, the chronological canonical cycle, logical-sequence
identity, inferred boundary, selected window, calibration label, and policy
version. Per-family audit companions live in
`datasets/SmartHealth_raw/audit/`.

The calibration pipeline produces `label_capacity_Ah` directly or by
between-calibration interpolation. Paper RawMamba and Only-F experiments both
use `label_capacity_Ah / fixed nominal capacity` as the target: 40 Ah for C1,
280 Ah for C2/C3. The historical per-sequence `Q_ref` remains provenance only;
it is not the E1/E2 target denominator.

C-rate and DOD are operating/aging conditions inside their manufacturer/model
domain, not separate domains. The preprocessor deliberately avoids a
cross-condition physical-cell merge and never synthesizes temperature.

## Enterprise domains

No enterprise data are present. A future enterprise family must add semantic
domain metadata, a real raw adapter, normalization, label policy, physical-cell
split provenance, and audit output before entering E4. No placeholder data,
split, or result is valid.

## Prohibited paths

- Do not use XJTU_raw_t_v1_aligned or MIT_t_v1_aligned in Paper configs.
- Do not fabricate raw CC/CV/T from feature tables.
- Do not auto-fallback to a historical aligned intermediate.
- Do not use true cycle-life/lifetime information as an inference feature.
- Do not add raw/feature datasets or runtime outputs to Git without confirmed
  source redistribution permission and an intentional release plan.
