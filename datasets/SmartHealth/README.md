# SmartHealth canonical CC/CV/calibration-SOH v2

The immutable server-local source is
`/data1/chenyanxi/lb_project/datasets/SmartHealth`: 1,128 GB18030 CSV chunks
with point-level cycle, step, current, voltage, charge/discharge capacity, and
temperature data. The v2 implementation is split into six family-specific
commands under `code/sqj_soh/preprocess/`; the resulting canonical RAW and
Only-F feature products live in the sibling `SmartHealth_raw/` and
`SmartHealth_features/` directories and are intentionally Git-ignored.

| Domain ID | Paper alias | Source family | Canonical outputs |
|---|---|---|---|
| `smarthealth_lishen40` | C1 | LISHEN 40 Ah | `datasets/SmartHealth_{raw,features}/smarthealth_lishen40/` |
| `smarthealth_catl280` | C2 | CATL 280 Ah | `datasets/SmartHealth_{raw,features}/smarthealth_catl280/` |
| `smarthealth_eve280` | C3 | EVE 280 Ah | `datasets/SmartHealth_{raw,features}/smarthealth_eve280/` |

## Auditable policy (`smarthealth_cccv_calibration_v2`)

1. One `logical_sequence_id` is `domain + source serial + C-rate + DOD`;
   numeric chunk suffixes remain source-file provenance only. C-rate and DOD
   are in-domain operating conditions and never cross-merged.
2. The principal charge event is the longest contiguous `恒流恒压充电` event.
   CC is its high-current prefix. CV begins at the first current taper that
   persists for 30 source points, is at least 1% below the early 90th-percentile
   CC reference, and is within 0.02 V of that cycle's charge-voltage maximum.
   Model points are then selected only from inferred CC `3.45–3.58 V` and
   inferred CV `0.25C–0.05C`, using nominal capacity for C-rate.
3. Discharge capacity is the `max - min` span of `放电容量(Ah)` in the principal
   contiguous `恒流放电` event. Reliable calibration capacities define
   `Q_ref=median(first three)`. SOH is direct calibration or linear
   between-calibration interpolation only; partial-DOD capacity/40/280 Ah,
   RUL, EOL, and extrapolation are not generated.
4. Chunk-overlap candidates are ranked deterministically by boundary success,
   selected-point temperature, selected CC/CV coverage, point count, and chunk
   number.
   The selected and rejected candidates are both retained in cycle provenance.
5. Temperature is never synthesized. In particular, the nine EVE chunks with
   no `temp1_1` field remain auditable but cannot enter the temperature-aware
   raw or feature export.

`SmartHealthRawAdapter` reads the canonical raw products and rejects a direct
GB18030 source root for model training. Each v3 split-schema JSON sorts stable
SHA256 ranks per valid condition, selects two development logical sequences and
one held-out test sequence, then uses seed-420 80/20 mixed development cycles.
An inventory other than exactly three eligible logical sequences is recorded as
a manual-confirmation issue and rejected by the shared training split loader.
Per-family audit files under
`datasets/SmartHealth_raw/audit/` retain input inventory, source files,
boundary, selected windows, calibration labels, and split provenance.
