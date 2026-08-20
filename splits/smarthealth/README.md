# SmartHealth split provenance

The matching family-specific v2 RAW preprocessor emits one v3 split-schema JSON file per
SmartHealth domain:

- `smarthealth_lishen40_cell_split.json`
- `smarthealth_catl280_cell_split.json`
- `smarthealth_eve280_cell_split.json`

Each file is an explicit record rather than a filename heuristic. A logical
sequence is `domain + source serial + C-rate + DOD`; numeric suffixes are only
source chunks. For every condition with exactly three eligible logical
sequences, ascending deterministic SHA256 ranking assigns ranks 0/1 to
development and rank 2 to held-out test. All development sequences in the
battery family then share the seed-420, 80/20 mixed-cycle train/validation
protocol. Test is disjoint from development; train and validation intentionally
share development sequences. Any other logical-sequence inventory is recorded
under `manual_confirmation_conditions` with no fallback assignment, and the
shared split loader rejects it until manually resolved.

The JSON records source serial, condition, logical sequence, development cells,
test cell, deterministic selection rank/input, seed, strategy versions, and
source-manifest signature. Excluded cycles, duplicate
decisions, phase boundaries, temperature coverage, direct/interpolated
calibration labels, and no-extrapolation reasons are kept under
`datasets/SmartHealth_raw/audit/`, for example
`SMARTHEALTH_LISHEN40_CYCLE_PROVENANCE.csv` and
`SMARTHEALTH_LISHEN40_CELL_PROVENANCE.csv`.
