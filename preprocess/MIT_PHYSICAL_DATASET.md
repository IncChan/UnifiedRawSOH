# MIT physical-cell canonicalization

`run_preprocess.sh mit all` creates the canonical MIT physical-cell products without
changing the previous 140-source-file archives:

```bash
bash preprocess/run_preprocess.sh mit all --workers 4
```

The default `paper124` cohort follows the original author implementation
[`LoadData.m`](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m):

- append batch-2 cells `8, 9, 10, 16, 17` to batch-1 cells `1, 2, 3, 4, 5`,
  respectively, and never treat the appended records as independent cells;
- apply the documented batch-1 and batch-3 removals needed to obtain the
  paper's 124 physical cells;
- use `mit_p###` as the identity and a global physical `cycle` across each
  continuation boundary.

Every canonical raw and feature row records `physical_cell_id`, global
`cycle`, `source_batch_date`, `source_cell`, and `source_cycle`.  The raw
directory contains `MIT_PHYSICAL_PROVENANCE.json/.csv`, an extraction report,
and a physical-cell split JSON.  The feature directory contains a pointer to
that provenance file.

The proposed/canonical **raw** export first infers a persistent CC→CV taper
near the charge-voltage maximum, then keeps inferred-CC `3.45–3.60 V` and
inferred-CV nominal `0.25C–0.05C` (`abs(current_A) / 1.1 Ah`).  It records
the phase decision and nominal C-rate on every raw row.  The paired
Only-F feature table retains the validated 16 electrical and 8 temperature statistics, but
calculates them from those exact accepted raw points. It therefore uses the
same inferred CC `3.45–3.60 V` and inferred-CV nominal `0.25C–0.05C` signal
support as RawMamba; it does not apply a second historical window to the HDF5
source.

The default skips physical cycle 1 once per physical cell.  It does **not**
skip the first local cycle of a continuation file.  The known source spike
`2017-05-12 / cell-019 / cycle-039` is excluded from both products by default
and appears in the provenance and split data.

Cycle-life metadata mirrors the MIT paper: EOL is the first `QDischarge <
0.88 Ah` (`0.8 * 1.1 Ah`).  A record with no observed crossing is right
censored and receives the original `LoadData.m` fallback label
`observed_capacity_cycle_count + 1`; file termination alone is not an EOL
event.

For a non-curated, continuation-only 135-physical-cell archive, use:

```bash
python preprocess/extract_mit_physical_with_temperature.py --cohort source135 \
  --input-root /path/to/A123_Dataset \
  --raw-output-dir /path/to/MIT_raw_source135 \
  --feature-output-dir /path/to/MIT_features_source135
```

For a small, non-destructive smoke run, override both output paths and limit
the cohort:

```bash
MIT_RAW_OUTPUT=/tmp/mit_raw_smoke \
MIT_FEATURE_OUTPUT=/tmp/mit_feature_smoke \
bash preprocess/run_preprocess.sh mit all --workers 1 --max-physical-cells 1 --max-cycles 3
```
