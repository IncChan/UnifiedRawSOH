# Rebuild local dataset products

`preprocess/` contains the source-to-product code for the three public data
sources. It is deliberately separate from training: preprocessing never starts
a model run, and every generated raw/feature product stays ignored by Git.

Use the unified launcher rather than invoking historical scripts from the old
monorepo. The launcher passes explicit source/output paths to the real
extractors, so a clone has no embedded developer-specific data location.

## 1. Configure locations once

From the repository root:

```bash
cp preprocess/paths.env.example preprocess/paths.env
```

Edit `preprocess/paths.env` and set these three immutable source roots:

| Variable | Expected source content |
| --- | --- |
| `XJTU_SOURCE_ROOT` | Original XJTU `.mat` hierarchy with `Batch-*` directories. |
| `MIT_SOURCE_ROOT` | A123/MIT root containing the three batch HDF5/MAT files dated `2017-05-12`, `2017-06-30`, and `2018-04-12`. |
| `SMARTHEALTH_SOURCE_ROOT` | SmartHealth root containing `LISHEN/`, `CATL/`, and `EVE/` GB18030 CSV trees. |

The output variables may point anywhere writable. Their default destinations
are the canonical local products used by this repository:

```text
datasets/XJTU_raw          datasets/XJTU_features
datasets/MIT_raw           datasets/MIT_features
datasets/SmartHealth_raw   datasets/SmartHealth_features
```

They are all ignored by Git. Do not set an output directory inside `results/`:
that directory is only for compact, manually curated paper results.

`MIT_SPLIT_OUTPUT` and `SMARTHEALTH_SPLITS_OUTPUT` default to the versioned
split locations. A full re-extraction may rewrite their JSON only when you
explicitly pass `--overwrite`; inspect any resulting Git diff before committing
a changed split policy.

## 2. Dependencies

Use a Python environment with:

```bash
python -m pip install -r preprocess/requirements.txt
```

MIT full raw extraction requires `h5py` because the A123 batches are MATLAB
v7.3/HDF5. Select a preprocessing environment containing the packages in
`preprocess/requirements.txt` through `PYTHON_BIN` when needed.

## 3. Commands

Each command below runs preprocessing only. `all` means the listed raw and
feature steps, not all training experiments.

```bash
# XJTU: phase windows used by the existing Paper-v1 XJTU products.
bash preprocess/run_preprocess.sh xjtu all --workers 4

# MIT: canonical paper124 physical cells; RAW and paired baseline features are
# generated together from the same accepted phase-aware points.
bash preprocess/run_preprocess.sh mit all --workers 4

# SmartHealth: process one family or all public SmartHealth families.  Use
# --overwrite when replacing the former local-cycle-ID products with v3.
bash preprocess/run_preprocess.sh smarthealth_lishen40 all --workers 8 --overwrite
bash preprocess/run_preprocess.sh smarthealth all --workers 8 --overwrite

# Validate existing/generated SmartHealth raw, feature, provenance, and splits.
bash preprocess/run_preprocess.sh smarthealth_validate validate
```

To use a configuration outside the repository, keep it private and select it
explicitly:

```bash
bash preprocess/run_preprocess.sh --config /secure/path/paper_paths.env mit all --workers 4
```

Existing products are protected. XJTU, MIT, and SmartHealth require a
deliberate rerun request when their output already exists:

```bash
bash preprocess/run_preprocess.sh mit all --workers 4 --overwrite
```

Before overwriting, use a fresh output path for a one-cell or limited source
smoke when possible. Do not use `--overwrite` merely to work around a schema
or policy mismatch.

## 4. Parallel execution

`--workers N` is the only parallelism control exposed by the launcher:

| Source | Parallel unit | Recommendation |
| --- | --- | --- |
| XJTU RAW and features | One independent `.mat` battery file per process. | Start with 2–4 workers; workers write disjoint CSVs and the aggregate report is restored to source order. |
| MIT physical124 RAW + paired features | One physical cell per spawned HDF5 process. | Start with 2–4 workers. `--workers 1` is the debugging mode. |
| SmartHealth RAW | Independent source CSV scans, then one logical-sequence export per worker. | Start with 4–8 workers depending on RAM and storage throughput. |
| SmartHealth features | Canonical-RAW-only serial feature pass. | It intentionally does not reopen source CSVs or use the RAW worker pool. |

Do not launch multiple preprocessing commands that write the same domain output
at the same time. In particular, run the three SmartHealth families through
`smarthealth all` sequentially; each family internally uses its requested
worker pool. Generated rows, manifest order, and split decisions are designed
to be invariant to worker completion order.

## 5. Canonical policies preserved here

- **XJTU:** current Paper-v1 end-of-charge contract, raw CC `4.0–4.195 V`, CV
  `0.5–0.1 A`; the paired Only-F export has the validated 16 electrical + 8
  temperature statistics and excludes `test capacity` cycles as model inputs.
- **MIT:** `paper124` continuation-aware physical identities. Infer actual
  CC/CV phase first, then use inferred CC `3.45–3.60 V` and inferred CV
  `abs(current_A)/1.1 Ah = 0.25C–0.05C` with ±`0.002C` selection tolerance.
  The Only-F product retains its validated 16 electrical + 8 temperature
  statistic definitions, but computes them from those same accepted canonical
  raw rows.
- **SmartHealth:** family-specific `smarthealth_lishen40`,
  `smarthealth_catl280`, and `smarthealth_eve280`; infer CC→CV first, then use
  CC `3.45–3.58 V` and nominal-C-rate CV `0.25C–0.05C` with ±`0.002C`
  tolerance. Source-local `循环号` is provenance only: v3 derives the canonical
  chronology from each logical sequence's `绝对时间` start/end interval before
  calibration labels are assigned. Labels are direct calibration or bounded
  calibration interpolation only.

See `MIT_PHYSICAL_DATASET.md` and `SMARTHEALTH_DATASET.md` for the detailed
identity, label, phase, and audit contracts.
