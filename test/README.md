# SmartHealth source-signal diagnostics

`plot_smarthealth_charge_profiles.py` reads the original GB18030 source tree,
not the canonical dataset consumed by RawMamba.  It draws the **full principal
combined-charge event** from every original `(source file, source cycle)`
event, so it can answer whether the proposed `3.58 V` CC upper bound is truly
inside the source CC plateau.

Each condition occupies one row of a domain figure.  All valid source events
contribute to a mean ± one-standard-deviation band; dashed curves are the
source-serial/logical-cell means.  The existing persistent-taper rule is only
used to annotate the inferred CC→CV boundary and count how often inferred CC
reaches 3.58 V versus 3.60 V.  It does not read canonical raw data, SOH labels,
or a model window.

Source cycle numbers can repeat or reset across numbered chunks.  The script
intentionally does **not** deduplicate or concatenate them: it retains the
actual source-event identity `(source file, source cycle)`.

Run the convenience wrapper from the repository root:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  bash test/run_smarthealth_charge_profile_audit.sh
```

Useful overrides:

```bash
# Use a different original-source root or output location.
SMARTHEALTH_SOURCE_ROOT=/path/to/SmartHealth \
SMARTHEALTH_PROFILE_OUTPUT_DIR=/path/to/figures \
  bash test/run_smarthealth_charge_profile_audit.sh

# Plot only one battery domain, or use a small schema smoke before the full
# 88-GB scan.  The latter is not a scientific figure.
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  python test/plot_smarthealth_charge_profiles.py \
  --domains smarthealth_lishen40 --workers 4

PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  python test/plot_smarthealth_charge_profiles.py \
  --domains smarthealth_lishen40 --max-source-files 2 --workers 1
```

Generated files remain local under `test/outputs/smarthealth_source_charge_profiles/`:

- `<domain>_source_charge_profiles.png`: full-source voltage/current figure;
- `summary.json`: per-condition/source-cell event counts, full-charge profile
  coverage, inferred boundary statistics, and the proportions whose inferred
  CC reaches 3.58 V and 3.60 V.

The source corpus is about 88 GB.  The script streams one source file per
worker and retains only online resampled-curve moments, so it does not load
the corpus into memory.  Begin with the default four workers; increase only
if the storage throughput supports it.

## Canonical capacity-versus-cycle trajectories

`plot_smarthealth_capacity_trajectories.py` reads the already-exported
canonical `audit/*_CYCLE_PROVENANCE.csv` labels, rather than original
point-level CSVs.  It makes one mosaic per battery domain, with one condition
per panel and all cells under that condition on the same axes.

Only `selected_candidate=true` and `output_status=exported` rows are plotted.
Thus every point is the exported `label_capacity_Ah`. Solid lines are
development cells, dashed lines are test cells, dots denote direct calibration
labels, and the gray dotted horizontal line is the domain's fixed nominal
capacity. The horizontal axis is the **current canonical
source-cycle order**, intentionally not a reconstructed absolute chronology;
that makes session/chunk ordering problems visible instead of hiding them.

Run all three domains:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  bash test/run_smarthealth_capacity_trajectory_audit.sh
```

Or inspect one domain / use another canonical output root:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  bash test/run_smarthealth_capacity_trajectory_audit.sh \
  --domains smarthealth_lishen40

SMARTHEALTH_RAW_ROOT=/path/to/SmartHealth_raw \
SMARTHEALTH_CAPACITY_OUTPUT_DIR=/path/to/figures \
  bash test/run_smarthealth_capacity_trajectory_audit.sh
```

It writes `<domain>_capacity_vs_cycle.png` and a compact `summary.json` to
`test/outputs/smarthealth_capacity_trajectories/` by default.
