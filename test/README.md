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

Exported model-input rows and `calibration_anchor_only=true` provenance rows
are plotted. Solid lines are development cells, dashed lines are test cells,
filled dots denote direct normal-cycle labels, hollow dots denote interpolated
normal-cycle labels, crosses denote non-exported partial-DOD calibration
anchors, and the gray dotted horizontal line is fixed nominal capacity. Lines
are explicitly broken when consecutive exported model inputs skip a canonical
cycle, so a calibration anchor or rejected input is not drawn as an observed
model input. The horizontal axis is the **current canonical
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
`test/outputs/smarthealth_capacity_trajectories/` by default. The summary also
reports each cell's longest missing exported-cycle run and largest interval
between capacity-reference observations; these are different quantities and should not be
interpreted interchangeably.

## XJTU capacity-versus-cycle trajectories

`plot_xjtu_capacity_trajectories.py` reads the canonical point-level
`datasets/XJTU_raw` product. The extractor-produced `SOH` column is capacity
in Ah in this product, so the script reduces each battery file to one
validated capacity label per source `cycle`. It does not use `XJTU_features`,
because that table has no cycle identifier and can have different valid-cycle
coverage.

The default split is `splits/xjtu/paper_v1_mixed_split.json`: battery 4/8 are
dashed test cells; all other batteries are solid development cells.
Development train/val is a mixed-cycle protocol, so the figure does not
assign an entire development battery to only train or only validation.

Run from the repository root:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  bash test/run_xjtu_capacity_trajectory_audit.sh
```

The default output directory is `test/outputs/xjtu_capacity_trajectories/`,
containing `xjtu_capacity_vs_cycle.png` and `summary.json`.

Useful overrides:

```bash
XJTU_RAW_ROOT=/path/to/XJTU_raw \
XJTU_CAPACITY_OUTPUT_DIR=/path/to/figures \
  bash test/run_xjtu_capacity_trajectory_audit.sh

PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  bash test/run_xjtu_capacity_trajectory_audit.sh \
  --conditions 2C 3C --hide-cell-legend
```

## MIT physical-cell capacity-versus-global-cycle trajectories

`plot_mit_capacity_trajectories.py` reads canonical `datasets/MIT_raw` physical
cell files and plots `capacity_Ah` against global physical `cycle`. The
default split is `splits/mit/mit_paper_physical124_v2_split.json`:
`mit_p###` cells whose numeric suffix is divisible by five are dashed test
cells; other cells are solid development cells. The declared invalid cycle
(currently `mit_p015/cycle 39`) is removed before plotting. The default MIT
figure uses only a compact style legend; add `--show-cell-legend` for one
entry per physical cell.

Run from the repository root:

```bash
PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  bash test/run_mit_capacity_trajectory_audit.sh
```

The default output directory is `test/outputs/mit_capacity_trajectories/`,
containing `mit_capacity_vs_cycle.png` and `summary.json`. Conditions are the
three primary MIT batch dates, and the x-axis preserves the canonical global
cycle across continuation boundaries.

Useful overrides:

```bash
MIT_RAW_ROOT=/path/to/MIT_raw \
MIT_CAPACITY_OUTPUT_DIR=/path/to/figures \
  bash test/run_mit_capacity_trajectory_audit.sh

PYTHON_BIN=/home/chenyanxi/.conda/envs/pinn/bin/python \
  bash test/run_mit_capacity_trajectory_audit.sh \
  --conditions 2017-05-12 --show-cell-legend
```

For an intentionally incomplete local smoke inventory, add `--allow-subset`;
omit it for the canonical 124-cell output.
