# MIT splits

## Historical PINN4SOH split (reference only)

`PINN4SOH/main_MIT_subseq_mamba.py` used `cell_id % 5 == 0` on its historical
source-file inventory. That inventory and its length-7 window split are not
the Paper-v1 cohort and are retained only to understand lineage.

## Paper-v1 protocol

Apply `physical_id % 5 == 0` to the continuation-aware 124 physical cells.
This yields 24 independent test cells and 100 development cells. The explicit
list in `mit_paper_physical124_v2_split.json` is the canonical audit snapshot.
On development cells, split cycle records per primary date group with the
Paper-v1 mixed protocol:

- validation ratio: 0.20;
- permutation seed: 420;
- train and validation may share batteries;
- test batteries never enter train or validation;
- raw samples apply the invalid-cycle list by canonical battery/cycle ID;
  Only-F applies its validated source-native all-column 3-sigma cleaning;
- build full-life metadata before filtering and splitting.

The concrete rule and current test IDs are in
`mit_paper_physical124_v2_split.json`. The historical window-level split must
not be mixed with the Paper-v1 raw protocol in the same reported result.
