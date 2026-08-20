# XJTU splits

Paper-v1 uses `paper_v1_mixed_split.json` as the source of truth. It explicitly
assigns battery-4 and battery-8 to test for every XJTU condition, then applies
the mixed-cycle train/validation protocol independently per condition:

- validation ratio: `0.20`;
- random state: `420`;
- train and validation may share batteries;
- test batteries never enter train or validation.

`c5b_battery_split.json` is retained as a legacy fixed-role artifact for
equivalence checks only. It is not required by the Paper-v1 mixed loader.
