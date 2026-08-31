# Paper-Backup configurations

All JSON files in this directory are isolated from `paper_v1` and `paper_v2`.
They use `Paper-Backup` output roots and the SOH-only contract:

- `train.lambda_cycle = 0`;
- no cycle/lifetime target or predicted lifetime coordinate;
- battery, strategy, domain and cycle IDs remain provenance metadata;
- all production jobs use the offline `paper_backup_preprocessed_v1` product;
  raw models read its Terminal arrays unless the config selects `full_cccv`.

The current E1 matrix is deliberately limited to the requested three methods:
`HI-MLP`, `Transformer`, and `Ours`, independently for XJTU, MIT, LISHEN,
CATL, and EVE. E2 contains the five charging-view jobs for XJTU, LISHEN, and
CATL. E3 contains strategy-specific and family-pooled Ours jobs for XJTU and
LISHEN.

The E2 full configs consume the independently materialized FULL arrays. All E2
views are restricted to the same `full_matched` physical-cycle cohort. Run the
Paper-Backup FULL preprocessing and validator before launching E2.
