# Paper-Backup configurations

All JSON files in this directory are isolated from `paper_v1` and `paper_v2`.
They use `Paper-Backup` output roots and the SOH-only contract:

- `train.lambda_cycle = 0`;
- no cycle/lifetime target or predicted lifetime coordinate;
- battery, strategy, domain and cycle IDs remain provenance metadata;
- raw models use the canonical terminal product unless the config explicitly
  selects `full_cccv`.

The current E1 matrix is deliberately limited to the requested three methods:
`HI-MLP`, `Transformer`, and `Ours`, independently for XJTU, MIT, LISHEN,
CATL, and EVE. E2 contains the five charging-view jobs for XJTU, LISHEN, and
CATL. E3 contains strategy-specific and family-pooled Ours jobs for XJTU and
LISHEN.

The E2 full configs are marked `blocked_by_data` until `data.full_data_root`
points to a real, provenance-linkable point-level source. They must not be
changed to the canonical terminal directory.
