# One-cell head-only configurations

`base.json` inherits the complete E2-FULL-D w/o cycle auxiliary model and
five-domain raw-data contracts. It defines the common one-cell protocol and
head-only optimization settings. The five runnable target configs define only
the target domain, support groups, and support-selection policy.

XJTU and MIT use stable SHA256-based seed rotation for seeds 42, 52, and 62.
When a support group has at least three development cells, the three seeds
select different cells. SmartHealth uses the exact development-cell order in
its split JSON: A is the first cell and B is the second.

Checkpoint paths are intentionally absent from JSON. They are machine-specific
and are filled in the launcher script.
