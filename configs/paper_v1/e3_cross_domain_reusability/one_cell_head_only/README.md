# One-cell head-only configurations

`base.json` inherits the complete E2-FULL-D w/o cycle auxiliary model and
five-domain raw-data contracts. It defines the common one-cell protocol and
head-only optimization settings. The five runnable target configs define only
the target domain, support groups, and support-selection policy.

XJTU and MIT use stable SHA256-based seed rotation for seeds 42, 52, and 62.
The launcher strictly pairs each LODO checkpoint seed with the same support
seed. When a support group has at least three development cells, the three
paired seeds select different cells. SmartHealth uses the exact
development-cell order in its split JSON: A is the first cell and B is the
second, and both are evaluated for each checkpoint seed.

Checkpoint paths are intentionally absent from JSON. Machine-specific LODO
runtime roots are filled in the launcher script; checkpoints are resolved as
`<runtime_root>/seed_<seed>/best.pt`.
