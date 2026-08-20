# Curated experiment results

This directory is intentionally Git-tracked.  Put only compact,
publication-ready artifacts here: aggregate metric tables, final figure source
data, figures, and small result JSON files.

It intentionally starts empty. Existing runtime output is not copied here
automatically, because it can contain obsolete extraction versions as well as
non-public training detail. Promote a result only after its dataset version and
paper protocol have been checked.

Keep full training runtime directories under `outputs/`.  They are local-only
because they contain checkpoints, per-cycle predictions, copied runtime
configuration, and potentially large histories.  Do not copy raw or feature
data into this directory.

Before committing any result, remove machine-local absolute paths and verify
that the corresponding dataset's redistribution terms allow publication of
the derived artifact.
