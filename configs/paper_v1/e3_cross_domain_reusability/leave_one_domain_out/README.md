# E3 leave-one-domain-out

The runnable no-cycle LODO family contains one shared base configuration and
five folds:

- `lodo_no_cycle_aux_xjtu.json`
- `lodo_no_cycle_aux_mit.json`
- `lodo_no_cycle_aux_smarthealth_lishen40.json`
- `lodo_no_cycle_aux_smarthealth_catl280.json`
- `lodo_no_cycle_aux_smarthealth_eve280.json`

Each fold inherits the model, fixed physical normalizations, optimizer,
scheduler, early stopping, and hierarchical domain-then-battery sampling from
E2-FULL-D w/o cycle auxiliary.

The split routing is fixed and leakage-safe:

- training: the four source domains' existing Paper-v1 `train` partitions;
- checkpoint selection: the four source domains' existing `val` partitions;
- final evaluation: only the left-out domain's existing `test` partition;
- excluded: source-domain test partitions and left-out-domain train/val
  partitions.

No battery is reassigned and no new random train/test partition is generated.
The older `lodo_smarthealth_eve280.json` remains the historical blocked
interface and is not used by the runnable no-cycle launcher.
