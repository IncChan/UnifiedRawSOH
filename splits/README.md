# Battery-domain split specifications

Paper-v1 keeps domain-specific split assignments in JSON and keeps the loader
domain-agnostic. A domain is a battery family; a source can contain many
C-rates/DODs/conditions without splitting them into separate domains.

For a mixed-cycle dataset, a split file contains:

```json
{
  "domain_id": "new_battery_domain",
  "test_batteries": ["condition_battery-1"],
  "development_split": {
    "mode": "mixed_cycle",
    "scope": "per_condition_then_pool",
    "val_ratio": 0.2,
    "random_state": 420,
    "train_val_battery_overlap_expected": true
  }
}
```

Use `test_batteries_by_condition` when test IDs differ by condition. A generic
`test_rule` may be used when a source has a deterministic battery-ID rule, as
MIT does with `cell_id_modulo`. The runtime resolver applies such rules to the
observed inventory, while an explicit ID list remains an audit snapshot.

To add a new domain, register metadata, provide its adapter, point its config
at a split JSON, and make the adapter emit the common raw-cycle contract. No
new domain name or battery-ID rule should be added to datasets/loaders.py.
