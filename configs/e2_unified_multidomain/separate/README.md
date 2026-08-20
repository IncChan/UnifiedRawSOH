# E2 Separate reference

`Separate(domain)` is deliberately not a duplicate implementation.  It means
the exact E1 per-domain benchmark configuration and its own best checkpoint:

```text
xjtu  -> e1_raw_soh_learning/benchmark/raw_mamba_xjtu.json
mit   -> e1_raw_soh_learning/benchmark/raw_mamba_mit.json
SmartHealth C1/C2/C3 -> E1 configs after their raw adapters are validated
```

E2 compares those per-domain test metrics against a single configuration under
`../unified/`, reporting both results **by domain**.  The comparison must not
pool all cycles into one score, because a large or long-lived domain could
otherwise dominate the conclusion.
