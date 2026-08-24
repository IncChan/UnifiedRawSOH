# E1 ablation configurations

The default benchmark configuration is the proposed model:

```text
phase-specific CC/CV + CC→CV bridge + T0 + ΔT + degradation auxiliary
```

Runnable controls reuse the same loader, mixed split, normalization, optimizer,
and condition-macro checkpoint selection:

- `temperature_vi_only.json`: no sequence temperature or T0;
- `temperature_delta_t.json`: ΔT only;
- `temperature_t0.json`: T0 only;
- `independent_cc_cv.json`: phase-specific CC/CV branches without the bridge;
- `no_degradation_aux.json`: SOH-only supervision.

`raw_mamba_*.json` is the `T0 + ΔT` and bridge-on reference.  The current
model preserves relative-time input in every control; the `V/I` label here
means that the optional temperature channels are removed, not that physical
time ordering is discarded.

The planned joint-CC/CV and true Vanilla-Mamba controls are intentionally not
represented as runnable configs yet.  The current implementation has two
phase branches even when the bridge is disabled; calling that a joint or a
vanilla Mamba baseline would be misleading.
