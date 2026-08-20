# E2 Unified Multi-domain Learning

E2 asks whether heterogeneous **battery domains**, rather than merely
different operating conditions, can share one raw representation learner.

- `unified/public_xjtu_mit.json` is the currently runnable two-domain path.
  It uses the existing XJTU/MIT adapters and domain/battery-balanced sampling.
- `unified/public_all_domains.json` is the final public composition, but is
  intentionally blocked until SmartHealth preprocessing/splits are validated.
- `separate/` points back to the E1 domain-specific benchmark configs instead
  of copying training logic.

The model exposes `encode()`/`z_health`; result aggregation stores
`per_domain` metrics and domain-macro validation RMSE for this stage.
