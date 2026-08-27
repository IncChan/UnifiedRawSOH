# Paper-v2 trainers

ERM optimizes

```text
L = L_SOH + lambda_balance * L_balance
```

First-order MLDG performs one manual inner update

```text
w' = w - alpha * grad_w L_meta-train(w)
L = L_ERM(w) + beta * L_pseudo-target(w') + lambda_balance * L_balance
```

The fast weights are detached after the inner step. Pseudo-target gradients
are calculated with those fast weights and combined with the original ERM
gradient by parameter name. No optimizer is passed to the inner update, so its
state is not polluted. This is a first-order approximation, not full
second-order MAML. `erm_loss`, `meta_train_loss`, `pseudo_target_loss`, and
`balance_loss` are recorded separately.

| Component | Status | Config | Command | Output | Tests | Last verified | Limitations |
|---|---|---|---|---|---|---|---|
| Base/Dense/MoE ERM | smoke-tested | E2/E3 ERM configs | `python scripts/paper_v2/train.py --validate_only --config <config>` | `best.pt`, metrics | model/CPU smoke | 2026-08-27 | no formal training started |
| First-order MLDG | smoke-tested | E3 `moe_dg` folds | `DRY_RUN=1 bash scripts/paper_v2/run_e3_zero_cell.sh` | `episode_audit.json` | MLDG numerical tests | 2026-08-27 | one inner step; no P3 router adaptation |
