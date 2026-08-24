# E3 Cross-domain Reusability

E3 evaluates a representation learned on training-unseen battery domains:

1. `leave_one_domain_out/` — generic source-domain list to one held-out domain;
2. `cross_dataset_holdout/` — A+B pretraining with all SmartHealth domains held out;
3. `adaptation/` — zero/few-shot/few-cell target reuse versus scratch at the
   identical target budget.

These are protocol configurations.  They are not dispatched through the E1
trainer, because doing so would accidentally train on target data or select a
checkpoint from target test data.  The configuration validator lives in
`UnifiedRawSOH.trainers.reusability` until the dedicated trainer is added.
