# E3 whole-dataset holdout

This stronger protocol pretrains on public domains A+B (`xjtu`, `mit`) and
keeps the entire SmartHealth source unseen:

```text
xjtu + mit -> smarthealth_lishen40 / smarthealth_catl280 / smarthealth_eve280
```

The supplied configuration remains an interface, not a runnable experiment.
SmartHealth v2 phase/label/split code is ready, but its generated canonical
products still need validation and the E3 trainer itself is not implemented.
