# E3 leave-one-domain-out

Each configuration specifies `source_domain_ids` and one
`target_domain_id`; no trainer hard-codes a particular C1/C2/C3 combination.
For example, the supplied EVE configuration represents:

```text
xjtu + mit + smarthealth_lishen40 + smarthealth_catl280
    -> smarthealth_eve280
```

Its current status is blocked because generated SmartHealth v2 canonical
products still need validation and the E3 trainer is not implemented. Once
available, add analogous JSON configs by changing only source/target lists.
