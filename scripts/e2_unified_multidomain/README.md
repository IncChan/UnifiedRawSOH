# E2 unified launcher

run_public_xjtu_mit.sh is the currently available two-domain E2 launcher. It
uses the explicit xjtu+mit configuration, retains each domain's split and
normalization, samples with domain/battery balancing, and writes results under
the e2_unified_multidomain output namespace.

It is not invoked automatically. The all-public A+B+C1+C2+C3 configuration is
blocked until SmartHealth preprocessing and split provenance are validated.
