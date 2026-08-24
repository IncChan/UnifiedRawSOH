# Paper-v1 launchers

The E1-E4 and conditional directories preserve the existing Paper-v1 command
surface under one explicit version namespace. Launchers still call the shared
utilities in the parent `scripts/` directory and still write to
`outputs/Paper-v1/` according to their resolved config.

`diagnostics/` is the read-only post-training entry point for the three V1 E2
diagnostics. Its reports use a separate `outputs/Paper-v1/v1_diagnostics/`
namespace.

Use paths documented in the repository root `README.md`.
