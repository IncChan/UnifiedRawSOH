# E3 adaptation execution boundary

When the dedicated transfer trainer is implemented, its script will consume
the adaptation configs directly. It must support both cycle_fraction and
physical_cell_count budgets and write the selected target cells/samples to
provenance before fitting. No E3 adaptation result is launched or claimed now.
