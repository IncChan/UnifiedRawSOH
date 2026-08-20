"""Dataset adapter contracts shared by all Paper-v1 datasets."""

UNIFIED_SAMPLE_KEYS = ("cc", "cv", "t0", "soh", "battery_id", "dataset_id", "domain_id")
RAW_CYCLE_KEYS = (
    "dataset_id",
    "domain_id",
    "condition",
    "battery_id",
    "cycle_id",
    "segment",
    "time",
    "voltage",
    "current",
    "temperature",
    "soh",
    "soh_raw",
    "source_file",
)


class RawTerminalSignalUnavailable(RuntimeError):
    """Raised when an intermediate dataset cannot provide raw CC/CV samples."""
