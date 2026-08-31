"""Isolated data views and composition utilities for Paper-Backup."""

from .full_cccv import FullSourceUnavailable, match_full_terminal_records, materialize_full_records
from .sequence_views import (
    ALL_VIEW_IDS,
    TERMINAL_VIEW_IDS,
    SequenceViewDataset,
    build_sequence_loaders,
)
from .strategy_pooling import build_strategy_loaders, pooled_strategy_splits
from .preprocessed import (
    PreprocessedStore,
    build_preprocessed_feature_loaders,
    load_preprocessed_records,
)

__all__ = [
    "ALL_VIEW_IDS",
    "FullSourceUnavailable",
    "SequenceViewDataset",
    "PreprocessedStore",
    "TERMINAL_VIEW_IDS",
    "build_sequence_loaders",
    "build_preprocessed_feature_loaders",
    "build_strategy_loaders",
    "match_full_terminal_records",
    "materialize_full_records",
    "load_preprocessed_records",
    "pooled_strategy_splits",
]
