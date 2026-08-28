"""Paper-Backup training namespace."""

from .config_contract import validate_config
from .config_loader import load_config
from .trainer import run_epoch, train_from_config

__all__ = ["load_config", "run_epoch", "train_from_config", "validate_config"]
