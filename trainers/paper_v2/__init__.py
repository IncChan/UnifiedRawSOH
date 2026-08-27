"""Independent Paper-v2 trainers."""

from .config_contract import validate_v2_config


def __getattr__(name):
    if name == "train_from_config":
        from .seen_domain import train_from_config

        return train_from_config
    raise AttributeError(name)


__all__ = ["train_from_config", "validate_v2_config"]
