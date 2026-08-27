"""Training entry points for Paper-v1 experiments.

The public V1 export remains available, but is imported lazily so isolated
Paper-v2 config validation does not require importing the CUDA/Torch trainer.
"""


def __getattr__(name):
    if name == "train_from_config":
        from .c5b_trainer import train_from_config

        return train_from_config
    raise AttributeError(name)


__all__ = ["train_from_config"]
