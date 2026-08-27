"""Independent Paper-v2 model namespace."""

from .dense_adapter import (
    DenseResidualAdapter,
    choose_parameter_matched_dense_bottleneck,
    dense_adapter_parameter_count,
)
from .raw_mamba_moe import (
    PaperV2RawMambaModel,
    V2_MODEL_VARIANTS,
    build_paper_v2_model,
)
from .residual_moe import (
    BottleneckResidualExpert,
    ResidualMoEAdapter,
    count_trainable_parameters,
    residual_moe_parameter_count,
)

__all__ = [
    "BottleneckResidualExpert",
    "DenseResidualAdapter",
    "PaperV2RawMambaModel",
    "ResidualMoEAdapter",
    "V2_MODEL_VARIANTS",
    "build_paper_v2_model",
    "choose_parameter_matched_dense_bottleneck",
    "count_trainable_parameters",
    "dense_adapter_parameter_count",
    "residual_moe_parameter_count",
]
