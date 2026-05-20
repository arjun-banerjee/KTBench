"""Auto-import all scenario modules so @scenario decorators fire."""
from . import softmax_a100_to_h100  # noqa: F401
from . import judge_softmax_a100_to_h100  # noqa: F401
from . import swiglu_activation_a100_to_h100  # noqa: F401
from . import softmax_h200_to_triton  # noqa: F401
from . import causal_conv1d_silu_a100_to_h100  # noqa: F401
from . import chunk_decay_scan_a100_to_h100  # noqa: F401
from . import fused_rms_norm_residual_a100_to_h100  # noqa: F401
from . import hadamard_transform_a100_to_h100  # noqa: F401
from . import wkv_recurrence_a100_to_h100  # noqa: F401
