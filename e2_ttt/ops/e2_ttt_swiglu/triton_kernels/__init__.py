from .layernorm_loss_bwd import layernorm_loss_bwd
from .layernorm_residual import layernorm_affine_residual
from .swiglu import fused_swiglu_readout, swiglu_backward_from_grad_pre

try:
    from .l2norm_triton_kernels import l2_norm_add_fused
except Exception as exc:
    _l2_norm_add_fused_import_error = exc

    def l2_norm_add_fused(*args, **kwargs):
        raise RuntimeError(
            "l2_norm_add_fused is unavailable because its Triton kernels could "
            "not be imported"
        ) from _l2_norm_add_fused_import_error

__all__ = [
    "fused_swiglu_readout",
    "layernorm_affine_residual",
    "layernorm_loss_bwd",
    "swiglu_backward_from_grad_pre",
    "l2_norm_add_fused",
]
