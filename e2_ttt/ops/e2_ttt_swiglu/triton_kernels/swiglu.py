import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

try:
    from .lact_swiglu_ffn import fused_swiglu_ffn_fwd
    from .triton_swiglu_bwd_kernels import (
        swiglu_backward_three_bmm_triton,
    )
except Exception:
    fused_swiglu_ffn_fwd = None
    swiglu_backward_three_bmm_triton = None

try:
    from .triton_pointwise_kernels import (
        triton_swiglu_bwd_bwd_fused_cat_inp_out,
    )
except Exception:
    triton_swiglu_bwd_bwd_fused_cat_inp_out = None


def _can_use_lact_kernel(*tensors):
    return (
        fused_swiglu_ffn_fwd is not None
        and swiglu_backward_three_bmm_triton is not None
        and all(t.is_cuda for t in tensors)
    )


def _can_use_lact_bwd_bwd_kernel(*tensors):
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    return (
        triton_swiglu_bwd_bwd_fused_cat_inp_out is not None
        and all(t.is_cuda for t in tensors)
        and all(t.dtype in supported_dtypes for t in tensors)
    )


def _grad_enabled_for(*tensors):
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def fused_swiglu_readout(w0_w2, w1, q_chunk):
    """
    Args:
        w0_w2:  [B, 2 * DH, D]
        w1:     [B, D, DH]
        q_chunk:[B, D, L]
    Returns:
        out:    [B, D, L]
    """
    if _can_use_lact_kernel(w0_w2, w1, q_chunk):
        out = fused_swiglu_ffn_fwd(
            w0_w2.to(torch.bfloat16).contiguous(),
            w1.to(torch.bfloat16).contiguous(),
            q_chunk.transpose(1, 2).to(torch.bfloat16).contiguous(),
        )
        return out.transpose(1, 2).to(q_chunk.dtype)

    w0, w2 = w0_w2.chunk(2, dim=1)
    h = torch.bmm(w2, q_chunk)
    gate = F.silu(torch.bmm(w0, q_chunk))
    return torch.bmm(w1, gate * h)


def _swiglu_backward_from_grad_pre_ref(
    w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens
):
    compute_dtype = w0_w2_bf16.dtype
    w1_bf16 = w1_bf16.to(compute_dtype)
    k_tokens = k_tokens.to(compute_dtype)
    grad_pre_tokens = grad_pre_tokens.to(compute_dtype)
    w0, w2 = w0_w2_bf16.chunk(2, dim=1)
    k_t = k_tokens.transpose(1, 2)
    z0 = torch.bmm(w0, k_t)
    z2 = torch.bmm(w2, k_t)
    g = F.silu(z0)
    h = g * z2
    b = torch.bmm(w1_bf16.transpose(1, 2), grad_pre_tokens.transpose(1, 2))
    sig = torch.sigmoid(z0)
    silu_prime = sig * (1.0 + z0 * (1.0 - sig))
    dz0 = b * z2 * silu_prime
    dz2 = b * g
    return torch.cat([dz0, dz2], dim=1), h


def _swiglu_backward_from_grad_pre_forward_impl(
    w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens
):
    """
    Args:
        w0_w2_bf16:      [B, 2 * DH, D]
        w1_bf16:         [B, D, DH]
        k_tokens:        [B, L, D]
        grad_pre_tokens: [B, L, D]
    Returns:
        dZ0_dZ2: [B, 2 * DH, L]
        H:       [B, DH, L]
    """
    if _can_use_lact_kernel(w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens):
        return swiglu_backward_three_bmm_triton(
            w0_w2_bf16.to(torch.bfloat16).contiguous(),
            w1_bf16.to(torch.bfloat16).contiguous(),
            k_tokens.to(torch.bfloat16).contiguous(),
            grad_pre_tokens.to(torch.bfloat16).contiguous(),
        )

    return _swiglu_backward_from_grad_pre_ref(
        w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens
    )


def _swiglu_bwd_bwd_pointwise_no_lr_ref(dhidden, z0_z2, grad_dz, grad_h):
    z0, z2 = z0_z2.chunk(2, dim=1)
    grad_dz0, grad_dz2 = grad_dz.chunk(2, dim=1)

    sig = torch.sigmoid(z0)
    silu_z0 = F.silu(z0)
    silu_prime = sig * (1.0 + z0 * (1.0 - sig))

    grad_dhidden = grad_dz0 * z2 * silu_prime + grad_dz2 * silu_z0
    grad_z2 = grad_dz0 * dhidden * silu_prime + grad_h * silu_z0

    grad_sig = grad_dz0 * dhidden * z2 * (1.0 + z0 - 2.0 * sig * z0)
    grad_z0_naive = (
        grad_dz2 * dhidden + grad_h * z2
    ) * silu_prime + grad_dz0 * dhidden * z2 * sig * (1.0 - sig)
    grad_z0 = grad_z0_naive + grad_sig * sig * (1.0 - sig)

    return grad_dhidden, torch.cat([grad_z0, grad_z2], dim=1)


def _swiglu_bwd_bwd_pointwise_no_lr(dhidden, z0_z2, grad_dz, grad_h):
    if _can_use_lact_bwd_bwd_kernel(dhidden, z0_z2, grad_dz, grad_h):
        lr = torch.ones(
            (dhidden.shape[0], dhidden.shape[2]),
            device=dhidden.device,
            dtype=torch.float32,
        )
        grad_dhidden, grad_z0_z2, *_ = triton_swiglu_bwd_bwd_fused_cat_inp_out(
            dhidden.contiguous(),
            z0_z2.contiguous(),
            lr,
            lr,
            lr,
            grad_dz.contiguous(),
            grad_h.contiguous(),
        )
        return grad_dhidden, grad_z0_z2

    return _swiglu_bwd_bwd_pointwise_no_lr_ref(dhidden, z0_z2, grad_dz, grad_h)


class _SwiGLUBackwardFromGradPreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens):
        ctx.save_for_backward(w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens)
        return _swiglu_backward_from_grad_pre_forward_impl(
            w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dz, grad_h):
        if grad_dz is None and grad_h is None:
            return None, None, None, None

        w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens = ctx.saved_tensors
        needs = ctx.needs_input_grad
        grad_inputs = [None] * 4

        compute_dtype = w0_w2_bf16.dtype
        w1 = w1_bf16.to(compute_dtype)
        k = k_tokens.to(compute_dtype)
        grad_pre = grad_pre_tokens.to(compute_dtype)

        z0_z2 = torch.bmm(w0_w2_bf16, k.transpose(1, 2))
        dhidden = torch.bmm(w1.transpose(1, 2), grad_pre.transpose(1, 2))

        if grad_dz is None:
            grad_dz = torch.zeros_like(z0_z2)
        else:
            grad_dz = grad_dz.to(compute_dtype)

        if grad_h is None:
            grad_h = torch.zeros_like(dhidden)
        else:
            grad_h = grad_h.to(compute_dtype)

        grad_dhidden, grad_z0_z2 = _swiglu_bwd_bwd_pointwise_no_lr(
            dhidden, z0_z2, grad_dz, grad_h
        )

        if needs[0]:
            grad_inputs[0] = torch.bmm(grad_z0_z2, k).to(w0_w2_bf16.dtype)

        if needs[2]:
            grad_inputs[2] = torch.bmm(
                grad_z0_z2.transpose(1, 2), w0_w2_bf16
            ).to(k_tokens.dtype)

        if needs[1]:
            grad_inputs[1] = torch.bmm(
                grad_pre.transpose(1, 2), grad_dhidden.transpose(1, 2)
            ).to(w1_bf16.dtype)

        if needs[3]:
            grad_inputs[3] = torch.bmm(
                grad_dhidden.transpose(1, 2), w1.transpose(1, 2)
            ).to(grad_pre_tokens.dtype)

        return tuple(grad_inputs)


def swiglu_backward_from_grad_pre(w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens):
    if not _grad_enabled_for(w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens):
        return _swiglu_backward_from_grad_pre_forward_impl(
            w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens
        )

    return _SwiGLUBackwardFromGradPreFunction.apply(
        w0_w2_bf16, w1_bf16, k_tokens, grad_pre_tokens
    )
