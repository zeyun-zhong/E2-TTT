from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _layernorm_affine_residual_fwd_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        mean_ptr,
        rstd_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        L: tl.constexpr,
        stride_x_b: tl.constexpr,
        stride_x_d: tl.constexpr,
        stride_x_l: tl.constexpr,
        stride_res_b: tl.constexpr,
        stride_res_d: tl.constexpr,
        stride_res_l: tl.constexpr,
        stride_w_b: tl.constexpr,
        stride_w_d: tl.constexpr,
        stride_b_b: tl.constexpr,
        stride_b_d: tl.constexpr,
        stride_out_b: tl.constexpr,
        stride_out_l: tl.constexpr,
        stride_out_d: tl.constexpr,
        eps: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        STORE_STATS: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        l_block = tl.program_id(1)

        offs_l = l_block * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_d = tl.arange(0, BLOCK_D)
        mask_l = offs_l < L
        mask_d = offs_d < D
        mask = mask_l[:, None] & mask_d[None, :]

        x = tl.load(
            x_ptr
            + b * stride_x_b
            + offs_d[None, :] * stride_x_d
            + offs_l[:, None] * stride_x_l,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        mean = tl.sum(tl.where(mask, x, 0.0), axis=1) / D
        x_centered = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(x_centered * x_centered, axis=1) / D
        rstd = tl.rsqrt(var + eps)
        x_hat = x_centered * rstd[:, None]

        weight = tl.load(
            weight_ptr + b * stride_w_b + offs_d * stride_w_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + b * stride_b_b + offs_d * stride_b_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        y = x_hat * weight[None, :] + bias[None, :]

        if HAS_RESIDUAL:
            residual = tl.load(
                residual_ptr
                + b * stride_res_b
                + offs_d[None, :] * stride_res_d
                + offs_l[:, None] * stride_res_l,
                mask=mask,
                other=0.0,
            )
            y = y.to(residual.dtype) + residual

        if STORE_STATS:
            tl.store(mean_ptr + b * L + offs_l, mean, mask=mask_l)
            tl.store(rstd_ptr + b * L + offs_l, rstd, mask=mask_l)
        tl.store(
            out_ptr
            + b * stride_out_b
            + offs_l[:, None] * stride_out_l
            + offs_d[None, :] * stride_out_d,
            y,
            mask=mask,
        )

    @triton.jit
    def _layernorm_affine_residual_bwd_kernel(
        x_ptr,
        dy_ptr,
        residual_grad_ptr,
        weight_ptr,
        mean_ptr,
        rstd_ptr,
        dx_ptr,
        dweight_partial_ptr,
        dbias_partial_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        L: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
        stride_x_b: tl.constexpr,
        stride_x_d: tl.constexpr,
        stride_x_l: tl.constexpr,
        stride_dy_b: tl.constexpr,
        stride_dy_l: tl.constexpr,
        stride_dy_d: tl.constexpr,
        stride_resg_b: tl.constexpr,
        stride_resg_d: tl.constexpr,
        stride_resg_l: tl.constexpr,
        stride_w_b: tl.constexpr,
        stride_w_d: tl.constexpr,
        stride_dx_b: tl.constexpr,
        stride_dx_d: tl.constexpr,
        stride_dx_l: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        HAS_DWEIGHT: tl.constexpr,
        HAS_DBIAS: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        l_block = tl.program_id(1)

        offs_l = l_block * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_d = tl.arange(0, BLOCK_D)
        mask_l = offs_l < L
        mask_d = offs_d < D
        mask = mask_l[:, None] & mask_d[None, :]

        x = tl.load(
            x_ptr
            + b * stride_x_b
            + offs_d[None, :] * stride_x_d
            + offs_l[:, None] * stride_x_l,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dy = tl.load(
            dy_ptr
            + b * stride_dy_b
            + offs_l[:, None] * stride_dy_l
            + offs_d[None, :] * stride_dy_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        mean = tl.load(mean_ptr + b * L + offs_l, mask=mask_l, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + b * L + offs_l, mask=mask_l, other=0.0).to(tl.float32)
        weight = tl.load(
            weight_ptr + b * stride_w_b + offs_d * stride_w_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        x_hat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        wdy = dy * weight[None, :]

        c1 = tl.sum(x_hat * wdy, axis=1) / D
        c2 = tl.sum(wdy, axis=1) / D
        dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd[:, None]

        tl.store(
            dx_ptr
            + b * stride_dx_b
            + offs_d[None, :] * stride_dx_d
            + offs_l[:, None] * stride_dx_l,
            dx,
            mask=mask,
        )

        if HAS_RESIDUAL:
            tl.store(
                residual_grad_ptr
                + b * stride_resg_b
                + offs_d[None, :] * stride_resg_d
                + offs_l[:, None] * stride_resg_l,
                dy,
                mask=mask,
            )

        partial_base = (b * NUM_L_BLOCKS + l_block) * D + offs_d
        if HAS_DWEIGHT:
            dweight = tl.sum(tl.where(mask, dy * x_hat, 0.0), axis=0)
            tl.store(dweight_partial_ptr + partial_base, dweight, mask=mask_d)
        if HAS_DBIAS:
            dbias = tl.sum(tl.where(mask, dy, 0.0), axis=0)
            tl.store(dbias_partial_ptr + partial_base, dbias, mask=mask_d)


def _layernorm_affine_residual_ref(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    add_residual: bool,
):
    x_tokens = x.transpose(1, 2)
    y = F.layer_norm(x_tokens.float(), (x_tokens.shape[-1],), eps=eps)
    y = y * weight.unsqueeze(1).float() + bias.unsqueeze(1).float()
    if add_residual:
        return y.to(residual.dtype) + residual.transpose(1, 2)
    return y.to(x.dtype)


def _can_use_triton(*tensors):
    return triton is not None and all(t is not None and t.is_cuda for t in tensors)


def _block_l(D: int) -> int:
    if D <= 128:
        return 64
    if D <= 256:
        return 32
    if D <= 512:
        return 16
    if D <= 1024:
        return 8
    if D <= 2048:
        return 4
    if D <= 4096:
        return 2
    return 1


def _layernorm_affine_residual_triton_fwd(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    add_residual: bool,
    store_stats: bool,
):
    B, D, L = x.shape
    max_fused_size = 65536 // x.element_size()
    block_d = min(max_fused_size, triton.next_power_of_2(D))
    if D > block_d:
        raise RuntimeError("layernorm_affine_residual Triton path does not support this D")

    block_l = _block_l(D)
    num_l_blocks = triton.cdiv(L, block_l)
    out_dtype = residual.dtype if add_residual else x.dtype
    out = torch.empty((B, L, D), device=x.device, dtype=out_dtype)
    mean = torch.empty((B, L), device=x.device, dtype=torch.float32) if store_stats else out
    rstd = torch.empty((B, L), device=x.device, dtype=torch.float32) if store_stats else out
    residual_arg = residual if add_residual else x

    _layernorm_affine_residual_fwd_kernel[(B, num_l_blocks)](
        x,
        residual_arg,
        weight,
        bias,
        out,
        mean,
        rstd,
        B=B,
        D=D,
        L=L,
        stride_x_b=x.stride(0),
        stride_x_d=x.stride(1),
        stride_x_l=x.stride(2),
        stride_res_b=residual_arg.stride(0),
        stride_res_d=residual_arg.stride(1),
        stride_res_l=residual_arg.stride(2),
        stride_w_b=weight.stride(0),
        stride_w_d=weight.stride(1),
        stride_b_b=bias.stride(0),
        stride_b_d=bias.stride(1),
        stride_out_b=out.stride(0),
        stride_out_l=out.stride(1),
        stride_out_d=out.stride(2),
        eps=eps,
        HAS_RESIDUAL=add_residual,
        STORE_STATS=store_stats,
        BLOCK_L=block_l,
        BLOCK_D=block_d,
    )
    if store_stats:
        return out, mean, rstd, block_l, block_d, num_l_blocks
    return out


class _LayerNormAffineResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, bias, eps: float, add_residual: bool):
        out, mean, rstd, block_l, block_d, num_l_blocks = _layernorm_affine_residual_triton_fwd(
            x,
            residual,
            weight,
            bias,
            eps,
            add_residual,
            store_stats=True,
        )

        residual_save = residual if residual is not None else x
        ctx.save_for_backward(x, residual_save, weight, bias, mean, rstd)
        ctx.add_residual = add_residual
        ctx.block_l = block_l
        ctx.block_d = block_d
        ctx.num_l_blocks = num_l_blocks
        return out

    @staticmethod
    def backward(ctx, dy):
        x, residual, weight, bias, mean, rstd = ctx.saved_tensors
        B, D, L = x.shape

        need_dx, need_dresidual, need_dweight, need_dbias = ctx.needs_input_grad[:4]
        dx = torch.empty_like(x)
        dresidual = (
            torch.empty_like(residual)
            if ctx.add_residual and need_dresidual
            else torch.empty_strided((1,), (1,), device=x.device, dtype=x.dtype)
        )
        dweight_partial = (
            torch.empty((B, ctx.num_l_blocks, D), device=x.device, dtype=torch.float32)
            if need_dweight
            else torch.empty_strided((1,), (1,), device=x.device, dtype=torch.float32)
        )
        dbias_partial = (
            torch.empty((B, ctx.num_l_blocks, D), device=x.device, dtype=torch.float32)
            if need_dbias
            else torch.empty_strided((1,), (1,), device=x.device, dtype=torch.float32)
        )

        # The kernel must always produce dx when x requires grad. It can skip
        # residual/affine gradient stores independently.
        _layernorm_affine_residual_bwd_kernel[(B, ctx.num_l_blocks)](
            x,
            dy,
            dresidual,
            weight,
            mean,
            rstd,
            dx,
            dweight_partial,
            dbias_partial,
            B=B,
            D=D,
            L=L,
            NUM_L_BLOCKS=ctx.num_l_blocks,
            stride_x_b=x.stride(0),
            stride_x_d=x.stride(1),
            stride_x_l=x.stride(2),
            stride_dy_b=dy.stride(0),
            stride_dy_l=dy.stride(1),
            stride_dy_d=dy.stride(2),
            stride_resg_b=dresidual.stride(0),
            stride_resg_d=dresidual.stride(1) if dresidual.dim() == 3 else 0,
            stride_resg_l=dresidual.stride(2) if dresidual.dim() == 3 else 0,
            stride_w_b=weight.stride(0),
            stride_w_d=weight.stride(1),
            stride_dx_b=dx.stride(0),
            stride_dx_d=dx.stride(1) if dx.dim() == 3 else 0,
            stride_dx_l=dx.stride(2) if dx.dim() == 3 else 0,
            HAS_RESIDUAL=ctx.add_residual and need_dresidual,
            HAS_DWEIGHT=need_dweight,
            HAS_DBIAS=need_dbias,
            BLOCK_L=ctx.block_l,
            BLOCK_D=ctx.block_d,
        )

        dweight = dweight_partial.sum(dim=1).to(weight.dtype) if need_dweight else None
        dbias = dbias_partial.sum(dim=1).to(bias.dtype) if need_dbias else None
        return (
            dx if need_dx else None,
            dresidual if ctx.add_residual and need_dresidual else None,
            dweight,
            dbias,
            None,
            None,
        )


def layernorm_affine_residual(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-8,
    add_residual: bool = True,
):
    """
    Fuses this block:

        x_tokens = x.transpose(1, 2)
        y = F.layer_norm(x_tokens.float(), (x_tokens.shape[-1],), eps=eps)
        y = y * weight.unsqueeze(1).float() + bias.unsqueeze(1).float()
        if add_residual:
            y = y.to(residual.dtype) + residual.transpose(1, 2)
        else:
            y = y.to(x.dtype)

    Args:
        x:        [B, D, L]
        residual: [B, D, L], or None when add_residual=False
        weight:   [B, D]
        bias:     [B, D]
        add_residual: whether to add residual after layernorm + affine

    Returns:
        Tensor with shape [B, L, D]. The dtype is residual.dtype when
        add_residual=True, otherwise x.dtype.
    """
    if not add_residual:
        residual = None

    if x.dim() != 3:
        raise ValueError("x must have shape [B, D, L]")
    if add_residual:
        if residual is None or residual.dim() != 3:
            raise ValueError("residual must have shape [B, D, L] when add_residual=True")
        if x.shape != residual.shape:
            raise ValueError("x and residual must have the same shape")
    if weight.dim() != 2 or bias.dim() != 2:
        raise ValueError("weight and bias must have shape [B, D]")

    B, D, _ = x.shape
    if weight.shape != (B, D) or bias.shape != (B, D):
        raise ValueError("weight and bias must have shape [B, D]")

    triton_inputs = (x, weight, bias) if residual is None else (x, residual, weight, bias)
    if not _can_use_triton(*triton_inputs):
        return _layernorm_affine_residual_ref(x, residual, weight, bias, eps, add_residual)

    max_fused_size = 65536 // x.element_size()
    block_d = min(max_fused_size, triton.next_power_of_2(D))
    if D > block_d:
        return _layernorm_affine_residual_ref(x, residual, weight, bias, eps, add_residual)

    if not torch.is_grad_enabled() or not any(t.requires_grad for t in triton_inputs):
        return _layernorm_affine_residual_triton_fwd(
            x,
            residual,
            weight,
            bias,
            eps,
            add_residual,
            store_stats=False,
        )

    return _LayerNormAffineResidualFunction.apply(
        x,
        residual,
        weight,
        bias,
        eps,
        add_residual,
    )
