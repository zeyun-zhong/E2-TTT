import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _layernorm_loss_bwd_kernel(
        pre_ptr,
        v_ptr,
        weight_ptr,
        bias_ptr,
        alpha_lr_ptr,
        out_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        L: tl.constexpr,
        stride_pre_b: tl.constexpr,
        stride_pre_d: tl.constexpr,
        stride_pre_l: tl.constexpr,
        stride_v_b: tl.constexpr,
        stride_v_d: tl.constexpr,
        stride_v_l: tl.constexpr,
        stride_w_b: tl.constexpr,
        stride_w_d: tl.constexpr,
        stride_alpha_b: tl.constexpr,
        stride_alpha_l: tl.constexpr,
        stride_out_b: tl.constexpr,
        stride_out_d: tl.constexpr,
        stride_out_l: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // L
        l = pid - b * L
        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        pre = tl.load(
            pre_ptr + b * stride_pre_b + offs_d * stride_pre_d + l * stride_pre_l,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        mean = tl.sum(pre, axis=0) / D
        centered = tl.where(mask, pre - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / D
        inv_std = tl.rsqrt(var + eps)
        xhat = centered * inv_std

        weight_raw = tl.load(
            weight_ptr + b * stride_w_b + offs_d * stride_w_d,
            mask=mask,
            other=0.0,
        )
        bias_raw = tl.load(
            bias_ptr + b * stride_w_b + offs_d * stride_w_d,
            mask=mask,
            other=0.0,
        )
        v_raw = tl.load(
            v_ptr + b * stride_v_b + offs_d * stride_v_d + l * stride_v_l,
            mask=mask,
            other=0.0,
        )
        alpha_lr_raw = tl.load(alpha_lr_ptr + b * stride_alpha_b + l * stride_alpha_l)

        weight = weight_raw.to(tl.float32)
        bias = bias_raw.to(tl.float32)

        # Match layernorm_fwd -> Err -> Err_alpha in the reference path:
        # layernorm output is rounded to gamma dtype before subtracting v, and
        # the weighted error is rounded to alpha_lr dtype before LN backward.
        y = (xhat * weight + bias).to(weight_raw.dtype)
        err = y - v_raw
        dy = (err * alpha_lr_raw).to(alpha_lr_raw.dtype).to(tl.float32)
        gh = dy * weight
        sum_gh = tl.sum(tl.where(mask, gh, 0.0), axis=0)
        sum_gh_xhat = tl.sum(tl.where(mask, gh * xhat, 0.0), axis=0)
        grad = (gh - (sum_gh + xhat * sum_gh_xhat) / D) * inv_std

        tl.store(
            out_ptr + b * stride_out_b + offs_d * stride_out_d + l * stride_out_l,
            grad,
            mask=mask,
        )

    @triton.jit
    def _layernorm_loss_bwd_vjp_kernel(
        # forward inputs
        pre_ptr,
        v_ptr,
        weight_ptr,
        bias_ptr,
        alpha_lr_ptr,
        # vjp input
        grad_out_ptr,
        # vjp outputs (grad_pre, grad_v: per (b,l); grad_gamma, grad_beta: atomic)
        grad_pre_ptr,
        grad_v_ptr,
        grad_gamma_ptr,
        grad_beta_ptr,
        grad_alpha_lr_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        L: tl.constexpr,
        stride_pre_b: tl.constexpr,
        stride_pre_d: tl.constexpr,
        stride_pre_l: tl.constexpr,
        stride_v_b: tl.constexpr,
        stride_v_d: tl.constexpr,
        stride_v_l: tl.constexpr,
        stride_w_b: tl.constexpr,
        stride_w_d: tl.constexpr,
        stride_alpha_b: tl.constexpr,
        stride_alpha_l: tl.constexpr,
        stride_gout_b: tl.constexpr,
        stride_gout_d: tl.constexpr,
        stride_gout_l: tl.constexpr,
        stride_gpre_b: tl.constexpr,
        stride_gpre_d: tl.constexpr,
        stride_gpre_l: tl.constexpr,
        stride_gv_b: tl.constexpr,
        stride_gv_d: tl.constexpr,
        stride_gv_l: tl.constexpr,
        stride_gg_b: tl.constexpr,
        stride_gg_d: tl.constexpr,
        stride_ga_b: tl.constexpr,
        stride_ga_l: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // L
        l = pid - b * L
        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        # --- Load forward inputs ---
        pre = tl.load(
            pre_ptr + b * stride_pre_b + offs_d * stride_pre_d + l * stride_pre_l,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v_raw = tl.load(
            v_ptr + b * stride_v_b + offs_d * stride_v_d + l * stride_v_l,
            mask=mask,
            other=0.0,
        )
        weight_raw = tl.load(
            weight_ptr + b * stride_w_b + offs_d * stride_w_d,
            mask=mask,
            other=0.0,
        )
        bias_raw = tl.load(
            bias_ptr + b * stride_w_b + offs_d * stride_w_d,
            mask=mask,
            other=0.0,
        )
        alpha_lr_raw = tl.load(alpha_lr_ptr + b * stride_alpha_b + l * stride_alpha_l)
        g = tl.load(
            grad_out_ptr + b * stride_gout_b + offs_d * stride_gout_d + l * stride_gout_l,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        weight = weight_raw.to(tl.float32)
        alpha_lr = alpha_lr_raw.to(tl.float32)

        # --- Recompute forward quantities ---
        mu = tl.sum(tl.where(mask, pre, 0.0), axis=0) / D
        centered = tl.where(mask, pre - mu, 0.0)
        var = tl.sum(centered * centered, axis=0) / D
        inv_std = tl.rsqrt(var + eps)
        xhat = centered * inv_std

        y = (xhat * weight + bias_raw.to(tl.float32)).to(weight_raw.dtype)
        err_f = y.to(tl.float32) - v_raw.to(tl.float32)
        dy = tl.where(mask, (err_f * alpha_lr).to(alpha_lr_raw.dtype).to(tl.float32), 0.0)
        q = tl.where(mask, dy * weight, 0.0)

        S1 = tl.sum(q, axis=0)
        S2 = tl.sum(q * xhat, axis=0)

        # --- VJP ---
        T1 = tl.sum(tl.where(mask, g, 0.0), axis=0)
        T2 = tl.sum(tl.where(mask, g * xhat, 0.0), axis=0)

        # Gradient through LNBackwardCore w.r.t. q: same functional form as forward
        grad_q = tl.where(mask, (g - (T1 + xhat * T2) / D) * inv_std, 0.0)

        # Gradient through LNBackwardCore w.r.t. xhat
        grad_xhat = tl.where(mask, (-inv_std / D) * (g * S2 + q * T2), 0.0)

        # q = dy * weight → grad_dy, partial grad_gamma
        grad_dy = tl.where(mask, grad_q * weight, 0.0)
        grad_weight_from_q = tl.where(mask, grad_q * dy, 0.0)

        # dy = cast(err * alpha_lr)  [STE through cast]
        grad_err = grad_dy * alpha_lr
        grad_alpha_lr_l = tl.sum(tl.where(mask, grad_dy * err_f, 0.0), axis=0)

        # err = y - v
        grad_v_f = -grad_err

        # y = cast(xhat * weight + bias)  [STE through cast]
        grad_xhat = grad_xhat + tl.where(mask, grad_err * weight, 0.0)
        grad_weight_from_y = tl.where(mask, grad_err * xhat, 0.0)
        grad_bias_f = tl.where(mask, grad_err, 0.0)

        # xhat = centered * inv_std
        grad_centered = grad_xhat * inv_std
        grad_inv_std_s = tl.sum(tl.where(mask, grad_xhat * centered, 0.0), axis=0)

        # out = A * inv_std, where A = q - (S1 + xhat*S2)/D  [direct path to inv_std]
        A = tl.where(mask, q - (S1 + xhat * S2) / D, 0.0)
        grad_inv_std_s = grad_inv_std_s + tl.sum(g * A, axis=0)

        # inv_std = rsqrt(var + eps)
        grad_var = grad_inv_std_s * (-inv_std * inv_std * inv_std * 0.5)

        # var = sum_D(centered^2) / D
        grad_centered = grad_centered + tl.where(mask, grad_var * 2.0 * centered / D, 0.0)

        # centered = pre - mu,  mu = sum_D(pre)/D
        grad_mu = -tl.sum(tl.where(mask, grad_centered, 0.0), axis=0)
        grad_pre_f = tl.where(mask, grad_centered + grad_mu / D, 0.0)

        # --- Store per-(b,l) outputs ---
        out_dtype = weight_raw.dtype
        tl.store(
            grad_pre_ptr + b * stride_gpre_b + offs_d * stride_gpre_d + l * stride_gpre_l,
            grad_pre_f.to(out_dtype),
            mask=mask,
        )
        tl.store(
            grad_v_ptr + b * stride_gv_b + offs_d * stride_gv_d + l * stride_gv_l,
            grad_v_f.to(out_dtype),
            mask=mask,
        )
        tl.store(
            grad_alpha_lr_ptr + b * stride_ga_b + l * stride_ga_l,
            grad_alpha_lr_l.to(alpha_lr_raw.dtype),
        )

        # --- Atomic-accumulate grad_gamma and grad_beta over L ---
        # Accumulators are float32 to avoid dtype issues with atomic_add.
        grad_weight_f = grad_weight_from_q + grad_weight_from_y
        tl.atomic_add(
            grad_gamma_ptr + b * stride_gg_b + offs_d * stride_gg_d,
            grad_weight_f,
            mask=mask,
        )
        tl.atomic_add(
            grad_beta_ptr + b * stride_gg_b + offs_d * stride_gg_d,
            grad_bias_f,
            mask=mask,
        )


def _grad_enabled_for(*tensors):
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def _layernorm_loss_bwd_ref(pre, v_chunk, norm_weight, norm_bias, alpha_lr, eps):
    # Compute in at least float32 (float64 preserved for testing / gradcheck).
    ct = torch.promote_types(pre.dtype, torch.float32)
    pre_c = pre.to(ct)
    gamma_c = norm_weight.to(ct)
    beta_c = norm_bias.to(ct)

    mu = pre_c.mean(dim=1, keepdim=True)
    centered = pre_c - mu
    inv_std = torch.rsqrt((centered * centered).mean(dim=1, keepdim=True) + eps)
    xhat = centered * inv_std
    y = (xhat * gamma_c + beta_c).to(norm_weight.dtype)
    dy = (y - v_chunk) * alpha_lr.transpose(1, 2)
    gh = dy.to(ct) * gamma_c
    dim = pre.shape[1]
    grad = (
        gh
        - (gh.sum(dim=1, keepdim=True) + xhat * (gh * xhat).sum(dim=1, keepdim=True))
        / dim
    ) * inv_std
    return grad.to(norm_weight.dtype)


def _layernorm_loss_bwd_vjp_ref(pre, v, gamma, beta, alpha_lr, grad_out, eps):
    """
    Explicit VJP of layernorm_loss_bwd (pure PyTorch, readable reference).

    Given grad_out flowing back through the forward pass, returns:
        (grad_pre, grad_v, grad_gamma, grad_beta, grad_alpha_lr)
    """
    # Compute in at least float32 (float64 preserved for testing / gradcheck).
    ct = torch.promote_types(pre.dtype, torch.float32)
    D = pre.shape[1]

    # Recompute forward quantities
    pre_c = pre.to(ct)
    gamma_c = gamma.to(ct)
    beta_c = beta.to(ct)

    mu = pre_c.mean(dim=1, keepdim=True)                              # [B,1,L]
    centered = pre_c - mu                                              # [B,D,L]
    var = (centered * centered).mean(dim=1, keepdim=True)             # [B,1,L]
    inv_std = torch.rsqrt(var + eps)                                   # [B,1,L]
    xhat = centered * inv_std                                          # [B,D,L]

    y = (xhat * gamma_c + beta_c).to(gamma.dtype)                    # [B,D,L]
    err = y.to(ct) - v.to(ct)                                        # [B,D,L]
    alpha_lr_t = alpha_lr.transpose(1, 2).to(ct)                     # [B,1,L]
    dy = (err * alpha_lr_t).to(alpha_lr.dtype).to(ct)                # [B,D,L]
    q = dy * gamma_c                                                   # [B,D,L]

    S1 = q.sum(dim=1, keepdim=True)                                   # [B,1,L]
    S2 = (q * xhat).sum(dim=1, keepdim=True)                         # [B,1,L]

    g = grad_out.to(ct)
    T1 = g.sum(dim=1, keepdim=True)                                   # [B,1,L]
    T2 = (g * xhat).sum(dim=1, keepdim=True)                         # [B,1,L]

    # Gradient through LNBackwardCore w.r.t. q (same functional form as forward)
    grad_q = (g - (T1 + xhat * T2) / D) * inv_std                   # [B,D,L]

    # Gradient through LNBackwardCore w.r.t. xhat
    grad_xhat = (-inv_std / D) * (g * S2 + q * T2)                  # [B,D,L]

    # q = dy * gamma
    grad_dy = grad_q * gamma_c
    grad_gamma = (grad_q * dy).sum(dim=2, keepdim=True)              # [B,D,1]

    # dy = cast(err * alpha_lr_t)  [STE through cast]
    grad_err = grad_dy * alpha_lr_t
    grad_alpha_lr = (                                                  # [B,L,1]
        (grad_dy * err).sum(dim=1, keepdim=True).transpose(1, 2)
    )

    # err = y - v
    grad_v = -grad_err
    grad_y = grad_err

    # y = cast(xhat * gamma + beta)  [STE through cast]
    grad_xhat = grad_xhat + grad_y * gamma_c
    grad_gamma = grad_gamma + (grad_y * xhat).sum(dim=2, keepdim=True)
    grad_beta = grad_y.sum(dim=2, keepdim=True)                       # [B,D,1]

    # xhat = centered * inv_std
    grad_centered = grad_xhat * inv_std
    grad_inv_std = (grad_xhat * centered).sum(dim=1, keepdim=True)   # [B,1,L]

    # out = A * inv_std, where A = q - (S1 + xhat*S2)/D  [direct path to inv_std]
    A = q - (S1 + xhat * S2) / D
    grad_inv_std = grad_inv_std + (g * A).sum(dim=1, keepdim=True)

    # inv_std = rsqrt(var + eps)
    grad_var = grad_inv_std * (-inv_std ** 3 * 0.5)                  # [B,1,L]

    # var = sum_D(centered^2) / D
    grad_centered = grad_centered + grad_var * 2.0 * centered / D

    # centered = pre - mu,  mu = sum_D(pre) / D
    grad_mu = -grad_centered.sum(dim=1, keepdim=True)                # [B,1,L]
    grad_pre = grad_centered + grad_mu / D                            # [B,D,L]

    return (
        grad_pre.to(pre.dtype),
        grad_v.to(v.dtype),
        grad_gamma.to(gamma.dtype),
        grad_beta.to(beta.dtype),
        grad_alpha_lr.to(alpha_lr.dtype),
    )


def _layernorm_loss_bwd_forward_impl(pre, v_chunk, norm_weight, norm_bias, alpha_lr, eps=1e-8):
    if triton is None or not pre.is_cuda:
        return _layernorm_loss_bwd_ref(pre, v_chunk, norm_weight, norm_bias, alpha_lr, eps)

    B, D, L = pre.shape
    block_d = triton.next_power_of_2(D)
    if block_d > 131072:
        return _layernorm_loss_bwd_ref(pre, v_chunk, norm_weight, norm_bias, alpha_lr, eps)

    out = torch.empty_strided(
        pre.shape,
        pre.stride(),
        device=pre.device,
        dtype=norm_weight.dtype,
    )
    _layernorm_loss_bwd_kernel[(B * L,)](
        pre,
        v_chunk,
        norm_weight,
        norm_bias,
        alpha_lr,
        out,
        B,
        D,
        L,
        pre.stride(0),
        pre.stride(1),
        pre.stride(2),
        v_chunk.stride(0),
        v_chunk.stride(1),
        v_chunk.stride(2),
        norm_weight.stride(0),
        norm_weight.stride(1),
        alpha_lr.stride(0),
        alpha_lr.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        eps,
        BLOCK_D=block_d,
    )
    return out


def _layernorm_loss_bwd_vjp_impl(pre, v, gamma, beta, alpha_lr, grad_out, eps=1e-8):
    if triton is None or not pre.is_cuda:
        return _layernorm_loss_bwd_vjp_ref(pre, v, gamma, beta, alpha_lr, grad_out, eps)

    B, D, L = pre.shape
    block_d = triton.next_power_of_2(D)
    if block_d > 131072:
        return _layernorm_loss_bwd_vjp_ref(pre, v, gamma, beta, alpha_lr, grad_out, eps)

    out_dtype = gamma.dtype
    grad_pre = torch.empty_strided(
        pre.shape, pre.stride(), device=pre.device, dtype=out_dtype
    )
    grad_v = torch.empty_strided(
        v.shape, v.stride(), device=v.device, dtype=out_dtype
    )
    # Use float32 accumulators for gamma/beta to avoid dtype issues with atomic_add.
    grad_gamma_f32 = torch.zeros(
        gamma.shape, device=gamma.device, dtype=torch.float32
    )
    grad_beta_f32 = torch.zeros(
        beta.shape, device=beta.device, dtype=torch.float32
    )
    grad_alpha_lr = torch.empty_strided(
        alpha_lr.shape, alpha_lr.stride(), device=alpha_lr.device, dtype=alpha_lr.dtype
    )

    _layernorm_loss_bwd_vjp_kernel[(B * L,)](
        pre, v, gamma, beta, alpha_lr,
        grad_out,
        grad_pre, grad_v, grad_gamma_f32, grad_beta_f32, grad_alpha_lr,
        B, D, L,
        pre.stride(0), pre.stride(1), pre.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        gamma.stride(0), gamma.stride(1),
        alpha_lr.stride(0), alpha_lr.stride(1),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        grad_pre.stride(0), grad_pre.stride(1), grad_pre.stride(2),
        grad_v.stride(0), grad_v.stride(1), grad_v.stride(2),
        grad_gamma_f32.stride(0), grad_gamma_f32.stride(1),
        grad_alpha_lr.stride(0), grad_alpha_lr.stride(1),
        eps,
        BLOCK_D=block_d,
    )

    return (
        grad_pre,
        grad_v,
        grad_gamma_f32.to(out_dtype),
        grad_beta_f32.to(out_dtype),
        grad_alpha_lr,
    )


class _LayerNormLossBwdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre, v_chunk, norm_weight, norm_bias, alpha_lr, eps):
        ctx.eps = eps
        ctx.save_for_backward(pre, v_chunk, norm_weight, norm_bias, alpha_lr)
        return _layernorm_loss_bwd_forward_impl(
            pre, v_chunk, norm_weight, norm_bias, alpha_lr, eps
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None, None, None, None

        pre, v_chunk, norm_weight, norm_bias, alpha_lr = ctx.saved_tensors
        needs = ctx.needs_input_grad[:5]

        all_grads = _layernorm_loss_bwd_vjp_impl(
            pre, v_chunk, norm_weight, norm_bias, alpha_lr, grad_out, ctx.eps
        )
        grad_inputs = tuple(
            g if need else None for g, need in zip(all_grads, needs)
        )
        return (*grad_inputs, None)


def layernorm_loss_bwd(pre, v_chunk, norm_weight, norm_bias, alpha_lr, eps=1e-8):
    """
    Fuses layernorm forward, squared-error gradient weighting, and layernorm backward.

    Shapes:
        pre:         [B, D, L]
        v_chunk:     [B, D, L]
        norm_weight: [B, D, 1]
        norm_bias:   [B, D, 1]
        alpha_lr:    [B, L, 1]
    """
    if not _grad_enabled_for(pre, v_chunk, norm_weight, norm_bias, alpha_lr):
        return _layernorm_loss_bwd_forward_impl(
            pre, v_chunk, norm_weight, norm_bias, alpha_lr, eps
        )

    return _LayerNormLossBwdFunction.apply(
        pre, v_chunk, norm_weight, norm_bias, alpha_lr, eps
    )
