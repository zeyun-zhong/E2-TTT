import torch
from einops import rearrange, repeat

from e2_ttt.ops.scan_utils import build_token_weights_chunk_log
from .triton_kernels import (
    fused_swiglu_readout,
    layernorm_affine_residual,
    layernorm_loss_bwd,
    swiglu_backward_from_grad_pre,
    l2_norm_add_fused
)


def gradient_clip(grad, max_norm=1.0):
    assert grad.dim() == 3
    cur_norm = grad.norm(p=2, dim=(1,2), keepdim=True)
    if (cur_norm > max_norm).any():
        scale = (max_norm / (cur_norm + 1e-12)).clamp(max=1.0)
        grad = grad * scale
    return grad


def pick_tail_len_pow2(score, value_min=1e-9, window_min=64):
    seq_len = score.shape[1]
    mask = score < value_min

    num_prune = torch.cumprod(mask, dim=1).sum(dim=1).min()

    window_keep = seq_len - num_prune

    if window_keep > 0:
        # 1. Replicate "pow2_ceil" using PyTorch operations
        # Formula: 2^(ceil(log2(n)))
        # We assume window_keep is float for log2, then convert back
        target_exponent = torch.ceil(torch.log2(window_keep.float()))
        window_keep = torch.pow(2, target_exponent).long()

        # 2. Replicate "max(window_keep, window_min)"
        window_keep = torch.clamp(window_keep, min=window_min)

    return window_keep


def update_chunk_momentum(
    q_chunk, k_chunk, v_chunk,  # [b, d, l]
    lr_chunk, log_decay_chunk, log_beta_chunk,  # [b, l, 1]
    w0_w2, w1,  # w0_w2: [b, 2 * dh, d]; w1: [b, d, dh]
    m_init_w0_w2, m_init_w1,  # prev momentum states m_prev for each param
    w0_w2_norm, w1_norm,
    ttt_norm_weight, ttt_norm_bias,
    verbose=False,
    use_post_norm=True,
):
    # Retrieve memory (Read out)
    # layernorm + residual will be done once after all chunks are processed
    out = fused_swiglu_readout(w0_w2, w1, q_chunk)

    # Forward
    k_tokens = k_chunk.transpose(1, 2).contiguous()
    pre = fused_swiglu_readout(w0_w2, w1, k_chunk).float()

    # Closed-form time weights
    log_alpha, sbeta, carry_u, carry_m = build_token_weights_chunk_log(log_beta_chunk, log_decay_chunk)
    carry_u_bf16 = carry_u.to(k_chunk.dtype)
    carry_m_bf16 = carry_m.to(v_chunk.dtype)
    alpha = torch.exp(log_alpha)
    alpha_lr = (alpha * lr_chunk).to(k_chunk.dtype)

    # Fused layernorm forward, loss weighting, and layernorm backward.
    grad_pre = layernorm_loss_bwd(pre, v_chunk, ttt_norm_weight, ttt_norm_bias, alpha_lr)
    grad_pre_tokens = grad_pre.transpose(1, 2).contiguous()
    dZ0_dZ2, H = swiglu_backward_from_grad_pre(w0_w2, w1, k_tokens, grad_pre_tokens)

    dw0_w2 = -torch.bmm(dZ0_dZ2.to(k_chunk.dtype), k_tokens.to(k_chunk.dtype))
    dw1 = -torch.bmm(
        grad_pre_tokens.transpose(1, 2).to(k_chunk.dtype),
        H.transpose(1, 2).to(k_chunk.dtype),
    )
    del pre

    if verbose:
        rel_step = dw0_w2.norm() / w0_w2.norm()
        print(f"before clipping rel_step: {rel_step}, dw0_w2 norm {dw0_w2.norm()}, w0_w2 norm {w0_w2.norm()}")
        print(f"carry_u: {carry_u_bf16[:, 0, 0].tolist()}")
        print(f"carry_m: {carry_m_bf16[:, 0, 0].tolist()}")

    # Clip dw0 and dw2 independently, update w0_w2 with optional postnorm.
    N_half = w0_w2.shape[1] // 2
    dw0_part, dw2_part = dw0_w2[:, :N_half], dw0_w2[:, N_half:]
    dw0_clipped = gradient_clip(dw0_part)
    dw2_clipped = gradient_clip(dw2_part)
    dw0_w2_clipped = torch.cat([dw0_clipped, dw2_clipped], dim=1)
    cu = carry_u_bf16.reshape(-1, 1, 1)
    cm = carry_m_bf16.reshape(-1, 1, 1)
    w0_w2_base = cu.to(w0_w2.dtype) * w0_w2 + cm.to(w0_w2.dtype) * m_init_w0_w2
    if use_post_norm:
        upd_w0_w2 = l2_norm_add_fused(w0_w2_base, dw0_w2_clipped, w0_w2_norm.squeeze(-1), tgt_dtype=w0_w2.dtype)
    else:
        upd_w0_w2 = w0_w2_base + dw0_w2_clipped.to(w0_w2.dtype)

    # Clip dw1, update w1 with optional postnorm.
    dw1_clipped = gradient_clip(dw1)
    w1_base = cu.to(w1.dtype) * w1 + cm.to(w1.dtype) * m_init_w1
    if use_post_norm:
        upd_w1 = l2_norm_add_fused(w1_base, dw1_clipped, w1_norm.squeeze(-1), tgt_dtype=w1.dtype)
    else:
        upd_w1 = w1_base + dw1_clipped.to(w1.dtype)

    if verbose:
        rel_step = dw0_w2.norm() / w0_w2.norm()
        print(f"rel_step: {rel_step}, dw0_w2 norm {dw0_w2.norm()}, w0_w2 norm {w0_w2.norm()}")

    # We now calculate the momentum of the last token for the next chunk
    prod_beta = log_beta_chunk.sum(dim=1, keepdim=True).exp().to(k_chunk.dtype)  # [b,1,1]
    beta_lr = (sbeta * lr_chunk).to(k_chunk.dtype)

    # We might want to prune tokens which do not contribute to the momentum update
    n_keep = pick_tail_len_pow2(beta_lr)
    if n_keep > 0:
        k_tail_tokens = k_tokens[:, -n_keep:].contiguous()
        sbeta_tail = sbeta[:, -n_keep:]
        alpha_tail = alpha[:, -n_keep:]
        grad_pre_tail_tokens = grad_pre_tokens[:, -n_keep:].contiguous()

        ratio = (sbeta_tail / (alpha_tail + 1e-12)).to(k_chunk.dtype)  # [b, l, 1]
        ratio_t = ratio.transpose(1, 2)
        grad_pre_beta = grad_pre_tail_tokens.transpose(1, 2).to(k_chunk.dtype) * ratio_t
        dZ0_dZ2_beta = dZ0_dZ2[..., -n_keep:].to(k_chunk.dtype) * ratio_t
        H_tail = H[..., -n_keep:].to(k_chunk.dtype)

        m_w0_w2_add = -torch.bmm(dZ0_dZ2_beta.to(k_chunk.dtype), k_tail_tokens.to(k_chunk.dtype))
        m_w1_add = -torch.bmm(
            grad_pre_beta,
            H_tail.transpose(1, 2),
        )
        del dZ0_dZ2_beta, grad_pre_beta, H_tail

        if verbose:
            rel_step = m_w0_w2_add.norm() / m_init_w0_w2.norm()
            print(f"before clipping rel_step: {rel_step}, m add w0_w2 norm {m_w0_w2_add.norm()}, m init w0_w2 norm {m_init_w0_w2.norm()}")

        # Clip and update momentum; w0 and w2 clipped independently.
        pb = prod_beta.reshape(-1, 1, 1)
        m_w0_add, m_w2_add = m_w0_w2_add[:, :N_half], m_w0_w2_add[:, N_half:]
        m_w0_add_clipped = gradient_clip(m_w0_add)
        m_w2_add_clipped = gradient_clip(m_w2_add)
        m_w1_add_clipped = gradient_clip(m_w1_add)

        m_w0_w2_add_clipped = torch.cat([m_w0_add_clipped, m_w2_add_clipped], dim=1)
        m_next_w0_w2 = pb.to(m_init_w0_w2.dtype) * m_init_w0_w2 + m_w0_w2_add_clipped.to(m_init_w0_w2.dtype)
        m_next_w1 = pb.to(m_init_w1.dtype) * m_init_w1 + m_w1_add_clipped

        if verbose:
            clipped_add_w0 = m_next_w0_w2[:, :N_half] - prod_beta.reshape(-1, 1, 1).to(m_next_w0_w2.dtype) * m_init_w0_w2[:, :N_half]
            rel_step = clipped_add_w0.norm() / m_init_w0_w2[:, :N_half].norm()
            print(f"rel_step: {rel_step}, m add w0 norm {clipped_add_w0.norm()}, m init w0 norm {m_init_w0_w2[:, :N_half].norm()}")
    else:
        m_next_w1 = prod_beta.reshape(-1, 1, 1).to(m_init_w1.dtype) * m_init_w1  # [b, d, dh]
        m_next_w0_w2 = prod_beta.reshape(-1, 1, 1).to(m_init_w0_w2.dtype) * m_init_w0_w2  # [b, 2 * dh, d]

    return out, upd_w0_w2, upd_w1, m_next_w0_w2, m_next_w1


def ln_reconstruction_target(v, k, ttt_norm_weight, ttt_norm_bias):
    """
    Constructs the target for the TTT loss.
    """
    target = v - k  # [b, l, d]
    return layernorm_affine_residual(
        target.transpose(1, 2),
        None,
        ttt_norm_weight,
        ttt_norm_bias,
        add_residual=False,
    )


def chunk_ttt_momentum_with_cache(
    q, k, v,  # [b, l, h, d]
    lr, log_decay, log_momentum,  # [b, l, h, 1]
    recurrent_state,  # w0: [h, dh, d], w1: [h, d, dh], buf_k, buf_v, buf_lr
    ttt_norm_weight, ttt_norm_bias,  # [h, d]
    chunk_size=1024,
    verbose=False,
    use_closed_form=True,
    use_post_norm=True,
):
    b, l, h, d = q.shape
    assert use_closed_form, \
        'This kernel only implements the closed-form update; use e2_ttt.ops.e2_ttt_swiglu.chunk_momentum for use_closed_form=False'
    assert l == 1, 'This function is designed for token-by-token generation.'

    # We move head dimension into batch dimension
    q, k, v, lr = [
        rearrange(x, 'b l h d -> (b h) l d') for x in [q, k, v, lr]
    ]
    log_decay = rearrange(log_decay, 'b l h d -> (b h) l d') if log_decay is not None else None
    log_momentum = rearrange(log_momentum, 'b l h d -> (b h) l d')

    # expand weights
    ttt_norm_weight = repeat(ttt_norm_weight, 'h d -> (b h) d', b=b)
    ttt_norm_bias = repeat(ttt_norm_bias, 'h d -> (b h) d', b=b)

    v = ln_reconstruction_target(v, k, ttt_norm_weight, ttt_norm_bias)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # Read from cache
    w0, w1, w2, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1, m_next_w2 = recurrent_state

    if w0.shape[0] < b * h:
        w0 = repeat(w0, 'h dh d -> (b h) dh d', b=b)
        w1 = repeat(w1, 'h d dh -> (b h) d dh', b=b)
        w2 = repeat(w2, 'h dh d -> (b h) dh d', b=b)
    w0_w2 = torch.cat([w0, w2], dim=1).contiguous()

    # cache new k, v, and others
    buf_k_new = torch.cat([buf_k, k], dim=-1)
    buf_v_new = torch.cat([buf_v, v], dim=-1)
    buf_lr_new = torch.cat([buf_lr, lr], dim=1)
    buf_log_decay_new = torch.cat([buf_log_decay, log_decay], dim=1) if log_decay is not None else None
    buf_log_momentum_new = torch.cat([buf_log_momentum, log_momentum], dim=1)

    # We only cache new k,v tokens without updating the memory
    if buf_k_new.shape[-1] < chunk_size:
        # Forward with cached weights
        out = fused_swiglu_readout(w0_w2, w1, q)  # [b d l]
        out = layernorm_affine_residual(out, q, ttt_norm_weight, ttt_norm_bias)
        out = rearrange(out, '(b h) l d -> b l h d', b=b)

        return out, (w0, w1, w2, buf_k_new, buf_v_new, buf_lr_new, buf_log_decay_new, buf_log_momentum_new, m_next_w0, m_next_w1, m_next_w2)

    m_next_w0_w2 = torch.cat([m_next_w0, m_next_w2], dim=1).contiguous()
    out, w0_w2, w1, m_next_w0_w2, m_next_w1 = update_chunk_momentum(
        q, buf_k_new, buf_v_new,
        buf_lr_new, buf_log_decay_new, buf_log_momentum_new,
        w0_w2, w1,  # w0_w2: [b, 2 * dh, d]; w1: [b, d, dh]
        m_next_w0_w2, m_next_w1,
        w0_w2.norm(dim=-1, keepdim=True),
        w1.norm(dim=-1, keepdim=True),
        ttt_norm_weight.unsqueeze(-1), ttt_norm_bias.unsqueeze(-1),
        verbose=verbose,
        use_post_norm=use_post_norm,
    )

    # Do the layernorm + residual once
    out = layernorm_affine_residual(out, q, ttt_norm_weight, ttt_norm_bias)
    out = rearrange(out, '(b h) l d -> b l h d', b=b)

    # Clear the cache
    buf_k = q.new_zeros(b * h, d, 0)
    buf_v = q.new_zeros(b * h, d, 0)
    buf_lr = q.new_zeros(b * h, 0, lr.shape[-1])
    buf_log_decay = q.new_zeros(b * h, 0, log_decay.shape[-1]) if log_decay is not None else None
    buf_log_momentum = q.new_zeros(b * h, 0, log_momentum.shape[-1])

    w0, w2 = [x.contiguous() for x in w0_w2.chunk(2, dim=1)]
    m_next_w0, m_next_w2 = [x.contiguous() for x in m_next_w0_w2.chunk(2, dim=1)]
    return out, (w0, w1, w2, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1, m_next_w2)


def chunk_ttt_momentum(
    q, k, v,  # [b, l, h, d]
    lr, log_decay, log_momentum,  # [b, l, h, 1]
    initial_state,  # w0 & w2: [h, dh, d], w1: [h, d, dh]
    ttt_norm_weight, ttt_norm_bias,  # [h, d]
    chunk_size=1024,
    verbose=False,
    is_training=True,
    use_closed_form=True,
    use_post_norm=True,
):
    b, l, h, d = q.shape
    assert use_closed_form, \
        'This kernel only implements the closed-form update; use e2_ttt.ops.e2_ttt_swiglu.chunk_momentum for use_closed_form=False'
    rem = l % chunk_size

    w0, w1, w2 = initial_state

    # We move head dimension into batch dimension
    q, k, v, lr = [
        rearrange(x, 'b l h d -> (b h) l d') for x in [q, k, v, lr]
    ]
    log_decay = rearrange(log_decay, 'b l h d -> (b h) l d') if log_decay is not None else None
    log_momentum = rearrange(log_momentum, 'b l h d -> (b h) l d')

    # expand weights
    w0 = repeat(w0, 'h dh d -> (b h) dh d', b=b)
    w2 = repeat(w2, 'h dh d -> (b h) dh d', b=b)
    w1 = repeat(w1, 'h d dh -> (b h) d dh', b=b)
    w0_w2 = torch.cat([w0, w2], dim=1).contiguous()
    ttt_norm_weight = repeat(ttt_norm_weight, 'h d -> (b h) d', b=b)
    ttt_norm_bias = repeat(ttt_norm_bias, 'h d -> (b h) d', b=b)

    v = ln_reconstruction_target(v, k, ttt_norm_weight, ttt_norm_bias)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    w0_w2_norm = w0_w2.norm(dim=-1, keepdim=True)
    w1_norm = w1.norm(dim=-1, keepdim=True)

    m_next_w0_w2 = torch.zeros_like(w0_w2)
    m_next_w1 = torch.zeros_like(w1)

    q_chunks, k_chunks, v_chunks = [
        x.split(chunk_size, dim=-1) for x in [q, k, v]
    ]
    num_chunks = len(q_chunks)

    lr_chunks = lr.split(chunk_size, dim=1)
    log_decay_chunks = log_decay.split(chunk_size, dim=1) if log_decay is not None else [None] * num_chunks
    log_momentum_chunks = log_momentum.split(chunk_size, dim=1)

    s_index = 0
    output = torch.zeros_like(q)
    for i in range(num_chunks):
        e_index = s_index + chunk_size
        # We do not update the last chunk, as the updated weights do not contribute to the final loss
        if i == num_chunks - 1:
            if is_training or rem != 0:
                continue

        if verbose: print(f"{i + 1}-th chunk")

        out, w0_w2, w1, m_next_w0_w2, m_next_w1 = update_chunk_momentum(
            q_chunks[i], k_chunks[i], v_chunks[i],
            lr_chunks[i], log_decay_chunks[i], log_momentum_chunks[i],
            w0_w2, w1,  # w0_w2: [b, 2 * dh, d]; w1: [b, d, dh]
            m_next_w0_w2, m_next_w1,
            w0_w2_norm, w1_norm,
            ttt_norm_weight.unsqueeze(-1), ttt_norm_bias.unsqueeze(-1),
            verbose=verbose,
            use_post_norm=use_post_norm,
        )

        output[..., s_index:e_index] = out
        s_index = e_index

    # processing last chunk, direct read out
    if is_training or rem != 0:
        q_chunk = q_chunks[-1]
        tail = fused_swiglu_readout(w0_w2, w1, q_chunk)
        output[:, :, s_index:] = tail

    # Do the layernorm + residual once
    out = layernorm_affine_residual(output, q, ttt_norm_weight, ttt_norm_bias)
    out = rearrange(out, '(b h) l d -> b l h d', b=b)

    # Prepare buf
    if rem != 0:
        buf_k = k_chunks[-1]
        buf_v = v_chunks[-1]
        buf_lr = lr_chunks[-1]
        buf_log_decay = log_decay_chunks[-1]
        buf_log_momentum = log_momentum_chunks[-1]
    else:
        buf_k = q.new_zeros(b * h, d, 0)
        buf_v = q.new_zeros(b * h, d, 0)
        buf_lr = q.new_zeros(b * h, 0, lr.shape[-1])
        buf_log_decay = q.new_zeros(b * h, 0, log_decay.shape[-1]) if log_decay is not None else None
        buf_log_momentum = q.new_zeros(b * h, 0, log_momentum.shape[-1])

    w0, w2 = [x.contiguous() for x in w0_w2.chunk(2, dim=1)]
    m_next_w0, m_next_w2 = [x.contiguous() for x in m_next_w0_w2.chunk(2, dim=1)]

    return out, (w0, w1, w2, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1, m_next_w2)
