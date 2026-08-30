import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat

from e2_ttt.ops.norm_utils import layernorm_fwd, layernorm_bwd
from e2_ttt.ops.scan_utils import build_token_weights_chunk_log


def gradient_clip(grad, max_norm=1.0):
    assert grad.dim() == 3
    cur_norm = grad.norm(p=2, dim=(1,2), keepdim=True)
    if (cur_norm > max_norm).any():
        scale = (max_norm / (cur_norm + 1e-12)).clamp(max=1.0)
        grad = grad * scale
    return grad


def gelu_derivative(x):
    """
    Computes the derivative of the GELU function: d/dx (x * Phi(x)).
    Formula: Phi(x) + x * N(x; 0, 1)
    """
    # Cumulative Distribution Function (Phi)
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    # Probability Density Function (N)
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * x.pow(2))
    return cdf + x * pdf


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


def update_chunk_momentum_nonclosed(
    q_chunk, k_chunk, v_chunk,  # [b, d, l]
    lr_chunk, log_decay_chunk, log_beta_chunk,  # [b, l, 1]
    w0, w1,  # w0: [b, dh, d]; w1: [b, d, dh]
    m_init_w0, m_init_w1,
    w0_norm, w1_norm,
    ttt_norm_weight, ttt_norm_bias,
    verbose=False,
    use_grad_clip=True,
):
    """
    Standalone TTT update function for 2-layer MLP.
    Simplified: Chunk-wise averaged momentum and weight decay.
    The momentum passed to the next chunk is the total update vector of the current chunk.
    """
    # Retrieve memory (Read out)
    h = F.gelu(torch.bmm(w0, q_chunk))
    out = torch.bmm(w1, h)

    # Forward
    # MLPs
    Z = torch.bmm(w0, k_chunk)  # [b, dh, l]
    H = F.gelu(Z)  # [b, dh, l]
    pre = torch.bmm(w1, H).float()  # [b, d, l]

    # RMSNorm + Residual
    O, mu, inv_s = layernorm_fwd(pre, ttt_norm_weight, ttt_norm_bias)

    # MSE loss
    Err = O - v_chunk  # [b, d, l]

    alpha_lr = lr_chunk.to(k_chunk.dtype)
    Err_alpha = Err * alpha_lr.transpose(1, 2)

    # LayerNorm backward
    grad_pre = layernorm_bwd(pre, Err_alpha, ttt_norm_weight, mu, inv_s)

    # MLP Backward
    grad_H = torch.bmm(w1.transpose(1, 2), grad_pre)  # [b, dh, l]
    gelu_prime_val = gelu_derivative(Z)
    grad_Z = grad_H * gelu_prime_val

    dw1 = -torch.bmm(grad_pre, H.transpose(1, 2))
    k_t = k_chunk.transpose(1, 2)  # [b, l, d]
    dw0 = -torch.bmm(grad_Z, k_t)

    # Gradient clip
    if use_grad_clip:
        dw0 = gradient_clip(dw0)
        dw1 = gradient_clip(dw1)

    # 1. Calculate Chunk-wise Averages for Momentum and Decay
    m_avg = log_beta_chunk.exp().mean(dim=1, keepdim=True).to(k_chunk.dtype)   # [b, 1, 1]
    d_avg = log_decay_chunk.exp().mean(dim=1, keepdim=True).to(k_chunk.dtype)  # [b, 1, 1]

    # 2. Update Step (Velocity): New Gradient + Scaled Previous Momentum
    # This 'dw0' becomes the total update vector.
    dw0 = dw0 + m_init_w0 * m_avg
    dw1 = dw1 + m_init_w1 * m_avg

    # 3. Apply Update to Weights (with Averaged Decay)
    upd_w0 = w0 * d_avg + dw0
    upd_w1 = w1 * d_avg + dw1

    # 4. Next Momentum State
    # The momentum carried into the next chunk is the update vector just computed.
    m_next_w0 = dw0
    m_next_w1 = dw1

    # Weight normalization
    upd_w0 = upd_w0 / (upd_w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
    upd_w1 = upd_w1 / (upd_w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm

    return out, upd_w0, upd_w1, m_next_w0, m_next_w1


def update_chunk_momentum(
    q_chunk, k_chunk, v_chunk,  # [b, d, l]
    lr_chunk, log_decay_chunk, log_beta_chunk,  # [b, l, 1]
    w0, w1,  # w0: [b, dh, d]; w1: [b, d, dh]
    m_init_w0, m_init_w1,
    w0_norm, w1_norm,
    ttt_norm_weight, ttt_norm_bias,
    verbose=False,
    use_grad_clip=True,
):
    """
    Standalone TTT update function for 2-layer MLP: x + LN(w1 @ GeLU(w0 @ x))
    """
    # Retrieve memory (Read out)
    # layernorm + residual will be done once after all chunks are processed
    h = F.gelu(torch.bmm(w0, q_chunk))
    out = torch.bmm(w1, h)

    # Forward
    # MLPs
    Z = torch.bmm(w0, k_chunk)  # [b, dh, l]
    H = F.gelu(Z)  # [b, dh, l]
    pre = torch.bmm(w1, H).float()  # we use residual here  [b, d, l]
    # RMSNorm + Residual (Residual in already included in the v_chunk, see self.ln_restruction_target)
    O, mu, inv_s = layernorm_fwd(pre, ttt_norm_weight, ttt_norm_bias)

    # MSE loss
    Err = O - v_chunk  # [b, d, l]

    log_alpha, sbeta, carry_u, carry_m = build_token_weights_chunk_log(log_beta_chunk, log_decay_chunk)
    carry_u, carry_m = carry_u.to(k_chunk.dtype), carry_m.to(v_chunk.dtype)
    alpha = torch.exp(log_alpha)

    alpha_lr = (alpha * lr_chunk).to(k_chunk.dtype)
    Err_alpha = Err * alpha_lr.transpose(1, 2)

    # LayerNorm backward
    grad_pre = layernorm_bwd(pre, Err_alpha, ttt_norm_weight, mu, inv_s)
    grad_H = torch.bmm(w1.transpose(1, 2), grad_pre)  # [b, dh, l]
    gelu_prime_val = gelu_derivative(Z)
    grad_Z = grad_H * gelu_prime_val

    dw1 = -torch.bmm(grad_pre, H.transpose(1, 2))
    k_t = k_chunk.transpose(1, 2)  # [b, l, d]
    dw0 = -torch.bmm(grad_Z, k_t)

    if verbose:
        rel_step = dw0.norm() / w0.norm()
        print(f"before clipping rel_step: {rel_step}, dw0 norm {dw0.norm()}, w0 norm {w0.norm()}")
        print(f"carry_u: {carry_u[:, 0, 0].tolist()}")

    # gradient clip
    if use_grad_clip:
        dw0 = gradient_clip(dw0)
        dw1 = gradient_clip(dw1)

    momentum_w0 = carry_m * m_init_w0
    momentum_w1 = carry_m.transpose(1, 2) * m_init_w1

    # We now calculate the weights for the last token in the chunk
    upd_w0 = carry_u * w0 + momentum_w0 + dw0
    upd_w1 = carry_u * w1 + momentum_w1 + dw1

    if verbose:
        rel_step = dw0.norm() / w0.norm()
        print(f"rel_step: {rel_step}, dw0 norm {dw0.norm()}, w0 norm {w0.norm()}")
        print(f"MSE: {Err.pow(2).mean()}")

    # Weight normalization (same as in test training done right)
    upd_w0 = upd_w0 / (upd_w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
    upd_w1 = upd_w1 / (upd_w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm

    # We now calculate the momentum of the last token for the next chunk
    prod_beta = log_beta_chunk.sum(dim=1, keepdim=True).exp().to(k_chunk.dtype)  # [b,1,1]
    beta_lr = (sbeta * lr_chunk).to(k_chunk.dtype)

    # We might want to prune tokens which do not contribute to the momentum update
    n_keep = pick_tail_len_pow2(beta_lr)
    if n_keep > 0:
        H = H[..., -n_keep:]
        k_t = k_t[:, -n_keep:]
        grad_Z = grad_Z[..., -n_keep:]
        sbeta = sbeta[:, -n_keep:]
        alpha = alpha[:, -n_keep:]
        grad_pre = grad_pre[..., -n_keep:]

        ratio = (sbeta / (alpha + 1e-12)).transpose(1, 2).to(k_chunk.dtype)  # [b, 1, l]
        grad_pre_beta = grad_pre * ratio  # [b, d, l]
        grad_Z_beta = grad_Z * ratio

        m_w1_add = -torch.bmm(grad_pre_beta, H.transpose(1, 2))
        m_w0_add = -torch.bmm(grad_Z_beta, k_t)  # [b, dh, d]

        if verbose:
            rel_step = m_w0_add.norm() / m_init_w0.norm()
            print(f"before clipping rel_step: {rel_step}, m add w0 norm {m_w0_add.norm()}, m init w0 norm {m_init_w0.norm()}")

        if use_grad_clip:
            m_w0_add = gradient_clip(m_w0_add, max_norm=1.0)
            m_w1_add = gradient_clip(m_w1_add, max_norm=1.0)

        if verbose:
            rel_step = m_w0_add.norm() / m_init_w0.norm()
            print(f"rel_step: {rel_step}, m add w0 norm {m_w0_add.norm()}, m init w0 norm {m_init_w0.norm()}")

        m_next_w1 = prod_beta.transpose(1,2) * m_init_w1 + m_w1_add  # [b, d, dh]
        m_next_w0 = prod_beta * m_init_w0 + m_w0_add  # [b, dh, d]
    else:
        m_next_w1 = prod_beta.transpose(1,2) * m_init_w1  # [b, d, dh]
        m_next_w0 = prod_beta * m_init_w0  # [b, dh, d]

    return out, upd_w0, upd_w1, m_next_w0, m_next_w1


def ln_reconstruction_target(v, k, ttt_norm_weight, ttt_norm_bias):
    """
    Constructs the target for the TTT loss.
    """
    target = v - k  # [b, l, d]
    target = F.layer_norm(target.float(), normalized_shape=(target.size(-1),), eps=1e-8)
    w = ttt_norm_weight.unsqueeze(1).float()
    b = ttt_norm_bias.unsqueeze(1).float()

    target = target * w + b

    return target.to(v.dtype)


def chunk_ttt_momentum_with_cache(
    q, k, v,  # [b, l, h, d]
    lr, log_decay, log_momentum,  # [b, l, h, 1]
    recurrent_state,  # w0: [h, dh, d], w1: [h, d, dh], buf_k, buf_v, buf_lr
    ttt_norm_weight, ttt_norm_bias,  # [h, d]
    chunk_size=1024,
    verbose=False,
    use_closed_form=True,
    use_grad_clip=True,
):
    b, l, h, d = q.shape
    assert l == 1, 'This function is designed for token-by-token generation.'

    # We move head dimension into batch dimension
    q, k, v, lr = [
        rearrange(x, 'b l h d -> (b h) l d') for x in [q, k, v, lr]
    ]
    log_decay = rearrange(log_decay, 'b l h d -> (b h) l d') if log_decay is not None else None
    log_momentum = rearrange(log_momentum, 'b l h d -> (b h) l d')

    v = ln_reconstruction_target(v, k, ttt_norm_weight, ttt_norm_bias)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # Read from cache
    w0, w1, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1 = recurrent_state

    # expand weights
    ttt_norm_weight = repeat(ttt_norm_weight, 'h d -> (b h) d', b=b)
    ttt_norm_bias = repeat(ttt_norm_bias, 'h d -> (b h) d', b=b)

    if w0.shape[0] < b * h:
        w0 = repeat(w0, 'h dh d -> (b h) dh d', b=b)
        w1 = repeat(w1, 'h d dh -> (b h) d dh', b=b)

    # cache new k, v, and others
    buf_k_new = torch.cat([buf_k, k], dim=-1)
    buf_v_new = torch.cat([buf_v, v], dim=-1)
    buf_lr_new = torch.cat([buf_lr, lr], dim=1)
    buf_log_decay_new = torch.cat([buf_log_decay, log_decay], dim=1) if log_decay is not None else None
    buf_log_momentum_new = torch.cat([buf_log_momentum, log_momentum], dim=1)

    # We only cache new k,v tokens without updating the memory
    if buf_k_new.shape[-1] < chunk_size:
        # Forward with cached weights
        h = F.gelu(torch.bmm(w0, q))
        out = torch.bmm(w1, h)  # [b d l]
        x = out.transpose(1, 2)  # [b, l, d]
        x_norm = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-8)
        out = x_norm * ttt_norm_weight.unsqueeze(1).float() + ttt_norm_bias.unsqueeze(1).float()
        out = out.to(q.dtype) + q.transpose(1, 2)
        out = rearrange(out, '(b h) l d -> b l h d', b=b)

        return out, (w0, w1, buf_k_new, buf_v_new, buf_lr_new, buf_log_decay_new, buf_log_momentum_new, m_next_w0, m_next_w1)

    w0_norm = w0.norm(dim=-1, keepdim=True)
    w1_norm = w1.norm(dim=-1, keepdim=True)

    update_fn = update_chunk_momentum if use_closed_form else update_chunk_momentum_nonclosed
    out, w0, w1, m_next_w0, m_next_w1 = update_fn(
        q, buf_k_new, buf_v_new,
        buf_lr_new, buf_log_decay_new, buf_log_momentum_new,
        w0, w1,  # w0: [b, dh, d]; w1: [b, d, dh]
        m_next_w0, m_next_w1,
        w0_norm, w1_norm,
        ttt_norm_weight.unsqueeze(-1), ttt_norm_bias.unsqueeze(-1),
        verbose=verbose,
        use_grad_clip=use_grad_clip,
    )

    # Do the layernorm + residual once
    x = out.transpose(1, 2)  # [b, l, d]
    x_norm = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-8)
    out = x_norm * ttt_norm_weight.unsqueeze(1).float() + ttt_norm_bias.unsqueeze(1).float()
    out = out.to(q.dtype) + q.transpose(1, 2)
    out = rearrange(out, '(b h) l d -> b l h d', b=b)

    # Clear the cache
    buf_k = q.new_zeros(b * h, d, 0)
    buf_v = q.new_zeros(b * h, d, 0)
    buf_lr = q.new_zeros(b * h, 0, lr.shape[-1])
    buf_log_decay = q.new_zeros(b * h, 0, log_decay.shape[-1]) if log_decay is not None else None
    buf_log_momentum = q.new_zeros(b * h, 0, log_momentum.shape[-1])

    return out, (w0, w1, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1)


def chunk_ttt_momentum(
    q, k, v,  # [b, l, h, d]
    lr, log_decay, log_momentum,  # [b, l, h, 1]
    initial_state,  # w0: [h, dh, d], w1: [h, d, dh]
    ttt_norm_weight, ttt_norm_bias,  # [h, d]
    chunk_size=1024,
    verbose=False,
    is_training=True,
    use_closed_form=True,
    use_grad_clip=True,
):
    b, l, h, d = q.shape
    rem = l % chunk_size

    w0, w1 = initial_state

    # We move head dimension into batch dimension
    q, k, v, lr = [
        rearrange(x, 'b l h d -> (b h) l d') for x in [q, k, v, lr]
    ]
    log_decay = rearrange(log_decay, 'b l h d -> (b h) l d') if log_decay is not None else None
    log_momentum = rearrange(log_momentum, 'b l h d -> (b h) l d')

    # expand weights
    w0 = repeat(w0, 'h dh d -> (b h) dh d', b=b)
    w1 = repeat(w1, 'h d dh -> (b h) d dh', b=b)
    ttt_norm_weight = repeat(ttt_norm_weight, 'h d -> (b h) d', b=b)
    ttt_norm_bias = repeat(ttt_norm_bias, 'h d -> (b h) d', b=b)

    v = ln_reconstruction_target(v, k, ttt_norm_weight, ttt_norm_bias)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    w0_norm = w0.norm(dim=-1, keepdim=True)
    w1_norm = w1.norm(dim=-1, keepdim=True)

    m_next_w0 = torch.zeros_like(w0)
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

        update_fn = update_chunk_momentum if use_closed_form else update_chunk_momentum_nonclosed
        out, w0, w1, m_next_w0, m_next_w1 = update_fn(
            q_chunks[i], k_chunks[i], v_chunks[i],
            lr_chunks[i], log_decay_chunks[i], log_momentum_chunks[i],
            w0, w1,  # w0: [b, dh, d]; w1: [b, d, dh]
            m_next_w0, m_next_w1,
            w0_norm, w1_norm,
            ttt_norm_weight.unsqueeze(-1), ttt_norm_bias.unsqueeze(-1),
            verbose=verbose,
            use_grad_clip=use_grad_clip,
        )

        output[..., s_index:e_index] = out
        s_index = e_index

    # processing last chunk, direct read out
    if is_training or rem != 0:
        q_chunk = q_chunks[-1]
        hidden = F.gelu(torch.bmm(w0, q_chunk))
        tail = torch.bmm(w1, hidden)
        output[:, :, s_index:] = tail

    # Do the layernorm + residual once
    x = output.transpose(1, 2)  # [b, l, d]
    x_norm = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-8)
    out = x_norm * ttt_norm_weight.unsqueeze(1).float() + ttt_norm_bias.unsqueeze(1).float()
    out = out.to(q.dtype) + q.transpose(1, 2)

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

    return out, (w0, w1, buf_k, buf_v, buf_lr, buf_log_decay, buf_log_momentum, m_next_w0, m_next_w1)





