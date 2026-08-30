import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat

from e2_ttt.ops.norm_utils import layernorm_fwd, layernorm_bwd
from e2_ttt.ops.scan_utils import build_token_weights_chunk_log_nomomentum


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


def update_chunk_wo_momentum(
    q_chunk, k_chunk, v_chunk,  # [b, d, l]
    lr_chunk, log_decay_chunk,  # [b, l, 1]
    w0, w1,  # w0: [b, dh, d]; w1: [b, d, dh]
    w0_norm, w1_norm,
    ttt_norm_weight, ttt_norm_bias,
    verbose=False,
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

    alpha, carry_u = 1., 1.
    if log_decay_chunk is not None:
        log_alpha, carry_u = build_token_weights_chunk_log_nomomentum(log_decay_chunk)
        carry_u = carry_u.to(k_chunk.dtype)
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
    dw0 = gradient_clip(dw0)
    dw1 = gradient_clip(dw1)

    # We now calculate the weights for the last token in the chunk
    upd_w0 = carry_u * w0 + dw0
    upd_w1 = carry_u * w1 + dw1

    if verbose:
        rel_step = dw0.norm() / w0.norm()
        print(f"rel_step: {rel_step}, dw0 norm {dw0.norm()}, w0 norm {w0.norm()}")
        print(f"MSE: {Err.pow(2).mean()}")

    # Weight normalization (same as in test training done right)
    upd_w0 = upd_w0 / (upd_w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
    upd_w1 = upd_w1 / (upd_w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm

    return out, upd_w0, upd_w1


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


def chunk_ttt_wo_momentum_with_cache(
    q, k, v,  # [b, l, h, d]
    lr, log_decay, log_momentum,  # [b, l, h, 1]
    recurrent_state,  # w0: [h, dh, d], w1: [h, d, dh], buf_k, buf_v, buf_lr
    ttt_norm_weight, ttt_norm_bias,  # [h, d]
    chunk_size=1024,
    verbose=False,
):
    assert log_momentum is None
    b, l, h, d = q.shape
    assert l == 1, 'This function is designed for token-by-token generation.'

    # We move head dimension into batch dimension
    q, k, v, lr = [
        rearrange(x, 'b l h d -> (b h) l d') for x in [q, k, v, lr]
    ]
    log_decay = rearrange(log_decay, 'b l h d -> (b h) l d') if log_decay is not None else None

    v = ln_reconstruction_target(v, k, ttt_norm_weight, ttt_norm_bias)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # Read from cache
    w0, w1, buf_k, buf_v, buf_lr, buf_log_decay = recurrent_state

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

        return out, (w0, w1, buf_k_new, buf_v_new, buf_lr_new, buf_log_decay_new)

    w0_norm = w0.norm(dim=-1, keepdim=True)
    w1_norm = w1.norm(dim=-1, keepdim=True)

    out, w0, w1 = update_chunk_wo_momentum(
        q, buf_k_new, buf_v_new,
        buf_lr_new, buf_log_decay_new,
        w0, w1,  # w0: [b, dh, d]; w1: [b, d, dh]
        w0_norm, w1_norm,
        ttt_norm_weight.unsqueeze(-1), ttt_norm_bias.unsqueeze(-1),
        verbose=verbose,
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

    return out, (w0, w1, buf_k, buf_v, buf_lr, buf_log_decay)


def chunk_ttt_wo_momentum(
    q, k, v,  # [b, l, h, d]
    lr, log_decay, log_momentum,  # [b, l, h, 1]
    initial_state,  # w0: [h, dh, d], w1: [h, d, dh]
    ttt_norm_weight, ttt_norm_bias,  # [h, d]
    chunk_size=1024,
    verbose=False,
    is_training=True,
):
    assert log_momentum is None
    b, l, h, d = q.shape
    rem = l % chunk_size

    w0, w1 = initial_state

    # We move head dimension into batch dimension
    q, k, v, lr = [
        rearrange(x, 'b l h d -> (b h) l d') for x in [q, k, v, lr]
    ]
    log_decay = rearrange(log_decay, 'b l h d -> (b h) l d') if log_decay is not None else None

    # expand weights
    w0 = repeat(w0, 'h dh d -> (b h) dh d', b=b)
    w1 = repeat(w1, 'h d dh -> (b h) d dh', b=b)
    ttt_norm_weight = repeat(ttt_norm_weight, 'h d -> (b h) d', b=b)
    ttt_norm_bias = repeat(ttt_norm_bias, 'h d -> (b h) d', b=b)

    v = ln_reconstruction_target(v, k, ttt_norm_weight, ttt_norm_bias)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    w0_norm = w0.norm(dim=-1, keepdim=True)
    w1_norm = w1.norm(dim=-1, keepdim=True)

    q_chunks, k_chunks, v_chunks = [
        x.split(chunk_size, dim=-1) for x in [q, k, v]
    ]
    num_chunks = len(q_chunks)

    lr_chunks = lr.split(chunk_size, dim=1)
    log_decay_chunks = log_decay.split(chunk_size, dim=1) if log_decay is not None else [None] * num_chunks

    s_index = 0
    output = torch.zeros_like(q)
    for i in range(num_chunks):
        e_index = s_index + chunk_size
        # We do not update the last chunk if its length is smaller than chunk_size
        if i == num_chunks - 1 and rem != 0:
            continue

        if verbose: print(f"{i + 1}-th chunk")

        out, w0, w1 = update_chunk_wo_momentum(
            q_chunks[i], k_chunks[i], v_chunks[i],
            lr_chunks[i], log_decay_chunks[i],
            w0, w1,  # w0: [b, dh, d]; w1: [b, d, dh]
            w0_norm, w1_norm,
            ttt_norm_weight.unsqueeze(-1), ttt_norm_bias.unsqueeze(-1),
            verbose=verbose,
        )

        output[..., s_index:e_index] = out
        s_index = e_index

    if rem != 0:
        # processing last chunk, direct read out
        q_chunk = q_chunks[-1]
        h = F.gelu(torch.bmm(w0, q_chunk))
        tail = torch.bmm(w1, h)
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
    else:
        buf_k = q.new_zeros(b * h, d, 0)
        buf_v = q.new_zeros(b * h, d, 0)
        buf_lr = q.new_zeros(b * h, 0, lr.shape[-1])
        buf_log_decay = q.new_zeros(b * h, 0, log_decay.shape[-1]) if log_decay is not None else None

    return out, (w0, w1, buf_k, buf_v, buf_lr, buf_log_decay)





