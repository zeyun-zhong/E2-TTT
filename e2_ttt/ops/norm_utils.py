import torch


def layernorm_fwd(x_f32: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-8):
    """
    x_f32: [B, D, L]    (float32 compute recommended)
    gamma: [B, D, 1]
    beta:  [B, D, 1]
    returns:
      y:       [B, D, L]  (cast to gamma.dtype)
      mean:    [B, 1, L]
      inv_std: [B, 1, L]
    """
    out_dtype = gamma.dtype
    mu = x_f32.mean(dim=1, keepdim=True)
    xc = x_f32 - mu
    var = (xc * xc).mean(dim=1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    xhat = xc * inv_std
    y = xhat * gamma.float() + beta.float()  # fp32
    return y.to(out_dtype), mu, inv_std


# Backward: only dpre (input grad), using cached mean & inv_std
def layernorm_bwd(x32: torch.Tensor, dY: torch.Tensor, gamma: torch.Tensor, mean: torch.Tensor, inv_std: torch.Tensor):
    """
    x32:   [B, D, L] (same tensor used in forward, cast to float32)
    dY:    [B, D, L] (grad wrt output y)
    gamma: [B, D, 1] (detached when used here)
    mean:  [B, 1, L] (cached)
    inv_std: [B, 1, L] (cached)
    returns:
      dpre: [B, D, L] (cast to gamma.dtype)
    """
    B, D, L = x32.shape
    dY32 = dY.float()
    g32 = gamma.float()

    xhat = (x32 - mean) * inv_std                                # [B,D,L]

    gh = dY32 * g32                                              # [B,D,L]
    sum_gh = gh.sum(dim=1, keepdim=True)                         # [B,1,L]
    sum_gh_xhat = (gh * xhat).sum(dim=1, keepdim=True)           # [B,1,L]

    dpre = (gh - (sum_gh + xhat * sum_gh_xhat) / D) * inv_std    # [B,D,L]
    return dpre.to(gamma.dtype)