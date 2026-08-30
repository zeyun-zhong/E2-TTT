import torch


def build_token_weights_chunk_log(log_beta, log_gamma):
    """
    log_beta, log_gamma: [B, L, D] (log-space; typically <= 0)
    Returns (all broadcastable over [B, L, D] unless noted):
      log_alpha : [B, L, D]    where alpha = exp(log_alpha)
      sbeta     : [B, L, D]    = exp( sum_{j=t+1..L-1} log_beta_j )
      carry_u   : [B, 1, D]    = exp( sum_{j=0..L-1} log_gamma_j )
      carry_m   : [B, 1, D]    = exp( sum log_beta ) * sum_{q} exp( sum_{k=q+1..} (log_gamma - log_beta) )
    """
    # ---- make shapes [B, L] ----
    assert log_beta.dim() == 3 and log_gamma.dim() == 3
    log_beta = log_beta.float()
    log_gamma = log_gamma.float()

    # suffix log-products: log_sx[t] = Σ_{j=t+1..} log x_j
    # Implement via cumsum on reversed, padding a trailing 0 (exp(0)=1 sentinel)
    def suffix_logprod(logx):                              # logx: [B, L]
        pad  = torch.nn.functional.pad(logx, (0, 0, 0, 1), value=0.0)
        suf  = torch.flip(torch.cumsum(torch.flip(pad, [1]), dim=1), [1])
        return suf[:, 1:], suf[:, :1]

    log_sbeta,  log_prodb  = suffix_logprod(log_beta)     # [B, L], [B, 1]
    log_sgamma, log_prodg  = suffix_logprod(log_gamma)    # [B, L], [B, 1]

    # r_t = sγ_t / sbeta_t  => log_r = log_sgamma - log_sbeta
    log_r = log_sgamma - log_sbeta                        # [B, L]

    # R_t = Σ_{q=t..} r_q = suffix-sum(exp(log_r))
    # Use logcumsumexp on the reversed axis (loop-free, stable)
    R_log = torch.flip(torch.logcumsumexp(torch.flip(log_r, [1]), dim=1), [1])  # [B, L]

    # final weights
    sbeta   = torch.exp(log_sbeta)
    log_alpha = log_sbeta + R_log
    carry_u = torch.exp(log_prodg)
    carry_m = (log_prodb + R_log[:, :1]).exp()

    return log_alpha, sbeta, carry_u, carry_m


def build_token_weights_chunk_log_nomomentum(log_gamma):  # [B,L,d]
    assert log_gamma.dim() == 3
    log_gamma = log_gamma.float()
    pad  = torch.nn.functional.pad(log_gamma, (0, 0, 0, 1), value=0.0)
    suf = torch.flip(torch.cumsum(torch.flip(pad, [1]), dim=1), [1])
    log_sgamma, log_prodg = suf[:, 1:], suf[:, :1]
    alpha_log = log_sgamma
    carry_u = torch.exp(log_prodg)
    return alpha_log, carry_u