"""Check the closed-form chunk state transition against the token-wise recurrence.

Both paths consume the same fixed sequence of gradients, so this isolates the scan
algebra of ``build_token_weights_chunk_log`` -- not the chunk-start gradient
approximation, weight normalization, gradient clipping, bf16, or the Triton kernels.
"""

import argparse
import math

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def load_build_token_weights_chunk_log():
    try:
        from e2_ttt.ops.scan_utils import build_token_weights_chunk_log
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Could not import e2_ttt. Install the package first: `pip install -e .` "
            "from the repository root."
        ) from exc
    return build_token_weights_chunk_log


def relative_error(reference, value, eps: float = 1e-12):
    numerator = torch.linalg.vector_norm((value - reference).reshape(reference.shape[0], -1), dim=1)
    denominator = torch.linalg.vector_norm(reference.reshape(reference.shape[0], -1), dim=1).clamp_min(eps)
    return numerator / denominator


def make_token_inputs(
    batch_size: int,
    total_tokens: int,
    feature_dim: int,
    state_dim: int,
    eta_base: float,
    tau: float,
    alpha_base: float,
    device,
    dtype,
):
    scale = 1.0 / math.sqrt(feature_dim)

    x = torch.randn(batch_size, total_tokens, feature_dim, device=device, dtype=dtype)

    w_eta = torch.randn(feature_dim, 1, device=device, dtype=dtype) * scale
    w_beta = torch.randn(feature_dim, 1, device=device, dtype=dtype) * scale
    w_alpha = torch.randn(feature_dim, 1, device=device, dtype=dtype) * scale
    w_grad = torch.randn(feature_dim, state_dim * state_dim, device=device, dtype=dtype) * scale

    b_eta = torch.randn(1, device=device, dtype=dtype) * 0.1
    b_beta = torch.randn(1, device=device, dtype=dtype) * 0.1
    b_alpha = torch.randn(1, device=device, dtype=dtype) * 0.1
    b_grad = torch.randn(state_dim * state_dim, device=device, dtype=dtype) * 0.1

    eta_logits = x @ w_eta + b_eta
    beta_logits = x @ w_beta + b_beta
    alpha_logits = x @ w_alpha + b_alpha
    grad_logits = x @ w_grad + b_grad

    eta = eta_base * torch.sigmoid(eta_logits)
    beta = torch.sigmoid(beta_logits).pow(1.0 / tau)
    alpha = alpha_base * torch.sigmoid(alpha_logits)
    gamma = 1.0 - eta * alpha
    gradients = torch.tanh(grad_logits).reshape(batch_size, total_tokens, state_dim, state_dim)

    return eta, beta, gamma, gradients


def run_sequential(
    eta,
    beta,
    gamma,
    gradients,
    weight_init,
    momentum_init,
):
    weight = weight_init.clone()
    momentum = momentum_init.clone()
    weight_history = []
    momentum_history = []

    for token_idx in range(eta.shape[1]):
        beta_t = beta[:, token_idx].unsqueeze(-1)
        eta_t = eta[:, token_idx].unsqueeze(-1)
        gamma_t = gamma[:, token_idx].unsqueeze(-1)

        momentum = beta_t * momentum + eta_t * gradients[:, token_idx]
        weight = gamma_t * weight + momentum

        weight_history.append(weight.clone())
        momentum_history.append(momentum.clone())

    return weight, momentum, weight_history, momentum_history


def run_chunk_parallel(
    eta,
    beta,
    gamma,
    gradients,
    weight_init,
    momentum_init,
    chunk_size: int,
    build_token_weights_chunk_log,
):
    batch_size, total_tokens, _ = eta.shape
    if total_tokens % chunk_size != 0:
        raise ValueError(f"total_tokens={total_tokens} must be divisible by chunk_size={chunk_size}")

    weight = weight_init.clone()
    momentum = momentum_init.clone()
    weight_history = []
    momentum_history = []

    for start in range(0, total_tokens, chunk_size):
        end = start + chunk_size

        eta_chunk = eta[:, start:end]
        beta_chunk = beta[:, start:end]
        gamma_chunk = gamma[:, start:end]
        gradient_chunk = gradients[:, start:end]

        log_alpha, sbeta, carry_u, carry_m = build_token_weights_chunk_log(beta_chunk.log(), gamma_chunk.log())

        alpha = log_alpha.exp().to(eta_chunk.dtype)
        sbeta = sbeta.to(eta_chunk.dtype)
        carry_u = carry_u.squeeze(1).to(eta_chunk.dtype).unsqueeze(-1)
        carry_m = carry_m.squeeze(1).to(eta_chunk.dtype).unsqueeze(-1)

        weight_update = ((alpha * eta_chunk).unsqueeze(-1) * gradient_chunk).sum(dim=1)
        momentum_update = ((sbeta * eta_chunk).unsqueeze(-1) * gradient_chunk).sum(dim=1)
        prod_beta = beta_chunk.prod(dim=1).unsqueeze(-1)

        weight = carry_u * weight + carry_m * momentum + weight_update
        momentum = prod_beta * momentum + momentum_update

        weight_history.append(weight.clone())
        momentum_history.append(momentum.clone())

    return weight, momentum, weight_history, momentum_history


def verify(args: argparse.Namespace):
    if torch is None:
        raise SystemExit(
            "PyTorch is required to run this verifier. "
            "Install `torch` in the project environment and rerun the script."
        ) from TORCH_IMPORT_ERROR

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32

    total_tokens = args.num_chunks * args.chunk_size
    build_token_weights_chunk_log = load_build_token_weights_chunk_log()

    eta, beta, gamma, gradients = make_token_inputs(
        batch_size=args.batch_size,
        total_tokens=total_tokens,
        feature_dim=args.feature_dim,
        state_dim=args.state_dim,
        eta_base=args.eta_base,
        tau=args.tau,
        alpha_base=args.alpha_base,
        device=device,
        dtype=dtype,
    )

    weight_init = torch.randn(args.batch_size, args.state_dim, args.state_dim, device=device, dtype=dtype)
    momentum_init = torch.randn(args.batch_size, args.state_dim, args.state_dim, device=device, dtype=dtype)

    seq_weight, seq_momentum, seq_weight_history, seq_momentum_history = run_sequential(
        eta=eta,
        beta=beta,
        gamma=gamma,
        gradients=gradients,
        weight_init=weight_init,
        momentum_init=momentum_init,
    )

    chunk_weight, chunk_momentum, chunk_weight_history, chunk_momentum_history = run_chunk_parallel(
        eta=eta,
        beta=beta,
        gamma=gamma,
        gradients=gradients,
        weight_init=weight_init,
        momentum_init=momentum_init,
        chunk_size=args.chunk_size,
        build_token_weights_chunk_log=build_token_weights_chunk_log,
    )

    chunk_end_indices = list(range(args.chunk_size - 1, total_tokens, args.chunk_size))
    seq_chunk_weights = torch.stack([seq_weight_history[idx] for idx in chunk_end_indices], dim=1)
    seq_chunk_momenta = torch.stack([seq_momentum_history[idx] for idx in chunk_end_indices], dim=1)
    chunk_chunk_weights = torch.stack(chunk_weight_history, dim=1)
    chunk_chunk_momenta = torch.stack(chunk_momentum_history, dim=1)

    final_weight_rel_error = relative_error(seq_weight, chunk_weight)
    final_momentum_rel_error = relative_error(seq_momentum, chunk_momentum)
    chunk_weight_rel_error = relative_error(
        seq_chunk_weights.reshape(args.batch_size * args.num_chunks, -1),
        chunk_chunk_weights.reshape(args.batch_size * args.num_chunks, -1),
    )
    chunk_momentum_rel_error = relative_error(
        seq_chunk_momenta.reshape(args.batch_size * args.num_chunks, -1),
        chunk_chunk_momenta.reshape(args.batch_size * args.num_chunks, -1),
    )

    print("Closed-form state transition vs. token-wise recurrence (fixed gradients)")
    print(f"seed={args.seed}")
    print(f"device={device}")
    print(f"batch_size={args.batch_size}, num_chunks={args.num_chunks}, chunk_size={args.chunk_size}, total_tokens={total_tokens}")
    print(f"feature_dim={args.feature_dim}, state_dim={args.state_dim}")
    print(f"eta_base={args.eta_base}, tau={args.tau}, alpha_base={args.alpha_base}")
    print()
    print("End-of-sequence relative error")
    print(f"weight   max={final_weight_rel_error.max().item():.6e} mean={final_weight_rel_error.mean().item():.6e}")
    print(f"momentum max={final_momentum_rel_error.max().item():.6e} mean={final_momentum_rel_error.mean().item():.6e}")
    print()
    print("Chunk-boundary relative error")
    print(f"weight   max={chunk_weight_rel_error.max().item():.6e} mean={chunk_weight_rel_error.mean().item():.6e}")
    print(f"momentum max={chunk_momentum_rel_error.max().item():.6e} mean={chunk_momentum_rel_error.mean().item():.6e}")
    print()

    passed = (
        final_weight_rel_error.max().item() <= args.rtol
        and final_momentum_rel_error.max().item() <= args.rtol
    )
    print(f"PASS={passed} (rtol={args.rtol:.1e})")


def parse_args():
    parser = argparse.ArgumentParser(description="Check that the closed-form chunk state transition reproduces the token-wise recurrence for a fixed sequence of gradients.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--num-chunks", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--eta-base", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=32.0)
    parser.add_argument("--alpha-base", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    verify(parse_args())
