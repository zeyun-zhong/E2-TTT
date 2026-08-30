# -*- coding: utf-8 -*-
"""Error reporting for the ``check_correctness`` self-checks in this package.

Every kernel module here ends with a ``check_correctness()`` that compares the
Triton implementation against a PyTorch reference and is run directly, e.g.::

    python -m e2_ttt.ops.e2_ttt_swiglu.triton_kernels.triton_swiglu_kernels
"""

import torch


def report_error(a: torch.Tensor, b: torch.Tensor, name: str) -> float:
    """Print the max-absolute and relative L2 error between two tensors."""
    a, b = a.detach().float(), b.detach().float()
    max_abs = (a - b).abs().max().item()
    rel_l2 = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    print(f"  {name:<24} max_abs={max_abs:.3e}  rel_l2={rel_l2:.3e}")
    return rel_l2
