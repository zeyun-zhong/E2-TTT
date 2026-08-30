# -*- coding: utf-8 -*-
"""Chunked TTT update for the SwiGLU fast weight.

``chunk_ttt_momentum`` / ``chunk_ttt_momentum_with_cache`` are the reference
implementation and the one every result in the paper was produced with.

The ``fused_*`` aliases are an **experimental** Triton implementation of the
same update (see ``triton_kernels/``). They are not enabled by any shipped
config, were not used for any paper number, and are not covered by the smoke
test -- verify them yourself before relying on them.
"""

from .chunk_momentum import chunk_ttt_momentum, chunk_ttt_momentum_with_cache
from .chunk_momentum_fused_kernel import (
    chunk_ttt_momentum as fused_chunk_ttt_momentum,
    chunk_ttt_momentum_with_cache as fused_chunk_ttt_momentum_with_cache,
)

__all__ = [
    'chunk_ttt_momentum',
    'chunk_ttt_momentum_with_cache',
    'fused_chunk_ttt_momentum',
    'fused_chunk_ttt_momentum_with_cache'
]
