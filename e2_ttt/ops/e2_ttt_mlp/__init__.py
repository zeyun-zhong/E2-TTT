# -*- coding: utf-8 -*-
"""Chunked TTT update for the 2-layer-MLP fast weight.

Two variants, selected by the layer from ``use_ttt_momentum``:
``chunk_ttt_momentum`` (the default -- momentum, decay and a LayerNorm
reconstruction target) and ``chunk_ttt_wo_momentum`` (no momentum; it also
takes neither ``use_closed_form`` nor ``use_grad_clip``, since both describe
the momentum recurrence).

Each has a ``*_with_cache`` counterpart used for token-by-token generation.
"""

from .chunk_wo_momentum import chunk_ttt_wo_momentum, chunk_ttt_wo_momentum_with_cache
from .chunk_momentum import chunk_ttt_momentum, chunk_ttt_momentum_with_cache

__all__ = [
    'chunk_ttt_wo_momentum',
    'chunk_ttt_wo_momentum_with_cache',
    'chunk_ttt_momentum',
    'chunk_ttt_momentum_with_cache',
]
