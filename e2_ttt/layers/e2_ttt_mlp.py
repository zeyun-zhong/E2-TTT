# -*- coding: utf-8 -*-
# Derived from flash-linear-attention's DeltaNet layer:
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified by:
# Copyright (c) 2025 Zeyun Zhong, Joya Chen

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input, unpad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution, RotaryEmbedding
from e2_ttt.ops.e2_ttt_mlp import (
    chunk_ttt_momentum,
    chunk_ttt_momentum_with_cache,
    chunk_ttt_wo_momentum,
    chunk_ttt_wo_momentum_with_cache,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None


class E2TTTMLP(nn.Module):
    r"""
    E²-TTT with a 2-layer MLP fast weight ("E²-TTT_MLP" in the paper).

    Each layer runs two memories in parallel and combines their outputs:

    * a **sliding-window softmax attention** branch (window ``chunk_size``) --
      the short-term, exact memory, optionally with RoPE;
    * a **test-time-training** branch -- the long-term, compressive memory,
      whose fast weights ``(w0, w1)`` are updated chunk-wise by gradient
      descent on a reconstruction loss with per-token learning rate, momentum
      and decay.

    The paper's contribution lives in the TTT branch: with
    ``use_closed_form=True`` the chunk update reproduces the chunk-end state of
    the per-token recurrence exactly, instead of collapsing the per-token
    momentum and decay factors to chunk-level averages.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        use_gate (bool, Optional):
            Whether to gate the output norm. Default: `False`.
        use_local_pos_encoding (bool, Optional):
            Apply RoPE to the sliding-window attention branch. Default: `False`.
        use_memory_gate (bool, Optional):
            Fuse the two branches with a learned per-channel gate instead of a
            plain sum. Default: `False`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions on q/k/v. Default: `True`.
        conv_size (int, Optional):
            Kernel size of the short convolution, only used when
            `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use a bias in the short convolution. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layers. Default: 1e-5.
        chunk_size (int, Optional):
            The **sliding-window attention** window -- not to be confused with
            `ttt_chunk_size`. Default: 64.
        ttt_base_lr (float, Optional):
            Peak inner-loop learning rate. Default: 0.1.
        ttt_chunk_size (int, Optional):
            The **TTT** chunk C over which the fast weights are updated.
            Default: 1024.
        use_ttt_momentum (bool, Optional):
            Enable the inner-loop momentum term. Default: `False`.
        ttt_base_decay (float, Optional):
            Peak inner-loop weight decay. Default: 0.1.
        use_ttt_decay (bool, Optional):
            Enable the inner-loop weight-decay term. Default: `False`.
        use_closed_form (bool, Optional):
            Use the paper's exact closed-form chunk transition. Only meaningful
            together with `use_ttt_momentum`; ignored otherwise. Setting it to
            `False` gives the "chunk-averaged" baseline. Default: `True`.
        use_grad_clip (bool, Optional):
            Clip the inner-loop gradients. Default: `True`.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 4,
        use_gate: bool = False,
        use_local_pos_encoding: bool = False,
        use_memory_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        chunk_size: int = 64,
        ttt_base_lr: float = 0.1,
        ttt_chunk_size: int = 1024,
        use_ttt_momentum: bool = False,
        ttt_base_decay: float = 0.1,
        use_ttt_decay: bool = False,
        use_closed_form: bool = True,
        use_grad_clip: bool = True,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        self.use_local_pos_encoding = use_local_pos_encoding
        self.use_memory_gate = use_memory_gate
        self.max_position_embeddings = kwargs.get('max_position_embeddings', None)

        # The TTT branch reads and writes the same space, so q/k/v share one dim.
        self.key_dim = self.value_dim = hidden_size
        self.head_k_dim = self.head_v_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, f"hidden size must be divisible by num_heads of {num_heads}"

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_local_pos_encoding:
            self.rope_theta = 10000.  # hard coded atm
            self.rotary = RotaryEmbedding(dim=self.head_k_dim, base=self.rope_theta)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, activation=None)
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, activation=None)
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, activation='silu')
        if use_memory_gate:
            self.memory_gate_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # ttt related
        self.ttt_norm_weight = nn.Parameter(torch.ones(num_heads, self.head_k_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(num_heads, self.head_k_dim))
        inter_multi = 4
        self.ttt_inter_dim = inter_multi * self.head_k_dim
        self.w0 = nn.Parameter(torch.randn(num_heads, self.ttt_inter_dim, self.head_k_dim))
        self.w1 = nn.Parameter(torch.randn(num_heads, self.head_k_dim, self.ttt_inter_dim))
        self.lr_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.ttt_chunk_size = ttt_chunk_size
        self.base_lr = ttt_base_lr  # default value in ttt paper = 0.1
        self.use_ttt_momentum = use_ttt_momentum
        self.use_ttt_decay = use_ttt_decay
        self.base_decay = ttt_base_decay
        if use_ttt_momentum:
            self.momentum_proj = nn.Linear(hidden_size, num_heads, bias=True)
        if use_ttt_decay:
            self.decay_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # The closed form is a property of the momentum/decay recurrence; without
        # momentum there is nothing to close a form over, so it is simply unused.
        self.use_closed_form = use_closed_form
        self.use_grad_clip = use_grad_clip

    def reset_parameters(self):
        nn.init.normal_(self.w0, mean=0.0, std=1.0 / math.sqrt(self.head_k_dim))
        nn.init.normal_(self.w1, mean=0.0, std=1.0 / math.sqrt(self.ttt_inter_dim))
        nn.init.ones_(self.ttt_norm_weight)
        nn.init.zeros_(self.ttt_norm_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        # KV part
        if self.use_local_pos_encoding:
            # apply pos enc to kv part
            seqlen_offset, max_seqlen = 0, q_len
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                max_seqlen = q.shape[1] + seqlen_offset

                if attention_mask is not None:
                    # to deliminate the offsets of padding tokens
                    seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                    max_seqlen = q.shape[1] + max(seqlen_offset)

            if self.max_position_embeddings is not None:
                max_seqlen = max(max_seqlen, self.max_position_embeddings)
            q_kv, k_kv = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
        else:
            q_kv, k_kv = q, k
        v_kv = v

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            if self.training:
                assert not cache_has_content, 'This case (carry over context during training) is not implemented.'
            if not self.training:
                k_cached, v_cached = past_key_values.update(
                    attn_state=[k_kv.flatten(-2, -1), v_kv.flatten(-2, -1)],
                    layer_idx=self.layer_idx,
                    offset=q_len,
                    cache_kwargs=dict(window_size=self.chunk_size),
                )['attn_state']

            if cache_has_content:
                k_kv, v_kv = k_cached, v_cached
                k_kv = rearrange(k_kv, '... (h d) -> ... h d', d=self.head_k_dim)
                v_kv = rearrange(v_kv, '... (h d) -> ... h d', d=self.head_v_dim)
        else:
            cache_has_content = False

        if self.training:
            if attention_mask is not None:
                q_kv, (k_kv, v_kv), indices_q, cu_seqlens, max_seq_lens = unpad_input(q_kv, (k_kv, v_kv), attention_mask, q_len)
                cu_seqlens_q, cu_seqlens_k = cu_seqlens
                max_seqlen_q, max_seqlen_k = max_seq_lens
                o_kv = flash_attn_varlen_func(
                    q_kv, k_kv, v_kv,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    causal=True,
                    window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                )
                o_kv = pad_input(o_kv, indices_q, batch_size, q_len)
            elif cu_seqlens is not None:
                o_kv = flash_attn_varlen_func(
                    q_kv.squeeze(0), k_kv.squeeze(0), v_kv.squeeze(0),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                    window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                ).unsqueeze(0)
            else:
                o_kv = flash_attn_func(
                    q_kv, k_kv, v_kv,
                    causal=True,
                    window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                )
        else:
            o_kv = flash_attn_func(
                q_kv, k_kv, v_kv,
                causal=True,
                window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
            )

        # FW part
        q, k = F.silu(q), F.silu(k)
        eps = 1e-8
        q = F.normalize(q, p=2, dim=-1, eps=eps)
        k = F.normalize(k, p=2, dim=-1, eps=eps)

        lr = F.sigmoid(self.lr_proj(hidden_states).float()) * self.base_lr
        lr = lr.unsqueeze(-1)  # [b, l, h, 1]

        log_decay = None  # stays None unless the weight-decay term is enabled
        if self.use_ttt_decay:
            decay = F.sigmoid(self.decay_proj(hidden_states).float()) * self.base_decay
            log_decay = torch.log1p(-lr * decay.unsqueeze(-1))  # we use decay as a multiplicative term (1 - lr * wd) * weight

        log_momentum = None
        if self.use_ttt_momentum:
            log_momentum = F.logsigmoid(self.momentum_proj(hidden_states).float()) / 16  # we will use .exp() later, use logspace for numerical stability
            log_momentum = log_momentum.unsqueeze(-1)  # [b, l, h, 1]
            if log_decay is None:
                log_decay = torch.zeros_like(lr).float()  # logscale here, log(1.) == 0

        # scale the learning rate based on the chunk size
        lr = lr * (1 / self.ttt_chunk_size)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        # `use_closed_form` / `use_grad_clip` describe the momentum recurrence; the
        # no-momentum kernels neither accept nor need them.
        momentum_kwargs = dict(
            use_closed_form=self.use_closed_form,
            use_grad_clip=self.use_grad_clip,
        ) if self.use_ttt_momentum else {}

        if (not self.training) and (recurrent_state is not None):
            fn_ttt_cache = chunk_ttt_momentum_with_cache if self.use_ttt_momentum \
                else chunk_ttt_wo_momentum_with_cache

            o_fw, recurrent_state = fn_ttt_cache(
                q, k, v,  # [b, l, h, d]
                lr, log_decay, log_momentum,  # [b, l, h, 1]
                recurrent_state,
                self.ttt_norm_weight, self.ttt_norm_bias,  # [h, d]
                chunk_size=self.ttt_chunk_size,
                verbose=False,
                **momentum_kwargs,
            )
        else:
            if recurrent_state is None:
                recurrent_state = (self.w0, self.w1)

            fn_ttt = chunk_ttt_momentum if self.use_ttt_momentum else chunk_ttt_wo_momentum

            o_fw, recurrent_state = fn_ttt(
                q, k, v,  # [b, l, h, d]
                lr, log_decay, log_momentum,  # [b, l, h, 1]
                recurrent_state,
                self.ttt_norm_weight, self.ttt_norm_bias,  # [h, d]
                chunk_size=self.ttt_chunk_size,
                verbose=False,
                is_training=self.training,
                **momentum_kwargs,
            )

        if self.use_memory_gate:
            mg = rearrange(self.memory_gate_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim).sigmoid()
            o = mg * o_fw + (1 - mg) * o_kv
        else:
            o = o_fw + o_kv

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=0,
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
