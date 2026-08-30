# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class E2TTTMLPConfig(PretrainedConfig):
    """Configuration for E²-TTT_MLP.

    Defaults match the released 1.3B model (``training/configs/e2_ttt_mlp_1B.json``),
    so ``E2TTTMLPConfig()`` builds a sensible model without further arguments.

    Note the two window-like fields mean different things:
    ``chunk_size`` is the sliding-window attention window, ``ttt_chunk_size`` is
    the TTT chunk C.
    """

    model_type = 'e2_ttt_mlp'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        num_hidden_layers: int = 24,
        use_gate: bool = False,
        use_short_conv: bool = False,
        conv_size: int = 4,
        use_local_pos_encoding: bool = True,
        use_memory_gate: bool = True,
        chunk_size: int = 512,
        max_position_embeddings: int = 2048,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        vocab_size: int = 32000,
        ttt_base_lr: float = 0.01,
        ttt_chunk_size: int = 512,
        use_ttt_momentum: bool = True,
        ttt_base_decay: float = 0.1,
        use_ttt_decay: bool = True,
        use_closed_form: bool = True,
        use_grad_clip: bool = True,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_local_pos_encoding = use_local_pos_encoding
        self.use_memory_gate = use_memory_gate
        self.chunk_size = chunk_size
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.vocab_size = vocab_size

        self.ttt_base_lr = ttt_base_lr
        self.ttt_chunk_size = ttt_chunk_size
        self.use_ttt_momentum = use_ttt_momentum
        self.ttt_base_decay = ttt_base_decay
        self.use_ttt_decay = use_ttt_decay
        self.use_closed_form = use_closed_form
        self.use_grad_clip = use_grad_clip

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['qkv_bias'] = attn.get('qkv_bias', False)
            attn['window_size'] = attn.get('window_size', None)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
