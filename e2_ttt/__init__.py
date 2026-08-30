# -*- coding: utf-8 -*-
"""E2-TTT — Expressive and Efficient Test-Time Training.

Reference implementation for "Rethinking Expressivity and Efficiency in
Test-Time Training" (Zhong et al.).

Importing this package registers every model with HuggingFace's ``Auto*``
factories, so ``AutoModelForCausalLM.from_pretrained(...)`` resolves the
``e2_ttt_mlp`` / ``e2_ttt_swiglu`` model types.

Derived from flash-linear-attention (MIT) by Songlin Yang et al.
"""

from e2_ttt import baselines  # noqa: F401  (registers the LaCT baseline)
from e2_ttt.layers import E2TTTMLP, E2TTTSwiGLU
from e2_ttt.models import (
    E2TTTMLPConfig,
    E2TTTMLPForCausalLM,
    E2TTTMLPModel,
    E2TTTSwiGLUConfig,
    E2TTTSwiGLUForCausalLM,
    E2TTTSwiGLUModel,
)

__version__ = '1.0.0'

__all__ = [
    'E2TTTMLP',
    'E2TTTMLPConfig',
    'E2TTTMLPForCausalLM',
    'E2TTTMLPModel',
    'E2TTTSwiGLU',
    'E2TTTSwiGLUConfig',
    'E2TTTSwiGLUForCausalLM',
    'E2TTTSwiGLUModel',
    '__version__',
]
