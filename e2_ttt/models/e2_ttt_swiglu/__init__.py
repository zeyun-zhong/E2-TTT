# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from e2_ttt.models.e2_ttt_swiglu.configuration import E2TTTSwiGLUConfig
from e2_ttt.models.e2_ttt_swiglu.modeling import E2TTTSwiGLUForCausalLM, E2TTTSwiGLUModel

AutoConfig.register(E2TTTSwiGLUConfig.model_type, E2TTTSwiGLUConfig)
AutoModel.register(E2TTTSwiGLUConfig, E2TTTSwiGLUModel)
AutoModelForCausalLM.register(E2TTTSwiGLUConfig, E2TTTSwiGLUForCausalLM)

__all__ = ['E2TTTSwiGLUConfig', 'E2TTTSwiGLUForCausalLM', 'E2TTTSwiGLUModel']
