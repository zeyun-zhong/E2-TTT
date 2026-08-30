# -*- coding: utf-8 -*-
"""LaCT baseline (Zhang et al.), ported into this framework for the paper's comparisons."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from e2_ttt.baselines.lact.configuration import LaCTSWIGLUConfig
from e2_ttt.baselines.lact.modeling import LaCTForCausalLM, LaCTModel

AutoConfig.register(LaCTSWIGLUConfig.model_type, LaCTSWIGLUConfig)
AutoModel.register(LaCTSWIGLUConfig, LaCTModel)
AutoModelForCausalLM.register(LaCTSWIGLUConfig, LaCTForCausalLM)

__all__ = ['LaCTSWIGLUConfig', 'LaCTForCausalLM', 'LaCTModel']
