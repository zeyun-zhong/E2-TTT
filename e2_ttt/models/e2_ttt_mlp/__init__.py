# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from e2_ttt.models.e2_ttt_mlp.configuration import E2TTTMLPConfig
from e2_ttt.models.e2_ttt_mlp.modeling import E2TTTMLPForCausalLM, E2TTTMLPModel

AutoConfig.register(E2TTTMLPConfig.model_type, E2TTTMLPConfig)
AutoModel.register(E2TTTMLPConfig, E2TTTMLPModel)
AutoModelForCausalLM.register(E2TTTMLPConfig, E2TTTMLPForCausalLM)

__all__ = ['E2TTTMLPConfig', 'E2TTTMLPForCausalLM', 'E2TTTMLPModel']
