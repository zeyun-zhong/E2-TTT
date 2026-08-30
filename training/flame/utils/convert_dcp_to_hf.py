# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import io
import os
import tempfile
from datetime import timedelta

import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa
import e2_ttt  # noqa: F401  (registers E2-TTT and the LaCT baseline)
from torchtitan.tools.logging import init_logger, logger

# Compatibility shim: transformers>=5 expects each module's `_tied_weights_keys`
# to be a dict (it calls `.keys()` on it), but fla models declare it as a list
# (e.g. `["lm_head.weight"]`). Without this, `model.save_pretrained` crashes with
# `'list' object has no attribute 'keys'`. Normalize to support both forms.
import transformers.modeling_utils as _modeling_utils

if hasattr(_modeling_utils, "_get_tied_weight_keys"):
    def _get_tied_weight_keys(module):
        tied_weight_keys = []
        for name, submodule in module.named_modules():
            tied = getattr(submodule, "_tied_weights_keys", None) or {}
            keys = tied.keys() if isinstance(tied, dict) else tied
            tied_weight_keys.extend([f"{name}.{k}" if name else k for k in keys])
        return tied_weight_keys

    _modeling_utils._get_tied_weight_keys = _get_tied_weight_keys


@torch.inference_mode()
def save_pretrained(
    path: str,
    step: int,
    config: str,
    tokenizer: str
):
    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)

    logger.info(f"Saving the config to {path}")
    config.save_pretrained(path)
    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {path}")
    tokenizer.save_pretrained(path)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(path, f'checkpoint/step-{step}')
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config)
        logger.info(model)
        logger.info("Loading state dict from the checkpoint")

        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        # torch.load now with default weights_only=True will work
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])

        logger.info(f"Saving the model to {path}")
        model.save_pretrained(path)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Convert DCP format model weights to huggingface-style.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    args = parser.parse_args()
    save_pretrained(args.path, args.step, args.config, args.tokenizer)
