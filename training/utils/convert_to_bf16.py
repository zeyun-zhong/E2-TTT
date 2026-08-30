# -*- coding: utf-8 -*-
"""Cast HF-format checkpoints under `exp/` from fp32 to bf16.

The checkpoints written by `flame.utils.convert_dcp_to_hf` are fp32, which is
twice the size actually needed at eval time (`eval.sh` loads them with
`dtype=bfloat16` anyway). This rewrites the safetensors shards in bf16 and
copies everything else (config, tokenizer, generation config) alongside them.

The weights are streamed tensor-by-tensor straight from the safetensors files,
so nothing here needs the modeling code, `trust_remote_code`, or a GPU.

    # every checkpoint in exp/, each written to exp/<name>-bf16
    python utils/convert_to_bf16.py

    # a specific checkpoint, overwriting the fp32 files in place
    python utils/convert_to_bf16.py exp/e2_ttt_mlp_340M --in-place

    # a specific checkpoint, to a directory of your choosing
    python utils/convert_to_bf16.py exp/e2_ttt_swiglu_1B --output exp/swiglu_1B_bf16
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Files that describe the checkpoint but hold no weights; copied verbatim
# (except config.json / the index, which are patched below).
INDEX_FILE = "model.safetensors.index.json"
CONFIG_FILE = "config.json"


def find_checkpoints(path: str):
    """A checkpoint directory, or a directory of checkpoint directories."""
    if any(f.endswith(".safetensors") for f in os.listdir(path)):
        return [path]
    checkpoints = sorted(
        os.path.join(path, name) for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
        and any(f.endswith(".safetensors") for f in os.listdir(os.path.join(path, name)))
    )
    if not checkpoints:
        raise FileNotFoundError(f"No safetensors checkpoint found in or under {path}")
    return checkpoints


def convert_shard(src: str, dst: str, dtype: torch.dtype):
    """Rewrite one safetensors shard, casting its floating-point tensors."""
    num_bytes, converted = 0, 0
    with safe_open(src, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        state_dict = {}
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Leave integer/bool buffers (position ids, masks, ...) untouched.
            if tensor.is_floating_point() and tensor.dtype != dtype:
                tensor = tensor.to(dtype)
                converted += 1
            state_dict[key] = tensor
            num_bytes += tensor.numel() * tensor.element_size()
    # `save_pretrained` stamps format=pt; keep it so the shard reloads cleanly.
    metadata.setdefault("format", "pt")
    save_file(state_dict, dst, metadata=metadata)
    return num_bytes, converted, len(state_dict)


def patch_config(path: str, dtype_name: str):
    """Record the new dtype so `from_pretrained` restores it by default."""
    with open(path) as f:
        config = json.load(f)
    # transformers < 5 calls this `torch_dtype`, >= 4.56 writes `dtype`.
    for key in ("dtype", "torch_dtype"):
        if key in config:
            config[key] = dtype_name
    config.setdefault("dtype", dtype_name)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def patch_index(path: str, total_size: int):
    """The index advertises the on-disk byte count; it just halved."""
    with open(path) as f:
        index = json.load(f)
    index.setdefault("metadata", {})["total_size"] = total_size
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")


def convert(checkpoint: str, output: str, dtype: torch.dtype, dtype_name: str):
    print(f"Converting {checkpoint} -> {output} ({dtype_name})")
    os.makedirs(output, exist_ok=True)

    shards = sorted(f for f in os.listdir(checkpoint) if f.endswith(".safetensors"))
    total_size, total_converted, total_tensors = 0, 0, 0
    for shard in shards:
        src, dst = os.path.join(checkpoint, shard), os.path.join(output, shard)
        num_bytes, converted, num_tensors = convert_shard(src, dst, dtype)
        print(f"  {shard}: {num_tensors} tensors, {converted} cast, {num_bytes / 1e9:.2f} GB")
        total_size += num_bytes
        total_converted += converted
        total_tensors += num_tensors

    for name in sorted(os.listdir(checkpoint)):
        src, dst = os.path.join(checkpoint, name), os.path.join(output, name)
        if name.endswith(".safetensors") or not os.path.isfile(src):
            continue
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

    if os.path.isfile(os.path.join(output, CONFIG_FILE)):
        patch_config(os.path.join(output, CONFIG_FILE), dtype_name)
    if os.path.isfile(os.path.join(output, INDEX_FILE)):
        patch_index(os.path.join(output, INDEX_FILE), total_size)

    print(f"  done: {total_converted}/{total_tensors} tensors cast, {total_size / 1e9:.2f} GB total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cast HF checkpoints to bf16 (or fp16).")
    parser.add_argument("checkpoints", nargs="*", default=["exp"],
                        help="Checkpoint directories, or a directory holding them (default: exp).")
    parser.add_argument("--output", default=None,
                        help="Output directory; only valid with a single checkpoint.")
    parser.add_argument("--suffix", default="-bf16",
                        help="Suffix appended to each checkpoint name when --output is not given.")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite the fp32 files instead of writing a new directory.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Target dtype (default: bfloat16).")
    args = parser.parse_args()

    checkpoints = [c for path in args.checkpoints for c in find_checkpoints(path)]
    if args.output is not None and len(checkpoints) > 1:
        parser.error(f"--output takes a single checkpoint, but {len(checkpoints)} were resolved")
    if args.output is not None and args.in_place:
        parser.error("--output and --in-place are mutually exclusive")

    dtype = getattr(torch, args.dtype)
    for checkpoint in checkpoints:
        checkpoint = checkpoint.rstrip("/")
        if args.in_place:
            output = checkpoint
        elif args.output is not None:
            output = args.output
        else:
            output = checkpoint + args.suffix
        convert(checkpoint, output, dtype, args.dtype)
