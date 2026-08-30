# E²-TTT: Rethinking Expressivity and Efficiency in Test-Time Training

Zeyun Zhong<sup>1,3</sup> · Joya Chen<sup>2</sup> · Manuel Martin<sup>3</sup> · Frederik Diederichs<sup>3</sup> · Juergen Gall<sup>4,5</sup> · Juergen Beyerer<sup>1,3</sup>

<sup>1</sup>Karlsruhe Institute of Technology (KIT) · <sup>2</sup>National University of Singapore · <sup>3</sup>Fraunhofer IOSB · <sup>4</sup>Lamarr Institute for Machine Learning and Artificial Intelligence · <sup>5</sup>University of Bonn

[![arXiv](https://img.shields.io/badge/arXiv-2608.21308-b31b1b.svg)](https://arxiv.org/abs/2608.21308)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-ffd21e.svg)](https://huggingface.co/collections/zeyun-zhong/e2-ttt)

---

Test-Time Training (TTT) updates a fast-weight network during inference. Per-token updates are
expressive but serial; chunk-wise updates are fast but throw the per-token learning-rate, momentum
and decay dynamics away. **E²-TTT** closes the gap: under the standard approximation of taking
gradients at the chunk-start weights, we derive a **closed-form state transition that exactly
reproduces the chunk-end fast-weight and momentum states of the per-token recurrence**. Training
stays fully chunk-parallel; the update rule keeps its temporal structure.

This repository contains the language-modeling code: the `e2_ttt` package (`E²-TTT_MLP` and
`E²-TTT_SwiGLU`), the training/evaluation framework, and the configurations behind the paper's
E²-TTT and LaCT results.

## Use E²-TTT in your own model

`E2TTTMLP` and `E2TTTSwiGLU` are self-contained `nn.Module`s with the standard
`(hidden_states, ...) -> (output, None, cache)` signature, so they drop in wherever a
self-attention block goes (`pip install -e .` from the repository root is all they need):

```python
import torch
from e2_ttt import E2TTTMLP

block = E2TTTMLP(
    hidden_size=2048,
    num_heads=16,
    chunk_size=512,        # sliding-window attention window
    ttt_chunk_size=512,    # TTT chunk C
    ttt_base_lr=0.01,      # inner-loop peak learning rate
    ttt_base_decay=0.1,    # inner-loop peak weight decay
    use_ttt_momentum=True,
    use_ttt_decay=True,
    use_closed_form=True,  # the paper's exact chunk transition
    layer_idx=0,
).cuda()

x = torch.randn(2, 2048, 2048, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    o, _, cache = block(x)   # o: [2, 2048, 2048]
```

Both chunk sizes are 512 throughout the paper. A full causal LM is built the same way, from a
config:

```python
from e2_ttt import E2TTTMLPConfig, E2TTTMLPForCausalLM

config = E2TTTMLPConfig(hidden_size=2048, num_heads=16, num_hidden_layers=24, vocab_size=32000)
model = E2TTTMLPForCausalLM(config).cuda()
```


<details>
<summary><b>Experimental: fused Triton kernels</b></summary>

`E²-TTT_SwiGLU` also ships a fused Triton implementation of the chunk update
(`e2_ttt/ops/e2_ttt_swiglu/triton_kernels/`), enabled with `use_fused_ttt_kernel=True` on the layer
or the config:

```python
from e2_ttt import E2TTTSwiGLU

block = E2TTTSwiGLU(hidden_size=2048, num_heads=16, use_fused_ttt_kernel=True, layer_idx=0)
```

This is **experimental and not part of the paper**: every reported number was produced with the
default PyTorch chunk update (`use_fused_ttt_kernel=False`), no shipped config turns it on, and it
is not covered by the tests. In our preliminary experiments the fused kernels lowered peak GPU
memory somewhat, but were also somewhat slower in wall-clock time, so we left them off. Verify both
correctness and speed on your own setup before relying on them.

</details>


## Installation

Linux with an NVIDIA GPU and a driver new enough for CUDA 12.x; the environment ships prebuilt
CUDA 12.8 wheels, so no local CUDA toolkit is required.

```bash
conda env create -f environment.yml
conda activate e2ttt

pip install -e .                       # the e2_ttt package
pip install -e lm-evaluation-harness   # the vendored harness
```


### Check the closed-form state transition

The claim above — that with gradients taken at the chunk-start weights, the closed-form transition
reproduces the chunk-end states of the per-token recurrence — is checkable numerically (`--device
cpu` works too):

```bash
cd training
python verify_state_transition.py --device cuda \
  --num-chunks 128 --chunk-size 512 --feature-dim 1024 --state-dim 128 --rtol 1e-5
```

## Reproducing the paper

Training and evaluation live in `training/`; see [training/README.md](training/README.md) for the
data preparation, the full recipes, and the evaluation suites.

```bash
cd training
bash train_340M.sh configs/e2_ttt_mlp_340M.json   exp/e2_ttt_mlp_340M    # 340M, 15B tokens
bash train_1B.sh   configs/e2_ttt_swiglu_1B.json  exp/e2_ttt_swiglu_1B   # 1.3B, 15B tokens

bash eval.sh exp/e2_ttt_swiglu_1B general recall ruler longbench
```

Training reads `HuggingFaceFW/fineweb-edu` (`sample-100BT`) with the
`fla-hub/transformer-1.3B-100B` tokenizer. Both scripts take `<model-config> <dump-folder>` and
accept any config in `training/configs/`; `eval.sh` takes the same dump folder, which `train.sh`
leaves in Hugging Face format.


## Checkpoints

All four checkpoints are on the Hub, collected under
[🤗 E²-TTT](https://huggingface.co/collections/zeyun-zhong/e2-ttt):

| Fast weights | 340M | 1.3B |
| --- | --- | --- |
| MLP | [`e2-ttt-mlp-340M-15B`](https://huggingface.co/zeyun-zhong/e2-ttt-mlp-340M-15B) | [`e2-ttt-mlp-1.3B-15B`](https://huggingface.co/zeyun-zhong/e2-ttt-mlp-1.3B-15B) |
| SwiGLU | [`e2-ttt-swiglu-340M-15B`](https://huggingface.co/zeyun-zhong/e2-ttt-swiglu-340M-15B) | [`e2-ttt-swiglu-1.3B-15B`](https://huggingface.co/zeyun-zhong/e2-ttt-swiglu-1.3B-15B) |

All are trained on FineWeb-Edu for a 15B-token budget at 2K context, and released in bfloat16.

They load through the `Auto*` factories, which need the model types registered.

```python
import torch
import e2_ttt  # registers e2_ttt_mlp / e2_ttt_swiglu (and the LaCT baseline)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "zeyun-zhong/e2-ttt-swiglu-1.3B-15B",
    dtype=torch.bfloat16,
).cuda()
```

`eval.sh` takes a Hub id wherever it takes a dump folder:

```bash
cd training
bash eval.sh zeyun-zhong/e2-ttt-swiglu-1.3B-15B general
```

## Acknowledgements

Each vendored directory keeps its own LICENSE and NOTICE.

* [kazuki-irie/hybrid-memory](https://github.com/kazuki-irie/hybrid-memory) — repository
  organization, the language-modeling setup, and the HQLT baseline.
* [fla-org/flame](https://github.com/fla-org/flame) — `training/` is a fork.
* [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) — `e2_ttt/`
  follows its conventions and depends on it.
* [LaCT](https://github.com/tianyuanzhang/lact) — `e2_ttt/baselines/lact/` and the SwiGLU Triton
  kernels.
* [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and
  [HazyResearch/prefix-linear-attention](https://github.com/HazyResearch/prefix-linear-attention) —
  evaluation. `lm-evaluation-harness/` is vendored with a few local changes, listed in
  [training/README.md](training/README.md).

## Citation

```bibtex
@article{zhong2026e2ttt,
  title   = {Rethinking Expressivity and Efficiency in Test-Time Training},
  author  = {Zhong, Zeyun and Chen, Joya and Martin, Manuel and
             Diederichs, Frederik and Gall, Juergen and Beyerer, Juergen},
  journal = {arXiv preprint arXiv:2608.21308},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). Vendored and derived code keeps its own license: `training/LICENSE`
(flame, MIT), `lm-evaluation-harness/LICENSE.md`.
