# Training and evaluation

Everything needed to reproduce the paper's language-modeling results. `training/` is a fork of
[fla-org/flame](https://github.com/fla-org/flame), kept at the state used for the paper; for a
maintained training framework, use the upstream repository.

## Setup

Install the environment as described in the root `README.md`, then run every command below from
this directory. `train.sh` additionally needs `jq`.

Point `HF_HOME` at a disk with room for FineWeb-Edu `sample-100BT` (hundreds of GB) and the
evaluation datasets:

```bash
export HF_HOME=/path/to/huggingface/cache
```

## Data

Training reads `HuggingFaceFW/fineweb-edu` (`sample-100BT`) with the `fla-hub/transformer-1.3B-100B`
tokenizer. Nothing is vendored here; populate the cache once before the first run:

```python
from datasets import load_dataset

load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", num_proc=64)
```

Two helpers for constrained nodes:

* `save_shuffle_indices.py` writes `global_shuffle_indices.npy` on a large-memory machine.
  When that file is present in the working directory, `flame/data.py` applies it with `select()`
  instead of shuffling the dataset itself, which otherwise needs the whole index in RAM.
* `utils/reshard.py` re-saves a dataset with more shards, for when the shard count is below
  `dp_degree x num_workers`.

## Training

```bash
bash train_340M.sh configs/e2_ttt_mlp_340M.json   exp/e2_ttt_mlp_340M
bash train_1B.sh   configs/e2_ttt_swiglu_1B.json  exp/e2_ttt_swiglu_1B
```

Both take `<model-config> <dump-folder>` and work with any config in `configs/`.


## Evaluation

`eval.sh` takes a checkpoint — a dump folder converted by `train.sh`, or a Hub model id — and one
or more suites (default `general`), and writes to `results/<suite>/` (`OUTPUT` overrides `results`):

```bash
bash eval.sh exp/e2_ttt_swiglu_1B
bash eval.sh exp/e2_ttt_swiglu_1B general recall ruler longbench
```

| Suite | Tasks |
| --- | --- |
| `general` | WikiText, LAMBADA, PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge |
| `recall` | SQuAD completion, SWDE |
| `ruler` | RULER S-NIAH-1/2 at 512–16K context |
| `longbench` | the 14 LongBench tasks |

Everything runs through `evals/harness.py`, a thin wrapper that registers E²-TTT and the LaCT
baseline with the vendored lm-evaluation-harness. The models are trained at 2K context, so
everything RULER evaluates past 2,048 tokens is extrapolation.

<details>
<summary>Vendored lm-evaluation-harness</summary>

`lm-evaluation-harness/` at the repository root is
[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) at commit
[`c9772b9`](https://github.com/EleutherAI/lm-evaluation-harness/commit/c9772b90a8ee95b0df1ba76651443a0534c96aad)
(2025-12-12), with four local changes (upstream's `.github/` CI workflows are not vendored):

* `lm_eval/models/huggingface.py` — honour the model's own `config.use_cache` during generation
  instead of forcing `use_cache=True`.
* `lm_eval/tasks/{squad_completion,swde}/task.py` — strip surrounding whitespace from the prompt
  and the target, per [lm-evaluation-harness#2690](https://github.com/EleutherAI/lm-evaluation-harness/issues/2690).
* `lm_eval/tasks/winogrande/default.yaml` — `dataset_path: allenai/winogrande` (the Hub dataset
  moved).
* `lm_eval/tasks/longbench/{trec.yaml,_generate_config.py}` — use LongBench's own `trec` prompt
  (`{context}\n{question}{answer_prefix}`) rather than the harness's paraphrase.
</details>

**FDA** is not run here. It uses
[HazyResearch/prefix-linear-attention](https://github.com/HazyResearch/prefix-linear-attention),
following [GatedDeltaNet's recommendation](https://github.com/NVlabs/GatedDeltaNet?tab=readme-ov-file#5%EF%B8%8F%E2%83%A3-any-guidance-for-evaluating-the-models).
Clone it, then from its `lm-eval-harness` directory:

```bash
PYTHONPATH=. python launch_hf.py -m <checkpoint> -t based_fda \
  --batch-size 1 --context_length 1000 --answer_length 50 --cutting_context --limit -1 -p \
  --output_dir <out>
```
