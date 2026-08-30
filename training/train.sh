#!/usr/bin/bash
# torchrun launcher for flame. Every argument is forwarded to `flame.train`.
#
#   NGPU=4 bash train.sh --job.config_file flame/models/fla.toml \
#                        --job.dump_folder exp/my-run \
#                        --model.config configs/e2_ttt_mlp_340M.json ...
#
# See train_340M.sh / train_1B.sh for the two recipes used in the paper.
# NNODE/NGPU/MASTER_ADDR/MASTER_PORT are read from the environment so the same
# script works under a cluster scheduler.

params=""
if [ $# -ne 0 ]; then
    params="$*"
fi

NNODE=${NNODE:-"1"}
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}

if [[ -z "${MASTER_ADDR}" ]]; then
  export MASTER_ADDR="localhost"
fi
if [[ -z "${MASTER_PORT}" ]]; then
  export MASTER_PORT="0"
fi

echo "Launching training..."

set -x
path=$(grep -oP '(?<=--job.dump_folder )[^ ]+' <<< "$params")
steps=$(grep -oP '(?<=--training.steps )[^ ]+' <<< "$params")
config=$(grep -oP '(?<=--model.config )[^ ]+' <<< "$params")
tokenizer=$(grep -oP '(?<=--model.tokenizer_path )[^ ]+' <<< "$params")
# `e2_ttt` registers our models with transformers; `fla` registers the baselines.
model=$(
  python -c "import e2_ttt, fla, sys; from transformers import AutoConfig; print(AutoConfig.from_pretrained(sys.argv[1]).to_json_string())" "$config" | jq -r '.model_type'
)

# Snapshot the exact model config and training code next to the checkpoints, so a
# finished run stays interpretable after the working tree moves on.
mkdir -p "$path"
cp "$config" "$path"
cp -r flame "$path"

if [ "$date" == "" ]; then
  date=$(date +%Y%m%d%H%M)
fi
RUN_NAME="$model-$(basename $path)"
RUN_ID="$RUN_NAME-$date"

export WANDB_RESUME=allow
if [[ -z "${WANDB_PROJECT}" ]]; then
  export WANDB_PROJECT="e2-ttt"
fi
if [[ -z "${WANDB_NAME}" ]]; then
  export WANDB_NAME="$RUN_NAME"
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  export WANDB_RUN_ID="$RUN_ID"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=${NNODE} \
  --nproc_per_node=${NGPU} \
  --rdzv_backend c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  --local-ranks-filter ${LOG_RANK} \
  --role rank \
  --tee 3 \
  --log-dir $path/logs \
  -m flame.train \
  $params

echo "TRAINING DONE!"
echo "Converting the DCP checkpoints to HF format..."

python -m flame.utils.convert_dcp_to_hf \
  --path $path \
  --step $steps \
  --config $config \
  --tokenizer $tokenizer

echo "RUNNING DONE!"
