#!/usr/bin/bash
#   bash train_340M.sh configs/e2_ttt_mlp_340M.json exp/e2_ttt_mlp_340M
set -euo pipefail

MODEL_CONFIG=${1:?usage: train_340M.sh <model-config.json> <dump-folder>}
DUMP=${2:?usage: train_340M.sh <model-config.json> <dump-folder>}

export NGPU=4   # 4 x 16 x 4 = 256

TOKENIZER=fla-hub/transformer-1.3B-100B
DATASET=HuggingFaceFW/fineweb-edu
DATASET_NAME=sample-100BT
STEPS=28672   # 28672 x 256 x 2048 tokens ~= 15B

bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder ${DUMP} \
  --model.config ${MODEL_CONFIG} \
  --model.tokenizer_path ${TOKENIZER} \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 4 \
  --training.seq_len 2048 \
  --training.gradient_accumulation_steps 16 \
  --training.steps ${STEPS} \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset ${DATASET} \
  --training.dataset_name ${DATASET_NAME} \
  --training.dataset_split train \
  --training.num_workers 12 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 100
