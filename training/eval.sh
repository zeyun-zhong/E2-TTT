#!/usr/bin/bash
# Evaluation driver for a trained checkpoint (an HF-format folder, i.e. the
# output of `flame.utils.convert_dcp_to_hf`, or a Hub model id).
#
#   bash eval.sh exp/e2_ttt_mlp_340M                  # the `general` suite
#   bash eval.sh exp/e2_ttt_swiglu_1B ruler longbench
#
# Suites:
#   general    Wikitext / LAMBADA perplexity + commonsense reasoning
#   recall     SQuAD and SWDE real-world retrieval
#   ruler      RULER S-NIAH-1/2 at 512 - 16K context
#   longbench  the 14 LongBench tasks
#
# FDA is not run here; see the README.
set -euo pipefail

CHECKPOINT=${1:?usage: eval.sh <checkpoint> [suite ...]}
shift
SUITES=("$@")
if [ ${#SUITES[@]} -eq 0 ]; then SUITES=(general); fi

OUTPUT=${OUTPUT:-results}

COMMON=(--model_args "pretrained=${CHECKPOINT},dtype=bfloat16,trust_remote_code=True"
        --batch_size 1 --num_fewshot 0 --show_config --trust_remote_code)

GENERAL_TASKS=wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge
RECALL_TASKS=squad_completion,swde
RULER_TASKS=niah_single_1,niah_single_2
LONGBENCH_TASKS=longbench_narrativeqa,longbench_qasper,longbench_multifieldqa_en,longbench_hotpotqa,longbench_2wikimqa,longbench_musique,longbench_gov_report,longbench_qmsum,longbench_multi_news,longbench_trec,longbench_triviaqa,longbench_samsum,longbench_lcc,longbench_repobench-p

for suite in "${SUITES[@]}"; do
  echo "=============== ${suite} :: ${CHECKPOINT} ==============="
  case "${suite}" in
    general)
      accelerate launch -m evals.harness --output_path "${OUTPUT}/general" \
        --tasks ${GENERAL_TASKS} "${COMMON[@]}"
      ;;
    recall)
      # squad_completion and swde use the white-space-stripped variants; see
      # https://github.com/EleutherAI/lm-evaluation-harness/issues/2690
      accelerate launch -m evals.harness --output_path "${OUTPUT}/recall" \
        --tasks ${RECALL_TASKS} "${COMMON[@]}"
      ;;
    ruler)
      # max_length must exceed the longest evaluated context; these models are
      # trained at 2K, so everything past 2048 is extrapolation.
      accelerate launch -m evals.harness --output_path "${OUTPUT}/ruler" \
        --tasks ${RULER_TASKS} \
        --model_args "pretrained=${CHECKPOINT},dtype=bfloat16,max_length=32768,trust_remote_code=True" \
        --metadata='{"max_seq_lengths":[512, 1024, 2048, 4096, 8192, 16384]}' \
        --batch_size 1 --num_fewshot 0 --show_config --trust_remote_code
      ;;
    longbench)
      accelerate launch -m evals.harness --output_path "${OUTPUT}/longbench" \
        --tasks ${LONGBENCH_TASKS} \
        --model_args "pretrained=${CHECKPOINT},dtype=bfloat16,max_length=128000,trust_remote_code=True" \
        --batch_size 1 --num_fewshot 0 --show_config --trust_remote_code
      ;;
    *)
      echo "unknown suite '${suite}' (general | recall | ruler | longbench)" >&2
      exit 1
      ;;
  esac
done
