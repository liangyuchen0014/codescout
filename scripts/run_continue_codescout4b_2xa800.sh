#!/bin/bash

# 2x A800 launch script for continuing RL from OpenHands/CodeScout-4B.
#
# This is intended for small-budget continued-training ablations, not a full
# Qwen3-to-CodeScout reproduction. Use the same model, data, and compute budget
# across reward configs:
#   configs/reward_old_continue.yaml
#   configs/reward_atomic_outcome.yaml
#   configs/reward_atomic_outcome_process_v1.yaml
#
# Example:
#   bash scripts/run_continue_codescout4b_2xa800.sh \
#     -e configs/reward_atomic_outcome_process_v1.yaml \
#     -r codescout4b-continue-outcome-process-smoke \
#     -s /root/autodl-tmp/codescout_runs/codescout4b-outcome-process-smoke

set -euo pipefail
export WANDB_API_KEY="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"
while getopts ":m:n:d:s:l:o:i:t:b:c:r:w:e:x:u:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    n ) N_ROLLOUTS=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    s ) CKPT_PATH=$OPTARG;;
    l ) LCAL_PATH=$OPTARG;;
    o ) OTHER_OPTION=$OPTARG;;
    i ) NUM_INFERENCE_ENGINES=$OPTARG;;
    t ) NUM_TRAINING_ENGINES=$OPTARG;;
    b ) BATCH_SIZE=$OPTARG;;
    c ) MICRO_BATCH_SIZE=$OPTARG;;
    r ) RUN_NAME=$OPTARG;;
    w ) STEP_WISE=$OPTARG;;
    e ) EXP_CONFIG=$OPTARG;;
    x ) MAX_STEPS=$OPTARG;;
    u ) EXCLUDE_IDS_FILE=$OPTARG;;
    * )
      echo "Usage: $0 [-m model] [-e exp_config] [-x max_steps] [-u exclude_ids_file] [-n rollouts] [-b batch] [-c micro_batch] [-d data_path] [-s output_ckpt_path] [-l resume_ckpt_path] [-i inference_gpus] [-t training_gpus] [-r run_name] [-w step_wise] [-o hydra_option]" >&2
      exit 2
      ;;
  esac
done

MODEL="${MODEL:-OpenHands/CodeScout-4B}"
EXP_CONFIG="${EXP_CONFIG:-configs/reward_atomic_outcome_process_v1.yaml}"
MODEL_ALIAS=$(echo "$MODEL" | sed 's#[/:]#-#g')
EXP_ALIAS=$(basename "$EXP_CONFIG" .yaml)

NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -lt 2 ]; then
  echo "Expected at least 2 GPUs for this script, found ${NUM_GPUS}." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [ -d /root/autodl-tmp ]; then
  DEFAULT_STORAGE_ROOT="/root/autodl-tmp"
elif [ -d /data ]; then
  DEFAULT_STORAGE_ROOT="/data"
else
  DEFAULT_STORAGE_ROOT="$HOME"
fi
DEFAULT_RUN_ROOT="${DEFAULT_STORAGE_ROOT}/codescout_runs"
DEFAULT_CACHE_ROOT="${DEFAULT_STORAGE_ROOT}/codescout_cache"
DEFAULT_TMP_ROOT="${DEFAULT_STORAGE_ROOT}/codescout_tmp"
DEFAULT_WANDB_DIR="${DEFAULT_STORAGE_ROOT}/codescout_wandb"

DATA_PATH="${DATA_PATH:-./data/swe_smith}"
CKPT_PATH="${CKPT_PATH:-${DEFAULT_RUN_ROOT}/${MODEL_ALIAS}-${EXP_ALIAS}/}"
LCAL_PATH="${LCAL_PATH:-$CKPT_PATH}"
CKPT_DIR="${CKPT_PATH%/}/"
mkdir -p "$CKPT_DIR" "$DEFAULT_CACHE_ROOT" "$DEFAULT_TMP_ROOT" "$DEFAULT_WANDB_DIR"

N_ROLLOUTS="${N_ROLLOUTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-40960}"
MAX_TURNS="${MAX_TURNS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-1}"
NUM_TRAINING_ENGINES="${NUM_TRAINING_ENGINES:-1}"
INFERENCE_TENSOR_PARALLEL_SIZE="${INFERENCE_TENSOR_PARALLEL_SIZE:-1}"
STEP_WISE="${STEP_WISE:-false}"
COLOCATE_ALL="${COLOCATE_ALL:-false}"
RUN_ASYNC_TRAINER="${RUN_ASYNC_TRAINER:-true}"
LOGGER="${LOGGER:-wandb}"
TRAINER_SEED="${TRAINER_SEED:-42}"
WEIGHT_SYNC_BACKEND="${WEIGHT_SYNC_BACKEND:-nccl}"
DUMP_DATA_BATCH="${DUMP_DATA_BATCH:-false}"
CKPT_INTERVAL="${CKPT_INTERVAL:-20}"
HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-50}"
MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-2}"
EVAL_INTERVAL="${EVAL_INTERVAL:--1}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-false}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-20}"
EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
RUN_NAME="${RUN_NAME:-${MODEL_ALIAS}-${EXP_ALIAS}-${BATCH_SIZE}x${N_ROLLOUTS}}"
OTHER_OPTION="${OTHER_OPTION:-}"
EXCLUDE_IDS_FILE="${EXCLUDE_IDS_FILE:-}"

TRAIN_DATA_PATH="${DATA_PATH}"
if [ "${MAX_STEPS}" -gt 0 ]; then
  SUBSET_DATA_PATH="${CKPT_DIR}run_data"
  mkdir -p "${SUBSET_DATA_PATH}"
  MAX_TRAIN_ISSUES=$((MAX_STEPS * BATCH_SIZE))
  SOURCE_DATA_PATH="${DATA_PATH}" SUBSET_DATA_PATH="${SUBSET_DATA_PATH}" MAX_TRAIN_ISSUES="${MAX_TRAIN_ISSUES}" EXCLUDE_IDS_FILE="${EXCLUDE_IDS_FILE}" python - <<'PY'
import os
from pathlib import Path

import pandas as pd

source = Path(os.environ["SOURCE_DATA_PATH"])
target = Path(os.environ["SUBSET_DATA_PATH"])
max_train_issues = int(os.environ["MAX_TRAIN_ISSUES"])
exclude_ids_file = os.environ.get("EXCLUDE_IDS_FILE", "")

train = pd.read_parquet(source / "train.parquet")
source_count = len(train)
if exclude_ids_file:
    exclude_path = Path(exclude_ids_file)
    exclude_ids = {
        line.strip()
        for line in exclude_path.read_text().splitlines()
        if line.strip()
    }
    train = train[~train["instance_id"].isin(exclude_ids)].reset_index(drop=True)
    print(
        f"Excluded {source_count - len(train)} train issues using "
        f"{exclude_path} ({len(exclude_ids)} ids)."
    )

if len(train) < max_train_issues:
    raise ValueError(
        f"Not enough train issues for requested MAX_STEPS: "
        f"need {max_train_issues}, found {len(train)}"
    )

train.head(max_train_issues).to_parquet(target / "train.parquet", index=False)
validation_path = source / "validation.parquet"
if validation_path.exists():
    pd.read_parquet(validation_path).to_parquet(target / "validation.parquet", index=False)
print(f"Prepared fixed train subset: {max_train_issues} issues -> {target / 'train.parquet'}")
PY
  TRAIN_DATA_PATH="${SUBSET_DATA_PATH}"
fi

export VLLM_FLASH_ATTN_VERSION="${VLLM_FLASH_ATTN_VERSION:-2}"
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-600}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export SKYRL_DISABLE_NUMA_AFFINITY="${SKYRL_DISABLE_NUMA_AFFINITY:-1}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-${DEFAULT_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}/vllm}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${DEFAULT_CACHE_ROOT}/uv}"
export RAY_TMPDIR="${RAY_TMPDIR:-${DEFAULT_TMP_ROOT}/ray}"
export TMPDIR="${TMPDIR:-${DEFAULT_TMP_ROOT}/tmp}"
export WANDB_DIR="${WANDB_DIR:-${DEFAULT_WANDB_DIR}}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$UV_CACHE_DIR" "$RAY_TMPDIR" "$TMPDIR" "$WANDB_DIR"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,0.0.0.0"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

set -x

uv run --isolated --active -m src.train \
  +run_async_trainer=${RUN_ASYNC_TRAINER} \
  +generator.exp_config=${EXP_CONFIG} \
  data.train_data="['$TRAIN_DATA_PATH/train.parquet']" \
  data.val_data="['$TRAIN_DATA_PATH/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.seed=${TRAINER_SEED} \
  trainer.policy.model.path=${MODEL} \
  trainer.placement.colocate_all=${COLOCATE_ALL} \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  trainer.policy.fsdp_config.fsdp_size=-1 \
  trainer.fully_async.num_parallel_generation_workers=${BATCH_SIZE} \
  trainer.placement.policy_num_gpus_per_node=${NUM_TRAINING_ENGINES} \
  trainer.placement.ref_num_gpus_per_node=${NUM_TRAINING_ENGINES} \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.policy.sequence_parallel_size=1 \
  generator.num_inference_engines=${NUM_INFERENCE_ENGINES} \
  generator.inference_engine_tensor_parallel_size=${INFERENCE_TENSOR_PARALLEL_SIZE} \
  +generator.traj_dir=${CKPT_DIR}trajectories/ \
  +generator.engine_init_kwargs.enable_auto_tool_choice=true \
  +generator.engine_init_kwargs.tool_call_parser="hermes" \
  +generator.engine_init_kwargs.max_model_len=${MAX_MODEL_LEN} \
  +generator.prompts.system_prompt="templates/system_prompt_custom_finish.j2" \
  +generator.prompts.user_prompt="templates/file_module_custom_finish.j2" \
  +generator.engine_init_kwargs.disable_cascade_attn=true \
  trainer.epochs=1 \
  generator.eval_n_samples_per_prompt=${EVAL_N_SAMPLES_PER_PROMPT} \
  trainer.eval_batch_size=${EVAL_BATCH_SIZE} \
  trainer.eval_before_train=${EVAL_BEFORE_TRAIN} \
  trainer.eval_interval=${EVAL_INTERVAL} \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${BATCH_SIZE} \
  trainer.policy_mini_batch_size=${BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  trainer.dump_data_batch=${DUMP_DATA_BATCH} \
  trainer.export_path="${CKPT_DIR}exported_model/" \
  trainer.hf_save_interval=${HF_SAVE_INTERVAL} \
  trainer.ckpt_interval=${CKPT_INTERVAL} \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=${MAX_PROMPT_LENGTH} \
  trainer.algorithm.policy_loss_type="gspo" \
  trainer.algorithm.eps_clip_low=0.0003 \
  trainer.algorithm.eps_clip_high=0.0004 \
  trainer.algorithm.loss_reduction="sequence_mean" \
  generator.sampling_params.max_generate_length=${MAX_LENGTH} \
  generator.sampling_params.temperature=1.0 \
  generator.max_input_length=${MAX_PROMPT_LENGTH} \
  generator.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
  generator.max_turns=${MAX_TURNS} \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=False \
  trainer.algorithm.use_kl_in_reward=False \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host='127.0.0.1' \
  generator.http_endpoint_port=8080 \
  generator.weight_sync_backend=${WEIGHT_SYNC_BACKEND} \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=${N_ROLLOUTS} \
  generator.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
  generator.enforce_eager=false \
  trainer.step_wise_training=${STEP_WISE} \
  trainer.logger="${LOGGER}" \
  trainer.project_name="code_search" \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$LCAL_PATH" \
  trainer.max_ckpts_to_keep=${MAX_CKPTS_TO_KEEP} \
  $OTHER_OPTION
