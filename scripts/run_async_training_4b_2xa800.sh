#!/bin/bash

# 2x A800-80GB-NVLink launch script for reproducing the CodeScout-4B RL run.
#
# The upstream 4B script assumes an 8-GPU node with 4 rollout engines and
# 4 policy/ref training GPUs. This script keeps the CodeScout-4B algorithmic
# settings, but defaults to a 2-GPU layout that fits one A800 for vLLM rollout
# generation and one A800 for policy/ref training.
#
# Expected machine:
#   GPU: 2x A800 80GB with NVLink
#   CPU: 32 vCPU
#   RAM: 240GB
#   Disk: AutoDL system disk + /root/autodl-tmp data SSD
#
# Recommended invocation:
#   export WANDB_API_KEY=<your_key>
#   bash scripts/run_async_training_4b_2xa800.sh \
#     -m Qwen/Qwen3-4B-Instruct-2507 \
#     -d ./data/swe_smith \
#     -s /root/autodl-tmp/codescout_runs/qwen3-4b-a800 \
#     -r qwen3-4b-2xa800-gspo \
#     -o "+generator.reward=configs/reward_config_4b.yaml"

set -euo pipefail
export WANDB_API_KEY="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"
while getopts ":m:n:d:s:l:o:i:t:b:c:r:w:" opt; do
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
    * )
      echo "Usage: $0 [-m model] [-n rollouts] [-b batch] [-c micro_batch] [-d data_path] [-s ckpt_path] [-l resume_ckpt_path] [-i inference_gpus] [-t training_gpus] [-r run_name] [-w step_wise] [-o hydra_option]" >&2
      exit 2
      ;;
  esac
done

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_ALIAS=$(echo "$MODEL" | sed 's#[/:]#-#g')

NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -lt 2 ]; then
  echo "Expected at least 2 GPUs for this script, found ${NUM_GPUS}." >&2
  exit 1
fi
if [ "$NUM_GPUS" -gt 2 ]; then
  echo "Found ${NUM_GPUS} GPUs; this script will use the first 2 unless CUDA_VISIBLE_DEVICES is already set." >&2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Keep heavy artifacts out of the repository and system disk. On AutoDL,
# /root/autodl-tmp is the fast persistent data disk, but it is not saved into
# custom images.
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
CKPT_PATH="${CKPT_PATH:-${DEFAULT_RUN_ROOT}/${MODEL_ALIAS}-2xa800/}"
LCAL_PATH="${LCAL_PATH:-$CKPT_PATH}"
CKPT_DIR="${CKPT_PATH%/}/"
mkdir -p "$CKPT_DIR" "$DEFAULT_CACHE_ROOT" "$DEFAULT_TMP_ROOT" "$DEFAULT_WANDB_DIR"

FREE_KB=$(df -Pk "$CKPT_DIR" | awk 'NR==2 {print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))
if [ "$FREE_GB" -lt 80 ]; then
  echo "Warning: only ${FREE_GB}GB free at ${CKPT_DIR}. 4B checkpoints, model cache, trajectories, and Ray temp files can fill this disk quickly." >&2
fi

# Conservative 2-GPU defaults. Increase BATCH_SIZE/N_ROLLOUTS only after a
# smoke run succeeds and disk usage is understood.
N_ROLLOUTS="${N_ROLLOUTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-32768}"
MAX_TURNS="${MAX_TURNS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-1}"
NUM_TRAINING_ENGINES="${NUM_TRAINING_ENGINES:-1}"
INFERENCE_TENSOR_PARALLEL_SIZE="${INFERENCE_TENSOR_PARALLEL_SIZE:-1}"
STEP_WISE="${STEP_WISE:-false}"
COLOCATE_ALL="${COLOCATE_ALL:-false}"
RUN_ASYNC_TRAINER="${RUN_ASYNC_TRAINER:-true}"
LOGGER="${LOGGER:-wandb}"
WEIGHT_SYNC_BACKEND="${WEIGHT_SYNC_BACKEND:-nccl}"
DUMP_DATA_BATCH="${DUMP_DATA_BATCH:-false}"
CKPT_INTERVAL="${CKPT_INTERVAL:-50}"
HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-50}"
MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-2}"
EVAL_INTERVAL="${EVAL_INTERVAL:--1}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-false}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-20}"
EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"

RUN_NAME="${RUN_NAME:-${MODEL_ALIAS}-2xa800-${BATCH_SIZE}x${N_ROLLOUTS}}"
OTHER_OPTION="${OTHER_OPTION:-+generator.reward=configs/reward_config_4b.yaml}"

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

# Single-node A800 NVLink should allow NCCL P2P. Do not disable P2P by default.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

set -x

uv run --isolated --active -m src.train \
  +run_async_trainer=${RUN_ASYNC_TRAINER} \
  data.train_data="['$DATA_PATH/train.parquet']" \
  data.val_data="['$DATA_PATH/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.policy.model.path=${MODEL} \
  trainer.placement.colocate_all=${COLOCATE_ALL} \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
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
