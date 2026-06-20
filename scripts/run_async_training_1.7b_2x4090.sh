#!/bin/bash

# 2-GPU launch script for continuing CodeScout-1.7B-RFT training.
# Default placement: 1 GPU for vLLM rollout generation, 1 GPU for policy/ref training.
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
  esac
done

MODEL="${MODEL:-OpenHands/CodeScout-1.7B-RFT}"
MODEL_ALIAS=$(echo "$MODEL" | sed 's#[/:]#-#g')

NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -lt 2 ]; then
  echo "Expected at least 2 GPUs, found ${NUM_GPUS}." >&2
fi

# Start small. Increase N_ROLLOUTS to 4 after the run is stable.
N_ROLLOUTS="${N_ROLLOUTS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${MAX_CONTEXT:-32768}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-24576}"
MAX_TURNS="${MAX_TURNS:-5}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
RUN_ASYNC_TRAINER="${RUN_ASYNC_TRAINER:-false}"
COLOCATE_ALL="${COLOCATE_ALL:-true}"

RUN_NAME="${RUN_NAME:-${MODEL_ALIAS}-2x4090-${BATCH_SIZE}x${N_ROLLOUTS}}"
set -x

DATA_PATH="${DATA_PATH:-data/qwen3_1.7b_data}"

# Keep checkpoints and trajectories outside the repository. Ray may copy the
# repository working tree during training, and large artifacts slow that down.
CKPT_PATH="${CKPT_PATH:-$HOME/codescout_runs/${MODEL_ALIAS}/}"
LCAL_PATH="${LCAL_PATH:-$CKPT_PATH}"
mkdir -p "$CKPT_PATH"
CKPT_DIR="${CKPT_PATH%/}/"

NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-1}"
NUM_TRAINING_ENGINES="${NUM_TRAINING_ENGINES:-1}"
STEP_WISE="${STEP_WISE:-false}"

export VLLM_FLASH_ATTN_VERSION=2
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
export RAY_worker_register_timeout_seconds=600
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export HYDRA_FULL_ERROR=1
export SKYRL_DISABLE_NUMA_AFFINITY=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,0.0.0.0"

uv run --isolated -m src.train \
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
  generator.inference_engine_tensor_parallel_size=1 \
  +generator.traj_dir=${CKPT_DIR}trajectories/ \
  +generator.engine_init_kwargs.enable_auto_tool_choice=true \
  +generator.engine_init_kwargs.tool_call_parser="hermes" \
  +generator.engine_init_kwargs.max_model_len=${MAX_MODEL_LEN} \
  +generator.prompts.system_prompt="templates/system_prompt_custom_finish.j2" \
  +generator.prompts.user_prompt="templates/file_module_custom_finish.j2" \
  +generator.engine_init_kwargs.disable_cascade_attn=true \
  trainer.epochs=1 \
  trainer.eval_batch_size=20 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${BATCH_SIZE} \
  trainer.policy_mini_batch_size=${BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  trainer.dump_data_batch=true \
  trainer.export_path="${CKPT_DIR}exported_model/" \
  trainer.hf_save_interval=100 \
  trainer.ckpt_interval=25 \
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
  trainer.policy.optimizer_config.lr=5.0e-7 \
  trainer.algorithm.use_kl_loss=False \
  trainer.algorithm.use_kl_in_reward=False \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host='127.0.0.1' \
  generator.http_endpoint_port=8080 \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=${N_ROLLOUTS} \
  generator.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
  generator.enforce_eager=true \
  trainer.step_wise_training=${STEP_WISE} \
  trainer.logger="wandb" \
  trainer.project_name="code_search" \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$LCAL_PATH" \
  trainer.max_ckpts_to_keep=1 \
  $OTHER_OPTION
