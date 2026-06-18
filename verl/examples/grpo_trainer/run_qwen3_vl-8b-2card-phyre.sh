#!/usr/bin/env bash
set -euo pipefail
set -x

export ROCR_VISIBLE_DEVICES=""
export CUDA_LAUNCH_BLOCKING=0
export WANDB_MODE="${WANDB_MODE:-offline}"
export HYDRA_FULL_ERROR=1

ENGINE=${1:-vllm}
shift $(( $# > 0 ? 1 : 0 )) || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export REWARD_THREADS="${REWARD_THREADS:-4}"
export PHYRE_PORT="${PHYRE_PORT:-5001}"

# Resolve repository root from script location:
# examples/grpo_trainer/<this_file> -> repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

# --- Edit these paths before running ---
SFT_MODEL_PATH="/path/to/SFT_VLM/sft_within_template"
TRAIN_FILE="/path/to/vaora-dataset/train_data/VAORA_DATA/vaora_dataset_within_template/train.parquet"
VAL_FILE="/path/to/vaora-dataset/train_data/VAORA_DATA/vaora_dataset_within_template/test.parquet"

if [[ ! -e "${SFT_MODEL_PATH}" ]]; then
    echo "SFT checkpoint not found: ${SFT_MODEL_PATH}"
    echo "Edit SFT_MODEL_PATH in this script (see VAORA-VERL.md)."
    exit 1
fi

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
    echo "Training/validation parquet files not found."
    echo "TRAIN_FILE=${TRAIN_FILE}"
    echo "VAL_FILE=${VAL_FILE}"
    echo "Edit TRAIN_FILE and VAL_FILE in this script (see VAORA-VERL.md)."
    exit 1
fi

ulimit -n 65535

gunicorn --preload -w "${REWARD_THREADS}" --keep-alive 1 -b "0.0.0.0:${PHYRE_PORT}" --timeout 120 scripts.phyre_agent_server:app &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

sleep 35

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.gdpo=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=1280 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    +data.training_steps=300 \
    actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_llds_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_sft_gdpo_phyre' \
    trainer.experiment_name='qwen3_vl_8b_gdpo_within_template' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    "$@"
