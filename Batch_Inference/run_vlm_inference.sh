#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for VLM batch inference (API models and local checkpoints).
# See README.md for usage examples.
#
# Required: set BACKEND (or pass as the first argument):
#   chatgpt | claude | gemini | internvl | qwen3 | craft
#
# Optional: set ENV_TYPE (default depends on BACKEND):
#   phyre | minigrid | craft
#
# Extra CLI args are forwarded to the Python agent unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BACKEND="${BACKEND:-${1:-}}"
if [[ -z "${BACKEND}" ]]; then
  echo "Error: BACKEND is required." >&2
  echo "Usage: BACKEND=<chatgpt|claude|gemini|internvl|qwen3|craft> $0 [extra agent args...]" >&2
  echo "   or: $0 <backend> [extra agent args...]" >&2
  exit 1
fi

if [[ "${BACKEND}" == "${1:-}" ]]; then
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_PATH="${DATASET_PATH:-}"

case "${BACKEND}" in
  chatgpt|claude|gemini|internvl|qwen3|craft) ;;
  *)
    echo "Error: unknown BACKEND '${BACKEND}'." >&2
    echo "Supported: chatgpt, claude, gemini, internvl, qwen3, craft" >&2
    exit 1
    ;;
esac

if [[ "${BACKEND}" == "craft" ]]; then
  ENV_TYPE="craft"
else
  ENV_TYPE="${ENV_TYPE:-phyre}"
fi

# ── shared defaults ───────────────────────────────────────────────────────────
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
BATCH_SIZE="${BATCH_SIZE:-8}"
REPEAT_NUM="${REPEAT_NUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/explorer_outputs}"

# ── backend-specific defaults ─────────────────────────────────────────────────
case "${BACKEND}" in
  chatgpt)
    AGENT_SCRIPT="${SCRIPT_DIR}/agent/chatgpt_agent.py"
    MODEL_NAME="${MODEL_NAME:-gpt-4o}"
    LOG_MODEL_NAME="${LOG_MODEL_NAME:-gpt-4o}"
    LOG_DIR_BASE="${LOG_DIR_BASE:-${SCRIPT_DIR}/batch_inference_output/chatgpt/${ENV_TYPE}}"
    LOG_DIR_LABEL="${LOG_DIR_LABEL:-tmp_log/chatgpt/${ENV_TYPE}/}"
    MAX_RETRIES="${MAX_RETRIES:-3}"
    RETRY_DELAY="${RETRY_DELAY:-2.0}"
    ;;
  claude)
    AGENT_SCRIPT="${SCRIPT_DIR}/agent/claude_agent.py"
    MODEL_NAME="${MODEL_NAME:-claude-3-5-sonnet-20241022}"
    LOG_MODEL_NAME="${LOG_MODEL_NAME:-claude-3-5-sonnet}"
    LOG_DIR_BASE="${LOG_DIR_BASE:-${SCRIPT_DIR}/batch_inference_output/claude/${ENV_TYPE}}"
    LOG_DIR_LABEL="${LOG_DIR_LABEL:-tmp_log/claude/${ENV_TYPE}/}"
    MAX_RETRIES="${MAX_RETRIES:-3}"
    RETRY_DELAY="${RETRY_DELAY:-2.0}"
    ;;
  gemini)
    AGENT_SCRIPT="${SCRIPT_DIR}/agent/gemini_agent.py"
    MODEL_NAME="${MODEL_NAME:-gemini-2.0-flash}"
    LOG_MODEL_NAME="${LOG_MODEL_NAME:-gemini-2.0-flash}"
    LOG_DIR_BASE="${LOG_DIR_BASE:-${SCRIPT_DIR}/batch_inference_output/gemini/${ENV_TYPE}}"
    LOG_DIR_LABEL="${LOG_DIR_LABEL:-tmp_log/gemini/${ENV_TYPE}/}"
    MAX_RETRIES="${MAX_RETRIES:-3}"
    RETRY_DELAY="${RETRY_DELAY:-2.0}"
    ;;
  internvl)
    AGENT_SCRIPT="${SCRIPT_DIR}/agent/internvl_agent.py"
    MODEL_NAME="${MODEL_NAME:-OpenGVLab/InternVL3_5-8B}"
    LOCAL_LOAD_PATH="${LOCAL_LOAD_PATH:-}"
    LOG_MODEL_NAME="${LOG_MODEL_NAME:-InternVL3.5-8B}"
    LOG_DIR_BASE="${LOG_DIR_BASE:-${SCRIPT_DIR}/batch_inference_output/internvl/${ENV_TYPE}}"
    LOG_DIR_LABEL="${LOG_DIR_LABEL:-tmp_log/internvl/${ENV_TYPE}/}"
    BATCH_SIZE="${BATCH_SIZE:-80}"
    REPEAT_NUM="${REPEAT_NUM:-5}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
    MAX_NUM_TILES="${MAX_NUM_TILES:-12}"
    INPUT_SIZE="${INPUT_SIZE:-448}"
    ;;
  qwen3)
    AGENT_SCRIPT="${SCRIPT_DIR}/agent/qwen3_agent.py"
    MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
    LOCAL_LOAD_PATH="${LOCAL_LOAD_PATH:-}"
    LOG_MODEL_NAME="${LOG_MODEL_NAME:-Qwen3-VL-8B-Instruct}"
    LOG_DIR_BASE="${LOG_DIR_BASE:-${SCRIPT_DIR}/batch_inference_output/qwen3/phyre}"
    LOG_DIR_LABEL="${LOG_DIR_LABEL:-tmp_log/qwen3/phyre/}"
    BATCH_SIZE="${BATCH_SIZE:-160}"
    REPEAT_NUM="${REPEAT_NUM:-5}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
    TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
    USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-1}"
    ;;
  craft)
    AGENT_SCRIPT="${SCRIPT_DIR}/agent/craft_qwen3_agent.py"
    MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
    LOCAL_LOAD_PATH="${LOCAL_LOAD_PATH:-}"
    LOG_MODEL_NAME="${LOG_MODEL_NAME:-}"
    LOG_DIR_BASE="${LOG_DIR_BASE:-${SCRIPT_DIR}/batch_inference_output/craft}"
    BATCH_SIZE="${BATCH_SIZE:-160}"
    NUM_ITEMS="${NUM_ITEMS:-}"
    TEMPERATURE="${TEMPERATURE:-0.2}"
    TOP_P="${TOP_P:-0.95}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
    TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
    USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-0}"
    ;;
esac

if [[ -z "${DATASET_PATH}" ]]; then
  echo "Error: DATASET_PATH is required (no hardcoded default)." >&2
  echo "Please point DATASET_PATH to a JSON file under Hugging Face dataset 'vaora-proj/vaora-dataset' (test_data directory)." >&2
  exit 1
fi

# ── PHYRE / MiniGrid shared settings (API + local agents except craft) ────────
if [[ "${BACKEND}" != "craft" ]]; then
  EVAL_SETUPS="${EVAL_SETUPS:-ball_within_template}"
  FOLD_ID="${FOLD_ID:-0}"
  EVAL_TYPE="${EVAL_TYPE:-all}"
fi

# ── build command ─────────────────────────────────────────────────────────────
if [[ "${BACKEND}" == "craft" ]]; then
  CMD=(
    ${PYTHON_BIN}
    "${AGENT_SCRIPT}"
    --dataset_path "${DATASET_PATH}"
    --model_name "${MODEL_NAME}"
    --log_dir_base "${LOG_DIR_BASE}"
    --batch_size "${BATCH_SIZE}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --system_prompt "${SYSTEM_PROMPT}"
    --torch_dtype "${TORCH_DTYPE}"
  )

  if [[ -n "${LOCAL_LOAD_PATH}" ]]; then
    CMD+=(--local_load_path "${LOCAL_LOAD_PATH}")
  fi

  if [[ -n "${LOG_MODEL_NAME}" ]]; then
    CMD+=(--log_model_name "${LOG_MODEL_NAME}")
  fi

  if [[ -n "${NUM_ITEMS}" ]]; then
    CMD+=(--num_items "${NUM_ITEMS}")
  fi

  if [[ "${USE_FLASH_ATTENTION}" == "1" || "${USE_FLASH_ATTENTION}" == "true" || "${USE_FLASH_ATTENTION}" == "yes" ]]; then
    CMD+=(--use_flash_attention)
  fi
elif [[ "${BACKEND}" == "qwen3" ]]; then
  CMD=(
    ${PYTHON_BIN}
    "${AGENT_SCRIPT}"
    --eval_setups "${EVAL_SETUPS}"
    --fold_id "${FOLD_ID}"
    --eval_type "${EVAL_TYPE}"
    --model_name "${MODEL_NAME}"
    --log_dir_base "${LOG_DIR_BASE}"
    --log_dir_label "${LOG_DIR_LABEL}"
    --log_model_name "${LOG_MODEL_NAME}"
    --output_root "${OUTPUT_ROOT}"
    --batch_size "${BATCH_SIZE}"
    --dataset_path "${DATASET_PATH}"
    --repeat_num "${REPEAT_NUM}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --system_prompt "${SYSTEM_PROMPT}"
    --num_workers "${NUM_WORKERS}"
    --torch_dtype "${TORCH_DTYPE}"
  )

  if [[ -n "${LOCAL_LOAD_PATH}" ]]; then
    CMD+=(--local_load_path "${LOCAL_LOAD_PATH}")
  fi

  if [[ "${USE_FLASH_ATTENTION}" == "1" || "${USE_FLASH_ATTENTION}" == "true" || "${USE_FLASH_ATTENTION}" == "yes" ]]; then
    CMD+=(--use_flash_attention)
  fi
elif [[ "${BACKEND}" == "internvl" ]]; then
  CMD=(
    ${PYTHON_BIN}
    "${AGENT_SCRIPT}"
    --env_type "${ENV_TYPE}"
    --eval_setups "${EVAL_SETUPS}"
    --fold_id "${FOLD_ID}"
    --eval_type "${EVAL_TYPE}"
    --model_name "${MODEL_NAME}"
    --log_dir_base "${LOG_DIR_BASE}"
    --log_dir_label "${LOG_DIR_LABEL}"
    --log_model_name "${LOG_MODEL_NAME}"
    --output_root "${OUTPUT_ROOT}"
    --batch_size "${BATCH_SIZE}"
    --dataset_path "${DATASET_PATH}"
    --repeat_num "${REPEAT_NUM}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --system_prompt "${SYSTEM_PROMPT}"
    --num_workers "${NUM_WORKERS}"
    --max_num_tiles "${MAX_NUM_TILES}"
    --input_size "${INPUT_SIZE}"
    --use_flash_attention
  )

  if [[ -n "${LOCAL_LOAD_PATH}" ]]; then
    CMD+=(--local_load_path "${LOCAL_LOAD_PATH}")
  fi
else
  # chatgpt, claude, gemini
  CMD=(
    ${PYTHON_BIN}
    "${AGENT_SCRIPT}"
    --env_type "${ENV_TYPE}"
    --eval_setups "${EVAL_SETUPS}"
    --fold_id "${FOLD_ID}"
    --eval_type "${EVAL_TYPE}"
    --model_name "${MODEL_NAME}"
    --log_dir_base "${LOG_DIR_BASE}"
    --log_dir_label "${LOG_DIR_LABEL}"
    --log_model_name "${LOG_MODEL_NAME}"
    --output_root "${OUTPUT_ROOT}"
    --batch_size "${BATCH_SIZE}"
    --dataset_path "${DATASET_PATH}"
    --repeat_num "${REPEAT_NUM}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --system_prompt "${SYSTEM_PROMPT}"
    --num_workers "${NUM_WORKERS}"
    --max_retries "${MAX_RETRIES}"
    --retry_delay "${RETRY_DELAY}"
  )
fi

CMD+=("$@")

echo "BACKEND=${BACKEND} ENV_TYPE=${ENV_TYPE}"
echo "Running: ${CMD[*]}"
cd "${REPO_ROOT}"
exec "${CMD[@]}"
