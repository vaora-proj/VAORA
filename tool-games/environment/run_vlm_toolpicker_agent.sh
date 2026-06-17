#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

# Simple launcher for the tool-games VLM ToolPicker agent.
# Extra CLI args are forwarded to the Python runner.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=artifacts_paths.sh
source "${SCRIPT_DIR}/artifacts_paths.sh"

INPUT_PATH="${INPUT_PATH:-${SCRIPT_DIR}/Trials/Original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARTIFACTS_DIR}/vlm_toolpicker}"
BACKEND="${BACKEND:-qwen}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
LOCAL_LOAD_PATH="${LOCAL_LOAD_PATH:-}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
GEMINI_API_KEY_VALUE="${GEMINI_API_KEY_VALUE:-${GEMINI_API_KEY:-}}"
API_KEY_ENV="${API_KEY_ENV:-GEMINI_API_KEY}"
API_MAX_RETRIES="${API_MAX_RETRIES:-3}"
API_RETRY_DELAY="${API_RETRY_DELAY:-1.5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
TOOLS_PER_TASK="${TOOLS_PER_TASK:-3}"
MAX_TIME="${MAX_TIME:-20.0}"
BASIC_TIMESTEP="${BASIC_TIMESTEP:-0.1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
DO_SAMPLE="${DO_SAMPLE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/agent/vlm_toolpicker_agent.py"
  --backend "${BACKEND}"
  --input_path "${INPUT_PATH}"
  --output_root "${OUTPUT_ROOT}"
  --model_name "${MODEL_NAME}"
  --system_prompt "${SYSTEM_PROMPT}"
  --torch_dtype "${TORCH_DTYPE}"
  --api_key_env "${API_KEY_ENV}"
  --api_max_retries "${API_MAX_RETRIES}"
  --api_retry_delay "${API_RETRY_DELAY}"
  --max_attempts "${MAX_ATTEMPTS}"
  --tools_per_task "${TOOLS_PER_TASK}"
  --max_time "${MAX_TIME}"
  --basic_timestep "${BASIC_TIMESTEP}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
)

if [[ -n "${LOCAL_LOAD_PATH}" ]]; then
  CMD+=(--local_load_path "${LOCAL_LOAD_PATH}")
fi

if [[ -n "${GEMINI_API_KEY_VALUE}" ]]; then
  CMD+=(--api_key "${GEMINI_API_KEY_VALUE}")
fi

if [[ "${DO_SAMPLE}" == "1" ]]; then
  CMD+=(--do_sample)
fi

CMD+=("$@")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
