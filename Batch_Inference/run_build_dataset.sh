#!/usr/bin/env bash
set -euo pipefail

# Launcher for building PHYRE/MiniGrid datasets.
# Assumes the correct Python env is already active; extra CLI args are forwarded.
# No hardcoded output path: OUTPUT_PATH must be provided.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_TYPE="${ENV_TYPE:-phyre}"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${ENV_TYPE}" == "minigrid" ]]; then
  ENV_ID="${ENV_ID:-MiniGrid-DoorKey-5x5-v0}"
  SPLIT="${SPLIT:-test}"
  NUM_SAMPLES="${NUM_SAMPLES:-64}"
  SEED_START="${SEED_START:-10000}"
  REPEAT_NUM="${REPEAT_NUM:-1}"
  MAX_STEPS="${MAX_STEPS:-50}"
  TILE_SIZE="${TILE_SIZE:-32}"
  OUTPUT_PATH="${OUTPUT_PATH:-}"
  IMAGE_DIR="${IMAGE_DIR:-}"
  FOLD_ID="${FOLD_ID:-}"
  AGENT_POV="${AGENT_POV:-0}"

  CMD=(
    ${PYTHON_BIN}
    "${SCRIPT_DIR}/build_dataset/build_qwen3_minigrid_dataset.py"
    --env_id "${ENV_ID}"
    --split "${SPLIT}"
    --num_samples "${NUM_SAMPLES}"
    --seed_start "${SEED_START}"
    --repeat_num "${REPEAT_NUM}"
    --max_steps "${MAX_STEPS}"
    --tile_size "${TILE_SIZE}"
    --output_path "${OUTPUT_PATH}"
  )

  if [[ -n "${IMAGE_DIR}" ]]; then
    CMD+=(--image_dir "${IMAGE_DIR}")
  fi

  if [[ -n "${FOLD_ID}" ]]; then
    CMD+=(--fold_id "${FOLD_ID}")
  fi

  if [[ "${AGENT_POV}" == "1" || "${AGENT_POV}" == "true" || "${AGENT_POV}" == "yes" ]]; then
    CMD+=(--agent_pov)
  fi
else
  EVAL_SETUPS="${EVAL_SETUPS:-my_template_based_split}"
  FOLD_ID="${FOLD_ID:-4}"
  EVAL_TYPE="${EVAL_TYPE:-test}"
  NUM_TASKS="${NUM_TASKS:-}"
  REPEAT_NUM="${REPEAT_NUM:-1}"
  PROMPT_PATH="${PROMPT_PATH:-}"
  HINT="${HINT:-0}"
  HINTS_JSON="${HINTS_JSON:-}"
  HINT_SEED="${HINT_SEED:-42}"
  OUTPUT_PATH="${OUTPUT_PATH:-}"

  CMD=(
    ${PYTHON_BIN}
    "${SCRIPT_DIR}/build_dataset/build_qwen3_phyre_dataset.py"
    --eval_setups "${EVAL_SETUPS}"
    --fold_id "${FOLD_ID}"
    --eval_type "${EVAL_TYPE}"
    --output_path "${OUTPUT_PATH}"
  )

  if [[ -n "${NUM_TASKS}" ]]; then
    CMD+=(--num_tasks "${NUM_TASKS}")
  fi

  if [[ -n "${REPEAT_NUM}" ]]; then
    CMD+=(--repeat_num "${REPEAT_NUM}")
  fi

  if [[ -n "${PROMPT_PATH}" ]]; then
    CMD+=(--prompt_path "${PROMPT_PATH}")
  fi

  if [[ "${HINT}" == "1" || "${HINT}" == "true" || "${HINT}" == "yes" ]]; then
    CMD+=(--hint)
  fi

  if [[ -n "${HINTS_JSON}" ]]; then
    CMD+=(--hints_json "${HINTS_JSON}")
  fi

  if [[ -n "${HINT_SEED}" ]]; then
    CMD+=(--hint_seed "${HINT_SEED}")
  fi
fi

if [[ -z "${OUTPUT_PATH}" ]]; then
  echo "Error: OUTPUT_PATH is required (no hardcoded dataset path)." >&2
  echo "Set OUTPUT_PATH to a target JSON path under your dataset repo copy (e.g., vaora-proj/vaora-dataset/test_data)." >&2
  exit 1
fi

CMD+=("$@")

echo "Running: ${CMD[*]}"
cd "${REPO_ROOT}"
exec "${CMD[@]}"
