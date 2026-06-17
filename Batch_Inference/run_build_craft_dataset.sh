#!/usr/bin/env bash
set -euo pipefail
# Launcher for building the CRAFT VQA dataset.
# Assumes the correct Python env is already active.
#
# Environment variables:
#
#   DATASET_JSON   Path to the CRAFT source QA JSON
#                  required
#
#   FRAMES_DIR     Root directory containing per-split frame PNGs
#                  required
#
#   OUTPUT_PATH    Destination path for the built dataset JSON
#                  required
#
#   NUM_ITEMS      Cap on the number of entries to include (empty = no cap)
#
#   INCLUDE_MISSING  Set to 1/true/yes to keep entries whose frame is missing
#                    default: 0 (missing frames are skipped)
#
#   PYTHON_BIN     Python interpreter to use (default: python)
#
# Examples
# --------
# Custom QA JSON and frame directory:
#   DATASET_JSON=/path/to/my_questions.json \
#   FRAMES_DIR=/path/to/my_frames \
#   OUTPUT_PATH=/path/to/output.json \
#   bash run_build_craft_dataset.sh
#
# Cap at 500 entries:
#   NUM_ITEMS=500 bash run_build_craft_dataset.sh
#
# Include entries even when frame PNGs are absent:
#   INCLUDE_MISSING=1 bash run_build_craft_dataset.sh
#
# Extra CLI args are forwarded to the Python script unchanged:
#   bash run_build_craft_dataset.sh --num_items 100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_JSON="${DATASET_JSON:-}"
FRAMES_DIR="${FRAMES_DIR:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
NUM_ITEMS="${NUM_ITEMS:-}"
INCLUDE_MISSING="${INCLUDE_MISSING:-0}"

if [[ -z "${DATASET_JSON}" || -z "${FRAMES_DIR}" || -z "${OUTPUT_PATH}" ]]; then
  echo "Error: DATASET_JSON, FRAMES_DIR, and OUTPUT_PATH are required (no hardcoded dataset paths)." >&2
  echo "Use paths from your copy of Hugging Face dataset 'vaora-proj/vaora-dataset'." >&2
  exit 1
fi

CMD=(
  ${PYTHON_BIN}
  "${SCRIPT_DIR}/build_dataset/build_qwen3_craft_dataset.py"
  --dataset_json "${DATASET_JSON}"
  --frames_dir   "${FRAMES_DIR}"
  --output_path  "${OUTPUT_PATH}"
)

if [[ -n "${NUM_ITEMS}" ]]; then
  CMD+=(--num_items "${NUM_ITEMS}")
fi

if [[ "${INCLUDE_MISSING}" == "1" || "${INCLUDE_MISSING}" == "true" || "${INCLUDE_MISSING}" == "yes" ]]; then
  CMD+=(--include_missing)
fi

# Forward any extra CLI args to the Python script
CMD+=("$@")

echo "Running: ${CMD[*]}"
cd "${REPO_ROOT}"
exec "${CMD[@]}"
