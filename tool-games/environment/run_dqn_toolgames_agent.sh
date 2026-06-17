#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=artifacts_paths.sh
source "${SCRIPT_DIR}/artifacts_paths.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"

TOOL_GAMES_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_PATH="${INPUT_PATH:-${SCRIPT_DIR}/Trials/Original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARTIFACTS_DIR}/dqn_toolgames}"
ACTION_CACHE_PATH="${ACTION_CACHE_PATH:-${TOOL_GAMES_ROOT}/data/action_array_ball_seed42_100k.npy}"

if [[ -z "${DQN_LOAD_FROM:-}" ]]; then
  echo "Set DQN_LOAD_FROM to your PHYRE DQN checkpoint directory (must contain ckpt.* files)." >&2
  echo "Example: DQN_LOAD_FROM=/path/to/phyre/results/finals/dqn_10k/ball_within_template/0 bash $0" >&2
  exit 1
fi
DQN_RANK_SIZE="${DQN_RANK_SIZE:-10000}"
TOP_K="${TOP_K:-5}"
DQN_EVAL_BATCH_SIZE="${DQN_EVAL_BATCH_SIZE:-256}"
MAX_TIME="${MAX_TIME:-20.0}"
BASIC_TIMESTEP="${BASIC_TIMESTEP:-0.1}"
SAVE_ATTEMPT_VIDEOS="${SAVE_ATTEMPT_VIDEOS:-1}"
ATTEMPT_VIDEO_FPS="${ATTEMPT_VIDEO_FPS:-10}"

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/agent/dqn_toolgames_agent.py"
  --input_path "${INPUT_PATH}"
  --output_root "${OUTPUT_ROOT}"
  --dqn_load_from "${DQN_LOAD_FROM}"
  --action_cache_path "${ACTION_CACHE_PATH}"
  --dqn_rank_size "${DQN_RANK_SIZE}"
  --top_k "${TOP_K}"
  --dqn_eval_batch_size "${DQN_EVAL_BATCH_SIZE}"
  --max_time "${MAX_TIME}"
  --basic_timestep "${BASIC_TIMESTEP}"
)

if [[ "${SAVE_ATTEMPT_VIDEOS}" == "1" ]]; then
  CMD+=(--save_attempt_videos)
fi
CMD+=(--attempt_video_fps "${ATTEMPT_VIDEO_FPS}")

CMD+=("$@")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
