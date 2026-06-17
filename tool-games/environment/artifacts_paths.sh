# shellcheck shell=bash
# Shared artifacts root for tool-games shell launchers.
# Source from run_*.sh:  source "$(dirname "${BASH_SOURCE[0]}")/artifacts_paths.sh"

_TOOL_GAMES_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ARTIFACTS_NAME="${TOOL_GAMES_ARTIFACTS_DIR_NAME:-artifacts}"
ARTIFACTS_DIR="${TOOL_GAMES_ARTIFACTS_DIR:-${_TOOL_GAMES_ENV_DIR}/${_ARTIFACTS_NAME}}"
