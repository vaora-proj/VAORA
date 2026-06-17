"""Shared output paths for the tool-games environment.

Generated files (VLM inputs, run logs, exports) live under the artifacts root.
Override the root with ``TOOL_GAMES_ARTIFACTS_DIR``, or only the subdirectory
name with ``TOOL_GAMES_ARTIFACTS_DIR_NAME`` (default: ``artifacts``).
"""

from __future__ import annotations

import os

ENV_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR_NAME = os.environ.get("TOOL_GAMES_ARTIFACTS_DIR_NAME", "artifacts")


def get_artifacts_dir() -> str:
    """Absolute path to the artifacts root directory."""
    override = os.environ.get("TOOL_GAMES_ARTIFACTS_DIR")
    if override:
        return os.path.abspath(override)
    return os.path.join(ENV_DIR, ARTIFACTS_DIR_NAME)


def artifact_path(*parts: str) -> str:
    """Join path segments under the artifacts root."""
    return os.path.join(get_artifacts_dir(), *parts)
