"""
Render initial ToolPicker scenes for the Original trial set using semantic colors
from the JSON (red, green, blue, ...), without the PHYRE-style palette remap in
`pyGameWorld.helpers.applyColorRemap` (green→blue, red→green, blue→lightgrey).

Import order matters: patch helpers before `pyGameWorld.object` is first imported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Tuple

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

from paths import artifact_path  # noqa: E402


def _install_original_color_resolver(env_dir: str) -> None:
    """
    Patch `applyColorRemap` before `pyGameWorld.__init__` runs (it imports
    `world` → `pymunk`). We register a minimal package stub and load only
    `helpers.py`, patch it, then load `object` so PGObject uses original colors.
    """
    import importlib
    import importlib.util
    import types

    pkg = types.ModuleType("pyGameWorld")
    pkg.__path__ = [os.path.join(env_dir, "pyGameWorld")]
    sys.modules["pyGameWorld"] = pkg

    helpers_path = os.path.join(env_dir, "pyGameWorld", "helpers.py")
    spec = importlib.util.spec_from_file_location("pyGameWorld.helpers", helpers_path)
    helpers_mod = importlib.util.module_from_spec(spec)
    sys.modules["pyGameWorld.helpers"] = helpers_mod
    assert spec.loader is not None
    spec.loader.exec_module(helpers_mod)

    def apply_color_original(color: Any) -> Any:
        if color is None:
            return None
        if isinstance(color, str):
            cname = color.lower()
            if cname not in helpers_mod._BASE_COLOR_BY_NAME:
                raise Exception("Color name not known: " + cname)
            return helpers_mod._BASE_COLOR_BY_NAME[cname]
        return helpers_mod._to_rgba_tuple(color)

    helpers_mod.applyColorRemap = apply_color_original

    object_mod = importlib.import_module("pyGameWorld.object")
    object_mod.applyColorRemap = apply_color_original


_install_original_color_resolver(ENV_DIR)

from pyGameWorld.toolpicker_js import loadToolPicker  # noqa: E402


def _collect_json_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    files: List[str] = []
    for root, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith(".json"):
                files.append(os.path.join(root, name))
    return sorted(files)


def _is_toolpicker_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("world"), dict)
        and isinstance(payload.get("tools"), dict)
    )


def _save_rgb_array(image: np.ndarray, output_path: str) -> None:
    if image.max() <= 1.0:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    Image.fromarray(image).save(output_path)


def render_original_initial_scenes(
    input_path: str,
    output_dir: str,
    basic_timestep: float = 0.1,
) -> Tuple[int, int]:
    os.makedirs(output_dir, exist_ok=True)
    json_files = _collect_json_files(input_path)
    written = 0
    skipped = 0
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as err:
            print(f"[SKIP] {json_path} (read error: {err})")
            skipped += 1
            continue

        if not _is_toolpicker_payload(payload):
            skipped += 1
            continue

        try:
            tp = loadToolPicker(json_path, basicTimestep=basic_timestep)
            image = tp.drawPathSingleImage(tp._worlddict, path=None)
        except Exception as err:
            print(f"[SKIP] {json_path} (render error: {err})")
            skipped += 1
            continue

        level_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(output_dir, f"{level_name}.png")
        _save_rgb_array(image, output_path)
        print(f"[WRITE] {json_path} -> {output_path}")
        written += 1

    print(f"Done. wrote={written}, skipped={skipped}, output_dir={output_dir}")
    return written, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export initial-scene PNGs for ToolPicker JSON using original semantic "
            "colors (no PHYRE remap)."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join(ENV_DIR, "Trials", "Original"),
        help="ToolPicker JSON file or directory (default: Trials/Original).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=artifact_path("original_initial_scenes"),
        help="Directory for PNG output.",
    )
    parser.add_argument(
        "--basic_timestep",
        type=float,
        default=0.1,
        help="ToolPicker basic timestep.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_original_initial_scenes(
        args.input_path,
        args.output_dir,
        basic_timestep=args.basic_timestep,
    )
