import argparse
import json
import os
import sys
from typing import Any, List

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if ENV_DIR not in sys.path:
    sys.path.append(ENV_DIR)

from pyGameWorld import loadToolPicker  # noqa: E402
from pyGameWorld.vlm_agent import _ensure_vlm_input_size  # noqa: E402

from paths import artifact_path  # noqa: E402


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


def export_images(input_path: str, output_dir: str, basic_timestep: float = 0.1) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_files = _collect_json_files(input_path)
    if not json_files:
        print(f"No JSON files found: {input_path}")
        return

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
            # Match run_vlm_toolpicker_episode(): render then resize to 256x256 for the VLM.
            image = tp.drawPathSingleImage(tp._worlddict, path=None)
            image = _ensure_vlm_input_size(image, size=(256, 256))
        except Exception as err:
            print(f"[SKIP] {json_path} (render error: {err})")
            skipped += 1
            continue

        level_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(output_dir, f"{level_name}.png")
        _save_rgb_array(image, output_path)
        print(f"[WRITE] {json_path} -> {output_path}")
        written += 1

    print(
        f"Done. wrote={written}, skipped={skipped}, output_dir={output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the exact initial images fed to the VLM for ToolPicker tasks."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join(ENV_DIR, "Trials", "Original"),
        help="ToolPicker JSON file or directory of JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=artifact_path("vlm_input_images"),
        help="Directory where PNGs will be written.",
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
    export_images(args.input_path, args.output_dir, basic_timestep=args.basic_timestep)
