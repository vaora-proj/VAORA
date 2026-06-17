import argparse
import json
import os
import sys
from typing import Any, List

import pygame as pg


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if ENV_DIR not in sys.path:
    sys.path.append(ENV_DIR)

from pyGameWorld import loadToolPicker  # noqa: E402
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


def export_task_tools(input_path: str, output_dir: str, basic_timestep: float = 0.1) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_files = _collect_json_files(input_path)
    if not json_files:
        print(f"No JSON files found: {input_path}")
        return

    pg.init()

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
        except Exception as err:
            print(f"[SKIP] {json_path} (load error: {err})")
            skipped += 1
            continue

        task_name = os.path.splitext(os.path.basename(json_path))[0]
        task_out_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_out_dir, exist_ok=True)

        tool_names = list(tp.getToolNames())
        if not tool_names:
            print(f"[WARN] {task_name}: no tools found")
            continue

        metadata = {"task_name": task_name, "task_json": json_path, "tools": []}
        for idx, tool_name in enumerate(tool_names):
            surf = tp.drawTool(idx)
            out_path = os.path.join(task_out_dir, f"{idx:02d}_{tool_name}.png")
            pg.image.save(surf, out_path)
            metadata["tools"].append(
                {
                    "index": idx,
                    "name": tool_name,
                    "image_path": out_path,
                }
            )
            print(f"[WRITE] {task_name} tool={tool_name} -> {out_path}")
            written += 1

        meta_path = os.path.join(task_out_dir, "tools.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    pg.quit()
    print(f"Done. wrote={written}, skipped={skipped}, output_dir={output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export tool thumbnails for each ToolPicker task JSON."
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
        default=artifact_path("tool_catalog"),
        help="Directory where tool images will be written.",
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
    export_task_tools(args.input_path, args.output_dir, basic_timestep=args.basic_timestep)
