import argparse
import datetime
import importlib.util
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if ENV_DIR not in sys.path:
    sys.path.append(ENV_DIR)

from pyGameWorld import loadToolPicker, run_vlm_toolpicker_episode  # noqa: E402
from paths import artifact_path  # noqa: E402


def _load_prompt(prompt_path: Optional[str], default_prompt: str) -> str:
    if prompt_path:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return default_prompt


def _load_prompt_from_python(prompt_py_path: str) -> str:
    spec = importlib.util.spec_from_file_location("tool_games_prompt_module", prompt_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load prompt module: {prompt_py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt_text = getattr(module, "prompt", None)
    if not isinstance(prompt_text, str):
        raise ValueError(f"`prompt` string not found in: {prompt_py_path}")
    return prompt_text


def _load_tool_shapes_map(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result: Dict[str, Dict[str, str]] = {}
    if not isinstance(payload, dict):
        return result

    for task_name, task_info in payload.items():
        if not isinstance(task_name, str) or not isinstance(task_info, dict):
            continue
        tools_info = task_info.get("tools")
        if not isinstance(tools_info, dict):
            continue
        per_tool: Dict[str, str] = {}
        for tool_name, tool_entry in tools_info.items():
            if not isinstance(tool_name, str) or not isinstance(tool_entry, dict):
                continue
            shape = tool_entry.get("shape")
            if isinstance(shape, str) and shape:
                per_tool[tool_name] = shape.upper()
        if per_tool:
            result[task_name] = per_tool
    return result


def _collect_json_levels(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    result: List[str] = []
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.lower().endswith(".json"):
                result.append(os.path.join(root, filename))
    return sorted(result)


def _is_toolpicker_json(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("world"), dict)
        and isinstance(data.get("tools"), dict)
    )


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return token or "item"


def _export_attempt_simulation_gif(
    toolpicker: Any,
    toolname: str,
    position: List[int],
    output_path: str,
    maxtime: float,
    fps: int,
) -> Optional[str]:
    path, _, _, worlddict = toolpicker.observeFullPlacementPath(
        toolname=toolname,
        position=position,
        maxtime=maxtime,
        returnDict=True,
        stopOnGoal=True,
    )
    if path is None or worlddict is None:
        return "No simulation path (likely collision/out-of-bounds)."

    frames = toolpicker._get_image_array(worlddict, path)
    if not isinstance(frames, np.ndarray) or frames.size == 0:
        return "No frames available from simulator."

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_frames = [Image.fromarray(frame.astype(np.uint8), mode="RGB") for frame in frames]
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return None


def run_agent(args: argparse.Namespace) -> None:
    prompt = _load_prompt_from_python(args.prompt_py_path)
    if args.prompt_path:
        prompt = _load_prompt(args.prompt_path, prompt)

    tool_shapes_by_task = _load_tool_shapes_map(args.tool_shapes_json)

    json_paths = _collect_json_levels(args.input_path)
    if not json_paths:
        print(f"No JSON files found at: {args.input_path}")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(args.output_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    backend = args.backend.lower()
    if backend == "qwen":
        from qwen3 import Qwen3VLClient  # noqa: E402

        client = Qwen3VLClient(
            model_name=args.model_name,
            local_load_path=args.local_load_path,
            torch_dtype=args.torch_dtype,
            use_flash_attention=args.use_flash_attention,
            system_prompt=args.system_prompt,
        )
    elif backend == "gemini":
        from gemini import GeminiVLClient  # noqa: E402

        resolved_api_key = args.api_key or os.getenv(args.api_key_env)
        client = GeminiVLClient(
            model_name=args.model_name,
            api_key=resolved_api_key,
            system_prompt=args.system_prompt,
            max_retries=args.api_max_retries,
            retry_delay=args.api_retry_delay,
            thinking_level=args.gemini_thinking_level,
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    per_level_results: List[Dict[str, Any]] = []
    solved_count = 0
    evaluated_count = 0

    for json_path in json_paths:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as err:
            per_level_results.append(
                {
                    "level_path": json_path,
                    "status": "READ_ERROR",
                    "error": str(err),
                }
            )
            continue

        if not _is_toolpicker_json(payload):
            continue

        evaluated_count += 1
        level_name = os.path.splitext(os.path.basename(json_path))[0]
        level_tools = sorted(payload["tools"].keys())
        selected_tool_count = max(1, args.tools_per_task)
        selected_tools = level_tools[:selected_tool_count]
        per_tool_attempts = max(1, args.max_attempts)
        attempt_tools = [tool for tool in selected_tools for _ in range(per_tool_attempts)]
        if not selected_tools:
            per_level_results.append(
                {
                    "level_path": json_path,
                    "level_name": level_name,
                    "status": "NO_TOOLS",
                }
            )
            continue
        level_prompt = prompt
        tool_shapes_for_level = tool_shapes_by_task.get(level_name, {})

        tp = loadToolPicker(json_path, basicTimestep=args.basic_timestep)
        result = run_vlm_toolpicker_episode(
            toolpicker=tp,
            vlm_client=client,
            prompt=level_prompt,
            max_attempts=len(attempt_tools),
            maxtime=args.max_time,
            stop_on_goal=True,
            inference_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "do_sample": args.do_sample,
            },
            attempt_tools=attempt_tools,
            tool_shape_by_name=tool_shapes_for_level,
        )
        result = _to_builtin(result)
        result["level_path"] = json_path
        result["level_name"] = level_name

        if args.save_attempt_videos:
            level_video_dir = os.path.join(out_dir, "attempt_videos", level_name)
            for attempt in result.get("attempts", []):
                if attempt.get("status") != "SIMULATED":
                    continue
                forced_tool = attempt.get("forced_tool")
                world_xy = attempt.get("world_xy")
                attempt_number = attempt.get("attempt_number")
                if (
                    not isinstance(forced_tool, str)
                    or not isinstance(world_xy, list)
                    or len(world_xy) != 2
                    or not all(isinstance(v, (int, float)) for v in world_xy)
                    or not isinstance(attempt_number, int)
                ):
                    attempt["simulation_video_error"] = (
                        "Missing forced_tool/world_xy/attempt_number in attempt record."
                    )
                    continue

                safe_tool = _safe_filename_token(forced_tool)
                video_filename = f"attempt_{attempt_number:02d}_{safe_tool}.gif"
                video_path = os.path.join(level_video_dir, video_filename)
                error = _export_attempt_simulation_gif(
                    toolpicker=tp,
                    toolname=forced_tool,
                    position=[int(world_xy[0]), int(world_xy[1])],
                    output_path=video_path,
                    maxtime=args.max_time,
                    fps=args.attempt_video_fps,
                )
                if error is None:
                    attempt["simulation_video_path"] = video_path
                else:
                    attempt["simulation_video_error"] = error

        per_level_results.append(result)
        if result.get("solved"):
            solved_count += 1

        level_log_path = os.path.join(out_dir, f"{level_name}.result.json")
        with open(level_log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    summary = {
        "input_path": args.input_path,
        "evaluated_levels": evaluated_count,
        "solved_levels": solved_count,
        "success_rate": (solved_count / evaluated_count) if evaluated_count else 0.0,
        "model_name": args.local_load_path or args.model_name,
        "results_dir": out_dir,
    }

    with open(os.path.join(out_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(per_level_results, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch VLM runner for tool-games ToolPicker levels.")
    parser.add_argument(
        "--backend",
        type=str,
        default="qwen",
        choices=["qwen", "gemini"],
        help="VLM backend to use.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join(ENV_DIR, "unittest_files"),
        help="Path to a ToolPicker JSON file or a directory containing JSON files.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=artifact_path("vlm_toolpicker"),
        help="Directory where run logs are written.",
    )
    parser.add_argument("--prompt_path", type=str, default=None, help="Optional prompt text file.")
    parser.add_argument(
        "--prompt_py_path",
        type=str,
        default=os.path.join(ENV_DIR, "agent", "prompt.py"),
        help="Python prompt module path containing `prompt` variable.",
    )
    parser.add_argument(
        "--tool_shapes_json",
        type=str,
        default=os.path.join(ENV_DIR, "utils", "tool_shapes_by_task.json"),
        help="Task->tool shape mapping JSON file.",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-VL-3B-Instruct", help="HF model name."
    )
    parser.add_argument("--local_load_path", type=str, default=None, help="Optional local checkpoint.")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant for tool-games physics reasoning.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Optional Gemini API key. If unset, --api_key_env is used.",
    )
    parser.add_argument(
        "--api_key_env",
        type=str,
        default="GEMINI_API_KEY",
        help="Environment variable name used to read Gemini API key.",
    )
    parser.add_argument(
        "--api_max_retries",
        type=int,
        default=3,
        help="Max retries for Gemini API calls.",
    )
    parser.add_argument(
        "--api_retry_delay",
        type=float,
        default=1.5,
        help="Base retry delay (seconds) for Gemini API calls.",
    )
    parser.add_argument(
        "--gemini_thinking_level",
        type=str,
        default="low",
        choices=["none", "low", "medium", "high"],
        help=(
            "Gemini thinking level hint. "
            "Mapped to a model-specific thinking budget when supported."
        ),
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="Number of attempts per selected tool (default: 3).",
    )
    parser.add_argument(
        "--tools_per_task",
        type=int,
        default=3,
        help="Number of distinct tools to include per task (default: 3).",
    )
    parser.add_argument("--max_time", type=float, default=20.0)
    parser.add_argument("--basic_timestep", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--save_attempt_videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a simulation GIF for each simulated VLM attempt (default: enabled).",
    )
    parser.add_argument(
        "--attempt_video_fps",
        type=int,
        default=10,
        help="Frame rate used when writing attempt simulation GIFs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Ignored unless --do_sample is set.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling. Ignored unless --do_sample is set.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable stochastic sampling. Default is greedy decoding for stability.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_agent(parse_args())
