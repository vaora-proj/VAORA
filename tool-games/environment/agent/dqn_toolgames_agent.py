#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TOOL_GAMES_ROOT = os.path.abspath(os.path.join(ENV_DIR, ".."))
DEFAULT_ACTION_CACHE_PATH = os.path.join(
    TOOL_GAMES_ROOT, "data", "action_array_ball_seed42_100k.npy"
)

for path in (ENV_DIR, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from pyGameWorld import loadToolPicker  # noqa: E402
import phyre_dqn_compat  # noqa: E402
from paths import artifact_path  # noqa: E402


PHYRE_WAD_COLORS = np.array(
    [
        [255, 255, 255],  # white
        [243, 79, 70],    # red
        [107, 206, 187],  # green
        [24, 119, 242],   # blue
        [75, 74, 164],    # purple
        [185, 202, 210],  # gray
        [0, 0, 0],        # black
    ],
    dtype=np.uint8,
)


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
    return isinstance(data, dict) and isinstance(data.get("world"), dict) and isinstance(
        data.get("tools"), dict
    )


def _ensure_256_rgb(image: np.ndarray) -> np.ndarray:
    img = image
    if img.max() <= 1.0:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    if img.shape[:2] != (256, 256):
        img = np.asarray(Image.fromarray(img).resize((256, 256), Image.BILINEAR), dtype=np.uint8)
    return img


def _rgb_to_phyre_indices(rgb_image: np.ndarray) -> np.ndarray:
    img = _ensure_256_rgb(rgb_image)
    flat = img.reshape(-1, 3).astype(np.int16)
    palette = PHYRE_WAD_COLORS.astype(np.int16)
    distances = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    indices = distances.argmin(axis=1).astype(np.int64)
    return indices.reshape(256, 256)


def _remap_action_to_world_xy(
    action_xyz: Sequence[float], world_dims: Sequence[int]
) -> Tuple[List[int], Dict[str, float]]:
    x = float(action_xyz[0])
    y = float(action_xyz[1])
    r = float(action_xyz[2]) if len(action_xyz) >= 3 else 0.0
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    width = max(int(world_dims[0]), 1)
    height = max(int(world_dims[1]), 1)
    world_x = int(round(x * (width - 1)))
    world_y = int(round(y * (height - 1)))
    return [world_x, world_y], {"x": x, "y": y, "r": r}


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
    position: Sequence[int],
    output_path: str,
    maxtime: float,
    fps: int,
) -> Optional[str]:
    """
    Export a placement GIF for one (toolname, position) attempt.

    Returns:
      None on success, otherwise an error string.
    """
    path, _, _, worlddict = toolpicker.observeFullPlacementPath(
        toolname=toolname,
        position=list(position),
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


def run(args: argparse.Namespace) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_root, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    model = phyre_dqn_compat.load_model_from_phyre_ckpt(args.dqn_load_from)
    if not os.path.isfile(args.action_cache_path):
        raise FileNotFoundError(f"Action cache file not found: {args.action_cache_path}")
    action_array = np.load(args.action_cache_path)
    if action_array.ndim != 2 or action_array.shape[1] < 2:
        raise ValueError(
            f"Action cache must be a 2D array with >=2 dims per action, got {action_array.shape}"
        )
    if action_array.dtype != np.float32:
        action_array = action_array.astype(np.float32)
    rank_size = args.dqn_rank_size if args.dqn_rank_size > 0 else len(action_array)
    rank_size = min(rank_size, len(action_array))
    action_array = action_array[:rank_size]

    all_jsons = _collect_json_levels(args.input_path)
    if not all_jsons:
        raise FileNotFoundError(f"No json levels found in {args.input_path}")

    results: List[Dict[str, Any]] = []
    total_tasks = 0
    solved_at_1 = 0
    solved_at_3 = 0
    solved_at_5 = 0

    for json_path in all_jsons:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not _is_toolpicker_json(payload):
            continue

        total_tasks += 1
        level_name = os.path.splitext(os.path.basename(json_path))[0]
        toolpicker = loadToolPicker(json_path, basicTimestep=args.basic_timestep)
        world_dims = toolpicker.getWorldDims()
        tool_names = list(toolpicker.getToolNames())

        initial_rgb = toolpicker.drawPathSingleImage(toolpicker._worlddict, path=None)
        dqn_obs = _rgb_to_phyre_indices(initial_rgb)

        scores = phyre_dqn_compat.eval_actions(
            model, action_array, args.dqn_eval_batch_size, dqn_obs
        )
        sorted_ids = np.argsort(-scores)
        top_ids = sorted_ids[: args.top_k]

        per_action_attempts: List[Dict[str, Any]] = []
        best_success_rank: Optional[int] = None

        level_video_dir = os.path.join(output_dir, "attempt_videos", level_name)

        for rank, action_id in enumerate(top_ids, start=1):
            action_xyz = action_array[action_id]
            world_xy, remapped = _remap_action_to_world_xy(action_xyz, world_dims)

            tool_try_records: List[Dict[str, Any]] = []
            rank_success = False
            for tool_name in tool_names:
                success, end_time = toolpicker.runPlacement(
                    toolname=tool_name,
                    position=world_xy,
                    maxtime=args.max_time,
                    stopOnGoal=True,
                )
                is_collision = success is None and end_time == -1
                solved = bool(success) if success is not None else False
                rank_success = rank_success or solved
                attempt_record: Dict[str, Any] = {
                    "tool_name": tool_name,
                    "world_xy": world_xy,
                    "sim_status": "COLLISION_OR_OOB" if is_collision else "SIMULATED",
                    "solved": solved,
                    "simulation_end_time": end_time,
                }

                if args.save_attempt_videos and not is_collision:
                    safe_tool = _safe_filename_token(tool_name)
                    video_filename = f"rank_{rank:02d}_action_{int(action_id):06d}_tool_{safe_tool}.gif"
                    video_path = os.path.join(level_video_dir, video_filename)
                    video_error = _export_attempt_simulation_gif(
                        toolpicker=toolpicker,
                        toolname=tool_name,
                        position=world_xy,
                        output_path=video_path,
                        maxtime=args.max_time,
                        fps=args.attempt_video_fps,
                    )
                    if video_error is None:
                        attempt_record["simulation_video_path"] = video_path
                    else:
                        attempt_record["simulation_video_error"] = video_error

                tool_try_records.append(attempt_record)

            if rank_success and best_success_rank is None:
                best_success_rank = rank

            per_action_attempts.append(
                {
                    "rank": rank,
                    "action_id": int(action_id),
                    "predicted_score": float(scores[action_id]),
                    "dqn_action_xyz": [float(action_xyz[0]), float(action_xyz[1]), float(action_xyz[2])],
                    "remapped_action_xy": world_xy,
                    "remapped_meta": remapped,
                    "tool_attempts": tool_try_records,
                    "solved_at_this_rank": rank_success,
                }
            )

        task_pass_1 = best_success_rank is not None and best_success_rank <= 1
        task_pass_3 = best_success_rank is not None and best_success_rank <= 3
        task_pass_5 = best_success_rank is not None and best_success_rank <= 5
        solved_at_1 += int(task_pass_1)
        solved_at_3 += int(task_pass_3)
        solved_at_5 += int(task_pass_5)

        task_result = {
            "level_path": json_path,
            "level_name": level_name,
            "tool_names": tool_names,
            "top_k": args.top_k,
            "best_success_rank": best_success_rank,
            "pass_at_1": task_pass_1,
            "pass_at_3": task_pass_3,
            "pass_at_5": task_pass_5,
            "attempts": per_action_attempts,
        }
        results.append(task_result)

        with open(os.path.join(output_dir, f"{level_name}.result.json"), "w", encoding="utf-8") as f:
            json.dump(_to_builtin(task_result), f, indent=2)

    summary = {
        "input_path": args.input_path,
        "dqn_load_from": args.dqn_load_from,
        "action_cache_path": args.action_cache_path,
        "dqn_rank_size": rank_size,
        "top_k": args.top_k,
        "evaluated_levels": total_tasks,
        "pass@1": (solved_at_1 / total_tasks) if total_tasks else 0.0,
        "pass@3": (solved_at_3 / total_tasks) if total_tasks else 0.0,
        "pass@5": (solved_at_5 / total_tasks) if total_tasks else 0.0,
        "results_dir": output_dir,
    }

    with open(os.path.join(output_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(_to_builtin(results), f, indent=2)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_to_builtin(summary), f, indent=2)

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PHYRE DQN policy on tool-games with cross-tool top-k evaluation."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join(ENV_DIR, "Trials", "Original"),
        help="Tool-games level json file or directory.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=artifact_path("dqn_toolgames"),
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--dqn_load_from",
        type=str,
        required=True,
        help="Path to PHYRE DQN checkpoint folder (expects ckpt.* files).",
    )
    parser.add_argument(
        "--action_cache_path",
        type=str,
        default=DEFAULT_ACTION_CACHE_PATH,
        help="Path to pre-saved action cache .npy (shape: [N, 3]).",
    )
    parser.add_argument("--dqn_rank_size", type=int, default=10000)
    parser.add_argument("--dqn_eval_batch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_time", type=float, default=20.0)
    parser.add_argument("--basic_timestep", type=float, default=0.1)
    parser.add_argument(
        "--save_attempt_videos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, save a GIF for each successful tool placement attempt.",
    )
    parser.add_argument(
        "--attempt_video_fps",
        type=int,
        default=10,
        help="FPS used when writing attempt GIFs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
