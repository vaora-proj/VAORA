from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from PIL import Image


def parse_toolpicker_action(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a VLM response into a ToolPicker action.

    Expected action schema inside <action> ... </action> (preferred):
      [x, y, r]

    Returns:
      {"norm_xy": (float, float)} or None when parsing fails.
    """
    if not response_text:
        return None

    action_match = re.search(
        r"<action>\s*(.*?)\s*</action>", response_text, flags=re.IGNORECASE | re.DOTALL
    )
    payload = action_match.group(1).strip() if action_match else response_text.strip()

    numbers = re.findall(r"[-+]?\d*\.?\d+", payload)
    if len(numbers) >= 2:
        try:
            return {"norm_xy": (float(numbers[0]), float(numbers[1]))}
        except ValueError:
            return None

    return None


def _norm255_to_world_position(
    norm_xy: tuple[float, float],
    world_dims: tuple[int, int] | list[int],
) -> tuple[int, int]:
    """
    Convert model output [x, y] in 0..255 image coordinates into tool-games world coordinates.
    y is interpreted in image convention (0=top, 255=bottom), then inverted to world y-up.
    """
    width = max(int(world_dims[0]), 1)
    height = max(int(world_dims[1]), 1)
    x_norm, y_norm = norm_xy
    x_norm = max(0.0, min(255.0, x_norm))
    y_norm = max(0.0, min(255.0, y_norm))

    x_world = int(round((x_norm / 255.0) * (width - 1)))
    y_world = int(round(((255.0 - y_norm) / 255.0) * (height - 1)))
    return x_world, y_world


def _ensure_vlm_input_size(image: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Ensure image sent to VLM has fixed spatial size (H, W) = (256, 256).
    """
    if image.shape[0] == size[0] and image.shape[1] == size[1]:
        return image
    img = image
    if img.max() <= 1.0:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    resized = Image.fromarray(img).resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def run_vlm_toolpicker_episode(
    toolpicker,
    vlm_client: Any,
    prompt: str,
    max_attempts: int = 3,
    maxtime: float = 20.0,
    stop_on_goal: bool = True,
    parse_action_fn: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    inference_kwargs: Optional[Dict[str, Any]] = None,
    attempt_tools: Optional[List[str]] = None,
    tool_shape_by_name: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run a ToolPicker episode with a VLM policy in a PHYRE-style loop.

    The VLM client is expected to expose:
      inference_image(images=[np.ndarray], prompts=[str], **kwargs) -> (responses, input_dims)

    The action parser must return:
      {"norm_xy": (<x_0_to_255>, <y_0_to_255>)}
    """
    parser = parse_action_fn or parse_toolpicker_action
    valid_tools = list(toolpicker.getToolNames())
    inference_kwargs = inference_kwargs or {}
    world_dims = toolpicker.getWorldDims()

    if attempt_tools is None:
        attempt_tools = valid_tools[:max_attempts]
    else:
        attempt_tools = [t for t in attempt_tools if t in valid_tools]
        if max_attempts > 0:
            attempt_tools = attempt_tools[:max_attempts]

    attempt_logs: List[Dict[str, Any]] = []
    solved = False
    final_action: Optional[Dict[str, Any]] = None

    retry_suffix = ""
    for attempt_idx, forced_tool in enumerate(attempt_tools, start=1):
        # Render initial state image for the VLM.
        obs_image = toolpicker.drawPathSingleImage(toolpicker._worlddict, path=None)
        obs_image = _ensure_vlm_input_size(obs_image, size=(256, 256))
        tool_shape = None
        if tool_shape_by_name is not None:
            tool_shape = tool_shape_by_name.get(forced_tool)
        prompt_with_shape = prompt.replace("<RED_OBJECT_SHAPE>", tool_shape or "BALL")

        full_prompt = (
            prompt_with_shape
            + f"\n\nFor this attempt, use exactly this tool: {forced_tool}\n"
            "Output format must be exactly:\n"
            "<action>\n[x, y, r]\n</action>\n"
            + retry_suffix
        )
        responses, _ = vlm_client.inference_image(
            images=[obs_image],
            prompts=[full_prompt],
            **inference_kwargs,
        )
        response_text = responses[0] if responses else ""

        parsed = parser(response_text)
        attempt_record: Dict[str, Any] = {
            "attempt_number": attempt_idx,
            "forced_tool": forced_tool,
            "response_text": response_text,
            "parsed_action": parsed,
            "status": "PARSE_FAILED",
            "solved": False,
        }

        if not parsed:
            retry_suffix = (
                "\n\nPrevious output could not be parsed. Return exactly:\n"
                "<action>\n[x, y, r]\n</action>"
            )
            attempt_logs.append(attempt_record)
            continue

        norm_xy = parsed.get("norm_xy")
        if not isinstance(norm_xy, tuple) or len(norm_xy) != 2:
            attempt_record["status"] = "INVALID_ACTION_SCHEMA"
            retry_suffix = (
                "\n\nPrevious action was invalid. Return exactly:\n"
                "<action>\n[x, y, r]\n</action>"
            )
            attempt_logs.append(attempt_record)
            continue

        position = _norm255_to_world_position(norm_xy, world_dims)
        attempt_record["normalized_xy"] = [norm_xy[0], norm_xy[1]]
        attempt_record["world_xy"] = [position[0], position[1]]

        success, end_time = toolpicker.runPlacement(
            toolname=forced_tool,
            position=position,
            maxtime=maxtime,
            stopOnGoal=stop_on_goal,
        )
        if success is None and end_time == -1:
            attempt_record["status"] = "COLLISION_OR_OUT_OF_BOUNDS"
            retry_suffix = (
                "\n\nPrevious placement collided or was out-of-bounds. "
                "Try a different [x, y, r] and return only:\n"
                "<action>\n[x, y, r]\n</action>"
            )
            attempt_logs.append(attempt_record)
            continue

        attempt_record["status"] = "SIMULATED"
        attempt_record["simulation_end_time"] = end_time
        attempt_record["solved"] = bool(success)
        attempt_logs.append(attempt_record)
        final_action = {"tool": forced_tool, "norm_xy": norm_xy, "world_xy": position}

        if success:
            solved = True
            break

        retry_suffix = (
            "\n\nPrevious action did not solve the task. "
            "Try another [x, y, r] and return only:\n"
            "<action>\n[x, y, r]\n</action>"
        )

    return {
        "solved": solved,
        "final_action": final_action,
        "attempts": attempt_logs,
    }
