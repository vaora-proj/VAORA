"""
Shared agent utilities and run loop for PHYRE batch inference.

All model-specific logic is isolated in the individual client modules
(gemini.py, chatgpt.py, claude.py, internvl.py, qwen3.py).  Any client
that exposes

    inference_image(images, prompts, max_new_tokens, temperature, top_p)
        -> (List[str], List[Any])

can be dropped into run_phyre_agent.
"""

import datetime
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import phyre

BATCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BATCH_DIR not in sys.path:
    sys.path.append(BATCH_DIR)
REPO_ROOT = os.path.abspath(os.path.join(BATCH_DIR, ".."))

MODEL_WIDTH = 256
MODEL_HEIGHT = 256
MIN_RADIUS = 2
MAX_RADIUS = 32
_RAD_RANGE = MAX_RADIUS - MIN_RADIUS


# ---------------------------------------------------------------------------
# Text parsing helpers
# ---------------------------------------------------------------------------

def _extract_tag_block(text: str, tag_names: Sequence[str]) -> str | None:
    for tag_name in tag_names:
        match = re.search(
            rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>",
            text or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
    return None


def extract_action(text: str) -> List[float] | None:
    if not text:
        return None
    search_space = _extract_tag_block(text, ("action",)) or text
    numbers = re.findall(r"[-+]?\d*\.?\d+", search_space)
    if len(numbers) >= 3:
        return [float(numbers[0]), float(numbers[1]), float(numbers[2])]
    return None


def extract_reasoning_and_answer(text: str) -> Tuple[str, str]:
    reasoning = _extract_tag_block(
        text,
        ("scene_reasoning", "plan_reasoning", "causal_actions_reasoning", "placement_reasoning"),
    )
    answer = _extract_tag_block(text, ("action", "action_sequence"))
    return (reasoning or "No reasoning provided"), (answer or "No answer found")


def sanitize_parsed_action(action: Sequence[float] | None) -> List[float]:
    if not action:
        return []
    sanitized = []
    for value in action[:3]:
        value_f = float(value)
        sanitized.append(value_f if np.isfinite(value_f) else 0.0)
    return sanitized


# ---------------------------------------------------------------------------
# PHYRE action conversion
# ---------------------------------------------------------------------------

def convert_model_prediction_to_float_action_resized(model_prediction: Sequence[int]) -> List[float]:
    pred_x, pred_y, pred_r = model_prediction
    pred_y = MODEL_HEIGHT - 1 - pred_y
    x_action = pred_x / (MODEL_WIDTH - 1)
    y_action = pred_y / (MODEL_HEIGHT - 1)
    d_action = (pred_r - MIN_RADIUS) / _RAD_RANGE
    return np.array([x_action, y_action, d_action]).tolist()


def digit_mapping(action: Sequence[float]) -> List[float]:
    action = [int(a) for a in action]
    return convert_model_prediction_to_float_action_resized(action)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_image(image: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.max() <= 1.0:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    Image.fromarray(image).save(path)


def save_gif(frames: List[np.ndarray], path: str, duration: int = 100, loop: int = 0) -> None:
    if not frames:
        return
    pil_frames = []
    for frame in frames:
        if frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        pil_frames.append(Image.fromarray(frame))
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=duration, loop=loop)


def save_attempt_history_txt(path: str, attempt_history: List[Dict[str, Any]]) -> None:
    lines = []
    for attempt in attempt_history:
        lines.append(f"Attempt {attempt.get('attempt_number')}")
        lines.append(f"  Status: {attempt.get('simulation_status')}")
        lines.append(f"  Solved: {attempt.get('is_solved_in_this_attempt')}")
        lines.append(f"  Parsed Action: {attempt.get('parsed_action') or []}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _validate_dataset_item(item: Any) -> Dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if not item.get("task_id") or not item.get("prompt"):
        return None
    return item


def load_dataset(path: str, num_tasks: int | None, num_workers: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if num_tasks is not None and num_tasks > 0:
        data = data[:num_tasks]
    if num_workers <= 1:
        return [item for item in (_validate_dataset_item(i) for i in data) if item]
    max_workers = min(num_workers, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        items = list(executor.map(_validate_dataset_item, data))
    return [item for item in items if item]


def repeat_dataset(data: List[Dict[str, str]], repeat_num: int) -> List[Dict[str, str]]:
    if repeat_num <= 1:
        return data
    expanded = []
    for item in data:
        for _ in range(repeat_num):
            expanded.append(item.copy())
    return expanded


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def create_main_results_dir(log_dir: str, timestamp: str | None = None) -> str:
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_dir = os.path.join(log_dir, timestamp)
    os.makedirs(main_dir, exist_ok=True)
    return main_dir


def normalize_log_model_name(model_name: str, local_load_path: str | None) -> str:
    source = local_load_path or model_name
    base = os.path.basename(source) if source else model_name
    if not base and model_name:
        base = model_name.split("/")[-1]
    base = base.replace("-Instruct", "").replace("Instruct", "")
    return base or "model"


def prepare_deepphy_log_dirs(args: Any) -> Dict[str, str]:
    local_load_path = getattr(args, "local_load_path", None)
    log_model_name = getattr(args, "log_model_name", None) or normalize_log_model_name(
        args.model_name, local_load_path
    )
    log_dir = os.path.join(
        args.log_dir_base,
        log_model_name,
        args.eval_setups,
        args.eval_type,
        args.format,
    )
    base_dir = create_main_results_dir(log_dir)
    timestamp = os.path.basename(base_dir)
    label_root = args.log_dir_label.rstrip("/")
    label_base = os.path.join(
        label_root,
        log_model_name,
        args.eval_setups,
        args.eval_type,
        args.format,
        timestamp,
    )
    return {
        "base": base_dir,
        "label_base": label_base,
        "log_model_name": log_model_name,
        "timestamp": timestamp,
    }


def select_keyframe_indices(num_frames: int, num_keyframes: int = 5) -> List[int]:
    if num_frames <= 0:
        return []
    if num_frames <= num_keyframes:
        return list(range(num_frames))
    indices = np.linspace(0, num_frames - 1, num=num_keyframes)
    return sorted({int(idx) for idx in indices})


def log_simulation_results(pred_action, task_index, tasks, simulation):
    status = {-1: "NOT_SOLVED", 0: "INVALID_INPUT", 1: "SOLVED", 2: "UNSTABLY_SOLVED", 3: "STABLY_SOLVED"}
    print("Result of taking action", pred_action, "on task", tasks[task_index], "is:", status[simulation.status])
    return simulation.status.is_solved(), simulation.status.is_invalid()


# ---------------------------------------------------------------------------
# PHYRE eval helpers
# ---------------------------------------------------------------------------

def _select_eval_tasks(eval_setups: str, fold_id: int, eval_type: str) -> List[str]:
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setups, fold_id)
    eval_type_norm = eval_type.lower()
    if eval_type_norm == "train":
        return list(train_tasks)
    if eval_type_norm in {"dev", "val", "valid", "validation"}:
        return list(dev_tasks)
    if eval_type_norm == "test":
        return list(test_tasks)
    if eval_type_norm in {"all", "full"}:
        return list(train_tasks) + list(dev_tasks) + list(test_tasks)
    raise ValueError(f"Unsupported eval_type: {eval_type}")


# ---------------------------------------------------------------------------
# Core run loop (client is injected; not constructed here)
# ---------------------------------------------------------------------------

def run_phyre_agent(args: Any, client: Any) -> None:
    """Run PHYRE batch inference with the supplied *client* object.

    The client must implement:
        inference_image(images, prompts, max_new_tokens, temperature, top_p)
            -> (List[str], List[Any])
    """
    output_dirs = prepare_deepphy_log_dirs(args)

    dataset = load_dataset(args.dataset_path, None, args.num_workers)
    dataset = repeat_dataset(dataset, args.repeat_num)
    if not dataset:
        print(f"No dataset entries found in {args.dataset_path}")
        return

    eval_tasks = _select_eval_tasks(args.eval_setups, args.fold_id, args.eval_type)
    task_id_to_index = {tid: idx for idx, tid in enumerate(eval_tasks)}

    valid_items = [item for item in dataset if item.get("task_id") in task_id_to_index]
    if not valid_items:
        print("No valid dataset items match the selected eval setup/fold.")
        return

    action_tier = phyre.eval_setup_to_action_tier(args.eval_setups)
    simulator = phyre.initialize_simulator(eval_tasks, action_tier)

    batch_items = list(valid_items)
    solved_counter = 0
    all_results: List[Dict[str, Any]] = []
    attempt_counters: Dict[str, int] = {}

    task_contexts: Dict[str, Dict[str, Any]] = {}
    for item in batch_items:
        task_id = item["task_id"]
        if task_id in task_contexts:
            continue
        task_index = task_id_to_index[task_id]
        initial_scene = simulator.initial_scenes[task_index]
        init_img = phyre.observations_to_float_rgb(initial_scene)
        task_slug = task_id.replace(":", "-")
        task_group_dir = os.path.join(output_dirs["base"], f"imgs_{task_slug}")
        task_dir = os.path.join(task_group_dir, task_slug)
        os.makedirs(task_dir, exist_ok=True)

        initial_image_path = os.path.join(task_dir, "initial_scene.png")
        if args.save_images:
            save_image(init_img, initial_image_path)

        task_dir_label = os.path.join(output_dirs["label_base"], f"imgs_{task_slug}", task_slug)
        task_contexts[task_id] = {
            "task_index": task_index,
            "initial_image": init_img,
            "task_dir": task_dir,
            "task_dir_label": task_dir_label,
            "initial_image_path": initial_image_path,
            "initial_image_label": os.path.join(task_dir_label, "initial_scene.png"),
        }

    task_results: Dict[str, Dict[str, Any]] = {}
    ordered_attempts: List[Dict[str, Any]] = []
    for item in batch_items:
        task_id = item["task_id"]
        attempt_idx = attempt_counters.get(task_id, 0) + 1
        attempt_counters[task_id] = attempt_idx
        ordered_attempts.append({"task_id": task_id, "attempt_idx": attempt_idx, "item": item})
        if task_id not in task_results:
            task_results[task_id] = {
                "seed": task_id,
                "is_solved": False,
                "total_attempts": 0,
                "output_dir": task_contexts[task_id]["task_dir_label"],
                "attempt_history": [],
            }

    print(f"Processing {len(ordered_attempts)} attempts from dataset")
    for start in tqdm(range(0, len(ordered_attempts), args.batch_size), desc="Inference"):
        chunk = ordered_attempts[start : start + args.batch_size]
        init_imgs = [task_contexts[entry["task_id"]]["initial_image"] for entry in chunk]
        prompts = [entry["item"]["prompt"] for entry in chunk]

        responses, input_dims = client.inference_image(
            images=init_imgs,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        for local_idx, entry in enumerate(chunk):
            item = entry["item"]
            task_id = entry["task_id"]
            attempt_idx = entry["attempt_idx"]
            task_context = task_contexts[task_id]
            task_index = task_context["task_index"]
            meta = item.get("meta", {}) or {}
            attempt_prefix = f"attempt_{attempt_idx}"
            task_dir = task_context["task_dir"]
            task_dir_label = task_context["task_dir_label"]

            response_text = responses[local_idx]
            reasoning_text, answer_text = extract_reasoning_and_answer(response_text)

            with open(os.path.join(task_dir, f"vlm_prompts_attempt_{attempt_idx}.txt"), "w", encoding="utf-8") as f:
                f.write("=== System Prompt ===\n")
                f.write(args.system_prompt + "\n\n")
                f.write("=== User Prompt ===\n")
                f.write(item["prompt"] + "\n\n")
                f.write("=== Image Descriptions ===\n")
                f.write(f"Path: {task_context['initial_image_path']}\nLabel: Image 1 (Initial Scene)\n")

            with open(os.path.join(task_dir, f"vlm_response_attempt_{attempt_idx}.txt"), "w", encoding="utf-8") as f:
                f.write(response_text)

            parsed_action = sanitize_parsed_action(extract_action(response_text))
            parsed_action_int = [int(round(x)) for x in parsed_action] if parsed_action else []
            normalized_action = digit_mapping(parsed_action) if parsed_action else None

            simulation_status = "INVALID_ACTION_FORMAT" if normalized_action is None else "NOT_RUN"
            keyframes: List[Dict[str, Any]] = []
            gif_path = None
            solved = False

            if normalized_action:
                pred_action = np.array(normalized_action, dtype=np.float32)
                simulation = simulator.simulate_action(
                    task_index, pred_action, need_images=True, need_featurized_objects=True, stride=10
                )
                solved, invalid = log_simulation_results(pred_action, task_index, simulator.task_ids, simulation)
                simulation_status = str(simulation.status) if not invalid else "INVALID_ACTION"

                if solved:
                    solved_counter += 1

                if args.save_images and simulation.images is not None and len(simulation.images) > 0:
                    frame_arrays = [phyre.observations_to_float_rgb(frame) for frame in simulation.images]
                    gif_path = os.path.join(task_dir, f"{attempt_prefix}simulation.gif")
                    save_gif(frame_arrays, gif_path)
                    indices = select_keyframe_indices(len(frame_arrays), num_keyframes=5)
                    for frame_idx, frame_pos in enumerate(indices):
                        frame_name = f"{attempt_prefix}_frame_{frame_idx:03d}.png"
                        frame_path = os.path.join(task_dir, frame_name)
                        save_image(frame_arrays[frame_pos], frame_path)
                        keyframes.append(
                            {"path": os.path.join(task_dir_label, frame_name), "label": f"Keyframe {frame_idx}"}
                        )

            attempt_record: Dict[str, Any] = {
                "attempt_number": attempt_idx,
                "vlm_response": response_text,
                "extracted_reasoning": reasoning_text,
                "extracted_answer": answer_text,
                "simulation_status": simulation_status,
                "simulation_keyframes": keyframes,
                "is_solved_in_this_attempt": bool(solved),
                "image_descriptions": [
                    {"path": task_context["initial_image_label"], "label": "Image 1 (Initial Scene)"}
                ],
                "parsed_x": parsed_action_int[0] if len(parsed_action_int) > 0 else None,
                "parsed_y": parsed_action_int[1] if len(parsed_action_int) > 1 else None,
                "parsed_r": parsed_action_int[2] if len(parsed_action_int) > 2 else None,
                "parsed_radius_size": parsed_action_int[2] if len(parsed_action_int) > 2 else None,
                "parsed_action": parsed_action_int,
            }
            if gif_path:
                attempt_record["gif_path"] = os.path.join(task_dir_label, os.path.basename(gif_path))

            task_results[task_id]["attempt_history"].append(attempt_record)
            task_results[task_id]["total_attempts"] += 1
            if solved:
                task_results[task_id]["is_solved"] = True

            all_results.append(
                {
                    "task_id": task_id,
                    "attempt": attempt_idx,
                    "response": response_text,
                    "raw_action": parsed_action_int,
                    "normalized_action": normalized_action,
                    "input_dim": input_dims[local_idx],
                    "prompt": item["prompt"],
                    "meta": meta,
                    "simulation_status": simulation_status,
                    "solved": bool(solved),
                }
            )

    total = len(valid_items)
    solved_total = min(total, solved_counter)
    log_label = args.log_dir_label
    if not log_label.endswith("/"):
        log_label += "/"
    summary_results: List[Any] = [
        {
            "model": output_dirs["log_model_name"],
            "eval_setup": args.eval_setups,
            "format": args.format,
            "debug": False,
            "start_id": 0,
            "reuse_log_dir": None,
            "eval_type": args.eval_type,
            "LOG": log_label,
        }
    ]
    for task_id, result in task_results.items():
        task_dir = task_contexts[task_id]["task_dir"]
        save_attempt_history_txt(os.path.join(task_dir, "attempt_history.txt"), result["attempt_history"])
        summary_results.append(result)

    summary = {
        "model_name": output_dirs["log_model_name"],
        "eval_setup": args.eval_setups,
        "fold_id": args.fold_id,
        "num_tasks_configured": total,
        "solved_tasks": solved_total,
        "success_rate": solved_total / total if total else 0.0,
        "dataset_path": args.dataset_path,
    }
    save_json(all_results, os.path.join(output_dirs["base"], "all_results.json"))
    save_json(summary_results, os.path.join(output_dirs["base"], "summary_results.json"))
    save_json(summary, os.path.join(output_dirs["base"], "summary.json"))
    print(f"Finished. Summary saved to {os.path.join(output_dirs['base'], 'summary.json')}")


def run_agent(args: Any, client: Any) -> None:
    run_phyre_agent(args, client)
