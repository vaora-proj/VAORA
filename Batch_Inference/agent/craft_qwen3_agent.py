"""
Batch inference agent for the CRAFT VQA dataset using Qwen3-VL.

Unlike the PHYRE agent this script:
  - Loads images directly from the ``image_path`` field in the dataset JSON
    (no PHYRE simulator involved at all).
  - Extracts ``<final_answer>`` from the VLM response and compares it against
    the ``answer`` field in the dataset to compute accuracy.
  - Writes per-item prompts/responses and a summary JSON to disk.

Usage
-----
python craft_qwen3_agent.py \\
    --dataset_path ../build_dataset/craft_dataset_sid1.json \\
    --model_name   Qwen/Qwen3-VL-7B-Instruct \\
    --log_dir_base /tmp/craft_logs \\
    --batch_size   8
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

BATCH_DIR = Path(__file__).resolve().parent.parent
if str(BATCH_DIR) not in sys.path:
    sys.path.insert(0, str(BATCH_DIR))

from agent.qwen3 import Qwen3VLClient  # noqa: E402


# ── answer helpers ─────────────────────────────────────────────────────────────

_FINAL_ANSWER_RE = re.compile(
    r"<final_answer>\s*(.*?)\s*</final_answer>",
    re.IGNORECASE | re.DOTALL,
)


def extract_final_answer(text: str) -> str:
    """Return content of <final_answer> tag, or the last non-empty line."""
    m = _FINAL_ANSWER_RE.search(text or "")
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def load_image_as_array(image_path: str) -> np.ndarray:
    """Load a PNG from disk as an RGB uint8 numpy array.

    This matches the format that the PHYRE agent uses (phyre.observations_to_float_rgb
    returns float arrays; here we use uint8, but qwen3.py's encode_image handles both).
    Passing arrays instead of file-path strings avoids a deduplication issue in
    process_vision_info: when multiple batch items share the same image file, identical
    base64 strings in the message content blocks can be collapsed into a single image
    tensor, leaving most items without a valid image.  Each np.ndarray object is
    distinct even when loaded from the same file, preventing that collapse.
    """
    return np.array(Image.open(image_path).convert("RGB"))


def normalize_answer(value: Any) -> str:
    """Normalise an answer value to a lowercase string for comparison."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def is_correct(predicted: str, ground_truth: Any) -> bool:
    return normalize_answer(predicted) == normalize_answer(ground_truth)


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _slug(task_id: str) -> str:
    """Convert task_id to a filesystem-safe string."""
    return re.sub(r"[^\w\-]", "_", task_id)


# ── dataset loading ────────────────────────────────────────────────────────────

def _validate_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if not item.get("task_id") or not item.get("prompt") or not item.get("image_path"):
        return None
    return item


def load_dataset(path: str, num_items: int | None = None) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if num_items and num_items > 0:
        data = data[:num_items]
    valid = [_validate_item(item) for item in data]
    return [item for item in valid if item]


# ── log directory helpers ──────────────────────────────────────────────────────

def _normalize_model_name(model_name: str, local_load_path: str | None) -> str:
    source = local_load_path or model_name
    base = os.path.basename(source.rstrip("/")) if source else model_name
    base = base.replace("-Instruct", "").replace("Instruct", "")
    return base or "model"


def prepare_log_dirs(args: argparse.Namespace) -> dict[str, str]:
    log_model_name = args.log_model_name or _normalize_model_name(
        args.model_name, args.local_load_path
    )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(args.log_dir_base, log_model_name, "craft", timestamp)
    os.makedirs(base_dir, exist_ok=True)
    return {"base": base_dir, "log_model_name": log_model_name, "timestamp": timestamp}


# ── core run loop ──────────────────────────────────────────────────────────────

def run_craft_agent(args: argparse.Namespace) -> None:
    output_dirs = prepare_log_dirs(args)
    base_dir = output_dirs["base"]

    dataset = load_dataset(args.dataset_path, args.num_items)
    if not dataset:
        print(f"No valid dataset entries found in {args.dataset_path}")
        return
    print(f"Loaded {len(dataset)} entries from {args.dataset_path}")

    # Check images exist and warn about missing ones
    missing = [item for item in dataset if not Path(item["image_path"]).exists()]
    if missing:
        print(f"WARNING: {len(missing)} entries have missing image files and will be skipped.")
        dataset = [item for item in dataset if Path(item["image_path"]).exists()]

    client = Qwen3VLClient(
        model_name=args.model_name,
        local_load_path=args.local_load_path,
        torch_dtype=args.torch_dtype,
        use_flash_attention=args.use_flash_attention,
        system_prompt=args.system_prompt,
    )

    all_results: list[dict[str, Any]] = []
    correct_count = 0
    total = len(dataset)

    for start in tqdm(range(0, total, args.batch_size), desc="Inference"):
        chunk = dataset[start : start + args.batch_size]

        image_paths = [item["image_path"] for item in chunk]
        prompts = [item["prompt"] for item in chunk]
        print(image_paths)

        responses, _ = client.inference_image(
            images=image_paths,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        for item, response_text in zip(chunk, responses):
            task_id = item["task_id"]
            ground_truth = item.get("answer")
            meta = item.get("meta", {}) or {}

            predicted_raw = extract_final_answer(response_text)
            correct = is_correct(predicted_raw, ground_truth)
            if correct:
                correct_count += 1

            # Per-item log files
            item_dir = os.path.join(base_dir, "items", _slug(task_id))
            _write_text(
                os.path.join(item_dir, "vlm_prompt.txt"),
                f"=== System Prompt ===\n{args.system_prompt}\n\n"
                f"=== User Prompt ===\n{item['prompt']}\n\n"
                f"=== Image ===\n{item['image_path']}\n",
            )
            _write_text(os.path.join(item_dir, "vlm_response.txt"), response_text)

            result: dict[str, Any] = {
                "task_id": task_id,
                "image_path": item["image_path"],
                "question": meta.get("question", ""),
                "ground_truth": ground_truth,
                "predicted_raw": predicted_raw,
                "predicted_normalized": normalize_answer(predicted_raw),
                "ground_truth_normalized": normalize_answer(ground_truth),
                "correct": correct,
                "vlm_response": response_text,
                "meta": meta,
            }
            all_results.append(result)

    accuracy = correct_count / total if total else 0.0
    summary = {
        "model_name": output_dirs["log_model_name"],
        "dataset_path": args.dataset_path,
        "total_items": total,
        "correct": correct_count,
        "accuracy": round(accuracy, 4),
        "timestamp": output_dirs["timestamp"],
        "args": {
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
    }

    _write_json(os.path.join(base_dir, "all_results.json"), all_results)
    _write_json(os.path.join(base_dir, "summary.json"), summary)

    print(
        json.dumps(
            {
                "output_dir": base_dir,
                "total_items": total,
                "correct": correct_count,
                "accuracy": round(accuracy, 4),
            },
            indent=2,
        )
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    default_dataset = str(
        BATCH_DIR / "build_dataset" / "craft_dataset_sid1.json"
    )
    parser = argparse.ArgumentParser(
        description="Batch Qwen3-VL inference agent for the CRAFT VQA dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=default_dataset,
        help="Path to the craft dataset JSON built by build_craft_dataset.py.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-7B-Instruct",
        help="HuggingFace model ID for the Qwen3-VL model.",
    )
    parser.add_argument(
        "--local_load_path",
        type=str,
        default=None,
        help="Path to a local model checkpoint; overrides --model_name for loading.",
    )
    parser.add_argument(
        "--log_model_name",
        type=str,
        default=None,
        help="Short name used in log directory path. Inferred from model if omitted.",
    )
    parser.add_argument(
        "--log_dir_base",
        type=str,
        default=str(BATCH_DIR / "batch_inference_output" / "craft"),
        help="Root directory for all output logs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of (image, prompt) pairs per inference call.",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=None,
        help="Optional cap on the number of dataset entries to process.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful physics reasoning assistant.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype for model weights (bfloat16, float16, float32, auto).",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Enable Flash Attention 2 (requires flash-attn package).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_craft_agent(parse_args())
