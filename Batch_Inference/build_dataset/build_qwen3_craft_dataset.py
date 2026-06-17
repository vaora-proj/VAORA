#!/usr/bin/env python3
"""Build a CRAFT dataset JSON for VLM batch inference.

Each entry combines:
  - The craft_prompt from prompt_generator/prompt.py with <QUESTION_TEXT> filled in
  - The absolute path to the first-frame PNG corresponding to the video
  - The ground-truth answer

Usage example
-------------
python build_craft_dataset.py \
    --dataset_json /home/var2025/repo/craft/dataset_sid1.json \
    --frames_dir   /home/var2025/repo/craft/frames \
    --output_path  /home/var2025/repo/Batch_Inference/build_dataset/craft_dataset_sid1.json

Only entries whose corresponding frame PNG exists on disk are included.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ── importability ──────────────────────────────────────────────────────────────
BATCH_DIR = Path(__file__).resolve().parent.parent
if str(BATCH_DIR) not in sys.path:
    sys.path.insert(0, str(BATCH_DIR))

from prompt_generator.prompt import craft_prompt as CRAFT_PROMPT  # noqa: E402

_QUESTION_PLACEHOLDER = "<QUESTION_TEXT>"


# ── helpers ────────────────────────────────────────────────────────────────────

def _video_path_to_frame(video_path: str, frames_dir: Path) -> Path:
    """Map a video_path like './videos/sid_1/000000.mpg' to an absolute frame PNG.

    The mapping strips the leading './videos/' prefix, replaces the extension
    with '.png', and resolves against *frames_dir*:
        ./videos/sid_1/000000.mpg  →  <frames_dir>/sid_1/000000.png
    """
    # Normalise path separators and strip any leading './'
    normalised = video_path.replace("\\", "/").lstrip("./")
    # Remove a leading 'videos/' segment if present
    normalised = re.sub(r"^videos/", "", normalised)
    # Swap extension
    stem = os.path.splitext(normalised)[0]
    return frames_dir / f"{stem}.png"


def _task_id_from_video(video_path: str, question_idx: int) -> str:
    """Build a stable task_id from the video path and question index.

    e.g. './videos/sid_1/000000.mpg', 3  →  'sid_1/000000:3'
    """
    normalised = video_path.replace("\\", "/").lstrip("./")
    normalised = re.sub(r"^videos/", "", normalised)
    stem = os.path.splitext(normalised)[0]        # e.g. 'sid_1/000000'
    return f"{stem}:{question_idx}"


def _fill_prompt(question: str) -> str:
    if _QUESTION_PLACEHOLDER not in CRAFT_PROMPT:
        raise ValueError(
            f"craft_prompt does not contain the placeholder '{_QUESTION_PLACEHOLDER}'."
        )
    return CRAFT_PROMPT.replace(_QUESTION_PLACEHOLDER, question)


# ── core builder ───────────────────────────────────────────────────────────────

def build_dataset(
    dataset_json: Path,
    frames_dir: Path,
    num_items: int | None = None,
    skip_missing: bool = True,
) -> list[dict[str, Any]]:
    """Load *dataset_json*, resolve frames, and produce dataset entries."""
    with open(dataset_json, encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {dataset_json}")

    # Group by video_path to assign stable per-video question indices
    question_counter: dict[str, int] = {}

    dataset: list[dict[str, Any]] = []
    skipped = 0

    for entry in raw:
        video_path: str = entry.get("video_path", "")
        question: str = entry.get("question", "")
        answer: Any = entry.get("answer")
        meta_info: Any = entry.get("meta_info", {})

        if not video_path or not question:
            skipped += 1
            continue

        # Increment question index per video
        q_idx = question_counter.get(video_path, 0)
        question_counter[video_path] = q_idx + 1

        frame_path = _video_path_to_frame(video_path, frames_dir)

        if not frame_path.exists():
            if skip_missing:
                skipped += 1
                continue
            # Still include entry but flag the missing image
            image_path = str(frame_path)
        else:
            image_path = str(frame_path.resolve())

        task_id = _task_id_from_video(video_path, q_idx)
        prompt = _fill_prompt(question)

        item: dict[str, Any] = {
            "task_id": task_id,
            "prompt": prompt,
            "image_path": image_path,
            "answer": answer,
            "meta_info": meta_info if isinstance(meta_info, dict) else {"question_type": meta_info},
        }
        dataset.append(item)

        if num_items is not None and len(dataset) >= num_items:
            break

    if skipped:
        print(f"[build_craft_dataset] Skipped {skipped} entries (missing frames or empty fields).")

    return dataset


def save_dataset(dataset: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    repo_root = BATCH_DIR.parent
    default_dataset = repo_root / "craft" / "dataset_sid1.json"
    default_frames = repo_root / "craft" / "frames"
    default_output = Path(__file__).with_name("craft_dataset_sid1.json")

    parser = argparse.ArgumentParser(
        description="Build a CRAFT VQA dataset JSON for VLM batch inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_json",
        type=Path,
        default=default_dataset,
        help="Path to the CRAFT source JSON (e.g. dataset_sid1.json).",
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        default=default_frames,
        help="Root directory containing per-split frame PNGs (e.g. craft/frames).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=default_output,
        help="Destination path for the built dataset JSON.",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=None,
        help="Optional cap on the number of dataset entries to include.",
    )
    parser.add_argument(
        "--include_missing",
        action="store_true",
        help="Include entries even when the corresponding frame PNG is not found on disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_json: Path = args.dataset_json.resolve()
    frames_dir: Path = args.frames_dir.resolve()
    output_path: Path = args.output_path.resolve()

    if not dataset_json.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    dataset = build_dataset(
        dataset_json=dataset_json,
        frames_dir=frames_dir,
        num_items=args.num_items,
        skip_missing=not args.include_missing,
    )

    save_dataset(dataset, output_path)

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "num_items": len(dataset),
                "dataset_json": str(dataset_json),
                "frames_dir": str(frames_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
