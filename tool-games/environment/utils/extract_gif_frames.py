#!/usr/bin/env python3
"""Extract all frames from an input GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageSequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract each frame from a GIF into PNG files."
    )
    parser.add_argument("gif_path", type=Path, help="Path to the source GIF file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for extracted frames. Defaults to <gif_name>_frames next to GIF.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Filename prefix for exported frames. Default: frame",
    )
    return parser.parse_args()


def extract_frames(gif_path: Path, output_dir: Path, prefix: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(gif_path) as gif:
        frame_count = 0
        for i, frame in enumerate(ImageSequence.Iterator(gif)):
            frame_path = output_dir / f"{prefix}_{i:04d}.png"
            frame.convert("RGBA").save(frame_path, "PNG")
            frame_count += 1
    return frame_count


def main() -> None:
    args = parse_args()
    gif_path = args.gif_path

    if not gif_path.exists():
        raise FileNotFoundError(f"GIF not found: {gif_path}")
    if gif_path.suffix.lower() != ".gif":
        raise ValueError(f"Expected a .gif file, got: {gif_path}")

    output_dir = args.output_dir or gif_path.with_name(f"{gif_path.stem}_frames")
    frame_count = extract_frames(gif_path=gif_path, output_dir=output_dir, prefix=args.prefix)

    print(f"Extracted {frame_count} frames to: {output_dir}")


if __name__ == "__main__":
    main()
