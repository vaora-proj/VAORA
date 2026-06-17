#!/usr/bin/env python3
"""Evaluate per-tool attempt success from ToolPicker all_results.json files.

This script reads one or more ToolPicker `all_results.json` files, extracts the
success/failure outcome for each task, tool, and within-tool attempt index, then:

1. Writes a detailed CSV with one row per task/tool/attempt.
2. Writes a summary CSV with aggregated success rates across tasks.
3. Saves a comparison plot with one line per input JSON file.

Example:
    python eval_toolpicker_attempt_success.py \
      --run qwen artifacts/vlm_toolpicker/Original/<timestamp>/all_results.json \
      --run gemini artifacts/vlm_toolpicker/Original/<timestamp>/all_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

_ENV_DIR = Path(__file__).resolve().parent
if str(_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(_ENV_DIR))
from paths import artifact_path  # noqa: E402


DetailRow = Dict[str, Any]
SummaryRow = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ToolPicker success rates for attempt 1/2/3 of each tool "
            "decision and plot one or more all_results.json runs."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "ALL_RESULTS_JSON"),
        required=True,
        help="Run label and path to all_results.json (repeat for multiple runs).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(artifact_path("vlm_toolpicker_eval")),
        help="Directory for CSV summaries and the output plot.",
    )
    parser.add_argument(
        "--plot_name",
        default="attempt_success_rates.png",
        help="Filename for the output plot inside --output_dir.",
    )
    parser.add_argument(
        "--max_attempt_index",
        type=int,
        default=3,
        help="Maximum within-tool attempt index to include in the summary plot.",
    )
    return parser.parse_args()


def discover_input_files(input_json_dict: Dict[str, str]) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for run_label, raw_path in input_json_dict.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Expected an all_results.json file: {path}")
        files.append((run_label, path))
    if not files:
        raise FileNotFoundError("No input all_results.json files found.")
    return files


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}, got {type(payload).__name__}")
    return payload


def make_run_label(path: Path) -> str:
    parent = path.parent.name
    if parent and parent != path.stem:
        return parent
    return path.stem


def natural_sort_key(text: str) -> Tuple[Any, ...]:
    parts: List[Any] = []
    current = ""
    is_digit = None
    for char in text:
        char_is_digit = char.isdigit()
        if is_digit is None or char_is_digit == is_digit:
            current += char
        else:
            parts.append(int(current) if is_digit else current.lower())
            current = char
        is_digit = char_is_digit
    if current:
        parts.append(int(current) if is_digit else current.lower())
    return tuple(parts)


def extract_detail_rows(path: Path, run_label: str) -> List[DetailRow]:
    results = load_results(path)
    detail_rows: List[DetailRow] = []

    for level in results:
        task_name = level.get("level_name")
        attempts = level.get("attempts", [])
        if not isinstance(task_name, str) or not isinstance(attempts, list):
            continue

        tool_occurrence_counts: Dict[str, int] = defaultdict(int)
        tool_first_order: Dict[str, int] = {}

        for attempt in attempts:
            tool_name = attempt.get("forced_tool")
            if not isinstance(tool_name, str):
                continue

            if tool_name not in tool_first_order:
                tool_first_order[tool_name] = len(tool_first_order) + 1

            tool_occurrence_counts[tool_name] += 1
            within_tool_attempt = tool_occurrence_counts[tool_name]

            detail_rows.append(
                {
                    "run_label": run_label,
                    "source_file": str(path),
                    "task_name": task_name,
                    "tool_name": tool_name,
                    "tool_order": tool_first_order[tool_name],
                    "global_attempt_number": attempt.get("attempt_number"),
                    "within_tool_attempt": within_tool_attempt,
                    "status": attempt.get("status", ""),
                    "success": 1 if attempt.get("solved", False) else 0,
                }
            )

    return detail_rows


def summarize_detail_rows(
    detail_rows: Iterable[DetailRow], max_attempt_index: int
) -> List[SummaryRow]:
    grouped: Dict[Tuple[str, str, int, int], List[int]] = defaultdict(list)

    for row in detail_rows:
        within_tool_attempt = row["within_tool_attempt"]
        if within_tool_attempt > max_attempt_index:
            continue
        key = (
            row["run_label"],
            row["tool_name"],
            row["tool_order"],
            within_tool_attempt,
        )
        grouped[key].append(int(row["success"]))

    summary_rows: List[SummaryRow] = []
    for (run_label, tool_name, tool_order, within_tool_attempt), outcomes in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][2], natural_sort_key(item[0][1]), item[0][3]),
    ):
        total = len(outcomes)
        success_rate = (sum(outcomes) / total) if total else math.nan
        summary_rows.append(
            {
                "run_label": run_label,
                "tool_name": tool_name,
                "tool_order": tool_order,
                "within_tool_attempt": within_tool_attempt,
                "num_tasks": total,
                "num_successes": sum(outcomes),
                "success_rate": success_rate,
            }
        )

    return summary_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows: Sequence[SummaryRow], output_path: Path) -> None:
    if not summary_rows:
        raise ValueError("No summary rows available to plot.")

    run_labels = sorted({row["run_label"] for row in summary_rows}, key=natural_sort_key)
    categories = sorted(
        {
            (row["tool_order"], row["tool_name"], row["within_tool_attempt"])
            for row in summary_rows
        },
        key=lambda item: (item[0], natural_sort_key(item[1]), item[2]),
    )

    category_labels = [f"{tool_name}-A{attempt_idx}" for _, tool_name, attempt_idx in categories]
    category_to_x = {category: idx for idx, category in enumerate(categories)}
    summary_map = {
        (
            row["run_label"],
            row["tool_order"],
            row["tool_name"],
            row["within_tool_attempt"],
        ): row["success_rate"]
        for row in summary_rows
    }

    plt.figure(figsize=(max(10, len(categories) * 1.2), 6))
    for run_label in run_labels:
        y_values = [
            summary_map.get((run_label, tool_order, tool_name, attempt_idx), math.nan)
            for tool_order, tool_name, attempt_idx in categories
        ]
        x_values = [category_to_x[category] for category in categories]
        plt.plot(x_values, y_values, marker="o", linewidth=2, label=run_label)

    plt.xticks(range(len(category_labels)), category_labels, rotation=45, ha="right")
    plt.ylim(-0.02, 1.02)
    plt.ylabel("Success rate")
    plt.xlabel("Tool decision / within-tool attempt")
    plt.title("ToolPicker success rate by tool and attempt")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    input_files = discover_input_files(dict(args.run))

    detail_rows: List[DetailRow] = []
    for run_label, input_file in input_files:
        detail_rows.extend(extract_detail_rows(input_file, run_label))

    if not detail_rows:
        raise ValueError("No task attempts were found in the provided inputs.")

    summary_rows = summarize_detail_rows(detail_rows, args.max_attempt_index)
    output_dir = args.output_dir.resolve()

    write_csv(
        output_dir / "per_task_attempt_results.csv",
        detail_rows,
        [
            "run_label",
            "source_file",
            "task_name",
            "tool_name",
            "tool_order",
            "global_attempt_number",
            "within_tool_attempt",
            "status",
            "success",
        ],
    )
    write_csv(
        output_dir / "summary_success_rates.csv",
        summary_rows,
        [
            "run_label",
            "tool_name",
            "tool_order",
            "within_tool_attempt",
            "num_tasks",
            "num_successes",
            "success_rate",
        ],
    )
    plot_summary(summary_rows, output_dir / args.plot_name)

    print(f"Processed {len(input_files)} input file(s).")
    print(f"Wrote detail CSV: {output_dir / 'per_task_attempt_results.csv'}")
    print(f"Wrote summary CSV: {output_dir / 'summary_success_rates.csv'}")
    print(f"Wrote plot: {output_dir / args.plot_name}")


if __name__ == "__main__":
    main()
