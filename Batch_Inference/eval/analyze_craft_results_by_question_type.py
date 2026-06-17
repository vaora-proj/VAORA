#!/usr/bin/env python3
"""
Join CRAFT batch inference results with the build dataset (by task_id), read
question_type directly from dataset metadata, and report accuracy per type.

Example:
  python3 analyze_craft_results_by_question_type.py \\
    --results ../batch_inference_output/craft/Qwen3-VL-8B-Instruct/craft/2026-04-15_15-40-57/all_results.json

  Writes summary_per_type.json next to all_results.json unless --summary-json is set.
  Default --dataset: craft_dataset_1000_merged_infer.json under
  Batch_Inference/build_dataset/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Default CRAFT build JSON (task_id + meta_info.question_type).
_DEFAULT_CRAFT_DATASET = Path(
    "/home/u5597173/repo/Batch_Inference/build_dataset/craft_dataset_1000_merged_infer.json"
)


def extract_question_from_prompt(prompt: str) -> str:
    """Return the sentence after 'Question:' in the INPUT section (normalized whitespace)."""
    if not prompt:
        return ""
    m = re.search(
        r"Question:\s*(.+?)(?=\n\s*\n\*\*TASK|\n\*\*TASK)",
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        return " ".join(m.group(1).split()).strip()
    m2 = re.search(r"Question:\s*(.+)", prompt, flags=re.IGNORECASE)
    if m2:
        line = m2.group(1).strip()
        line = line.split("\n", 1)[0].strip()
        return " ".join(line.split()).strip()
    return ""


def load_task_metadata(dataset_path: Path) -> Dict[str, Dict[str, str]]:
    with dataset_path.open() as f:
        rows: List[Dict[str, Any]] = json.load(f)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        tid = row.get("task_id")
        if tid is not None:
            meta_info = row.get("meta_info") or {}
            out[str(tid)] = {
                "prompt": row.get("prompt") or "",
                "question_type": str(meta_info.get("question_type") or "").strip(),
            }
    return out


def analyze(
    results: List[Dict[str, Any]],
    task_id_to_meta: Optional[Dict[str, Dict[str, str]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """
    Returns (per_row_annotations, aggregates) where aggregates[qtype] =
    {"n": int, "correct": int}.
    """
    aggregates: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    annotated: List[Dict[str, Any]] = []

    missing_prompt = 0
    missing_question_type = 0
    for row in results:
        tid = str(row.get("task_id", ""))
        ds_meta = task_id_to_meta.get(tid, {}) if task_id_to_meta is not None else {}
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            prompt = str(ds_meta.get("prompt", ""))
        if not prompt:
            missing_prompt += 1

        question = extract_question_from_prompt(prompt)
        qtype = str(ds_meta.get("question_type", "")).strip() or "Unknown"
        if qtype == "Unknown":
            missing_question_type += 1
        correct = bool(row.get("correct"))

        aggregates[qtype]["n"] += 1
        if correct:
            aggregates[qtype]["correct"] += 1

        annotated.append(
            {
                "task_id": tid,
                "question": question,
                "question_type": qtype,
                "correct": correct,
            }
        )

    if missing_prompt:
        print(f"Warning: {missing_prompt} rows had no prompt (question text may be empty).", file=sys.stderr)
    if missing_question_type:
        print(
            f"Warning: {missing_question_type} rows had no dataset question_type (counted as Unknown).",
            file=sys.stderr,
        )

    return annotated, dict(aggregates)


def print_table(aggregates: Dict[str, Dict[str, int]], total_rows: int, total_correct: int) -> None:
    order = ["Descriptive", "Counterfactual", "Enable", "Cause", "Prevent", "Unknown"]
    print(f"{'question_type':<18} {'count':>8} {'correct':>8} {'accuracy':>10}")
    for k in order:
        if k not in aggregates:
            continue
        v = aggregates[k]
        n, c = v["n"], v["correct"]
        acc = f"{100.0 * c / n:.2f}%" if n else "n/a"
        print(f"{k:<18} {n:>8} {c:>8} {acc:>10}")
    print()
    print(f"Overall: {100.0 * total_correct / total_rows:.2f}% ({total_correct}/{total_rows})")


def write_summary_json(path: Path, aggregates: Dict[str, Any], meta: Dict[str, Any]) -> None:
    payload = {"meta": meta, "by_question_type": aggregates}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_per_row_csv(path: Path, rows: Iterable[Dict[str, Any]], results: List[Dict[str, Any]]) -> None:
    """Merge annotation fields with a few original columns for auditing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_tid = {str(r.get("task_id")): r for r in results}
    fieldnames = [
        "task_id",
        "question_type",
        "question",
        "correct",
        "ground_truth",
        "predicted_normalized",
        "image_path",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for ann in rows:
            tid = ann["task_id"]
            base = by_tid.get(tid, {})
            w.writerow(
                {
                    "task_id": tid,
                    "question_type": ann["question_type"],
                    "question": ann["question"],
                    "correct": ann["correct"],
                    "ground_truth": base.get("ground_truth", ""),
                    "predicted_normalized": base.get("predicted_normalized", ""),
                    "image_path": base.get("image_path", ""),
                }
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to all_results.json (list of dicts with task_id, correct, optional prompt).",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_CRAFT_DATASET,
        help=(
            "Path to craft_dataset_*.json with task_id, meta_info.question_type, and prompt "
            "(prompt is used only for per-row CSV question text). "
            f"Default: {_DEFAULT_CRAFT_DATASET}"
        ),
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Override path for aggregate JSON. Default: <directory of --results>/summary_per_type.json",
    )
    p.add_argument(
        "--no-summary-json",
        action="store_true",
        help="Do not write summary_per_type.json (or any summary file).",
    )
    p.add_argument(
        "--per-row-csv",
        type=Path,
        default=None,
        help="Write task_id, question_type, question, correct, and key result columns to CSV.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with args.results.open() as f:
        results: List[Dict[str, Any]] = json.load(f)

    if not args.dataset.is_file():
        print(f"Error: dataset file not found: {args.dataset}", file=sys.stderr)
        return 1
    task_id_to_meta = load_task_metadata(args.dataset)

    annotated, aggregates = analyze(results, task_id_to_meta)
    total_correct = sum(1 for r in results if r.get("correct"))
    print_table(aggregates, len(results), total_correct)

    meta = {
        "results_path": str(args.results.resolve()),
        "dataset_path": str(args.dataset.resolve()) if args.dataset else None,
        "num_rows": len(results),
        "total_correct": total_correct,
        "overall_accuracy": total_correct / len(results) if results else 0.0,
    }

    if not args.no_summary_json:
        summary_path = args.summary_json if args.summary_json is not None else (
            args.results.parent / "summary_per_type.json"
        )
        out_agg = {}
        for qtype, v in aggregates.items():
            n, c = v["n"], v["correct"]
            out_agg[qtype] = {
                "count": n,
                "correct": c,
                "accuracy": (c / n) if n else None,
            }
        write_summary_json(summary_path, out_agg, meta)
        print(f"Wrote summary: {summary_path}")

    if args.per_row_csv:
        write_per_row_csv(args.per_row_csv, annotated, results)
        print(f"Wrote per-row CSV: {args.per_row_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
