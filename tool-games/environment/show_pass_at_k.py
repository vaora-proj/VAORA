#!/usr/bin/env python3
"""ToolPicker pass@K from ``all_results.json`` (within-tool attempt index N).

Schedule (see ``agent/vlm_toolpicker_agent.py``): each task uses several tools in
order, and each tool gets the same number of tries ``m`` (``--max-attempts``,
often 5). Global ``attempt_number`` in the JSON runs 1 … T×m in that fixed order.

Definition of N for pass@K (your setting):
  When the task is first solved on global attempt G using some tool, define
  **N = (G - 1) % m + 1**, i.e. the **1-based attempt index within that tool’s
  block** (1 … m), not G itself.

  Example: global G=7 with m=5 is the 2nd try on the 2nd tool → N=2 → counts
  toward pass@2 (and pass@3, …).

pass@K:
  (# tasks with first success and N ≤ K) / (# tasks). Unsolved tasks → 0 for every K.

With m=5, pass@5 over solved tasks aligns with “solved before exhausting any tool’s
quota on the successful tool”; pass@5 rate 4/18 matches ``summary.json`` success
rate for runs where every solve has N≤5 (as in the sample Original run).

Optional ``--metric global`` keeps the older behaviour (N = G).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Tuple


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON array at top level, got {type(data).__name__}")
    return data


def infer_attempts_per_tool(levels: List[Dict[str, Any]]) -> Optional[int]:
    """Infer m from the first global attempt where ``forced_tool`` changes (m = G-1)."""
    for lev in levels:
        attempts = lev.get("attempts") or []
        if len(attempts) < 2:
            continue
        first_tool = attempts[0].get("forced_tool")
        for att in attempts[1:]:
            if att.get("forced_tool") != first_tool:
                g = att.get("attempt_number")
                if isinstance(g, int) and g >= 2:
                    return int(g) - 1
    return None


def first_solve_global_attempt(level: Dict[str, Any]) -> Optional[int]:
    solved_ns: List[int] = []
    for att in level.get("attempts") or []:
        if att.get("solved") is not True:
            continue
        n = att.get("attempt_number")
        if n is None:
            continue
        solved_ns.append(int(n))
    return min(solved_ns) if solved_ns else None


def global_to_within_tool(global_n: int, attempts_per_tool: int) -> int:
    if attempts_per_tool <= 0:
        raise ValueError("attempts_per_tool must be positive")
    return (int(global_n) - 1) % attempts_per_tool + 1


def first_solve_pass_n(
    level: Dict[str, Any],
    attempts_per_tool: int,
    metric: Literal["within_tool", "global"],
) -> Optional[int]:
    """N used for pass@K: within-tool index, or global G."""
    g = first_solve_global_attempt(level)
    if g is None:
        return None
    if metric == "global":
        return g
    return global_to_within_tool(g, attempts_per_tool)


def first_solve_details(
    level: Dict[str, Any], attempts_per_tool: int, metric: Literal["within_tool", "global"]
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Returns (N_for_pass, global_G, forced_tool_at_success)."""
    g = first_solve_global_attempt(level)
    if g is None:
        return None, None, None
    tool = None
    for att in level.get("attempts") or []:
        if att.get("attempt_number") == g and att.get("forced_tool") is not None:
            tool = str(att.get("forced_tool"))
            break
    n = g if metric == "global" else global_to_within_tool(g, attempts_per_tool)
    return n, g, tool


def level_pass_at_k(
    level: Dict[str, Any],
    k: int,
    attempts_per_tool: int,
    metric: Literal["within_tool", "global"],
) -> bool:
    n0 = first_solve_pass_n(level, attempts_per_tool, metric)
    return n0 is not None and n0 <= k


def max_attempt_number(level: Dict[str, Any]) -> int:
    nums = [int(a["attempt_number"]) for a in (level.get("attempts") or []) if "attempt_number" in a]
    return max(nums) if nums else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_json",
        type=Path,
        help="Path to all_results.json from a VLM ToolPicker run.",
    )
    parser.add_argument(
        "--attempts-per-tool",
        type=int,
        default=5,
        metavar="M",
        help="m in N=(G-1)%%m+1 when using within-tool metric (default: 5).",
    )
    parser.add_argument(
        "--infer-attempts-per-tool",
        action="store_true",
        help="Override --attempts-per-tool by scanning for first forced_tool change (if found).",
    )
    parser.add_argument(
        "--metric",
        choices=("within_tool", "global"),
        default="within_tool",
        help="within_tool: N=(G-1)%%m+1 (default). global: N=G (JSON attempt_number).",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="K values for pass@K (default: 1 3 5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Per task: N, global G, tool at success, final solved.",
    )
    args = parser.parse_args()
    path = args.results_json.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    levels = load_results(path)
    n = len(levels)
    if n == 0:
        print(f"{path}: no levels")
        return

    m = int(args.attempts_per_tool)
    if args.infer_attempts_per_tool:
        inferred = infer_attempts_per_tool(levels)
        if inferred is not None:
            m = inferred
            print(f"inferred attempts_per_tool m={m} (from first tool switch in logs)")
        else:
            print(f"could not infer m from tool switches; using --attempts-per-tool={m}")

    ks = sorted(set(args.ks))
    print(f"file: {path}")
    print(f"levels: {n}")
    print(f"metric: {args.metric}" + (f", attempts_per_tool m={m}" if args.metric == "within_tool" else ""))
    if args.metric == "within_tool":
        print(
            "pass@K = (# tasks with first success and N=(G-1)%m+1 ≤ K) / (# tasks), "
            "G = global attempt_number on the winning row."
        )
    else:
        print("pass@K = (# tasks with first success and G ≤ K) / (# tasks).")

    for k in ks:
        passed = sum(1 for lev in levels if level_pass_at_k(lev, k, m, args.metric))
        rate = passed / n
        print(f"pass@{k}: {passed}/{n} = {rate:.6f}")

    if args.verbose:
        print("\nper-level: N (for pass@*), global G, tool, final_solved")
        for i, lev in enumerate(levels):
            name = lev.get("level_name") or lev.get("level_path") or f"level_{i}"
            n_pass, g, tool = first_solve_details(lev, m, args.metric)
            print(
                f"  [{i}] {name}: N={n_pass} G={g} tool={tool} "
                f"max_G={max_attempt_number(lev)} final_solved={lev.get('solved')}"
            )


if __name__ == "__main__":
    main()
