import argparse
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional

import phyre

from phyre.metrics import (
    EvalSetup,
    TRAIN_SHARE,
    _register_eval_setup_builder,
    get_task_ids_in_tier,
    _get_task_per_tpl,
    create_dev_set,
)
@_register_eval_setup_builder
def my_template_based_split(seed=1, dev_seed=None) -> EvalSetup:
    """Custom split based on specific template IDs."""
    # Get all ball tasks
    all_task_ids = get_task_ids_in_tier('ball')
    
    test_template = {
        1: {'00001', '00002', '00003','00006', '00013'},
        2: {'00000', '00003', '00001', '00014', '00012'},
        3: {'00000', '00001', '00002', '00009', '00015'},
    }
    test_template_ids = test_template[seed]
    # Split tasks by template
    train_ids = []
    test_ids = []
    
    for task_id in all_task_ids:
        template_id = task_id.split(':')[0]
        if template_id in test_template_ids:
            test_ids.append(task_id)
        else:
            train_ids.append(task_id)
    
    
    # Handle dev split if needed
    if dev_seed is not None:
        train_ids = phyre.util.stable_shuffle(
            train_ids, f'dev_split_{dev_seed}')
        train_size = int(len(train_ids) * TRAIN_SHARE)
        train_ids, dev_ids = train_ids[:train_size], train_ids[train_size:]
        return [(tuple(train_ids), [tuple(dev_ids)])]
    
    return [(tuple(train_ids), [tuple(test_ids)])]

# Make batch_inference importable when running as a script from repo root.
BATCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BATCH_DIR not in sys.path:
    sys.path.append(BATCH_DIR)

from prompt_generator.prompt import prompt as BASE_PROMPT
from prompt_generator.prompt import prompt_w_hint as BASE_PROMPT_W_HINT

_OVERALL_HINT_RE = re.compile(r"<overall_hint>\s*(.*?)\s*</overall_hint>", re.DOTALL)
_HINT_PLACEHOLDER = "<HINT>"


def _extract_overall_hint(gpt_text: str) -> str:
    m = _OVERALL_HINT_RE.search(gpt_text or "")
    if m:
        return m.group(1).strip()
    return (gpt_text or "").strip()


def load_hints_by_task_type(json_path: str) -> Dict[str, List[str]]:
    """Load SFT JSON; group overall hints by task_type (e.g. 00000). Multiple rows per type => multiple hints."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {json_path}")
    by_type: Dict[str, List[str]] = {}
    for entry in data:
        tt = entry.get("task_type")
        if not tt:
            continue
        convs = entry.get("conversations") or []
        gpt_val = None
        for c in convs:
            if c.get("from") == "gpt":
                gpt_val = c.get("value")
                break
        if not gpt_val:
            continue
        hint = _extract_overall_hint(gpt_val)
        if hint:
            by_type.setdefault(str(tt), []).append(hint)
    return by_type


def _task_type_from_task_id(task_id: str) -> str:
    return task_id.split(":", 1)[0]


def _prompt_with_hint(
    template: str,
    hint_text: str,
) -> str:
    if _HINT_PLACEHOLDER not in template:
        raise ValueError(
            f"Hint template must contain {_HINT_PLACEHOLDER} for substitution."
        )
    return template.replace(_HINT_PLACEHOLDER, hint_text)


def _select_tasks(eval_setups: str, fold_id: int, eval_type: str) -> List[str]:
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


def build_dataset(
    eval_setups: str,
    fold_id: int,
    eval_type: str,
    prompt_text: str,
    num_tasks: int | None = None,
    repeat_num: int = 1,
    *,
    use_hint: bool = False,
    hints_by_task_type: Optional[Dict[str, List[str]]] = None,
    hint_prompt_template: Optional[str] = None,
    hint_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    task_ids = _select_tasks(eval_setups, fold_id, eval_type)

    if num_tasks is not None and num_tasks > 0:
        task_ids = task_ids[:num_tasks]

    rng = random.Random(hint_seed) if hint_seed is not None else random

    dataset = []
    repeat = max(1, int(repeat_num))
    if eval_setups == "my_template_based_split":
        eval_setups = "ball_within_template"

    if use_hint:
        if not hints_by_task_type:
            raise ValueError("use_hint=True requires hints_by_task_type.")
        tpl = hint_prompt_template or BASE_PROMPT_W_HINT
        for task_id in task_ids:
            tt = _task_type_from_task_id(task_id)
            pool = hints_by_task_type.get(tt)
            if not pool:
                keys = sorted(hints_by_task_type.keys())
                preview = keys[:20]
                suffix = f" (+{len(keys) - 20} more)" if len(keys) > 20 else ""
                raise ValueError(
                    f"No hints for task_type={tt!r} (task_id={task_id!r}). "
                    f"Known types ({len(keys)}): {preview}{suffix}"
                )
            hint_text = rng.choice(pool)
            final_prompt = _prompt_with_hint(tpl, hint_text)
            item = {
                "task_id": task_id,
                "prompt": final_prompt,
                "meta": {
                    "eval_setup": eval_setups,
                    "eval_type": eval_type,
                    "fold_id": fold_id,
                    "task_type": tt,
                },
            }
            for _ in range(repeat):
                dataset.append(item.copy())
        return dataset

    for task_id in task_ids:
        item = {
            "task_id": task_id,
            "prompt": prompt_text,
            "meta": {
                "eval_setup": eval_setups,
                "eval_type": eval_type,
                "fold_id": fold_id,
            },
        }
        for _ in range(repeat):
            dataset.append(item.copy())
    return dataset


def save_dataset(dataset: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PHYRE dataset for Qwen3-VL agent.")
    parser.add_argument("--eval_setups", type=str, default="my_template_based_split")
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument("--eval_type", type=str, default="test")
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "qwen3_dataset.json"),
        help="Path to write dataset JSON.",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=None,
        help="Optional cap on number of tasks to include.",
    )
    parser.add_argument(
        "--repeat_num",
        type=int,
        default=1,
        help="Repeat each dataset item sequentially this many times.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Optional path to a prompt file; overrides prompt.py contents.",
    )
    parser.add_argument(
        "--hint",
        action="store_true",
        help="Use BASE_PROMPT_W_HINT and inject a random overall hint per task from --hints_json.",
    )
    default_hints = os.path.abspath(
        os.path.join(
            os.path.dirname(BATCH_DIR),
            "verl_new",
            "data",
            "sft_dataset_gemini_for_overall_hint_by_task_type.json",
        )
    )
    parser.add_argument(
        "--hints_json",
        type=str,
        default=default_hints,
        help="JSON with task_type and gpt overall_hint (default: verl_new/data/...).",
    )
    parser.add_argument(
        "--hint_seed",
        type=int,
        default=None,
        help="RNG seed for choosing hints per task (default: nondeterministic).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_text = BASE_PROMPT
    if args.prompt_path and not args.hint:
        with open(args.prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()

    hints_by_task_type = None
    hint_prompt_template = BASE_PROMPT_W_HINT
    if args.hint:
        if args.prompt_path:
            with open(args.prompt_path, "r", encoding="utf-8") as f:
                hint_prompt_template = f.read()
        hints_by_task_type = load_hints_by_task_type(args.hints_json)

    dataset = build_dataset(
        eval_setups=args.eval_setups,
        fold_id=args.fold_id,
        eval_type=args.eval_type,
        prompt_text=prompt_text,
        num_tasks=args.num_tasks,
        repeat_num=args.repeat_num,
        use_hint=args.hint,
        hints_by_task_type=hints_by_task_type,
        hint_prompt_template=hint_prompt_template if args.hint else None,
        hint_seed=args.hint_seed,
    )
    save_dataset(dataset, args.output_path)
    print(
        json.dumps(
            {
                "output_path": args.output_path,
                "num_items": len(dataset),
                "eval_setup": args.eval_setups,
                "eval_type": args.eval_type,
                "fold_id": args.fold_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
