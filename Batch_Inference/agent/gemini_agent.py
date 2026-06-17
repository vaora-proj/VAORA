"""
Batch inference agent using Google Gemini vision-language API.

Usage:
    GEMINI_API_KEY=<key> python gemini_agent.py \
        --env_type phyre \
        --model_name gemini-1.5-flash \
        --dataset_path ../build_dataset/qwen3_dataset_my_cross_fold_1.json \
        --log_dir_base /tmp/gemini_logs/phyre \
        --log_dir_label tmp_log/gemini/
"""

import argparse
import os
import sys

BATCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BATCH_DIR not in sys.path:
    sys.path.append(BATCH_DIR)

from agent.gemini import GeminiVLClient  # noqa: E402
from agent import base_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch Gemini explorer for PHYRE and MiniGrid."
    )
    parser.add_argument("--env_type", type=str, default="phyre", choices=["phyre", "minigrid"])
    parser.add_argument("--eval_setups", type=str, default="ball_within_template")
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-1.5-flash",
        help="Gemini model name, e.g. gemini-1.5-flash or gemini-2.0-flash.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Gemini API key (falls back to GEMINI_API_KEY / GOOGLE_API_KEY env var).",
    )
    parser.add_argument("--eval_type", type=str, default="test")
    parser.add_argument("--format", type=str, default="USER")
    parser.add_argument(
        "--log_dir_base",
        type=str,
        default="/home/u5597173/repo/Batch_Inference/batch_inference_output/gemini/phyre",
    )
    parser.add_argument("--log_dir_label", type=str, default="tmp_log/gemini/phyre/")
    parser.add_argument("--log_model_name", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="./explorer_outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "build_dataset", "qwen3_dataset.json"
        ),
    )
    parser.add_argument("--repeat_num", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant for PHYRE physics reasoning.",
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--save_images", dest="save_images", action="store_true")
    parser.add_argument("--no-save_images", dest="save_images", action="store_false")
    parser.set_defaults(save_images=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    client = GeminiVLClient(
        model_name=args.model_name,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    base_agent.run_agent(args, client)
