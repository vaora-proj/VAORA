"""
Batch inference agent using InternVL-series local models.

Usage:
    python internvl_agent.py \
        --env_type phyre \
        --model_name OpenGVLab/InternVL2-8B \
        --dataset_path ../build_dataset/qwen3_dataset_my_cross_fold_1.json \
        --log_dir_base /tmp/internvl_logs/phyre \
        --log_dir_label tmp_log/internvl/
"""

import argparse
import os
import sys

BATCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BATCH_DIR not in sys.path:
    sys.path.append(BATCH_DIR)

from agent.internvl import InternVLClient  # noqa: E402
from agent import base_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch InternVL explorer for PHYRE and MiniGrid."
    )
    parser.add_argument("--env_type", type=str, default="phyre", choices=["phyre", "minigrid"])
    parser.add_argument("--eval_setups", type=str, default="ball_within_template")
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument(
        "--model_name",
        type=str,
        default="OpenGVLab/InternVL2-8B",
        help=(
            "InternVL model name or HuggingFace hub ID, e.g. "
            "OpenGVLab/InternVL2-8B, OpenGVLab/InternVL2_5-8B, "
            "OpenGVLab/InternVL3-8B."
        ),
    )
    parser.add_argument(
        "--local_load_path",
        type=str,
        default=None,
        help="Path to a local checkpoint directory (overrides --model_name if set).",
    )
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--max_num_tiles", type=int, default=12,
                        help="Maximum number of image tiles for dynamic preprocessing.")
    parser.add_argument("--input_size", type=int, default=448,
                        help="Tile resolution in pixels (448 for InternVL2/2.5/3).")
    parser.add_argument("--eval_type", type=str, default="test")
    parser.add_argument("--format", type=str, default="USER")
    parser.add_argument(
        "--log_dir_base",
        type=str,
        default="/home/u5597173/repo/Batch_Inference/batch_inference_output/internvl/phyre",
    )
    parser.add_argument("--log_dir_label", type=str, default="tmp_log/internvl/phyre/")
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
    parser.add_argument("--save_images", dest="save_images", action="store_true")
    parser.add_argument("--no-save_images", dest="save_images", action="store_false")
    parser.set_defaults(save_images=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    client = InternVLClient(
        model_name=args.model_name,
        local_load_path=args.local_load_path,
        torch_dtype=args.torch_dtype,
        use_flash_attention=args.use_flash_attention,
        max_num_tiles=args.max_num_tiles,
        input_size=args.input_size,
        system_prompt=args.system_prompt,
    )
    base_agent.run_agent(args, client)
