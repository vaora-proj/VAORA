#!/usr/bin/env python3
"""
Script to prepare Phyre dataset for VERL multimodal RL training with INTERACTION ONLY.

This script creates parquet files for interaction-based learning (no tools).
The interaction system provides feedback and rewards directly.

Required Fields:
    - data_source (str): Dataset identifier for reward function routing
    - prompt (list[dict]): Chat-formatted messages (system + user)
    - images (list[np.ndarray]): Scene images as numpy uint8 arrays
    - ability (str): Task category
    - reward_model (dict): {"style": "env", "ground_truth": None}
    - extra_info (dict): Metadata including:
        - interaction_kwargs (dict): Interaction configurations (NO tools_kwargs!)

The dataset loader will:
    1. Apply chat template to messages
    2. Process images with vision_utils.process_image()
    3. Replace <image> placeholders with actual image content
    4. Extract interaction_kwargs for multi-turn feedback

Usage:
    python scripts/prepare_phyre_dataset.py \\
        --eval_setup ball_within_template \\
        --fold_id 0 \\
        --num_train_tasks 200 \\
        --num_test_tasks 50 \\
        --output_dir data/phyre_dataset_interaction
"""

import os
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path
import random
from io import BytesIO
from PIL import Image
from verl.utils.reward_score import phyre_util
from tqdm import tqdm

try:
    import phyre
    import numpy as np
    import datasets
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install phyre numpy datasets pyarrow")
    exit(1)

from phyre.metrics import (
    EvalSetup,
    TRAIN_SHARE,
    _register_eval_setup_builder,
    get_task_ids_in_tier,
    _get_task_per_tpl,
    create_dev_set,
)

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

@_register_eval_setup_builder
def my_template_based_split(seed=1, dev_seed=None) -> EvalSetup:
    """Custom split based on specific template IDs."""
    # Get all ball tasks
    all_task_ids = get_task_ids_in_tier('ball')
    
    # Define which templates should be in test
    test_template = {
        1: {'00001', '00002', '00003', '00006', '00013'},
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


def _difficulty_from_task_id(task_id: str) -> str:
    """Determine difficulty level from task template ID."""
    try:
        template_id = int(task_id.split(":")[0])
    except Exception:
        return "medium"
    if template_id < 5:
        return "easy"
    if template_id < 15:
        return "medium"
    return "hard"

# NEW helper function to select tasks based on the specified mode
def filter_tasks(task_list: List[str], 
                 mode: str, 
                 filter_values: List[str] = None, 
                 max_tasks: int = 0) -> List[str]:
    """Filters a list of Phyre tasks based on the selection mode."""
    if mode == "number":
        print(f"  -> Mode 'number': Selecting up to {max_tasks} tasks.")
        return task_list[:max_tasks] if max_tasks else task_list

    if mode == "difficulty":
        if not filter_values:
            return []
        print(f"  -> Mode 'difficulty': Filtering for {filter_values}.")
        return [
            tid for tid in task_list 
            if _difficulty_from_task_id(tid) in filter_values
        ]

    if mode == "template_id":
        if not filter_values:
            return []
        print(f"  -> Mode 'template_id': Filtering for template IDs {filter_values}.")
        # The template ID is the part before the colon, e.g., "00000" in "00000:000"
        return [
            tid for tid in task_list 
            if tid.split(":")[0] in filter_values
        ]
    
    # Default case if mode is unrecognized
    return []

def create_dataset_entry(eval_setup: str,
                         fold_id: int,
                         task_id: str, 
                         initial_scene: np.ndarray,
                         split: str,
                         index: int,
                         data_source: str = "phyre/ball_within_template") -> Dict[str, Any]:
    """
    Create a single dataset entry by embedding image bytes directly.
    This creates a self-contained dataset file.
    """
    
    # 1. Convert the Phyre scene into a PIL Image object
    scene_rgb = phyre.observations_to_float_rgb(initial_scene)
    scene_uint8 = (scene_rgb * 255).astype(np.uint8)
    pil_image = Image.fromarray(scene_uint8)

    # 2. Save the image to an in-memory byte buffer
    with BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    
    # --- Rest of the function is mostly the same ---
    template_id, variant = task_id.split(":")
    difficulty = _difficulty_from_task_id(task_id)
    user_text = phyre_util.USER_PROMPT
    system_content = phyre_util.SYSTEM_PROMPT
    
    temp = (phyre_util.causal_action_collisions + 
            phyre_util.causal_action_rotations + 
            phyre_util.causal_action_blocks + 
            phyre_util.causal_action_supports)
    random.shuffle(temp)
    user_text = user_text.replace("<CAUSAL_ACTIONS>", '/'.join(temp))

    temp = phyre_util.global_spatial[:]
    random.shuffle(temp)
    user_text = user_text.replace("<GLOBAL_SPATIAL_OPTIONS>", str(temp))

    # --- Main Templates ---
    # Template A: Collision
    temp1 = phyre_util.causal_action_collisions[:]
    random.shuffle(temp1)
    temp2 = phyre_util.collision_directions[:]
    random.shuffle(temp2)
    template_a = phyre_util.causal_action_collision_template.replace("<COLLISION_ACTIONS>", str(temp1)).replace("<COLLISION_DIRECTIONS>", str(temp2))

    # Template B: Rotation
    temp1 = phyre_util.causal_action_rotations[:]
    random.shuffle(temp1)
    temp2 = phyre_util.rotation_directions[:]
    random.shuffle(temp2)
    template_b = phyre_util.causal_action_rotation_template.replace("<ROTATION_ACTIONS>", str(temp1)).replace("<ROTATION_DIRECTIONS>", str(temp2))

    # Template C: Block
    temp1 = phyre_util.causal_action_blocks[:]
    random.shuffle(temp1)
    template_c = phyre_util.causal_action_block_template.replace("<BLOCK_ACTIONS>", str(temp1))

    # Template D: Support
    temp1 = phyre_util.causal_action_supports[:]
    random.shuffle(temp1)
    template_d = phyre_util.causal_action_support_template.replace("<SUPPORT_ACTIONS>", str(temp1))

    templates = [template_a, template_b, template_c, template_d]
    random.shuffle(templates)
    user_text = user_text.replace("<CAUSAL_TEMPLATE_1>", templates[0])
    user_text = user_text.replace("<CAUSAL_TEMPLATE_2>", templates[1])
    user_text = user_text.replace("<CAUSAL_TEMPLATE_3>", templates[2])
    user_text = user_text.replace("<CAUSAL_TEMPLATE_4>", templates[3])
    
    # --- Additional Templates ---
    # Additional A (Collision Chain)
    temp1 = phyre_util.causal_action_collisions[:]
    random.shuffle(temp1)
    temp2 = phyre_util.collision_directions[:]
    random.shuffle(temp2)
    additional_template_a = phyre_util.additional_causal_action_collision_template.replace("<COLLISION_ACTIONS>", str(temp1)).replace("<COLLISION_DIRECTIONS>", str(temp2))

    # Additional B (Rotation Chain)
    temp1 = phyre_util.causal_action_rotations[:]
    random.shuffle(temp1)
    temp2 = phyre_util.rotation_directions[:]
    random.shuffle(temp2)
    additional_template_b = phyre_util.additional_causal_action_rotation_template.replace("<ROTATION_ACTIONS>", str(temp1)).replace("<ROTATION_DIRECTIONS>", str(temp2))

    # Additional C (Block Chain)
    temp1 = phyre_util.chain_action_block_states[:]
    random.shuffle(temp1)
    additional_template_c = phyre_util.additional_causal_action_block_template.replace("<CHAIN_BLOCK_STATES>", str(temp1))

    additional_templates = [additional_template_a, additional_template_b, additional_template_c]
    random.shuffle(additional_templates)
    user_text = user_text.replace("<ADDITIONAL_CAUSAL_TEMPLATE_1>", additional_templates[0])
    user_text = user_text.replace("<ADDITIONAL_CAUSAL_TEMPLATE_2>", additional_templates[1])
    user_text = user_text.replace("<ADDITIONAL_CAUSAL_TEMPLATE_3>", additional_templates[2])

    temp = phyre_util.relative_spatial
    random.shuffle(temp)
    user_text = user_text.replace("<RELATVE_SPATIAL_OPTIONS>", str(temp))

    if split == 'test':
        eval_setup = 'ball_within_template'
        fold_id = 0

    entry = {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_text}
        ],
        # 3. Store the image bytes in the dataset in the expected format
        "images": [{"bytes": image_bytes}],
        "ability": "physics",
        "reward_model": {
            "style": "env",
            "ground_truth": task_id
        },
        "extra_info": {
            "eval_setup": eval_setup,
            "fold_id": fold_id,
            "split": split,
            "index": index,
            "task_id": task_id,
            "template_id": template_id,
            "variant": variant,
            "difficulty": difficulty,
            "interaction_kwargs": {
                "name": "phyre_physics",
                "task_id": task_id,
                "template_id": template_id,
                "variant": variant,
                "difficulty": difficulty,
            }
        }
    }
    
    return entry


def prepare_phyre_dataset(eval_setup: str = "ball_within_template",
                         fold_id: int = 0,
                         num_train_tasks: int = 100,
                         num_test_tasks: int = 20,
                         output_dir: str = "data/phyre_dataset",
                         seed: int = 42,
                         # ADD new arguments for data mode
                         data_mode: str = "number",
                         filter_values: List[str] = None,
                         repeat_count: int = 1) -> None:
    """Prepare Phyre dataset for VERL training in standard format.
    
    Creates parquet files with the following structure per entry:
    - data_source: Dataset identifier
    - prompt: Chat-formatted messages
    - images: List of numpy arrays (uint8, shape HxWxC)
    - ability: Task category
    - reward_model: Reward computation metadata
    - extra_info: Metadata including tools_kwargs and interaction_kwargs
    """
    
    random.seed(seed)
    np.random.seed(seed)
    
    # --- ADD validation for filter values ---
    if data_mode in ["difficulty", "template_id"] and not filter_values:
        print(f"Error: --data_mode '{data_mode}' requires --filter_values.")
        print("Please provide a list of values to filter by.")
        exit(1)
    
    data_source = f"phyre/{eval_setup}"
    
    print(f"Preparing Phyre dataset with setup: {eval_setup}, fold: {fold_id}")
    print(f"Data source: {data_source}")
    
    # Load Phyre tasks
    try:
        train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
        train_tasks = train_tasks+dev_tasks
        print(f"Loaded {len(train_tasks)} train tasks, {len(dev_tasks)} dev tasks, {len(test_tasks)} test tasks")
    except Exception as e:
        print(f"Error loading Phyre tasks: {e}")
        return
    
    # --- MODIFIED: Replace simple slicing with the new filter function ---
    print("\nFiltering training tasks...")
    print(train_tasks)
    train_tasks_filtered = filter_tasks(
        task_list=train_tasks, 
        mode=data_mode, 
        filter_values=filter_values, 
        max_tasks=num_train_tasks
    )
    
    print("Filtering test tasks...")
    print(test_tasks)
    test_tasks_filtered = filter_tasks(
        task_list=test_tasks, 
        mode=data_mode, 
        filter_values=filter_values, 
        max_tasks=num_test_tasks
    )
    
    print(f"\nSelected {len(train_tasks_filtered)} train tasks and {len(test_tasks_filtered)} test tasks after filtering.")

    
    def process_tasks(task_list, split_name):
        """Process a list of tasks and return dataset entries."""
        if not task_list:
            return []
            
        try:
            action_tier = phyre.eval_setup_to_action_tier('ball_within_template')
            simulator = phyre.initialize_simulator(task_list, action_tier)
            print(f"Initialized {split_name} simulator with {len(task_list)} tasks")
        except Exception as e:
            print(f"Error initializing simulator for {split_name}: {e}")
            return []
        
        dataset_entries = []
        total_entries = len(task_list) * repeat_count
        
        with tqdm(total=total_entries) as pbar:
            for rep in range(repeat_count):
                print(f"  > Processing repetition {rep + 1}/{repeat_count} for {split_name}...")
                for i in range(len(task_list)):
                    try:
                        task_id = simulator.task_ids[i]
                        initial_scene = simulator.initial_scenes[i]
                        
                        # Calculate a unique index across all repetitions
                        global_index = i + (rep * len(task_list))
                        
                        # Create entry following VERL standards
                        entry = create_dataset_entry(
                            eval_setup=eval_setup,
                            fold_id=fold_id,
                            task_id=task_id,
                            initial_scene=initial_scene,
                            split=split_name,
                            index=global_index,
                            data_source=data_source
                        )
                        dataset_entries.append(entry)
                        
                        # if (i + 1) % 20 == 0:
                        #     print(f"Processed {i + 1}/{len(task_list)} {split_name} tasks")
                            
                    except Exception as e:
                        print(f"Error processing {split_name} task {i}: {e}")
                        continue
                
            print(f"Finished {split_name}: {len(dataset_entries)} total entries generated.")
            return dataset_entries
    
    # Process train and test datasets
    print("\nProcessing training dataset...")
    train_entries = process_tasks(train_tasks_filtered, "train")
    
    print("\nProcessing test dataset...")
    test_entries = process_tasks(test_tasks_filtered, "test")
    
    if not train_entries and not test_entries:
        print("Error: No dataset entries were created!")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreated {len(train_entries)} train entries and {len(test_entries)} test entries")
    print(f"\nDataset entry structure (INTERACTION ONLY):")
    print(f"  - data_source: '{data_source}'")
    print(f"  - prompt: Chat-formatted messages (system + user)")
    print(f"  - images: List of numpy arrays (uint8 format)")
    print(f"  - ability: 'physics'")
    print(f"  - reward_model: {{'style': 'env', 'ground_truth': None}}")
    print(f"  - extra_info:")
    print(f"      - NO tools_kwargs (interaction-based learning)")
    print(f"      - interaction_kwargs.name: 'phyre_physics'")
    
    # Convert to HuggingFace Dataset and save as parquet
    if train_entries:
        train_dataset = datasets.Dataset.from_list(train_entries)
        train_parquet_path = output_path / "train.parquet"
        train_dataset.to_parquet(train_parquet_path)
        
        train_size = os.path.getsize(train_parquet_path) / (1024 * 1024)  # MB
        print(f"\n✓ Train dataset saved to: {train_parquet_path}")
        print(f"  File size: {train_size:.2f} MB")
        
        # Save a sample JSON for inspection
        sample_output = output_path / "train_sample.json"
        with open(sample_output, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            sample = train_entries[0].copy()
            if 'images' in sample:
                sample['images'] = [f"<image string>" for img in sample['images']]
            json.dump(sample, f, indent=2)
        print(f"✓ Train sample saved to: {sample_output}")
    
    if test_entries:
        test_dataset = datasets.Dataset.from_list(test_entries)
        test_parquet_path = output_path / "test.parquet"
        test_dataset.to_parquet(test_parquet_path)
        
        test_size = os.path.getsize(test_parquet_path) / (1024 * 1024)  # MB
        print(f"\n✓ Test dataset saved to: {test_parquet_path}")
        print(f"  File size: {test_size:.2f} MB")
        
        # Save a sample JSON for inspection
        sample_output = output_path / "test_sample.json"
        with open(sample_output, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            sample = test_entries[0].copy()
            if 'images' in sample:
                sample['images'] = [f"<image string>" for img in sample['images']]
            json.dump(sample, f, indent=2)
        print(f"✓ Test sample saved to: {sample_output}")
    
    print("\n" + "="*60)
    print("✓ Dataset is ready for VERL interaction-based RL training!")
    print("="*60)
    print("\nConfiguration:")
    print(f"  data.train_files: ['{output_path / 'train.parquet'}']")
    print(f"  data.val_files: ['{output_path / 'test.parquet'}']")
    print(f"  data.prompt_key: 'prompt'")
    print(f"  data.image_key: 'images'")
    print(f"  data.return_raw_chat: true")
    print(f"\n  # Interaction-based (NO TOOLS)")
    print(f"  actor_rollout_ref.rollout.multi_turn.enable: true")
    print(f"  actor_rollout_ref.rollout.multi_turn.interaction_config_path: 'config/phyre_interaction_config.yaml'")
    print(f"  # Do NOT set tool_config_path!")


def main():
    parser = argparse.ArgumentParser(description="Prepare Phyre dataset for verl training")
    parser.add_argument("--eval_setup", default="my_template_based_split", 
                       help="Phyre evaluation setup")
    parser.add_argument("--fold_id", type=int, default=0, 
                       help="Fold ID for train/dev/test split")
    parser.add_argument("--num_train_tasks", type=int, default=2000, 
                       help="Number of training tasks to include")
    parser.add_argument("--num_test_tasks", type=int, default=500, 
                       help="Number of test tasks to include")
    parser.add_argument("--output_dir", default="data/phyre_ball_within_template_fold_0", 
                       help="Output directory path for train.parquet and test.parquet")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")

    # --- ADD new arguments ---
    parser.add_argument("--data_mode", type=str, default="number",
                        choices=["number", "difficulty", "template_id"],
                        help="Data selection mode.")
    parser.add_argument("--filter_values", nargs='+', default=None,
                        help="List of values for filtering (e.g., 'easy' 'medium' for difficulty mode).")
    parser.add_argument("--repeat_count", type=int, default=1,
                        help="Number of times to repeat the dataset (generates varied prompts per task).")

    args = parser.parse_args()

    prepare_phyre_dataset(
        eval_setup=args.eval_setup,
        fold_id=args.fold_id,
        num_train_tasks=args.num_train_tasks,
        num_test_tasks=args.num_test_tasks,
        output_dir=args.output_dir,
        seed=args.seed,
        # Add these two lines
        data_mode=args.data_mode,
        filter_values=args.filter_values,
        repeat_count=args.repeat_count
    )


if __name__ == "__main__":
    main()
