# phyre_util.py
import math
import re, random, os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)

import numpy as np
from tqdm import tqdm

from verl.utils.reward_score.phyre_DQN_agent import DQNInferenceModel
from verl.utils.reward_score.phyre_placement import *
from verl.utils.reward_score.phyre_collision import *
from verl.utils.reward_score.phyre_prompts import *

# --- Scene constants (from original code) ---
SCENE_WIDTH = 256
SCENE_HEIGHT = 256
MIN_RADIUS = 2
# MAX_RADIUS = max(256, 256) // 8 = 32
MAX_RADIUS = 32 

# --- Scaling ranges for 256x256 scene ---
# Note: The original code scales position from 0 to 255
_X_RANGE_256 = SCENE_WIDTH - 1   # 255
_Y_RANGE_256 = SCENE_HEIGHT - 1  # 255
_RAD_RANGE = MAX_RADIUS - MIN_RADIUS # 30

# --- Model constants (your new resolution) ---
MODEL_WIDTH = 256
MODEL_HEIGHT = 256

# --- Resizing scale factors ---
# Scale factor to go from 256-space -> 252-space
_SCALE_256_TO_252 = MODEL_WIDTH / SCENE_WIDTH  # 252 / 256 = 0.984375

# Scale factor to go from 252-space -> 256-space
_SCALE_252_TO_256 = SCENE_WIDTH / MODEL_WIDTH  # 256 / 252 = 1.015873...

try:
    import phyre
    from phyre.metrics import (
    EvalSetup,
    TRAIN_SHARE,
    _register_eval_setup_builder,
    get_task_ids_in_tier,
    _get_task_per_tpl,
    create_dev_set,
    )
except ImportError:
    # This allows the file to be imported even if phyre is not installed,
    # though functions using it will fail.
    phyre = None


def _sample_responses(mode: str) -> List[str]:
    """Selects a random response string based on the simulation outcome."""
    responses = {
        "wrong_format": [
            "The action format is incorrect. Please provide the action in the format: `<answer>[x, y, r]</answer>`, where `x` and `y` are the coordinates (0-252) and `r` is the radius (2-32)."
        ],
        "invalid": [
            "That placement is invalid. The action was likely **out of the simulation's boundaries** or **occluding an existing object**. Please try again.",
        ],
        "solved": [
            "Success! The task has been solved.",
        ],
        "not_solved": [
            # Version 1: Direct Command
            "The attempt failed. Analyze the video replay: 1.**Motion:** Describe the object movement. 2.**Cause:** State the reason for failure. 3.**Fix:** Propose the corrected action.\nHere is a video replay of the simulation: ",        ]
    }

    selected_list = responses.get(mode, ["An unknown error occurred."])
    return random.choice(selected_list)

def _digit_mapping(action: list) -> List:
    action = [int(a) for a in action]
    return convert_model_prediction_to_float_action_resized(action)


def convert_model_prediction_to_float_action_resized(
    model_prediction: list[int]
) -> list[float]:
    """
    Converts a 252x252 integer pixel prediction [px, py, pr]
    back into a normalized 0.0-1.0 float action [x, y, d].
    """
    pred_x, pred_y, pred_r = model_prediction
    pred_y = MODEL_HEIGHT - 1 - pred_y  # Invert y-axis for model output
    
    x_action = pred_x / (MODEL_WIDTH - 1)
    y_action = pred_y / (MODEL_HEIGHT - 1)
    d_action = (pred_r - MIN_RADIUS) / _RAD_RANGE

    # Clip to ensure valid 0.0-1.0 range
    action_np = np.array([x_action, y_action, d_action], dtype=np.float64)

    # Return as list
    return action_np.tolist()


def _extract_action(text: str) -> Optional[List[float]]:
    """
    Extracts a normalized action [x, y, r] from the model's text output.
    Uses regex to find a pattern like [0.123, 0.456, 0.789].
    """
    # Regex to find a list of 3 floating point numbers
    match = re.search(r"<action>\s*(.*?)\s*</action>", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            # Convert matched groups to floats
            inner_text = match.group(1)
            original_actions = re.findall(r"[-+]?\d*\.\d+|\d+", inner_text)
            if len(original_actions) != 3:
                return None, None
            mapped_actions = _digit_mapping(original_actions)
            if any(float(val) < 0 or float(val) > 1 for val in mapped_actions):
                return None, None
            return original_actions, mapped_actions
        except (ValueError, TypeError):
            return None, None
    return None, None

def parse_numbers(text):
    """
    Helper to find all numbers (integers, floats, negatives) in a string
    and return them as a list of floats.
    """
    # Matches: -0.5, 0.5, 100, -20
    return [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)]

def coord_to_phyre(coord_list):
    """
    Converts a list of coordinates from string format to Phyre normalized float format.
    Input: [x_pixel, y_pixel] in 256-space
    Output: [x_norm, y_norm] in 0.0-1.0 range
    """
    if len(coord_list) != 2 or not all(isinstance(x, (int, float)) for x in coord_list):
        return []
    x_pixel, y_pixel = coord_list
    y_pixel = MODEL_HEIGHT - y_pixel  # Invert y-axis
    x_norm = x_pixel / (MODEL_WIDTH - 1)
    y_norm = y_pixel / (MODEL_HEIGHT - 1)
    return [x_norm, y_norm]

def diameter_to_phyre(diameter_pixel):
    """
    Converts a radius from pixel format to Phyre normalized float format.
    Input: radius in pixels
    Output: radius in 0.0-1.0 range
    """
    return [i / MODEL_WIDTH for i in diameter_pixel]

def extract_scene_answer(text):
    """
    Extracts object state from <scene_answer> tags.
    Now handles the format: [OBJECT] is at [LABEL] [COORDS] with size [SIZE].
    Returns coordinates and size as lists of floats.
    """
    scene_pattern = r"<scene_answer>(.*?)</scene_answer>"
    scene_content = re.search(scene_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if not scene_content:
        return None
        
    content = scene_content.group(1).strip()
    
    # Updated Regex to capture the Position Label separately from the Coordinates
    # Group 1: Object
    # Group 2: Position Label (e.g., TOP-CENTER)
    # Group 3: Position Coords (e.g., 0.5, 0.8)
    # Group 4: Size (e.g., 0.4)
    pattern = r"\[(.*?)\]\s+is at\s+\[(.*?)\]\s+\[(.*?)\]\s+with size\s+\[(.*?)\]"
    
    matches = re.findall(pattern, content, re.IGNORECASE)
    
    results = []
    for match in matches:
        obj_name, pos_label, pos_coords_str, size_str = match
        
        # --- FIX START: robust parsing for size ---
        parsed_size = parse_numbers(size_str)
        if not parsed_size:
            # Skip this entry if size cannot be parsed, or assign a default
            parsed_size = []
        
        position = coord_to_phyre(parse_numbers(pos_coords_str))
        if len(position) != 2 or not all(isinstance(x, (float, int)) for x in position):
            position = []

        results.append({
            "object": obj_name.strip(),
            "position_label": pos_label.strip(),
            # Ensure the position is a list of floats/ints, else use []
            "position": position,
            "size": diameter_to_phyre(parsed_size)            # Converts "0.4" -> [0.4]
        })
        
    return results

def normalize_event(event_str):
    """
    Restores your original logic to classify the event string.
    """
    evt = event_str.strip().upper()
    if evt in ["PUSH", "COLLIDE WITH", "HIT", "STRIKE"]:
        return "collision"
    elif evt in ["TILT", "ROTATE", "SPIN"]:
        return "rotation"
    return "unknown"


def extract_causal_actions(text):
    """
    Extracts logic from <causal_actions_answer> tags.
    Returns simplified dictionaries containing only necessary keys.
    """
    # 1. Extract content inside tags (if present), otherwise assume text is the content
    causal_pattern = r"<causal_actions_answer>(.*?)</causal_actions_answer>"
    causal_content = re.search(causal_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if causal_content:
        content = causal_content.group(1).strip()
    else:
        content = text.strip()
    lines = content.split("\n")
    results = []

    # Helper: Returns a list [x, y] for coords, or a string for labels
    def parse_value(val):
        val = val.strip()
        # Check for coordinates format "35, 220"
        if re.match(r"^\d+\s*,\s*\d+$", val):
            return [int(n) for n in val.split(",")]
        return val

    for line in lines:
        line_is_record = False
        # --- Pattern 1: Collision / Push ---
        # Matches: The [RED BALL] should [STRIKE] the [GREEN BALL] at [13, 246] to push it towards [RIGHT].
        collision_pattern_0 = (
            r"The \[RED BALL\] should\s+\[(.*?)\]"
            r"\s+the\s+\[(.*?)\]"
            r"\s+at\s+\[(.*?)\]"
            r"\s+to\s+push\s+it\s+(?:to|towards)\s+\[(.*?)\]"
        )
        
        for match in re.finditer(collision_pattern_0, line, re.IGNORECASE):
            raw_action, target, contact_raw, dir_lbl = match.groups()

            if "BAR" in target:
                continue
            
            results.append({
                "action": normalize_event(raw_action.strip()),
                "target": target.strip(),
                "contact": coord_to_phyre(parse_value(contact_raw)),
                "direction": dir_lbl.strip(),
                "active": True
            })
            line_is_record = True
        
        if line_is_record:
            continue
        
        # The [RED BALL] should [BLOCK] the [GREEN BALL] at [22, 91].
        collision_pattern_1 = (
            r"The \[RED BALL\] should\s+\[(.*?)\]"                     # 1. Action (BLOCK)
            r"\s+the\s+\[(.*?)\]"         # 2. Target (GREEN BALL)
            r"\s+at\s+\[(.*?)\]" # 3. Contact
        )

        for match in re.finditer(collision_pattern_1, line, re.IGNORECASE):
            raw_action, target, contact_raw = match.groups()

            if "BAR" in target:
                continue

            mapping_dict_temp = {
                "BLOCK": "BLOCKED",
                "STOP": "STOPPED",
                "DEFLECT": "DEFLECTED",
            }
            
            results.append({
                "action": "collision",
                "target": target.strip(),
                "contact": coord_to_phyre(parse_value(contact_raw)),
                "direction": mapping_dict_temp.get(raw_action.strip().upper(), "unknown"),
                "active": True
            })
            line_is_record = True
        
        if line_is_record:
            continue
        # The chain actions cause the [GREEN BALL] to move [LEFT] at point [249, 250].
        collision_pattern_2 = (
            r"The chain actions cause the\s+\[(.*?)\]"                     # 1. Target (GREEN BALL)
            r"\s+to move\s+\[(.*?)\]"                     # 2. Direction (LEFT)
            r"\s+at\s+point\s+\[(.*?)\]" # 3. Contact
        )
        
        for match in re.finditer(collision_pattern_2, line, re.IGNORECASE):
            target, dir_lbl, contact_raw = match.groups()

            if "BAR" in target:
                continue
            
            results.append({
                "action": "collision",
                "target": target.strip(),
                "contact": coord_to_phyre(parse_value(contact_raw)),
                "direction": dir_lbl.strip(),
                "active": False
            })
            line_is_record = True
        
        if line_is_record:
            continue
        # The chain actions cause the [GREEN BALL] to be [BLOCKED] at point [51, 182].
        collision_pattern_3 = (
            r"The chain actions cause the\s+\[(.*?)\]"                     # 1. Target (GREEN BALL)
            r"\s+to be\s+\[(.*?)\]"                     # 2. Direction (LEFT)
            r"\s+at\s+point\s+\[(.*?)\]" # 3. Contact
        )
        
        for match in re.finditer(collision_pattern_3, line, re.IGNORECASE):
            target, direction, contact_raw = match.groups()

            if "BAR" in target:
                continue
            
            results.append({
                "action": "collision",
                "target": target.strip(),
                "contact": coord_to_phyre(parse_value(contact_raw)),
                "direction": direction,
                "active": False
            })
            line_is_record = True
        
        if line_is_record:
            continue
        # --- Pattern 2: Rotation / Tilt ---
        # Matches: "The [RED BALL] should [ROTATE] the [GREEN BAR] at [170, 92] to rotate it [COUNTER-CLOCKWISE]."
        rotation_pattern_0 = (
            r"The \[RED BALL\] should\s+\[(.*?)\]" # 1. Action (ROTATE)
            r"\s+the\s+\[(.*?)\]" # 2. Target (GREEN BAR)
            r"\s+at\s+\[(.*?)\]" # 3. Contact
            r"\s+to\s+rotate\s+it\s+\[(.*?)\]" # 4. Rotation Dir
        )

        for match in re.finditer(rotation_pattern_0, line, re.IGNORECASE):
            raw_action, target, contact_raw, rot_dir = match.groups()

            if "BAR" not in target:
                continue

            results.append({
                "action": normalize_event(raw_action.strip()),
                "target": target.strip(),
                "contact": coord_to_phyre(parse_value(contact_raw)),
                "direction": rot_dir.strip(),
                "active": True
            })
            line_is_record = True
        
        if line_is_record:
            continue
        # The [RED BALL] should [SUPPORT] the [BLUE BAR] at [42, 149].
        rotation_pattern_1 = (
            r"The \[RED BALL\] should\s+\[(.*?)\]"
            r"\s+the\s+\[(.*?)\]"
            r"\s+at\s+\[(.*?)\]"
        )
        
        for match in re.finditer(rotation_pattern_1, line, re.IGNORECASE):
            direction, target, contact_raw = match.groups()

            if "BAR" not in target:
                continue
            
            results.append({
                "action": "rotation",
                "target": target.strip(),
                "contact": coord_to_phyre(parse_value(contact_raw)),
                "direction": direction,
                "active": True
            })
            line_is_record = True
        
        if line_is_record:
            continue
        # The chain actions cause the [GREEN BAR] to move [COUNTER-CLOCKWISE] at point [73, 226].
        rotation_pattern_2 = (
            r"The chain actions cause the\s+\[(.*?)\]"
            r"\s+to move\s+\[(.*?)\]"
            r"\s+at\s+point\s+\[(.*?)\]"
        )
        
        for match in re.finditer(rotation_pattern_2, line, re.IGNORECASE):
            target, direction, contact_raw = match.groups()

            if "BAR" not in target:
                continue
            
            results.append({
                "action": "rotation",
                "target": target.strip(),
                "contact": coord_to_phyre(parse_value(contact_raw)),
                "direction": direction,
                "active": False
            })
            line_is_record = True
        
        if line_is_record:
            continue

    return results


def extract_spatial_relationships(text):
    """
    Extracts spatial relationships from text within <placement> tags.
    Now supports variable shapes (e.g., GREEN BALL, GREEN CUBE, GREEN OBJECT).
    """
    
    # 1. Find content inside <placement> tags
    placement_pattern = r"<placement_answer>(.*?)</placement_answer>"
    placements = re.findall(placement_pattern, text, re.DOTALL | re.IGNORECASE)
    
    # extracted_data = {}
    extracted_data = []
    
    # Updated pattern with mandatory brackets as discussed previously
    relationship_pattern = r"\[(.*?)\]\s+is located at the\s+\[(.*?)\]\s+of the\s+\[(.*?)\][\.]"
    
    for content in placements:
        content = content.strip()
        matches = re.findall(relationship_pattern, content, re.IGNORECASE)
        
        if not matches:
            continue # Skip if no matches in this block
        
        for match in matches:
            obj, position, reference = match
            
            obj_upper = obj.strip().upper()
            ref_upper = reference.strip().upper()
            
            if "RED" in obj_upper:
                # extracted_data[ref_upper] = position.strip()
                extracted_data.append({
                    "object": ref_upper,
                    "position": position.strip(),
                })
            
    return extracted_data


def merge_features(featurized_objects):
    colors = featurized_objects.colors
    shapes = featurized_objects.shapes
    merged_features = [f"{colors[i]} {shapes[i]}" for i in range(len(colors))]        
    return merged_features

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


class PhyreAgentEvaluator:
    """
    An efficient Phyre evaluator that provides nuanced, ratio-based rewards,
    aligned with the PhyreInteraction logic.
    """
    
    def __init__(
        self,
        eval_setup: str,
        fold_id: int,
        model_path: str,
        device: str = 'cpu',
        no_action_cache_path: str = None  # <--- New Argument
    ):
        print(f"Initializing PhyreEvaluator for setup='{eval_setup}', fold={fold_id}...")
        self.eval_setup = eval_setup
        # self.eval_setup = "ball_within_template"
        self.fold_id = fold_id
        self.agent = DQNInferenceModel(model_path, device=device)
        self.agent.model.eval()
        
        try:
            train_tasks, dev_tasks, test_tasks = phyre.get_fold(self.eval_setup, self.fold_id)
            self.task_ids: List[str] = train_tasks + dev_tasks + test_tasks
            print("Task IDs:", self.task_ids[0])
        except Exception as e:
            raise RuntimeError(f"Failed to load Phyre tasks: {e}")

        action_tier = phyre.eval_setup_to_action_tier("ball_within_template")
        self.simulator = phyre.initialize_simulator(self.task_ids, action_tier)
        
        self.task_id_to_index_map: Dict[str, int] = {
            task_id: i for i, task_id in enumerate(self.task_ids)
        }

        self.max_step_for_punishment = 1000
        self.punishment_reward_mild = -0.1
        self.punishment_reward_medium = -0.5
        self.punishment_reward_severe = -1.0
        self.min_scheduler_reward = 0.0
        self.max_scheduler_reward = 1.0
        self.single_scene_reward = 0.2
        self.single_action_reward = 0.6
        self.single_placement_reward = 0.3
        self.format_reward = 0.2
        
        # Assuming self.simulator.initial_scenes is a list/array of images
        print("⚡ Pre-loading scene tensors to device...")
        self.scene_cache = {}
        for idx, scene in enumerate(self.simulator.initial_scenes):
            # Convert to tensor once. Adjust transformation based on your agent's needs.
            # Example: forcing float32 and adding batch dim if needed
            tensor_scene = torch.tensor(scene, dtype=torch.float64, device=device).unsqueeze(0)
                
            self.scene_cache[idx] = tensor_scene
            
        # --- CACHE LOGIC STARTS HERE ---
        self.no_action_cache = {}
        cache_loaded = False

        # 1. Try to load existing cache
        if no_action_cache_path and os.path.exists(no_action_cache_path):
            print(f"📂 Found no-action cache at '{no_action_cache_path}'. Loading...")
            try:
                # allow_pickle=True is required because we are loading a Python dict wrapped in numpy
                loaded_data = np.load(no_action_cache_path, allow_pickle=True).item()
                
                # 2. Validate the cache matches the current tasks (important if you change fold_id)
                if all(t_id in loaded_data for t_id in self.task_ids):
                    self.no_action_cache = loaded_data
                    cache_loaded = True
                    print("✅ Cache loaded and verified successfully.")
                else:
                    print("⚠️ Cache found, but it doesn't contain all required task IDs for this fold. Regenerating...")
            except Exception as e:
                print(f"⚠️ Failed to load cache file (it might be corrupted): {e}. Regenerating...")

        # 3. Run simulation if cache was not loaded
        if not cache_loaded:
            print("⚡ Generating no-action features (Simulation needed)...")
            
            # Optimization: Load the compiled task dict ONCE outside the loop
            all_compiled_tasks = phyre.loader.load_compiled_task_dict()
            
            # Using a temporary dict to build data
            # If we partially loaded data (e.g. from a different fold), you might want to merge,
            # but usually it's safer to regenerate the specific list we need.
            for idx, task_id in tqdm(enumerate(self.task_ids)):
                # Check if we already have it in memory to avoid re-simulating
                if task_id in self.no_action_cache:
                    continue

                task_tmp = all_compiled_tasks[task_id]
                no_action_sim = phyre.simulator.simulate_task(
                    task_tmp, 
                    steps=phyre.simulator.DEFAULT_MAX_STEPS, 
                    stride=1
                )
                feats = []
                for scene in no_action_sim.sceneList:
                    fo = phyre.simulator.scene_to_featurized_objects(scene)
                    feats.append(fo.features[0])  # (N,14)
                self.no_action_cache[task_id] = np.array(feats, dtype=np.float64)  # (T,N,14)

            # 4. Save to disk if a path was provided
            if no_action_cache_path:
                print(f"💾 Saving generated cache to '{no_action_cache_path}'...")
                try:
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(no_action_cache_path), exist_ok=True)
                    np.save(no_action_cache_path, self.no_action_cache)
                    print("✅ Cache saved.")
                except Exception as e:
                    print(f"❌ Failed to save cache: {e}")

        print(f"✅ PhyreEvaluator initialized with {len(self.task_ids)} tasks.")
    
    def get_scheduler(self, item_counter: int) -> float:
        step = max(item_counter, self.max_step_for_punishment)
        punishment_reward = self.max_scheduler_reward - 0.5*(self.max_scheduler_reward - self.min_scheduler_reward)*(1-np.cos(np.pi*step/self.max_step_for_punishment))
        return punishment_reward

    def score(self, task_id: str, solution_str: str, item_counter: int = 0) -> dict[str, float]:
        """
        Scores an action with ratio-based rewards.
        - Returns 1.0 if the task is solved.
        - Returns -0.1 if the action is invalid.
        - Otherwise, returns a ratio-based reward if dynamic objects move.
        """
        prob_weight = self.get_scheduler(item_counter)
        
        score_dict = {"score": 0.0, "placement_reward": 0.0, "collision_reward": 0.0, "grounding_reward": 0.0, "predicted_prob": 0.0}
        
        
        if task_id not in self.task_id_to_index_map:
            print(f"Warning: Task ID '{task_id}' not found in the loaded set.")
            return score_dict
        
        # Using the placeholder _extract_action defined above for consistency
        _, mapped_action = _extract_action(solution_str)
        # print("Mapped Action:", mapped_action)
        if mapped_action is None:
            score_dict["score"] = -1
            print("Mapped Action is None.")
            print("Solution String:", solution_str)
            return score_dict
            
        
        task_index = self.task_id_to_index_map[task_id]
        action_array = np.array(mapped_action, dtype=np.float64)
        # t_sim_start = time.perf_counter()
        sim_result = self.simulator.simulate_action(
            task_index=task_index,
            action=action_array,
            need_images=False,
            need_featurized_objects=True,
            stride=1
        )

        if sim_result.status.is_invalid():
            merged_features = merge_features(self.simulator.initial_featurized_objects[task_index])
        else:
            merged_features = merge_features(sim_result.featurized_objects)

        extract_objects = extract_scene_answer(solution_str)
        extract_collisions = extract_causal_actions(solution_str)
        extract_relations = extract_spatial_relationships(solution_str)
        featurized_objects = sim_result.featurized_objects
        RED_BALL_IDX = -1
        # Evaluate Grounding Reward
        if extract_objects is None or len(extract_objects) == 0:
            print("Scene extraction failed.")
            grounding_reward = 0.0
        else:
            grounding_reward = self.format_reward
            
            # 1. Initialize a set to track processed anchors
            processed_anchors = set()

            if sim_result.status.is_invalid():
                init_features = self.simulator.initial_featurized_objects[task_index].features[0]
            else:
                init_features = featurized_objects.features[0]
            
            for obj_info in extract_objects:
                # [{'object': 'GREEN BALL', 'position_label': 'TOP-CENTER', 'position': [0.5, 0.8], 'size': [0.4]}]
                anchor = obj_info['object'].upper()
                
                # 2. Skip/Punish if duplicate or non-existent
                if anchor in processed_anchors:
                    print("Anchor already processed:", anchor)
                    print("Task ID:", task_id, mapped_action)
                    grounding_reward += self.punishment_reward_mild
                    continue
                if anchor not in merged_features:
                    print("Anchor not in merged features:", anchor)
                    print("Task ID:", task_id, mapped_action)
                    grounding_reward += self.punishment_reward_mild
                    continue
                    
                processed_anchors.add(anchor)
                
                anchor_idx = merged_features.index(anchor)
                anchor_features = init_features[anchor_idx]
                
                # --- A. Calculate TEXT Score (Soft Grid) ---
                global_covered, true_position = get_global_position(anchor_features)
                pred_position_label = obj_info['position_label'].upper()
                
                text_score = self._calculate_soft_grid_score(
                    true_exact_label=true_position,
                    pred_label=pred_position_label, 
                    mode='global',
                    smooth_reward=True
                )
                # text_score is now between -1.0 and 1.0 (approx)

                # --- B. Calculate NUMERICAL Score (Euclidean) ---
                
                if len(obj_info['size']) != 1 or len(obj_info['position']) != 2:
                    # Malformed vector -> Heavy penalty on the numerical part
                    num_score = 0.0
                else:
                    predicted_pos = obj_info['position'] + obj_info['size']
                    gt_pos = anchor_features[:2].tolist() + [anchor_features[3].tolist()]
                    distance = math.sqrt(sum((predicted_pos[i] - gt_pos[i])**2 for i in range(3)))
                    max_distance = math.sqrt(3) # Max possible distance in normalized 3D space
                    
                    # Normalize: 0 distance -> 1.0 score, Max distance -> 0.0 score
                    # We map this to [-1, 1] range to match text_score scaling:
                    # 1.0 (perfect) to -1.0 (worst)
                    # Formula: 2 * (1 - normalized_dist) - 1
                    
                    normalized_dist = min(distance / max_distance, 1.0)
                    num_score = 1.0 - normalized_dist

                # --- C. Fuse Scores ---
                # We weight them equally (0.5 each). 
                # If text is perfect (1.0) and num is perfect (1.0) -> Total 1.0 * single_reward
                # If text is "Top-Left" vs "Top" (0.75) and num is decent (0.5) -> Total 0.625 * single_reward
                
                combined_score = (0.5 * text_score) + (0.5 * num_score)
                
                # Add to total reward
                # single_scene_reward is the "Max Value" this item contributes
                grounding_reward += self.single_scene_reward * combined_score

            # Clamp final result
            grounding_reward = max(0.0, grounding_reward)
            grounding_reward = min(1.0, grounding_reward)

        if sim_result.status.is_invalid():
            print("Action is invalid.")
            print("Task ID:", task_id, action_array*np.array([255, -255, 30])+np.array([0,255,2]))
            initial_scene = self.scene_cache[task_index]

            predicted_prob = 0.0

            score_dict["score"] = -1
            score_dict["placement_reward"] = 0.0
            score_dict["collision_reward"] = 0.0
            score_dict["grounding_reward"] = grounding_reward
            score_dict["predicted_prob"] = predicted_prob

            return score_dict

        # Evaluate collision reward
        processed_targets = set()
        if extract_collisions is None or len(extract_collisions) == 0:
            print("Causal action extraction failed.")
            collision_reward = 0.0
        else:
            collision_reward = self.format_reward
            no_action_feats = self.no_action_cache[task_id]
            deviation = compute_deviations(featurized_objects.features, no_action_feats)

            # Retrieve shapes for logic switching
            shapes = featurized_objects.shapes

            for causal_info in extract_collisions:
                causal_event = causal_info['action']
                if causal_event == 'unknown':
                    collision_reward += self.punishment_reward_mild
                    continue

                target = causal_info['target'].upper()
                
                # 2. Skip if we have already processed this object type
                if target in processed_targets:
                    collision_reward += self.punishment_reward_mild
                    continue
                if target not in merged_features:
                    collision_reward += self.punishment_reward_mild
                    continue
                if target == "RED BALL":
                    collision_reward += self.punishment_reward_mild
                    continue

                if causal_info['active'] == True:
                    processed_targets.add(target)
                target_indices = [i for i, name in enumerate(merged_features) if name == target]
                target_shape = shapes[target_indices[0]]

                # --- STEP 1: DETECT OBSERVED EVENT (Ground Truth Logic) ---
                # Mirroring generate_solutions.py logic:
                # Prioritize rotation for BAR objects.
                event_info = None
                
                # Determine event type based on shape
                scan_event_type = "rotation" if target_shape == 'BAR' else "collision"
                
                # Iterate over all candidates (handles duplicates and GRAY JAR parts)
                candidate_events = []
                for idx in target_indices:
                    ev = analyze_event(deviation, target_obj_id=idx, event_type=scan_event_type)
                    if ev is not None:
                        candidate_events.append(ev)
                
                if candidate_events:
                    # Pick the earliest event
                    event_info = min(candidate_events, key=lambda x: x['frame'])
                
                # If no event actually happened to this object
                if event_info is None:
                    collision_reward += self.punishment_reward_mild
                    continue

                # --- STEP 2: COMPARE PREDICTED vs OBSERVED ---
                # Check if the predicted event type matches what actually happened.
                # Note: 'causal_event' from extract_collisions is normalized to "collision" or "rotation" (or "unknown")
                
                # Allow 'unknown' predictions (chain reactions) to match any event type, 
                # but explicit predictions must match.
                if causal_event != event_info['event_type']:
                     # Predicted Collision but got Rotation, or vice versa
                     collision_reward += self.punishment_reward_mild
                     continue

                # --- STEP 3: SCORE ATTRIBUTES (Direction & Contact) ---
                if len(causal_info['contact']) != 2:
                    collision_reward += self.punishment_reward_mild
                    continue

                pred_contact_point = causal_info['contact']
                contact_point = event_info['ball_center_px']

                if event_info['direction'].upper() == causal_info['direction'].upper():
                    distance = math.sqrt(
                        (pred_contact_point[0] - contact_point[0]) ** 2 +
                        (pred_contact_point[1] - contact_point[1]) ** 2
                    )
                    max_distance = math.sqrt(2)  # Max possible distance in normalized space
                    normalized_distance = min(distance / max_distance, 1.0)
                    collision_reward += self.single_action_reward * (1 - normalized_distance)
                else:
                    collision_reward += self.punishment_reward_mild

            collision_reward = min(1.0, collision_reward)

        # Evaluate placement reward
        if extract_relations is None or len(extract_relations) == 0:
            print("Relation extraction failed.")
            placement_reward = 0.0
        else:
            placement_reward = self.format_reward

            init_features = featurized_objects.features[0]
            red_features = init_features[RED_BALL_IDX]
            
            anchor_set = set()

            for relation in extract_relations:
                anchor, pred_relation = relation['object'], relation['position']
                if anchor in anchor_set:
                    placement_reward += self.punishment_reward_mild
                    continue
                # --- CASE A: WHOLE SCENE ---
                if anchor == 'WHOLE SCENE':
                    anchor_set.add(anchor)
                    
                    # Capture BOTH covered blocks and the specific text label
                    global_covered, exact_text = get_global_position(red_features)
                    
                    score = self._calculate_soft_grid_score(
                        true_exact_label=exact_text,  # <--- Pass the exact label here
                        pred_label=pred_relation, 
                        mode='global',
                        smooth_reward=True
                    )
                    placement_reward += self.single_placement_reward * score
                    continue
                
                # --- CASE B: RELATIVE ---
                if anchor not in merged_features:
                    placement_reward += self.punishment_reward_mild
                    continue
                
                anchor_set.add(anchor)
                anchor_idx = merged_features.index(anchor)
                anchor_features = init_features[anchor_idx]
                
                # Capture BOTH covered blocks and the specific text label
                exact_text = get_relative_relationship(anchor_features, red_features)
                
                score = self._calculate_soft_grid_score(
                    true_exact_label=exact_text,  # <--- Pass the exact label here
                    pred_label=pred_relation, 
                    mode='relative',
                    smooth_reward=False
                )
                placement_reward += self.single_placement_reward * score

            for target in list(processed_targets) + ['WHOLE SCENE']:
                if target not in anchor_set:
                    placement_reward += self.punishment_reward_medium
            
            # Clamp final placement total
            placement_reward = max(0.0, placement_reward)
            placement_reward = min(1.0, placement_reward)
        # t_fmt_end = time.perf_counter()

        if sim_result.status.is_solved():
            score_dict["score"] = 1.0*(1 + placement_reward + collision_reward)
            score_dict["placement_reward"] = placement_reward
            score_dict["collision_reward"] = collision_reward
            score_dict["grounding_reward"] = grounding_reward
            score_dict["predicted_prob"] = 1.0
            return score_dict
        
        initial_scene = self.scene_cache[task_index]
        
        with torch.inference_mode():
            predicted_prob = self.agent.predict_pair_probability(initial_scene, action_array)
        

        score_dict["score"] = predicted_prob * (1 + placement_reward + collision_reward)
        score_dict["placement_reward"] = placement_reward
        score_dict["collision_reward"] = collision_reward
        score_dict["grounding_reward"] = grounding_reward
        score_dict["predicted_prob"] = predicted_prob

        print(f"Rewards => counter: {item_counter}, Placement: {placement_reward:.4f}, Collision: {collision_reward:.4f}, Grounding: {grounding_reward:.4f}")
        print(f"Task {task_id}: Predicted Prob={predicted_prob:.4f}, Total Reward={score_dict['score']:.4f}")
        
        return score_dict
    
    def _calculate_soft_grid_score(self, true_exact_label: str, pred_label: str, mode: str, smooth_reward: bool = False) -> float:
        """
        Calculates a tiered reward:
        - 1.0 if prediction matches true_exact_label (The 'Best' answer).
        - 0.75 if prediction is inside covered_blocks but not the exact label (Technically correct).
        - < 0.3 if prediction is outside, decaying based on distance.
        """
        pred_label = pred_label.upper()
        true_exact_label = true_exact_label.upper()
        
        # --- 1. Define Grid Mappings ---
        if mode == 'relative':
            idx_to_coord = {
                0: (0, 0), 1: (1, 0),
                2: (0, 1), 3: (1, 1),
            }
            label_to_idx = {
                'TOP-LEFT': 0, 'TOP-RIGHT': 1,
                'BOTTOM-LEFT': 2, 'BOTTOM-RIGHT': 3,
            }
        elif mode == 'global':
            idx_to_coord = {
                0: (0, 2), 1: (1, 2), 2: (2, 2),
                3: (0, 1), 4: (1, 1), 5: (2, 1),
                6: (0, 0), 7: (1, 0), 8: (2, 0)
            }
            label_to_idx = {
                'TOP-LEFT': 0, 'TOP': 1, 'TOP-RIGHT': 2,
                'LEFT': 3, 'CENTER': 4, 'RIGHT': 5,
                'BOTTOM-LEFT': 6, 'BOTTOM': 7, 'BOTTOM-RIGHT': 8
            }
        else:
            return 0.0


        # --- 2. Tier 1: Exact Match ---
        # "Best" answer gets max reward
        if true_exact_label not in label_to_idx:
            return 0.0

        if pred_label == true_exact_label:
            return 1.0
        if pred_label not in label_to_idx:
            return 0.0

        if not smooth_reward:
            return 0.0
        else:
            pred_idx = label_to_idx[pred_label]
            
            pred_coord = np.array(idx_to_coord[pred_idx], dtype=np.float64)
            true_coord = np.array(idx_to_coord[label_to_idx[true_exact_label]], dtype=np.float64)
            dist = np.linalg.norm(pred_coord - true_coord)
        
            # Standard neighbor distance is 1.0.
            # Formula: 1.0 - (dist / 1.414)
            # Dist 1.0 -> 0.29 Reward
            # Dist 1.41 -> 0.0 Reward
            # Dist 2.0  -> -0.41 Reward
            reward = 1.0 - (dist / 2.82)

        return max(0.0, min(1.0, reward))
    
    def _print_timing(self, t_sim_s, t_sim_e, t_rel_s, t_rel_e, 
                      t_fmt_s, t_fmt_e, t_pred_s, t_pred_e):
        """Helper to print a formatted timing table."""
        print("-" * 40)
        print(f"{'Operation':<20} | {'Time (ms)':<10}")
        print("-" * 40)
        print(f"{'Phyre Simulation':<20} | {(t_sim_e - t_sim_s)*1000:.2f} ms")
        print(f"{'Relation Extract':<20} | {(t_rel_e - t_rel_s)*1000:.2f} ms")
        print(f"{'Format Logic':<20} | {(t_fmt_e - t_fmt_s)*1000:.2f} ms")

        if t_pred_s is not None:
            print(f"{'Agent Prediction':<20} | {(t_pred_e - t_pred_s)*1000:.2f} ms")
        else:
            print(f"{'Agent Prediction':<20} | {'N/A':<10}")
        print("-" * 40)

