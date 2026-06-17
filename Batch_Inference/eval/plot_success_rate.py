import json
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phyre
from tqdm import tqdm

ACTION_STATUS_LABELS = ["NOT_SOLVED", "INVALID_ACTION", "INVALID_ACTION_FORMAT"]
GLOBAL_POSITION_LABELS = {
    "TOP-LEFT", "TOP", "TOP-RIGHT",
    "LEFT", "CENTER", "RIGHT",
    "BOTTOM-LEFT", "BOTTOM", "BOTTOM-RIGHT",
}
RELATIVE_POSITION_LABELS = {
    "TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT",
}

FORMAT_REWARD = 0.2
SINGLE_SCENE_REWARD = 0.2
SINGLE_ACTION_REWARD = 0.6
SINGLE_PLACEMENT_REWARD = 0.3
PUNISHMENT_REWARD_MILD = -0.1

INPUT_JSON_PATH = "/home/u5597173/repo/Batch_Inference/batch_inference_output/phyre/qwen3_vl_8b_r64_sft_gdpo_no_align_t30_my_cross_fold_10/ball_within_template/all/USER/2026-04-29_15-50-19/summary_results.json"
base_dir = os.path.dirname(INPUT_JSON_PATH)
json_files = {"Qwen3-VL-8B": os.path.basename(INPUT_JSON_PATH)}
OUTPUT_DIR = os.path.join(base_dir, "eval_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _coerce_attempt_number(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _get_task_type(seed):
    if not seed:
        return ""
    return seed.split(":", 1)[0]


def compute_deviations(feat_act, feat_no):
    min_objs = min(feat_act.shape[1], feat_no.shape[1])
    min_seq = min(feat_act.shape[0], feat_no.shape[0])
    diff_pos = feat_act[:min_seq, :min_objs, :2] - feat_no[:min_seq, :min_objs, :2]
    dist_diff = np.linalg.norm(diff_pos, axis=2)
    diff_x = feat_act[:min_seq, :min_objs, 0] - feat_no[:min_seq, :min_objs, 0]
    diff_y = feat_act[:min_seq, :min_objs, 1] - feat_no[:min_seq, :min_objs, 1]
    diff_rot = feat_act[:min_seq, :min_objs, 2] - feat_no[:min_seq, :min_objs, 2]
    return {
        "dist_diff": dist_diff,
        "diff_rot": diff_rot,
        "diff_x": diff_x,
        "diff_y": diff_y,
        "feat_act": feat_act,
    }


def analyze_event(
    deviations,
    target_obj_id,
    event_type,
    ball_idx=-1,
    pos_threshold=0.0,
    rot_threshold=0.0,
    frames_ahead=12,
):
    dist_diff = deviations["dist_diff"]
    diff_rot = deviations["diff_rot"]
    diff_x = deviations["diff_x"]
    feat_act = deviations["feat_act"]
    num_frames = dist_diff.shape[0]
    if event_type == "collision":
        trigger_frames = np.where(dist_diff[:, target_obj_id] > pos_threshold)[0]
        if len(trigger_frames) == 0:
            return None
        first_frame = trigger_frames[0]
        check_idx = min(first_frame + frames_ahead, num_frames - 1)
        dx = float(diff_x[check_idx, target_obj_id])
        direction = "RIGHT" if dx > 0 else "LEFT"
        x_act_start = float(feat_act[first_frame, target_obj_id, 0])
        x_act_end = float(feat_act[check_idx, target_obj_id, 0])
        dx_start = float(diff_x[first_frame, target_obj_id])
        dx_end = float(diff_x[check_idx, target_obj_id])
        x_no_start = x_act_start - dx_start
        x_no_end = x_act_end - dx_end
        vel_act = x_act_end - x_act_start
        vel_no = x_no_end - x_no_start
        vel_thres = 0.005
        if vel_no > vel_thres and dx < 0:
            if vel_act > vel_thres:
                direction = "BLOCKED"
            elif abs(vel_act) <= vel_thres:
                direction = "STOPPED"
            else:
                direction = "DEFLECTED"
        elif vel_no < -vel_thres and dx > 0:
            if vel_act < -vel_thres:
                direction = "BLOCKED"
            elif abs(vel_act) <= vel_thres:
                direction = "STOPPED"
            else:
                direction = "DEFLECTED"
    elif event_type == "rotation":
        trigger_frames = np.where(np.abs(diff_rot[:, target_obj_id]) > rot_threshold)[0]
        if len(trigger_frames) == 0:
            return None
        first_frame = trigger_frames[0]
        check_idx = min(first_frame + frames_ahead, num_frames - 1)
        d_theta = diff_rot[check_idx, target_obj_id]
        direction = "COUNTER-CLOCKWISE" if d_theta > 0 else "CLOCKWISE"
        theta_act_start = float(feat_act[first_frame, target_obj_id, 2])
        theta_act_end = float(feat_act[check_idx, target_obj_id, 2])
        d_rot_start = float(diff_rot[first_frame, target_obj_id])
        d_rot_end = float(diff_rot[check_idx, target_obj_id])
        theta_no_start = theta_act_start - d_rot_start
        theta_no_end = theta_act_end - d_rot_end

        def get_diff(a, b):
            d = a - b
            if d > 0.5:
                d -= 1.0
            elif d < -0.5:
                d += 1.0
            return d

        omega_act = get_diff(theta_act_end, theta_act_start)
        omega_no = get_diff(theta_no_end, theta_no_start)
        rot_speed_thres = 0.01
        if abs(omega_no) > rot_speed_thres and abs(omega_act) < abs(omega_no) * 0.5:
            direction = "SUPPORT"
    else:
        raise ValueError("event_type must be 'collision' or 'rotation'")
    bx = float(feat_act[first_frame, ball_idx, 0])
    by = float(feat_act[first_frame, ball_idx, 1])
    ox = float(feat_act[first_frame, target_obj_id, 0])
    oy = float(feat_act[first_frame, target_obj_id, 1])
    return {
        "event_type": event_type,
        "frame": int(first_frame),
        "object_id": int(target_obj_id),
        "direction": direction,
        "ball_center_px": [bx, by],
        "object_center_px": [ox, oy],
    }


def get_relative_relationship(anchor_feat, target_feat):
    ax, ay = anchor_feat[0], anchor_feat[1]
    tx, ty = target_feat[0], target_feat[1]
    dx = tx - ax
    dy = ty - ay
    is_top = dy > 0
    is_left = dx < 0
    if is_top:
        return "Top-Left" if is_left else "Top-Right"
    return "Bottom-Left" if is_left else "Bottom-Right"


def get_global_position(target_feat):
    global_names = {
        0: "Top-Left", 1: "Top", 2: "Top-Right",
        3: "Left", 4: "Center", 5: "Right",
        6: "Bottom-Left", 7: "Bottom", 8: "Bottom-Right",
    }
    cx, cy = target_feat[0], target_feat[1]
    th_1 = 1.0 / 3.0
    th_2 = 2.0 / 3.0
    if cx < th_1:
        col = 0
    elif cx < th_2:
        col = 1
    else:
        col = 2
    if cy > th_2:
        row = 0
    elif cy > th_1:
        row = 1
    else:
        row = 2
    grid_id = row * 3 + col
    return grid_id, global_names[grid_id]


def merge_features(featurized_objects):
    colors = featurized_objects.colors
    shapes = featurized_objects.shapes
    return [f"{colors[i]} {shapes[i]}" for i in range(len(colors))]


def _digit_mapping(action):
    action = [int(a) for a in action]
    pred_x, pred_y, pred_r = action
    pred_y = 256 - 1 - pred_y
    x_action = pred_x / (256 - 1)
    y_action = pred_y / (256 - 1)
    d_action = (pred_r - 2) / (32 - 2)
    return np.array([x_action, y_action, d_action], dtype=np.float64).tolist()


def _extract_action(text):
    match = re.search(r"<action>\s*(.*?)\s*</action>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None, None
    try:
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


def parse_numbers(text):
    return [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)]


def coord_to_phyre(coord_list):
    if len(coord_list) != 2 or not all(isinstance(x, (int, float)) for x in coord_list):
        return []
    x_pixel, y_pixel = coord_list
    y_pixel = 256 - y_pixel
    x_norm = x_pixel / (256 - 1)
    y_norm = y_pixel / (256 - 1)
    return [x_norm, y_norm]


def diameter_to_phyre(diameter_pixel):
    return [i / 256 for i in diameter_pixel]


def extract_scene_answer(text):
    scene_pattern = r"<scene_answer>(.*?)</scene_answer>"
    scene_content = re.search(scene_pattern, text, re.DOTALL | re.IGNORECASE)
    if not scene_content:
        return None
    content = scene_content.group(1).strip()
    pattern = r"\[(.*?)\]\s+is at\s+\[(.*?)\]\s+\[(.*?)\]\s+with size\s+\[(.*?)\]"
    matches = re.findall(pattern, content, re.IGNORECASE)
    results = []
    for obj_name, pos_label, pos_coords_str, size_str in matches:
        parsed_size = parse_numbers(size_str)
        if not parsed_size:
            parsed_size = []
        position = coord_to_phyre(parse_numbers(pos_coords_str))
        if len(position) != 2 or not all(isinstance(x, (float, int)) for x in position):
            position = []
        results.append(
            {
                "object": obj_name.strip(),
                "position_label": pos_label.strip(),
                "position": position,
                "size": diameter_to_phyre(parsed_size),
            }
        )
    return results


def normalize_event(event_str):
    evt = event_str.strip().upper()
    if evt in ["PUSH", "COLLIDE WITH", "HIT", "STRIKE"]:
        return "collision"
    if evt in ["TILT", "ROTATE", "SPIN"]:
        return "rotation"
    return "unknown"


def extract_causal_actions(text):
    causal_pattern = r"<causal_actions_answer>(.*?)</causal_actions_answer>"
    causal_content = re.search(causal_pattern, text, re.DOTALL | re.IGNORECASE)
    content = causal_content.group(1).strip() if causal_content else text.strip()
    lines = content.split("\n")
    results = []

    def parse_value(val):
        val = val.strip()
        if re.match(r"^\d+\s*,\s*\d+$", val):
            return [int(n) for n in val.split(",")]
        return val

    for line in lines:
        line_is_record = False
        collision_pattern_0 = (
            r"The \[RED BALL\] should\s+\[(.*?)\]\s+the\s+\[(.*?)\]\s+at\s+\[(.*?)\]"
            r"\s+to\s+push\s+it\s+(?:to|towards)\s+\[(.*?)\]"
        )
        for match in re.finditer(collision_pattern_0, line, re.IGNORECASE):
            raw_action, target, contact_raw, dir_lbl = match.groups()
            if "BAR" in target:
                continue
            results.append(
                {
                    "action": normalize_event(raw_action.strip()),
                    "target": target.strip(),
                    "contact": coord_to_phyre(parse_value(contact_raw)),
                    "direction": dir_lbl.strip(),
                    "active": True,
                }
            )
            line_is_record = True
        if line_is_record:
            continue
        collision_pattern_1 = (
            r"The \[RED BALL\] should\s+\[(.*?)\]\s+the\s+\[(.*?)\]\s+at\s+\[(.*?)\]"
        )
        for match in re.finditer(collision_pattern_1, line, re.IGNORECASE):
            raw_action, target, contact_raw = match.groups()
            if "BAR" in target:
                continue
            mapping_dict_temp = {"BLOCK": "BLOCKED", "STOP": "STOPPED", "DEFLECT": "DEFLECTED"}
            results.append(
                {
                    "action": "collision",
                    "target": target.strip(),
                    "contact": coord_to_phyre(parse_value(contact_raw)),
                    "direction": mapping_dict_temp.get(raw_action.strip().upper(), "unknown"),
                    "active": True,
                }
            )
            line_is_record = True
        if line_is_record:
            continue
        collision_pattern_2 = (
            r"The chain actions cause the\s+\[(.*?)\]\s+to move\s+\[(.*?)\]\s+at\s+point\s+\[(.*?)\]"
        )
        for match in re.finditer(collision_pattern_2, line, re.IGNORECASE):
            target, dir_lbl, contact_raw = match.groups()
            if "BAR" in target:
                continue
            results.append(
                {
                    "action": "collision",
                    "target": target.strip(),
                    "contact": coord_to_phyre(parse_value(contact_raw)),
                    "direction": dir_lbl.strip(),
                    "active": False,
                }
            )
            line_is_record = True
        if line_is_record:
            continue
        collision_pattern_3 = (
            r"The chain actions cause the\s+\[(.*?)\]\s+to be\s+\[(.*?)\]\s+at\s+point\s+\[(.*?)\]"
        )
        for match in re.finditer(collision_pattern_3, line, re.IGNORECASE):
            target, direction, contact_raw = match.groups()
            if "BAR" in target:
                continue
            results.append(
                {
                    "action": "collision",
                    "target": target.strip(),
                    "contact": coord_to_phyre(parse_value(contact_raw)),
                    "direction": direction,
                    "active": False,
                }
            )
            line_is_record = True
        if line_is_record:
            continue
        rotation_pattern_0 = (
            r"The \[RED BALL\] should\s+\[(.*?)\]\s+the\s+\[(.*?)\]\s+at\s+\[(.*?)\]\s+to\s+rotate\s+it\s+\[(.*?)\]"
        )
        for match in re.finditer(rotation_pattern_0, line, re.IGNORECASE):
            raw_action, target, contact_raw, rot_dir = match.groups()
            if "BAR" not in target:
                continue
            results.append(
                {
                    "action": normalize_event(raw_action.strip()),
                    "target": target.strip(),
                    "contact": coord_to_phyre(parse_value(contact_raw)),
                    "direction": rot_dir.strip(),
                    "active": True,
                }
            )
            line_is_record = True
        if line_is_record:
            continue
        rotation_pattern_1 = r"The \[RED BALL\] should\s+\[(.*?)\]\s+the\s+\[(.*?)\]\s+at\s+\[(.*?)\]"
        for match in re.finditer(rotation_pattern_1, line, re.IGNORECASE):
            direction, target, contact_raw = match.groups()
            if "BAR" not in target:
                continue
            results.append(
                {
                    "action": "rotation",
                    "target": target.strip(),
                    "contact": coord_to_phyre(parse_value(contact_raw)),
                    "direction": direction,
                    "active": True,
                }
            )
            line_is_record = True
        if line_is_record:
            continue
        rotation_pattern_2 = (
            r"The chain actions cause the\s+\[(.*?)\]\s+to move\s+\[(.*?)\]\s+at\s+point\s+\[(.*?)\]"
        )
        for match in re.finditer(rotation_pattern_2, line, re.IGNORECASE):
            target, direction, contact_raw = match.groups()
            if "BAR" not in target:
                continue
            results.append(
                {
                    "action": "rotation",
                    "target": target.strip(),
                    "contact": coord_to_phyre(parse_value(contact_raw)),
                    "direction": direction,
                    "active": False,
                }
            )
            line_is_record = True
    return results


def extract_spatial_relationships(text):
    placement_pattern = r"<placement_answer>(.*?)</placement_answer>"
    placements = re.findall(placement_pattern, text, re.DOTALL | re.IGNORECASE)
    extracted_data = []
    relationship_pattern = r"\[(.*?)\]\s+is located at the\s+\[(.*?)\]\s+of the\s+\[(.*?)\][\.]"
    for content in placements:
        matches = re.findall(relationship_pattern, content.strip(), re.IGNORECASE)
        for obj, position, reference in matches:
            if "RED" in obj.strip().upper():
                extracted_data.append({"object": reference.strip().upper(), "position": position.strip()})
    return extracted_data


class PhyreAttemptRewardScorer:
    def __init__(self, eval_setup, task_ids):
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)
        self.simulator = phyre.initialize_simulator(task_ids, action_tier)
        self.task_id_to_index_map = {task_id: idx for idx, task_id in enumerate(task_ids)}
        self.no_action_cache = {}
        all_compiled_tasks = phyre.loader.load_compiled_task_dict()
        for task_id in tqdm(task_ids, desc="Building no-action cache"):
            task_tmp = all_compiled_tasks[task_id]
            no_action_sim = phyre.simulator.simulate_task(
                task_tmp,
                steps=phyre.simulator.DEFAULT_MAX_STEPS,
                stride=1,
            )
            feats = []
            for scene in no_action_sim.sceneList:
                fo = phyre.simulator.scene_to_featurized_objects(scene)
                feats.append(fo.features[0])
            self.no_action_cache[task_id] = np.array(feats, dtype=np.float64)
        self.punishment_reward_mild = -0.1
        self.punishment_reward_medium = -0.5
        self.single_scene_reward = 0.2
        self.single_action_reward = 0.6
        self.single_placement_reward = 0.3
        self.format_reward = 0.2

    def _calculate_soft_grid_score(self, true_exact_label, pred_label, mode, smooth_reward=False):
        pred_label = pred_label.upper()
        true_exact_label = true_exact_label.upper()
        if mode == "relative":
            idx_to_coord = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
            label_to_idx = {
                "TOP-LEFT": 0,
                "TOP-RIGHT": 1,
                "BOTTOM-LEFT": 2,
                "BOTTOM-RIGHT": 3,
            }
        elif mode == "global":
            idx_to_coord = {
                0: (0, 2), 1: (1, 2), 2: (2, 2),
                3: (0, 1), 4: (1, 1), 5: (2, 1),
                6: (0, 0), 7: (1, 0), 8: (2, 0),
            }
            label_to_idx = {
                "TOP-LEFT": 0, "TOP": 1, "TOP-RIGHT": 2,
                "LEFT": 3, "CENTER": 4, "RIGHT": 5,
                "BOTTOM-LEFT": 6, "BOTTOM": 7, "BOTTOM-RIGHT": 8,
            }
        else:
            return 0.0
        if true_exact_label not in label_to_idx:
            return 0.0
        if pred_label == true_exact_label:
            return 1.0
        if pred_label not in label_to_idx:
            return 0.0
        if not smooth_reward:
            return 0.0
        pred_coord = np.array(idx_to_coord[label_to_idx[pred_label]], dtype=np.float64)
        true_coord = np.array(idx_to_coord[label_to_idx[true_exact_label]], dtype=np.float64)
        dist = np.linalg.norm(pred_coord - true_coord)
        reward = 1.0 - (dist / 2.82)
        return max(0.0, min(1.0, reward))

    def score_attempt(self, task_id, solution_str):
        score_dict = {"placement_reward": 0.0, "collision_reward": 0.0, "grounding_reward": 0.0}
        if task_id not in self.task_id_to_index_map:
            return score_dict
        _, mapped_action = _extract_action(solution_str)
        if mapped_action is None:
            return score_dict
        task_index = self.task_id_to_index_map[task_id]
        action_array = np.array(mapped_action, dtype=np.float64)
        sim_result = self.simulator.simulate_action(
            task_index=task_index,
            action=action_array,
            need_images=False,
            need_featurized_objects=True,
            stride=1,
        )
        if sim_result.status.is_invalid():
            merged_features = merge_features(self.simulator.initial_featurized_objects[task_index])
        else:
            merged_features = merge_features(sim_result.featurized_objects)
        extract_objects = extract_scene_answer(solution_str)
        extract_collisions = extract_causal_actions(solution_str)
        extract_relations = extract_spatial_relationships(solution_str)
        featurized_objects = sim_result.featurized_objects
        red_ball_idx = -1

        if extract_objects is None or len(extract_objects) == 0:
            grounding_reward = 0.0
        else:
            grounding_reward = self.format_reward
            processed_anchors = set()
            init_features = (
                self.simulator.initial_featurized_objects[task_index].features[0]
                if sim_result.status.is_invalid()
                else featurized_objects.features[0]
            )
            for obj_info in extract_objects:
                anchor = obj_info["object"].upper()
                if anchor in processed_anchors or anchor not in merged_features:
                    grounding_reward += self.punishment_reward_mild
                    continue
                processed_anchors.add(anchor)
                anchor_idx = merged_features.index(anchor)
                anchor_features = init_features[anchor_idx]
                _, true_position = get_global_position(anchor_features)
                pred_position_label = obj_info["position_label"].upper()
                text_score = self._calculate_soft_grid_score(true_position, pred_position_label, "global", True)
                if len(obj_info["size"]) != 1 or len(obj_info["position"]) != 2:
                    num_score = 0.0
                else:
                    predicted_pos = obj_info["position"] + obj_info["size"]
                    gt_pos = anchor_features[:2].tolist() + [anchor_features[3].tolist()]
                    distance = math.sqrt(sum((predicted_pos[i] - gt_pos[i]) ** 2 for i in range(3)))
                    num_score = 1.0 - min(distance / math.sqrt(3), 1.0)
                grounding_reward += self.single_scene_reward * ((0.5 * text_score) + (0.5 * num_score))
            grounding_reward = max(0.0, min(1.0, grounding_reward))

        if sim_result.status.is_invalid():
            score_dict["grounding_reward"] = grounding_reward
            return score_dict

        processed_targets = set()
        if extract_collisions is None or len(extract_collisions) == 0:
            collision_reward = 0.0
        else:
            collision_reward = self.format_reward
            no_action_feats = self.no_action_cache[task_id]
            deviation = compute_deviations(featurized_objects.features, no_action_feats)
            shapes = featurized_objects.shapes
            for causal_info in extract_collisions:
                causal_event = causal_info["action"]
                if causal_event == "unknown":
                    collision_reward += self.punishment_reward_mild
                    continue
                target = causal_info["target"].upper()
                if target in processed_targets or target not in merged_features or target == "RED BALL":
                    collision_reward += self.punishment_reward_mild
                    continue
                if causal_info["active"] is True:
                    processed_targets.add(target)
                target_indices = [i for i, name in enumerate(merged_features) if name == target]
                target_shape = shapes[target_indices[0]]
                scan_event_type = "rotation" if target_shape == "BAR" else "collision"
                candidate_events = []
                for idx in target_indices:
                    ev = analyze_event(deviation, target_obj_id=idx, event_type=scan_event_type)
                    if ev is not None:
                        candidate_events.append(ev)
                if not candidate_events:
                    collision_reward += self.punishment_reward_mild
                    continue
                event_info = min(candidate_events, key=lambda x: x["frame"])
                if causal_event != event_info["event_type"]:
                    collision_reward += self.punishment_reward_mild
                    continue
                if len(causal_info["contact"]) != 2:
                    collision_reward += self.punishment_reward_mild
                    continue
                pred_contact_point = causal_info["contact"]
                contact_point = event_info["ball_center_px"]
                if event_info["direction"].upper() == causal_info["direction"].upper():
                    distance = math.sqrt(
                        (pred_contact_point[0] - contact_point[0]) ** 2
                        + (pred_contact_point[1] - contact_point[1]) ** 2
                    )
                    collision_reward += self.single_action_reward * (1 - min(distance / math.sqrt(2), 1.0))
                else:
                    collision_reward += self.punishment_reward_mild
            collision_reward = min(1.0, collision_reward)

        if extract_relations is None or len(extract_relations) == 0:
            placement_reward = 0.0
        else:
            placement_reward = self.format_reward
            init_features = featurized_objects.features[0]
            red_features = init_features[red_ball_idx]
            anchor_set = set()
            for relation in extract_relations:
                anchor, pred_relation = relation["object"], relation["position"]
                if anchor in anchor_set:
                    placement_reward += self.punishment_reward_mild
                    continue
                if anchor == "WHOLE SCENE":
                    anchor_set.add(anchor)
                    _, exact_text = get_global_position(red_features)
                    score = self._calculate_soft_grid_score(exact_text, pred_relation, "global", True)
                    placement_reward += self.single_placement_reward * score
                    continue
                if anchor not in merged_features:
                    placement_reward += self.punishment_reward_mild
                    continue
                anchor_set.add(anchor)
                anchor_idx = merged_features.index(anchor)
                anchor_features = init_features[anchor_idx]
                exact_text = get_relative_relationship(anchor_features, red_features)
                score = self._calculate_soft_grid_score(exact_text, pred_relation, "relative", False)
                placement_reward += self.single_placement_reward * score
            for target in list(processed_targets) + ["WHOLE SCENE"]:
                if target not in anchor_set:
                    placement_reward += self.punishment_reward_medium
            placement_reward = max(0.0, min(1.0, placement_reward))
        score_dict["placement_reward"] = placement_reward
        score_dict["collision_reward"] = collision_reward
        score_dict["grounding_reward"] = grounding_reward
        return score_dict


def _calculate_attempt_rewards(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    metadata = data[0]
    eval_setup = metadata.get("eval_setup", "ball_within_template")
    tasks = data[1:]
    task_ids = [task.get("seed") for task in tasks if task.get("seed")]
    reward_scorer = PhyreAttemptRewardScorer(eval_setup=eval_setup, task_ids=task_ids)
    records = []
    for task in tasks:
        seed = task.get("seed")
        if not seed:
            continue
        for attempt in task.get("attempt_history", []):
            attempt_num = _coerce_attempt_number(attempt.get("attempt_number"))
            if attempt_num is None:
                continue
            vlm_response = attempt.get("vlm_response", "")
            reward_dict = reward_scorer.score_attempt(seed, vlm_response)
            grounding_reward = reward_dict["grounding_reward"]
            collision_reward = reward_dict["collision_reward"]
            placement_reward = reward_dict["placement_reward"]
            avg_reward = (placement_reward + collision_reward + grounding_reward) / 3.0
            records.append(
                {
                    "Task_ID": seed,
                    "Task_Type": _get_task_type(seed),
                    "Attempt_Number": attempt_num,
                    "Placement_Reward": placement_reward,
                    "Collision_Reward": collision_reward,
                    "Grounding_Reward": grounding_reward,
                    "Average_Reward": avg_reward,
                    "Simulation_Status": attempt.get("simulation_status", ""),
                    "Is_Solved_In_Attempt": bool(attempt.get("is_solved_in_this_attempt", False)),
                }
            )
    return pd.DataFrame(records)


def _get_solved_attempt(task):
    attempt_history = task.get("attempt_history", [])
    solved_attempts = []
    for attempt in attempt_history:
        if not attempt.get("is_solved_in_this_attempt", False):
            continue
        attempt_number = _coerce_attempt_number(attempt.get("attempt_number"))
        if attempt_number is not None:
            solved_attempts.append(attempt_number)
    if solved_attempts:
        return min(solved_attempts)
    if task.get("is_solved", False):
        attempt_numbers = [
            _coerce_attempt_number(attempt.get("attempt_number"))
            for attempt in attempt_history
        ]
        attempt_numbers = [n for n in attempt_numbers if n is not None]
        if attempt_numbers:
            return min(attempt_numbers)
    return None


def _load_tasks_by_id(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    tasks = data[1:]  # Skip metadata
    tasks_by_id = {}
    for task in tasks:
        seed = task.get("seed")
        if not seed:
            continue
        # if seed.split(":")[0] not in ["00000", "00002"]:
        #     continue
        solved_at_attempt = _get_solved_attempt(task)
        existing = tasks_by_id.get(seed)
        if existing is None:
            tasks_by_id[seed] = {"solved_at_attempt": solved_at_attempt}
            continue
        if solved_at_attempt is None:
            continue
        existing_attempt = existing.get("solved_at_attempt")
        if existing_attempt is None or solved_at_attempt < existing_attempt:
            existing["solved_at_attempt"] = solved_at_attempt
    return tasks_by_id

# Function to extract cumulative success rates
def extract_cumulative_success_rate(file_path):
    tasks_by_id = _load_tasks_by_id(file_path)
    total_tasks = len(tasks_by_id)
    cumulative_success = [0] * 10
    for task in tasks_by_id.values():
        solved_at_attempt = task.get("solved_at_attempt")
        if solved_at_attempt is None:
            continue
        if 1 <= solved_at_attempt <= 10:
            for i in range(solved_at_attempt - 1, 10):
                cumulative_success[i] += 1
    # Calculate success rate
    success_rate = [count / total_tasks if total_tasks else 0.0 for count in cumulative_success]
    return success_rate

# Function to extract cumulative success rates by task type
def extract_success_rate_by_task_type(file_path):
    tasks_by_id = _load_tasks_by_id(file_path)
    task_type_success = {}
    task_type_count = {}
    for task_id, task in tasks_by_id.items():
        task_type = _get_task_type(task_id)
        if task_type not in task_type_success:
            task_type_success[task_type] = 0
            task_type_count[task_type] = 0
        if task.get("solved_at_attempt") is not None:
            task_type_success[task_type] += 1
        task_type_count[task_type] += 1
    # Calculate success rate for each task type
    success_rate_by_task_type = {task_type: task_type_success[task_type] / task_type_count[task_type] for task_type in task_type_success}
    return success_rate_by_task_type

# Function to extract cumulative success rates for first N task types only
def extract_cumulative_success_rate_first_n_types(file_path, first_n_types):
    tasks_by_id = _load_tasks_by_id(file_path)
    # Filter tasks to only include first N task types
    filtered_tasks = {
        task_id: task
        for task_id, task in tasks_by_id.items()
        if _get_task_type(task_id) in first_n_types
    }
    total_tasks = len(filtered_tasks)
    if total_tasks == 0:
        return [0] * 10
    cumulative_success = [0] * 10
    for task in filtered_tasks.values():
        solved_at_attempt = task.get("solved_at_attempt")
        if solved_at_attempt is None:
            continue
        if 1 <= solved_at_attempt <= 10:
            for i in range(solved_at_attempt - 1, 10):
                cumulative_success[i] += 1
    # Calculate success rate
    success_rate = [count / total_tasks if total_tasks else 0.0 for count in cumulative_success]
    return success_rate

# Function to extract counts of key simulation statuses
def extract_action_status_counts(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    tasks = data[1:]  # Skip metadata
    status_counts = {status: 0 for status in ACTION_STATUS_LABELS}
    for task in tasks:
        for attempt in task.get("attempt_history", []):
            status = attempt.get("simulation_status")
            if status in status_counts:
                status_counts[status] += 1
    return status_counts

# Extract data for each setting
cumulative_success_data = {setting: extract_cumulative_success_rate(os.path.join(base_dir, file)) for setting, file in json_files.items()}
success_rate_data_by_task_type = {setting: extract_success_rate_by_task_type(os.path.join(base_dir, file)) for setting, file in json_files.items()}
action_status_counts = {setting: extract_action_status_counts(os.path.join(base_dir, file)) for setting, file in json_files.items()}
attempt_rewards_data = {
    setting: _calculate_attempt_rewards(os.path.join(base_dir, file))
    for setting, file in json_files.items()
}

# Plotting
plt.figure(figsize=(10, 6))
for setting, success_rates in cumulative_success_data.items():
    plt.plot(range(1, 11), success_rates, label=setting)

plt.title('Cumulative Success Rates for Different Settings')
plt.xlabel('Attempt Number')
plt.ylabel('Cumulative Success Rate')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_success_rates.png"))
plt.show()

# Save cumulative success rates to CSV
cumulative_df = pd.DataFrame(cumulative_success_data, index=range(1, 11))
cumulative_df.index.name = 'Attempt_Number'
cumulative_df.to_csv(os.path.join(OUTPUT_DIR, "cumulative_success_rates.csv"))
print("Cumulative success rates saved to cumulative_success_rates.csv")

# Plotting success rates by task type
plt.figure(figsize=(20, 8))
for setting, success_rates in success_rate_data_by_task_type.items():
    task_types = list(success_rates.keys())
    rates = list(success_rates.values())
    plt.plot(task_types, rates, label=setting)

plt.title('Success Rates by Task Type for Different Settings')
plt.xlabel('Task Type')
plt.ylabel('Success Rate')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig(os.path.join(OUTPUT_DIR, "success_rates_by_task_type.png"))
plt.show()

# Save success rates by task type to CSV
# First, get all unique task types across all settings
all_task_types = set()
for success_rates in success_rate_data_by_task_type.values():
    all_task_types.update(success_rates.keys())
all_task_types = sorted(list(all_task_types))

# Create DataFrame with all task types as index
task_type_df_data = {}
for setting, success_rates in success_rate_data_by_task_type.items():
    task_type_df_data[setting] = [success_rates.get(task_type, 0.0) for task_type in all_task_types]

task_type_df = pd.DataFrame(task_type_df_data, index=all_task_types)
task_type_df.index.name = 'Task_Type'
task_type_df.to_csv(os.path.join(OUTPUT_DIR, "success_rates_by_task_type.csv"))
print("Success rates by task type saved to success_rates_by_task_type.csv")

# ========== ACTION STATUS COUNTS ==========
action_status_df_data = {}
for setting, counts in action_status_counts.items():
    action_status_df_data[setting] = [counts.get(status, 0) for status in ACTION_STATUS_LABELS]

action_status_df = pd.DataFrame(action_status_df_data, index=ACTION_STATUS_LABELS)
action_status_df.index.name = 'Simulation_Status'
action_status_df.to_csv(os.path.join(OUTPUT_DIR, "action_status_counts.csv"))
print("Action status counts saved to action_status_counts.csv")

# ========== ATTEMPT REWARDS ==========
attempt_reward_summary_rows = []
attempt_reward_by_attempt_rows = []
for setting, reward_df in attempt_rewards_data.items():
    if reward_df.empty:
        continue
    reward_df_with_setting = reward_df.copy()
    reward_df_with_setting.insert(0, "Setting", setting)
    reward_df_with_setting.to_csv(
        os.path.join(OUTPUT_DIR, f"attempt_rewards_{setting.replace(' ', '_')}.csv"),
        index=False,
    )
    avg_dict = reward_df[
        ["Placement_Reward", "Collision_Reward", "Grounding_Reward", "Average_Reward"]
    ].mean().to_dict()
    attempt_reward_summary_rows.append(
        {
            "Setting": setting,
            "Placement_Reward_Avg": avg_dict["Placement_Reward"],
            "Collision_Reward_Avg": avg_dict["Collision_Reward"],
            "Grounding_Reward_Avg": avg_dict["Grounding_Reward"],
            "Average_Reward_Avg": avg_dict["Average_Reward"],
            "Num_Attempts": len(reward_df),
            "Num_Tasks": reward_df["Task_ID"].nunique(),
        }
    )
    by_attempt = (
        reward_df.groupby("Attempt_Number")[
            ["Placement_Reward", "Collision_Reward", "Grounding_Reward", "Average_Reward"]
        ]
        .mean()
        .reset_index()
    )
    by_attempt.insert(0, "Setting", setting)
    attempt_reward_by_attempt_rows.append(by_attempt)

if attempt_reward_summary_rows:
    attempt_reward_summary_df = pd.DataFrame(attempt_reward_summary_rows)
    attempt_reward_summary_df.to_csv(os.path.join(OUTPUT_DIR, "attempt_reward_summary.csv"), index=False)
    print("Attempt reward summary saved to attempt_reward_summary.csv")

if attempt_reward_by_attempt_rows:
    attempt_reward_by_attempt_df = pd.concat(attempt_reward_by_attempt_rows, ignore_index=True)
    attempt_reward_by_attempt_df.to_csv(
        os.path.join(OUTPUT_DIR, "attempt_reward_by_attempt_number.csv"),
        index=False,
    )
    print("Attempt reward by attempt number saved to attempt_reward_by_attempt_number.csv")