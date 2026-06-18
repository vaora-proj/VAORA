import numpy as np

def compute_deviations(feat_act, feat_no):
    """
    Step 1: Extract differences between Action and No-Action simulations.
    
    Args:
        feat_act: Numpy array of shape (Time, Num_Objects, Feat_Dim)
        feat_no:  Numpy array of shape (Time, Num_Objects, Feat_Dim)
        
    Returns:
        dict: A dictionary containing pre-computed deviation arrays.
    """
    # Ensure shapes match for element-wise subtraction
    # If feat_act includes an extra object (like the ball) at the end compared to feat_no, 
    # slice it to match feat_no.
    min_objs = min(feat_act.shape[1], feat_no.shape[1])
    min_seq = min(feat_act.shape[0], feat_no.shape[0])
    
    # Calculate differences (Action - NoAction) for relevant objects
    # positions are usually indices 0 and 1 (x, y)
    diff_pos = feat_act[:min_seq, :min_objs, :2] - feat_no[:min_seq, :min_objs, :2]
    
    # Magnitude of displacement (Euclidean distance)
    dist_diff = np.linalg.norm(diff_pos, axis=2)
    
    
    # X-axis difference (for Left/Right logic)
    diff_x = feat_act[:min_seq, :min_objs, 0] - feat_no[:min_seq, :min_objs, 0] 
    
    # Y-axis difference can be computed similarly if needed
    diff_y = feat_act[:min_seq, :min_objs, 1] - feat_no[:min_seq, :min_objs, 1]

    # Angular difference (assuming index 2 is rotation)
    diff_rot = feat_act[:min_seq, :min_objs, 2] - feat_no[:min_seq, :min_objs, 2] 
    
    return {
        "dist_diff": dist_diff,
        "diff_rot": diff_rot,
        "diff_x": diff_x,
        "diff_y": diff_y,
        "feat_act": feat_act  # Keep raw features for coordinate extraction
    }

def analyze_event(
    deviations, 
    target_obj_id, 
    event_type, 
    ball_idx=-1, 
    pos_threshold=0.0000, 
    rot_threshold=0.0000,
    frames_ahead=12
):
    """
    Step 2 & 3: Detect specific event for a specific object and return result.
    
    Args:
        deviations (dict): Output from compute_deviations.
        target_obj_id (int): The ID of the object to analyze.
        event_type (str): 'collision' or 'rotation'.
        
    Returns:
        dict: Details of the event if found, else None.
    """
    dist_diff = deviations["dist_diff"]
    diff_rot  = deviations["diff_rot"]
    diff_x    = deviations["diff_x"]
    feat_act  = deviations["feat_act"]
    
    num_frames = dist_diff.shape[0]

    # --- Logic Switch ---
    if event_type == "collision":
        # 1. Find frames where physical displacement > threshold
        trigger_frames = np.where(dist_diff[:, target_obj_id] > pos_threshold)[0]
        
        if len(trigger_frames) == 0:
            return None

        first_frame = trigger_frames[0]
        
        # Look 12 frames ahead for direction (Left/Right)
        check_idx = min(first_frame + frames_ahead, num_frames - 1)
        
        # --- ENHANCED LOGIC START ---
        # 1. Calculate Relative Deviation (Original Logic)
        dx = float(diff_x[check_idx, target_obj_id])
        direction = "RIGHT" if dx > 0 else "LEFT"
        
        # 2. Reconstruct Velocity Context (Absolute vs No-Action)
        # diff_x[t] = x_act[t] - x_no[t]  =>  x_no[t] = x_act[t] - diff_x[t]
        
        x_act_start = float(feat_act[first_frame, target_obj_id, 0])
        x_act_end   = float(feat_act[check_idx, target_obj_id, 0])
        
        dx_start    = float(diff_x[first_frame, target_obj_id])
        dx_end      = float(diff_x[check_idx, target_obj_id])
        
        x_no_start  = x_act_start - dx_start
        x_no_end    = x_act_end - dx_end
        
        vel_act     = x_act_end - x_act_start
        vel_no      = x_no_end - x_no_start
        
        VEL_THRES   = 0.005 # Approx 1-2 pixels in normalized space
        
        # Case A: Object was moving RIGHT (vel_no > 0)
        if vel_no > VEL_THRES:
            # If relative deviation is LEFT (dx < 0), it implies slowing down
            if dx < 0:
                if vel_act > VEL_THRES:
                    direction = "BLOCKED"   # Still moving right, but slowed
                elif abs(vel_act) <= VEL_THRES:
                    direction = "STOPPED"   # Stopped
                else: 
                    direction = "DEFLECTED" # Reversed/Bounced back
        
        # Case B: Object was moving LEFT (vel_no < 0)
        elif vel_no < -VEL_THRES:
            # If relative deviation is RIGHT (dx > 0), it implies slowing down
            if dx > 0:
                if vel_act < -VEL_THRES:
                    direction = "BLOCKED"   # Still moving left, but slowed
                elif abs(vel_act) <= VEL_THRES:
                    direction = "STOPPED"   # Stopped
                else:
                    direction = "DEFLECTED" # Reversed/Bounced back
        # --- ENHANCED LOGIC END ---

    elif event_type == "rotation":
        # 1. Find frames where angular deviation > threshold
        trigger_frames = np.where(np.abs(diff_rot[:, target_obj_id]) > rot_threshold)[0]
        
        if len(trigger_frames) == 0:
            return None
            
        first_frame = trigger_frames[0]
        
        # Look 2 frames ahead for direction (CW/CCW)
        check_idx = min(first_frame + frames_ahead, num_frames - 1)
        d_theta = diff_rot[check_idx, target_obj_id]
        direction = "COUNTER-CLOCKWISE" if d_theta > 0 else "CLOCKWISE"
        
        # --- ENHANCED LOGIC FOR SUPPORT ---
        # Reconstruct Angles
        theta_act_start = float(feat_act[first_frame, target_obj_id, 2])
        theta_act_end   = float(feat_act[check_idx, target_obj_id, 2])
        
        d_rot_start     = float(diff_rot[first_frame, target_obj_id])
        d_rot_end       = float(diff_rot[check_idx, target_obj_id])
        
        theta_no_start  = theta_act_start - d_rot_start
        theta_no_end    = theta_act_end - d_rot_end
        
        # Helper for handling 0-1 wrap around
        def get_diff(a, b):
            d = a - b
            if d > 0.5: d -= 1.0
            elif d < -0.5: d += 1.0
            return d
            
        omega_act = get_diff(theta_act_end, theta_act_start)
        omega_no  = get_diff(theta_no_end, theta_no_start)
        
        
        ROT_SPEED_THRES = 0.01 # Approx 3-4 degrees
        
        # If it was rotating significantly in No-Action
        if abs(omega_no) > ROT_SPEED_THRES:
            # And now it is rotating significantly LESS (e.g., < 50% of original speed)
            if abs(omega_act) < abs(omega_no) * 0.5:
                direction = "SUPPORT"
        # --- ENHANCED LOGIC END ---

    else:
        raise ValueError("event_type must be 'collision' or 'rotation'")

    # --- Common Return Logic ---
    # Get Ball Center at the moment of the event
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
        "object_center_px": [ox, oy]
    } 