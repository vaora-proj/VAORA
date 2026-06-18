def get_relative_relationship(anchor_feat, target_feat):
    """
    Determines the relative position of the target with respect to the anchor
    based purely on center points.
    
    Returns one of 4 labels: 'Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right'.
    """
    # Extract centers (x, y)
    ax, ay = anchor_feat[0], anchor_feat[1]
    tx, ty = target_feat[0], target_feat[1]

    # Calculate deltas
    dx = tx - ax
    dy = ty - ay

    # Determine vertical relationship (Top vs Bottom)
    is_top = dy > 0
    
    # Determine horizontal relationship (Left vs Right)
    is_left = dx < 0

    if is_top:
        return "Top-Left" if is_left else "Top-Right"
    else:
        return "Bottom-Left" if is_left else "Bottom-Right"

def get_global_position(target_feat):
    """
    Determines the global position of the target in the scene (3x3 grid)
    based on the object's center point.
    """
    GLOBAL_NAMES = {
        0: "Top-Left",    1: "Top",    2: "Top-Right",
        3: "Left",        4: "Center", 5: "Right",
        6: "Bottom-Left", 7: "Bottom", 8: "Bottom-Right"
    }

    # Extract center (x, y)
    cx, cy = target_feat[0], target_feat[1]

    # Grid thresholds
    TH_1 = 1.0 / 3.0
    TH_2 = 2.0 / 3.0

    # Determine Column (0, 1, 2)
    col = -1
    if cx < TH_1:
        col = 0 # Left
    elif cx < TH_2:
        col = 1 # Center
    else:
        col = 2 # Right

    # Determine Row (0=Top, 1=Mid, 2=Bot)
    # Note: Assuming y=0 is bottom and y=1 is top
    row = -1
    if cy > TH_2:
        row = 0 # Top
    elif cy > TH_1:
        row = 1 # Mid
    else:
        row = 2 # Bottom

    # Calculate Grid ID (Row-major: 0 to 8)
    grid_id = row * 3 + col
    
    return grid_id, GLOBAL_NAMES[grid_id]