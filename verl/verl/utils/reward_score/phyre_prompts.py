import random

SYSTEM_PROMPT = "You are a helpful assistant."

global_spatial = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT", "CENTER", "LEFT", "RIGHT", "TOP", "BOTTOM"]
relative_spatial = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"]
causal_action_collisions = ["PUSH", "COLLIDE WITH", "HIT", "STRIKE", "BLOCKED", "STOPPED", "DEFLECTED"]
causal_action_rotations = ["TILT", "ROTATE", "SPIN"]
collision_directions = ["LEFT", "RIGHT"]
rotation_directions = ["CLOCKWISE", "COUNTER-CLOCKWISE"]
causal_action_blocks = ["BLOCK", "STOP", "DEFLECT"]
causal_action_supports = ["SUPPORT"]
chain_action_block_states = ["BLOCKED", "STOPPED", "DEFLECTED"]

causal_action_collision_template = '- Template Collision (for * BALL or * JAR):\n- ACTION must be one of: <COLLISION_ACTIONS>\n- DIRECTION must be one of: <COLLISION_DIRECTIONS>\nThe [RED BALL] should [ACTION] the [TARGET OBJECT] at the contact point [x, y] to push it towards [DIRECTION].'

additional_causal_action_collision_template = '- Template Collision (for additional explaination of GREEN BALL or GREEN JAR):\n- DIRECTION must be one of: <COLLISION_DIRECTIONS>\nThe chain actions cause the [GREEN BALL or GREEN JAR] to move [DIRECTION] at point [x, y].'

causal_action_block_template = '- Template Block (for * BALL or * JAR):\n- ACTION must be one of: <BLOCK_ACTIONS>\nThe [RED BALL] should [ACTION] the [TARGET OBJECT] at the contact point [x, y].'

additional_causal_action_block_template = '- Template Block (for additional explaination of GREEN BALL or GREEN JAR):\n- STATE must be one of: <CHAIN_BLOCK_STATES>\nThe chain actions cause the [GREEN BALL or GREEN JAR] to be [STATE] at point [x, y].'

causal_action_rotation_template = '- Template Rotation (for * BAR):\n- ACTION must be one of: <ROTATION_ACTIONS>\n- DIRECTION must be one of: <ROTATION_DIRECTIONS>\nThe [RED BALL] should [ACTION] the [TARGET OBJECT] at the contact point [x, y] to rotate it [DIRECTION].'

additional_causal_action_rotation_template = '- Template Rotation (for additional explaination of GREEN BAR):\n- DIRECTION must be one of: <ROTATION_DIRECTIONS>\nThe chain actions cause the [GREEN BAR] to move [DIRECTION] at point [x, y].'

causal_action_support_template = '- Template Support (for * BAR):\n- ACTION must be one of: <SUPPORT_ACTIONS>\nThe [RED BALL] should [ACTION] the [TARGET OBJECT] at the contact point [x, y].'

USER_PROMPT = f"""You are a physics reasoning agent in the PHYRE environment.

ENVIRONMENT RULES
- The scene is 256x256 pixels with a white background.
- Objects are colored: black, gray, blue, green, purple.
- Static objects: BLACK and PURPLE are immovable.
- Dynamic objects: GRAY, BLUE, GREEN fall under gravity in 2D.
- Scene boundaries are walls/floor.
- Objects shapes: BALL, BAR, JAR.
- The goal: make any [GREEN *] object touch any [BLUE *] dynamic object OR any [PURPLE *] static object.
- You can add exactly one RED BALL as the action: [x, y, r]
  - x in [0, 255] (0=left edge, 255=right edge)
  - y in [0, 255] (0=top edge, 255=bottom edge)
  - r in [2, 32]

OBJECT NAMING CONVENTION (STRICT)
- When you mention an object, always use: [COLOR SHAPE] in UPPERCASE, e.g. [GREEN BALL], [BLUE BALL], [PURPLE BAR], [GRAY JAR].
- The added object must always be written exactly as: [RED BALL]
- **EXCEPTION: In reasoning sections only, you may use lowercase natural language (e.g., "the green ball", "blue bar") for readability.**

INPUT
This is the input scene: <image>

TASK (what to do)
1) Identify the goal pair: which [GREEN *] must touch which [BLUE *] or [PURPLE *].
2) Identify blockers: walls, gaps, covers, containers, slopes, platforms, and black obstacles.
3) Choose the main causal strategy (<CAUSAL_ACTIONS>).
4) Estimate coordinates of relevant objects (approximate is OK but must be plausible).
5) Describe the red ball placement relative to objects AND the whole scene.
6) Output the final action [x, y, r].
7) You are encouraged to proposed different reasoning in each section and different strategies to improve diversity.

OUTPUT FORMAT (MUST MATCH EXACTLY)
You MUST output EXACTLY these sections in EXACT order, and NOTHING else:

<scene_reasoning>
...
</scene_reasoning>
<scene_answer>
...
</scene_answer>
<causal_actions_reasoning>
...
</causal_actions_reasoning>
<causal_actions_answer>
...
</causal_actions_answer>
<placement_reasoning>
...
</placement_reasoning>
<placement_answer>
...
</placement_answer>
<action>
[x, y, r]
</action>

CONTENT RULES PER SECTION

<scene_reasoning>
**REASONING REQUIREMENTS:**
- Identify all relevant objects and estimate their approximate positions.
- State the goal clearly (which green touches which blue/purple).
- Analyze spatial relationships: alignment, distance, and location relative to the scene center (Left vs Right side).
- Identify obstacles between goal objects.
- Assess whether objects will naturally interact or need intervention.
</scene_reasoning>

<scene_answer>
- List the key goal objects (at least the [GREEN *], [GRAY *] and its goal [BLUE *]/[PURPLE *], good to include [BLACK *] or [GRAY *]).
- Include any critical blockers or intermediate objects referenced in your reasoning.
- Use THIS exact line style (copy the template):
  [OBJECT] is at [GLOBAL SPATIAL] [x, y] with size [d].
- GLOBAL SPATIAL must be one of:
  <GLOBAL_SPATIAL_OPTIONS>
- x,y,d must be numbers (integers preferred) and ranged from 0-255.
</scene_answer>

<causal_actions_reasoning>
**REASONING REQUIREMENTS:**
- Reference coordinates from <scene_answer> when describing positions.
- Choose and justify ONE causal strategy (<CAUSAL_ACTIONS>).
- Describe the event chain: red ball → interaction (COLLIDE OR SPIN) → target motion → goal achievement. 
- Explain the contact point location and why it creates the desired motion.
- Verify the approach avoids obstacles and reaches the goal.
</causal_actions_reasoning>

<causal_actions_answer>
- Output 1–2 lines total.
- Choose ONE of these templates:
<CAUSAL_TEMPLATE_1>

<CAUSAL_TEMPLATE_2>

<CAUSAL_TEMPLATE_3>

<CAUSAL_TEMPLATE_4>

If the causal action doesn't directly interact with the green object, use the following templates to further explain the chain action of the green object:

<ADDITIONAL_CAUSAL_TEMPLATE_1>

<ADDITIONAL_CAUSAL_TEMPLATE_2>

<ADDITIONAL_CAUSAL_TEMPLATE_3>

- TARGET OBJECT must be a real object in the scene.
- [x, y] must be the estimated global coordinates (0-255) where the objects touch.
</causal_actions_answer>

<placement_reasoning>
**REASONING REQUIREMENTS:**
- Reference target position from <scene_answer> and contact point from <causal_actions_answer>.
- Calculate red ball's initial position to achieve the desired contact point (either via gravity trajectory OR immediate static contact).
- Account for ball radius and gravity trajectory.
- Justify radius choice (larger for momentum, smaller for precision). If using SUPPORT/TILT, consider placing the ball at the BOTTOM to act as a static fulcrum.
- Verify placement avoids static obstacles. CRITICAL PHYSICS RULE: To push an object to a [DIRECTION], the [RED BALL] must be placed on the OPPOSITE side (e.g., to push LEFT, place on RIGHT).
</placement_reasoning>

<placement_answer>
- Output 1-3 lines, using these templates:

[RED BALL] is located at the [RELATIVE SPATIAL] of the [TARGET OBJECT A].
[RED BALL] is located at the [RELATIVE SPATIAL] of the [TARGET OBJECT B].
[RED BALL] is located at the [GLOBAL SPATIAL] of the [WHOLE SCENE].

- RELATIVE SPATIAL must be one of:
  <RELATVE_SPATIAL_OPTIONS>
- GLOBAL SPATIAL must be one of:
  <GLOBAL_SPATIAL_OPTIONS>
</placement_answer>

<action>
- Output ONLY: [x, y, r]
- Must be within valid ranges, x, y in [0, 255] and r in [2, 32]
- Do not add any explanation here.
</action>
""" 