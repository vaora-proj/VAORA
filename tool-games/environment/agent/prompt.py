prompt="""You are a physics reasoning agent in the PHYRE environment.

ENVIRONMENT RULES
- The scene is 256x256 pixels with a white background.
- Objects are colored: black, blue, green, gray.
- Static objects: BLACK is immovable.
- Dynamic objects: GREEN, GRAY fall under gravity in 2D.
- Scene boundaries are walls/floor.
- Objects shapes: BALL, BAR, JAR, TRIANGLE, TRAPEZOID.
- The goal: make any [GREEN *] object touch any [BLUE *] object.
- You can add exactly one RED OBJECT as the action: [x, y, r]
  - x in [0, 255] (0=left edge, 255=right edge)
  - y in [0, 255] (0=top edge, 255=bottom edge)
  - r in [2, 32]
- The RED OBJECT has shape <RED_OBJECT_SHAPE>.

OBJECT NAMING CONVENTION (STRICT)
- When you mention an object, always use: [COLOR SHAPE] in UPPERCASE, e.g. [GREEN BALL], [BLUE BAR], [GRAY BAR], [GRAY JAR], [GRAY TRIANGLE], [GRAY TRAPEZOID].
- The added object must always be written exactly as: [RED OBJECT]
- **EXCEPTION: In reasoning sections only, you may use lowercase natural language (e.g., "the green ball", "blue bar") for readability.**

INPUT
This is the input scene: <image>

TASK (what to do)
1) Identify the goal pair: which [GREEN *] must touch which [BLUE *].
2) Identify blockers: walls, gaps, covers, containers, slopes, platforms, and black obstacles.
3) Choose the main causal strategy (PUSH/ROTATE/HIT/TILT/BLOCKED/SPIN/COLLIDE WITH/DEFLECTED/DEFLECT/STOPPED/SUPPORT/BLOCK/STOP/STRIKE).
4) Estimate coordinates of relevant objects (approximate is OK but must be plausible).
5) Describe the red object placement relative to objects AND the whole scene.
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
- State the goal clearly (which green touches which blue).
- Analyze spatial relationships: alignment, distance, and location relative to the scene center (Left vs Right side).
- Identify obstacles between goal objects.
- Assess whether objects will naturally interact or need intervention.
</scene_reasoning>

<scene_answer>
- List the key goal objects (at least the [GREEN *] and its goal [BLUE *], good to include [BLACK *]).
- Include any critical blockers or intermediate objects referenced in your reasoning.
- Use THIS exact line style (copy the template):
  [OBJECT] is at [GLOBAL SPATIAL] [x, y] with size [d].
- GLOBAL SPATIAL must be one of:
  ['TOP-LEFT', 'TOP-RIGHT', 'CENTER', 'TOP', 'BOTTOM-RIGHT', 'BOTTOM-LEFT', 'LEFT', 'BOTTOM', 'RIGHT']
- x,y,d must be numbers (integers preferred) and ranged from 0-255.
</scene_answer>

<causal_actions_reasoning>
**REASONING REQUIREMENTS:**
- Reference coordinates from <scene_answer> when describing positions.
- Choose and justify ONE causal strategy (PUSH/ROTATE/HIT/TILT/BLOCKED/SPIN/COLLIDE WITH/DEFLECTED/DEFLECT/STOPPED/SUPPORT/BLOCK/STOP/STRIKE).
- Describe the event chain: red object → interaction (COLLIDE OR SPIN) → target motion → goal achievement.
- Explain the contact point location and why it creates the desired motion.
- Verify the approach avoids obstacles and reaches the goal.
</causal_actions_reasoning>

<causal_actions_answer>
- Output 1–2 lines total.
- Choose ONE of these templates:
- Template Block (for * BALL or * JAR):
- ACTION must be one of: ['DEFLECT', 'BLOCK', 'STOP']
The [RED OBJECT] should [ACTION] the [TARGET OBJECT] at the contact point [x, y].

- Template Collision (for * BALL or * JAR or * TRIANGLE or * TRAPEZOID):
- ACTION must be one of: ['DEFLECTED', 'COLLIDE WITH', 'BLOCKED', 'STOPPED', 'PUSH', 'STRIKE', 'HIT']
- DIRECTION must be one of: ['RIGHT', 'LEFT']
The [RED OBJECT] should [ACTION] the [TARGET OBJECT] at the contact point [x, y] to push it towards [DIRECTION].

- Template Rotation (for BAR * JAR or * TRIANGLE or * TRAPEZOID):
- ACTION must be one of: ['TILT', 'ROTATE', 'SPIN']
- DIRECTION must be one of: ['CLOCKWISE', 'COUNTER-CLOCKWISE']
The [RED OBJECT] should [ACTION] the [TARGET OBJECT] at the contact point [x, y] to rotate it [DIRECTION].

- Template Support (for BAR * JAR or * TRIANGLE or * TRAPEZOID):
- ACTION must be one of: ['SUPPORT']
The [RED OBJECT] should [ACTION] the [TARGET OBJECT] at the contact point [x, y].

If the causal action doesn't directly interact with the green object, use the following templates to further explain the chain action of the green object:

- Template Collision (for additional explaination of GREEN BALL):
- DIRECTION must be one of: ['LEFT', 'RIGHT']
The chain actions cause the [GREEN BALL] to move [DIRECTION] at point [x, y].

- Template Block (for additional explaination of GREEN BALL):
- STATE must be one of: ['STOPPED', 'DEFLECTED', 'BLOCKED']
The chain actions cause the [GREEN BALL] to be [STATE] at point [x, y].

- Template Rotation (for additional explaination of BAR * JAR or * TRIANGLE or * TRAPEZOID):
- DIRECTION must be one of: ['CLOCKWISE', 'COUNTER-CLOCKWISE']
The chain actions cause the [GREEN BAR] to move [DIRECTION] at point [x, y].

- TARGET OBJECT must be a real object in the scene.
- [x, y] must be the estimated global coordinates (0-255) where the objects touch.
</causal_actions_answer>

<placement_reasoning>
**REASONING REQUIREMENTS:**
- Reference target position from <scene_answer> and contact point from <causal_actions_answer>.
- Calculate red object's initial position to achieve the desired contact point (either via gravity trajectory OR immediate static contact).
- Account for object radius and gravity trajectory.
- Justify radius choice (larger for momentum, smaller for precision). If using SUPPORT/TILT, consider placing the object at the BOTTOM to act as a static fulcrum.
- Verify placement avoids static obstacles. CRITICAL PHYSICS RULE: To push an object to a [DIRECTION], the [RED OBJECT] must be placed on the OPPOSITE side (e.g., to push LEFT, place on RIGHT).
</placement_reasoning>

<placement_answer>
- Output 1-3 lines, using these templates:

[RED OBJECT] is located at the [RELATIVE SPATIAL] of the [TARGET OBJECT A].
[RED OBJECT] is located at the [RELATIVE SPATIAL] of the [TARGET OBJECT B].
[RED OBJECT] is located at the [GLOBAL SPATIAL] of the [WHOLE SCENE].

- RELATIVE SPATIAL must be one of:
  ['TOP-LEFT', 'TOP-RIGHT', 'BOTTOM-LEFT', 'BOTTOM-RIGHT']
- GLOBAL SPATIAL must be one of:
  ['TOP-LEFT', 'TOP-RIGHT', 'CENTER', 'TOP', 'BOTTOM-RIGHT', 'BOTTOM-LEFT', 'LEFT', 'BOTTOM', 'RIGHT']
</placement_answer>

<action>
- Output ONLY: [x, y, r]
- Must be within valid ranges, x, y in [0, 255] and r in [2, 32]
- Do not add any explanation here.
</action>
"""