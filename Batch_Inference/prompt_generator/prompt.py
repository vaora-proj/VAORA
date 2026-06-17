prompt="You are a physics reasoning agent in the PHYRE environment.\n\nENVIRONMENT RULES\n- The scene is 256x256 pixels with a white background.\n- Objects are colored: black, gray, blue, green, purple.\n- Static objects: BLACK and PURPLE are immovable.\n- Dynamic objects: GRAY, BLUE, GREEN fall under gravity in 2D.\n- Scene boundaries are walls/floor.\n- Objects shapes: BALL, BAR, JAR.\n- The goal: make any [GREEN *] object touch any [BLUE *] dynamic object OR any [PURPLE *] static object.\n- You can add exactly one RED BALL as the action: [x, y, r]\n  - x in [0, 255] (0=left edge, 255=right edge)\n  - y in [0, 255] (0=top edge, 255=bottom edge)\n  - r in [2, 32]\n\nOBJECT NAMING CONVENTION (STRICT)\n- When you mention an object, always use: [COLOR SHAPE] in UPPERCASE, e.g. [GREEN BALL], [BLUE BALL], [PURPLE BAR], [GRAY JAR].\n- The added object must always be written exactly as: [RED BALL]\n- **EXCEPTION: In reasoning sections only, you may use lowercase natural language (e.g., \"the green ball\", \"blue bar\") for readability.**\n\nINPUT\nThis is the input scene: <image>\n\nTASK (what to do)\n1) Identify the goal pair: which [GREEN *] must touch which [BLUE *] or [PURPLE *].\n2) Identify blockers: walls, gaps, covers, containers, slopes, platforms, and black obstacles.\n3) Choose the main causal strategy (PUSH/ROTATE/HIT/TILT/BLOCKED/SPIN/COLLIDE WITH/DEFLECTED/DEFLECT/STOPPED/SUPPORT/BLOCK/STOP/STRIKE).\n4) Estimate coordinates of relevant objects (approximate is OK but must be plausible).\n5) Describe the red ball placement relative to objects AND the whole scene.\n6) Output the final action [x, y, r].\n7) You are encouraged to proposed different reasoning in each section and different strategies to improve diversity.\n\nOUTPUT FORMAT (MUST MATCH EXACTLY)\nYou MUST output EXACTLY these sections in EXACT order, and NOTHING else:\n\n<scene_reasoning>\n...\n</scene_reasoning>\n<scene_answer>\n...\n</scene_answer>\n<causal_actions_reasoning>\n...\n</causal_actions_reasoning>\n<causal_actions_answer>\n...\n</causal_actions_answer>\n<placement_reasoning>\n...\n</placement_reasoning>\n<placement_answer>\n...\n</placement_answer>\n<action>\n[x, y, r]\n</action>\n\nCONTENT RULES PER SECTION\n\n<scene_reasoning>\n**REASONING REQUIREMENTS:**\n- Identify all relevant objects and estimate their approximate positions.\n- State the goal clearly (which green touches which blue/purple).\n- Analyze spatial relationships: alignment, distance, and location relative to the scene center (Left vs Right side).\n- Identify obstacles between goal objects.\n- Assess whether objects will naturally interact or need intervention.\n</scene_reasoning>\n\n<scene_answer>\n- List the key goal objects (at least the [GREEN *], [GRAY *] and its goal [BLUE *]/[PURPLE *], good to include [BLACK *] or [GRAY *]).\n- Include any critical blockers or intermediate objects referenced in your reasoning.\n- Use THIS exact line style (copy the template):\n  [OBJECT] is at [GLOBAL SPATIAL] [x, y] with size [d].\n- GLOBAL SPATIAL must be one of:\n  ['TOP-LEFT', 'TOP-RIGHT', 'CENTER', 'TOP', 'BOTTOM-RIGHT', 'BOTTOM-LEFT', 'LEFT', 'BOTTOM', 'RIGHT']\n- x,y,d must be numbers (integers preferred) and ranged from 0-255.\n</scene_answer>\n\n<causal_actions_reasoning>\n**REASONING REQUIREMENTS:**\n- Reference coordinates from <scene_answer> when describing positions.\n- Choose and justify ONE causal strategy (PUSH/ROTATE/HIT/TILT/BLOCKED/SPIN/COLLIDE WITH/DEFLECTED/DEFLECT/STOPPED/SUPPORT/BLOCK/STOP/STRIKE).\n- Describe the event chain: red ball → interaction (COLLIDE OR SPIN) → target motion → goal achievement. \n- Explain the contact point location and why it creates the desired motion.\n- Verify the approach avoids obstacles and reaches the goal.\n</causal_actions_reasoning>\n\n<causal_actions_answer>\n- Output 1–2 lines total.\n- Choose ONE of these templates:\n- Template Block (for * BALL or * JAR):\n- ACTION must be one of: ['DEFLECT', 'BLOCK', 'STOP']\nThe [RED BALL] should [ACTION] the [TARGET OBJECT] at the contact point [x, y].\n\n- Template Collision (for * BALL or * JAR):\n- ACTION must be one of: ['DEFLECTED', 'COLLIDE WITH', 'BLOCKED', 'STOPPED', 'PUSH', 'STRIKE', 'HIT']\n- DIRECTION must be one of: ['RIGHT', 'LEFT']\nThe [RED BALL] should [ACTION] the [TARGET OBJECT] at the contact point [x, y] to push it towards [DIRECTION].\n\n- Template Rotation (for * BAR):\n- ACTION must be one of: ['TILT', 'ROTATE', 'SPIN']\n- DIRECTION must be one of: ['CLOCKWISE', 'COUNTER-CLOCKWISE']\nThe [RED BALL] should [ACTION] the [TARGET OBJECT] at the contact point [x, y] to rotate it [DIRECTION].\n\n- Template Support (for * BAR):\n- ACTION must be one of: ['SUPPORT']\nThe [RED BALL] should [ACTION] the [TARGET OBJECT] at the contact point [x, y].\n\nIf the causal action doesn't directly interact with the green object, use the following templates to further explain the chain action of the green object:\n\n- Template Collision (for additional explaination of GREEN BALL or GREEN JAR):\n- DIRECTION must be one of: ['LEFT', 'RIGHT']\nThe chain actions cause the [GREEN BALL or GREEN JAR] to move [DIRECTION] at point [x, y].\n\n- Template Block (for additional explaination of GREEN BALL or GREEN JAR):\n- STATE must be one of: ['STOPPED', 'DEFLECTED', 'BLOCKED']\nThe chain actions cause the [GREEN BALL or GREEN JAR] to be [STATE] at point [x, y].\n\n- Template Rotation (for additional explaination of GREEN BAR):\n- DIRECTION must be one of: ['CLOCKWISE', 'COUNTER-CLOCKWISE']\nThe chain actions cause the [GREEN BAR] to move [DIRECTION] at point [x, y].\n\n- TARGET OBJECT must be a real object in the scene.\n- [x, y] must be the estimated global coordinates (0-255) where the objects touch.\n</causal_actions_answer>\n\n<placement_reasoning>\n**REASONING REQUIREMENTS:**\n- Reference target position from <scene_answer> and contact point from <causal_actions_answer>.\n- Calculate red ball's initial position to achieve the desired contact point (either via gravity trajectory OR immediate static contact).\n- Account for ball radius and gravity trajectory.\n- Justify radius choice (larger for momentum, smaller for precision). If using SUPPORT/TILT, consider placing the ball at the BOTTOM to act as a static fulcrum.\n- Verify placement avoids static obstacles. CRITICAL PHYSICS RULE: To push an object to a [DIRECTION], the [RED BALL] must be placed on the OPPOSITE side (e.g., to push LEFT, place on RIGHT).\n</placement_reasoning>\n\n<placement_answer>\n- Output 1-3 lines, using these templates:\n\n[RED BALL] is located at the [RELATIVE SPATIAL] of the [TARGET OBJECT A].\n[RED BALL] is located at the [RELATIVE SPATIAL] of the [TARGET OBJECT B].\n[RED BALL] is located at the [GLOBAL SPATIAL] of the [WHOLE SCENE].\n\n- RELATIVE SPATIAL must be one of:\n  ['TOP-LEFT', 'TOP-RIGHT', 'BOTTOM-LEFT', 'BOTTOM-RIGHT']\n- GLOBAL SPATIAL must be one of:\n  ['TOP-LEFT', 'TOP-RIGHT', 'CENTER', 'TOP', 'BOTTOM-RIGHT', 'BOTTOM-LEFT', 'LEFT', 'BOTTOM', 'RIGHT']\n</placement_answer>\n\n<action>\n- Output ONLY: [x, y, r]\n- Must be within valid ranges, x, y in [0, 255] and r in [2, 32]\n- Do not add any explanation here.\n</action>\n"



craft_prompt="""You are a physics reasoning agent in a simulation environment.

**ENVIRONMENT RULES**
- The scene is 256x256 pixels with a white background.
- Static objects: BLACK and PURPLE are immovable.
- Dynamic objects: GRAY, BLUE, GREEN, CYAN fall under gravity in 2D.
- Scene boundaries are walls/floor.
- Objects shapes: BALL, BAR, JAR, TRIANGLE.
- The goal: Use physical reasoning to predict outcomes and answer the Question.

**OBJECT NAMING CONVENTION (STRICT)**
- When you mention an object, always use: [COLOR SHAPE] in UPPERCASE, e.g. [GRAY TRIANGLE], [BLUE TRIANGLE], [PURPLE BAR], [BLACK BAR].
- **EXCEPTION: In reasoning sections only, you may use lowercase natural language (e.g., "the gray triangle", "blue bar") for readability.**

**INPUT**
This is the input scene: <image>
Question: <QUESTION_TEXT>

**TASK (what to do)**
1) Identify all relevant objects and their approximate positions.
2) Analyze the physical chain of events (gravity, collisions, rotations, and support).
3) Identify blockers: walls, gaps, covers, containers, slopes, and platforms.
4) Evaluate counterfactuals (e.g., "if objects are removed") or temporal sequences ("after X happens").
5) Formulate a step-by-step causal reasoning trace.
6) Output the final concise answer.

**OUTPUT FORMAT (MUST MATCH EXACTLY)**
You MUST output EXACTLY these sections in EXACT order, and NOTHING else:

<scene_reasoning>
- Identify all relevant objects and estimate their approximate positions.
- Analyze spatial relationships: alignment, distance, and location relative to the scene center.
- Identify the potential trajectory of dynamic objects under gravity.
- Address the specific context of the question (e.g., which objects are mentioned).
</scene_reasoning>

<scene_answer>
- List the key objects referenced in the question or the physical chain.
- Use THIS exact line style:
  [OBJECT] is at [GLOBAL SPATIAL] [x, y] with size [d].
- GLOBAL SPATIAL must be one of:
  ['TOP-LEFT', 'TOP-RIGHT', 'CENTER', 'TOP', 'BOTTOM-RIGHT', 'BOTTOM-LEFT', 'LEFT', 'BOTTOM', 'RIGHT']
- x,y,d must be numbers (integers preferred) and ranged from 0-255.
</scene_answer>

<causal_actions_reasoning>
- Reference coordinates from <scene_answer> when describing positions.
- Describe the event chain: object movement → interaction (COLLIDE/SPIN/HIT) → secondary motion → final state.
- If the question asks "if any other objects are removed," simulate the scene without static/dynamic blockers.
- Use causal verbs: PUSH, ROTATE, HIT, TILT, BLOCKED, COLLIDE WITH, DEFLECTED, SUPPORT.
- Justify the conclusion based on the simulated interaction.
</causal_actions_reasoning>

<causal_actions_answer>
- Output 1–2 lines total.
- Choose ONE of these templates:
- Template Interaction:
The [OBJECT A] will [ACTION] the [OBJECT B] at point [x, y].
- Template Final State:
The [OBJECT] will end at [GLOBAL SPATIAL] location [x, y] or enter the [CONTAINER/BASKET].
- Template Motion:
The chain actions cause the [OBJECT] to move [DIRECTION] at point [x, y].
</causal_actions_answer>

<final_answer>
- Provide ONLY the direct answer to the question (e.g., "true", "false", "2", "blue").
</final_answer>"""