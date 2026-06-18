import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from PIL import Image

try:
    import phyre
except ImportError as e:
    raise ImportError("Phyre is not installed. Please install it to use Phyre functionality.") from e

from verl.interactions.base import BaseInteraction
from verl.utils.dataset.vision_utils import process_image
from verl.tools.schemas import ToolResponse
from verl.utils.reward_score import phyre_util

logger = logging.getLogger(__name__)

class PhyreInteraction(BaseInteraction):
    """
    Interaction agent for Phyre that uses a ratio-based reward system for unsolved tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the agent, the shared simulator, and the reward pre-computation."""
        super().__init__(config)
        self.config = config
        
        # --- Task and Simulation Configuration ---
        self.eval_setup: str = config.get("eval_setup", "ball_within_template")
        self.fold_id: int = int(config.get("fold_id", 0))
        self.split: str = config.get("split", "train")
        self.max_steps: int = int(config.get("max_simulation_steps", 300))
        self.action_tier: str = phyre.eval_setup_to_action_tier(self.eval_setup)
        self.num_samples: int = int(config.get("num_samples", 5))

        # --- Ratio-Based Reward Configuration ---
        self.unsolved_reward_upper_bound: float = float(config.get("unsolved_reward_upper_bound", 0.8))
        self.x_reward_ratio: float = float(config.get("x_reward_ratio", 2.0))
        self.y_reward_ratio: float = float(config.get("y_reward_ratio", 1.0))
        self.angle_reward_ratio: float = float(config.get("angle_reward_ratio", 1.0))

        # --- Simulator and Pre-computation Attributes ---
        self.simulator = None
        self.task_ids: List[str] = []
        self.task_id_to_index_map: Dict[str, int] = {}
        # self.initial_states_map: Dict[str, np.ndarray] = {}
        self.reward_values_map: Dict[str, Tuple[float, float, float]] = {}
        
        self._instance_dict: Dict[str, Dict[str, Any]] = {}

        self._initialize_simulator_and_rewards()

    def _initialize_simulator_and_rewards(self):
        """Loads tasks, initializes the simulator, and pre-computes reward values."""
        try:
            # 1. Load tasks and initialize the simulator
            train_tasks, dev_tasks, test_tasks = phyre.get_fold(self.eval_setup, self.fold_id)
            self.task_ids = train_tasks + dev_tasks + test_tasks

            self.simulator = phyre.initialize_simulator(self.task_ids, self.action_tier)
            self.task_id_to_index_map = {task_id: i for i, task_id in enumerate(self.task_ids)}

            # 2. Pre-compute reward values for each task
            sum_of_ratios = self.x_reward_ratio + self.y_reward_ratio + self.angle_reward_ratio

            for i, task_id in enumerate(self.task_ids):
                initial_objects = self.simulator.initial_featurized_objects[i]
                # self.initial_states_map[task_id] = initial_objects.states[0]
                
                num_dynamic_objects = sum(
                    1 for color in initial_objects.colors[:initial_objects.num_scene_objects] 
                    if color in ['BLUE', 'GRAY', 'GREEN']
                )
                
                if num_dynamic_objects > 0 and sum_of_ratios > 0:
                    max_reward_per_object = self.unsolved_reward_upper_bound / num_dynamic_objects
                    reward_unit = max_reward_per_object / sum_of_ratios
                    
                    x_reward = reward_unit * self.x_reward_ratio
                    y_reward = reward_unit * self.y_reward_ratio
                    angle_reward = reward_unit * self.angle_reward_ratio
                    self.reward_values_map[task_id] = (x_reward, y_reward, angle_reward)
                else:
                    self.reward_values_map[task_id] = (0.0, 0.0, 0.0)

            logger.info(
                f"PhyreInteraction initialized successfully with ratio-based rewards. Loaded {len(self.task_ids)} tasks."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Phyre simulator and rewards: {e}")
            raise

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
    
        task_id = kwargs.get("task_id")
        if not task_id or task_id not in self.task_id_to_index_map:
            raise ValueError(f"A valid and loaded 'task_id' must be provided. Got: {task_id}")

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "best_reward": 0.0,
            "attempt_count": 0,
            "task_id": task_id,
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[bool, ToolResponse, float, Dict[str, Any]]:
        """Processes a model response, runs simulation, and returns structured feedback with nuanced rewards."""
        if instance_id not in self._instance_dict:
            return True, ToolResponse(text="Error: Interaction instance not found."), -1.0, {}
        
        instance_data = self._instance_dict[instance_id]
        instance_data["attempt_count"] += 1
        
        print("Task ID:", instance_data["task_id"])
        print(messages)
        
        last_message = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
        original_action, map_action = phyre_util._extract_action(last_message)
        
        if map_action is None:
            text = phyre_util._sample_responses("wrong_format")
            return False, ToolResponse(text=text), -0.1, {}
        
        x, y, r = map_action
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= r <= 1.0):
            text = phyre_util._sample_responses("wrong_format")
            return False, ToolResponse(text=text), -0.1, {}

        # Run simulation, requesting images and object data for feedback and scoring
        sim_result = self.simulator.simulate_action(
            self.task_id_to_index_map[instance_data["task_id"]],
            action=np.array(map_action, dtype=np.float32),
            need_images=True,
            need_featurized_objects=True, # Required for nuanced reward
            stride=max(1, self.max_steps // 10)
        )

        is_solved, is_invalid = sim_result.status.is_solved(), sim_result.status.is_invalid()

        # --- Updated Scoring Logic ---
        if is_invalid:
            score = -0.1
            status_text = "INVALID ACTION"
            explanation = phyre_util._sample_responses("invalid")
            return False, ToolResponse(text=f"Your action has been evaluated.\n{explanation}"), score, {}
        elif is_solved:
            score = 1.0
            status_text = "SOLVED"
            explanation = phyre_util._sample_responses("solved")
            return True, ToolResponse(text=f"Your action has been evaluated.\n{explanation}"), score, {}
        else: # NOT SOLVED - Calculate ratio-based reward
            status_text = "NOT SOLVED"
            explanation = phyre_util._sample_responses("not_solved")
            
            total_reward = 0.0
            task_id = instance_data["task_id"]
            x_reward, y_reward, angle_reward = self.reward_values_map[task_id]
            print(f"Reward Values - X: {x_reward:.4f}, Y: {y_reward:.4f}, Angle: {angle_reward:.4f}")
            
            initial_states = sim_result.featurized_objects.features[0]
            final_states = sim_result.featurized_objects.features[-1]
            colors = sim_result.featurized_objects.colors
            num_scene_objects = sim_result.featurized_objects.num_scene_objects
            print(f"Number of Scene Objects: {num_scene_objects}")

            for i in range(num_scene_objects):
                if colors[i] in ['BLUE', 'GRAY', 'GREEN']:
                    shift = np.abs(final_states[i] - initial_states[i])
                    print(f"Object {i} - Initial: {initial_states[i]}, Final: {final_states[i]}, Shift: {shift}")
                    if shift[0] > 1e-5: total_reward += x_reward
                    if shift[1] > 1e-5: total_reward += y_reward
                    if shift[2] > 1e-5: total_reward += angle_reward
            
            score = total_reward
        
        print(f"Action Result - Task: {instance_data['task_id']}, Attempt: {instance_data['attempt_count']}, Status: {status_text}, Score: {score:.4f}")

        instance_data["best_reward"] = max(instance_data["best_reward"], score)
        should_terminate = is_solved
        response_text = f"Your action has been evaluated.\n{explanation}"
        
        # --- Image Payload Generation (no change) ---
        image_payload = None
        if not should_terminate and sim_result.images is not None and len(sim_result.images) > 0:
            num_frames = len(sim_result.images)
            indices = np.linspace(0, num_frames - 1, self.num_samples, dtype=int) if num_frames > self.num_samples else range(num_frames)
            
            image_payload = [
                Image.fromarray((phyre.observations_to_float_rgb(sim_result.images[i]) * 255).astype(np.uint8))
                for i in indices
            ]
            image_payload = [process_image(img) for img in image_payload]

        response = ToolResponse(text=response_text, image=image_payload)
        return should_terminate, response, score, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Returns the best reward achieved so far for the instance."""
        return self._instance_dict.get(instance_id, {}).get("best_reward", 0.0)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Removes the state for the completed interaction instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]