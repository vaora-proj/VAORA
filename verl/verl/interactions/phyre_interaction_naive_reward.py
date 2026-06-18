# phyre_interaction.py

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
from verl.utils.dataset.vision_utils import process_image, process_video
# Import the ToolResponse class, assuming a similar schema path as phyre_tool.py
from verl.tools.schemas import ToolResponse

from verl.utils.reward_score import phyre_util

logger = logging.getLogger(__name__)

class PhyreInteraction(BaseInteraction):
    """
    Refined interaction agent for Phyre that uses a single, shared simulator.
    This version returns content using the structured ToolResponse class.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the agent and the shared Phyre simulator."""
        super().__init__(config)
        self.config = config
        
        self.eval_setup: str = config.get("eval_setup", "ball_within_template")
        self.fold_id: int = int(config.get("fold_id", 0))
        self.split: str = config.get("split", "train")
        self.max_steps: int = int(config.get("max_simulation_steps", 300))
        self.action_tier: str = phyre.eval_setup_to_action_tier(self.eval_setup)
        self.num_samples: int = int(config.get("num_samples", 5))

        self.simulator = None
        self.task_ids: List[str] = []
        self.task_id_to_index_map: Dict[str, int] = {}
        
        self._instance_dict: Dict[str, Dict[str, Any]] = {}

        self._initialize_shared_simulator()

    def _initialize_shared_simulator(self):
        """Loads tasks and initializes a single simulator for the configured split."""
        try:
            train_tasks, dev_tasks, test_tasks = phyre.metrics.get_fold(self.eval_setup, self.fold_id)
            task_map = {"train": train_tasks, "dev": dev_tasks, "test": test_tasks}
            # self.task_ids = task_map.get(self.split, [])
            self.task_ids = train_tasks + test_tasks

            if not self.task_ids:
                raise RuntimeError(f"No tasks found for split '{self.split}'.")

            self.simulator = phyre.initialize_simulator(self.task_ids, self.action_tier)
            self.task_id_to_index_map = {task_id: i for i, task_id in enumerate(self.task_ids)}
            logger.info(self.task_id_to_index_map)
            logger.info(
                f"PhyreInteraction initialized successfully. Loaded {len(self.task_ids)} tasks for "
                f"setup='{self.eval_setup}', fold={self.fold_id}, split='{self.split}'."
            )
        except Exception as e:
            logger.error(f"Failed to initialize shared Phyre simulator: {e}")
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
            "best_reward": 0.0,        # Correctly named and initialized
            "attempt_count": 0,        # Initialize attempt count
            "task_id": task_id,        # Store task_id
            "task_index": self.task_id_to_index_map[task_id], # Store task_index
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[bool, ToolResponse, float, Dict[str, Any]]:
        """Processes a model response, runs the action, and returns structured feedback."""
        if instance_id not in self._instance_dict:
            return True, ToolResponse(text="Error: Interaction instance not found."), -1.0, {}
        
        instance_data = self._instance_dict[instance_id]
        instance_data["attempt_count"] += 1
        
        last_message = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
        original_action, map_action = phyre_util._extract_action(last_message)
        ### Print the last message for debugging
        # print(f"Last assistant message: {last_message}")
        # print(f"Extracted action: {action}")
        
        # Handle invalid or out-of-bounds actions
        if map_action is None:
            text = "Your response did not contain a valid action format. Please provide the action as `[x, y, r]`."
            return False, ToolResponse(text=text), -0.1, {}
        
        x, y, r = map_action
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= r <= 1.0):
            text = f"Error: Action out of bounds. All values in `[x, y, r]` must be in the [0, 1] range. Got: [{x:.3f}, {y:.3f}, {r:.3f}]."
            return False, ToolResponse(text=text), -0.1, {}

        # Run the simulation
        pred_action = np.array(map_action, dtype=np.float32)
        sim_result = self.simulator.simulate_action(
            instance_data["task_index"], pred_action, need_images=True, stride=max(1, self.max_steps // 10)
        )

        is_solved, is_invalid = sim_result.status.is_solved(), sim_result.status.is_invalid()

        if is_invalid:
            score, status_text = -0.1, "INVALID ACTION"
            explanation = phyre_util._sample_responses("invalid") # <-- Updated
        elif is_solved:
            score, status_text = 1.0, "SOLVED"
            explanation = phyre_util._sample_responses("solved")   # <-- Updated
        else:
            score, status_text = 0.0, "NOT SOLVED"
            explanation = phyre_util._sample_responses("not_solved") # <-- Updated

        instance_data["best_reward"] = max(instance_data["best_reward"], score)
        should_terminate = is_solved

        response_text = (
            f"Your action has been evaluated.\n"
            f"{explanation}"
        )
        
        image_payload = None
        if not should_terminate:
            # This block runs for both INVALID and NOT SOLVED actions.
            
            # 1. Handle image availability
            iterable_images = sim_result.images if sim_result.images is not None else []
            
            # 2. Sample N images if they exist
            if len(iterable_images) > 0:
                num_frames = len(iterable_images)
                if num_frames > self.num_samples:
                    # Sample N frames evenly from the list of images
                    indices = np.linspace(0, num_frames - 1, self.num_samples, dtype=int)
                    sampled_raw_images = [iterable_images[i] for i in indices]
                else:
                    # If there are fewer frames than N, use all of them
                    sampled_raw_images = iterable_images

                # 3. Convert sampled raw images to PIL Image objects
                image_payload = [
                    Image.fromarray((phyre.observations_to_float_rgb(img) * 255).astype(np.uint8))
                    for img in sampled_raw_images
                ]
                image_payload = [process_image(img) for img in image_payload]
        
                response = ToolResponse(text=response_text, image=image_payload)
            
                return should_terminate, response, score, {}
            else:
                return should_terminate, ToolResponse(text=response_text), score, {}
        else:
            response = ToolResponse(text=response_text)
            return should_terminate, response, score, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Returns the best reward achieved so far for the instance."""
        return self._instance_dict.get(instance_id, {}).get("best_reward", 0.0)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Removes the state for the completed interaction instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
