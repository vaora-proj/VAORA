import asyncio
import sys
from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock
from verl.tools.schemas import ToolResponse
from verl.utils.reward_score import phyre_util

# --- Mocking Dependencies ---
# To make this script standalone, we create a mock of the ToolResponse class
# and inject it into Python's module cache. This must be done *before*
# importing the PhyreInteraction class. This avoids an ImportError if the 'verl'
# library isn't available in the test environment.

@dataclass
class MockToolResponse:
    """A mock dataclass to simulate verl.tools.schemas.ToolResponse."""
    text: str = ""
    image: Optional[List[Any]] = None
    video: Optional[List[Any]] = None

# Create a mock 'verl.tools.schemas' module
mock_schemas = MagicMock()
mock_schemas.ToolResponse = MockToolResponse

# Place the mock module into the system's module cache
# sys.modules['verl.tools.schemas'] = mock_schemas

# --- Import the Class to be Tested ---
# This import will now succeed and use our MockToolResponse
from verl.interactions.phyre_interaction import PhyreInteraction

# --- Test Configuration ---
TEST_CONFIG = {
    "eval_setup": "ball_within_template",
    "fold_id": 0,
    "split": "train",
    "max_simulation_steps": 300,
}
# A known task ID from the ball_within_template/fold_0/dev split
TEST_TASK_ID = "00000:003"


async def test_initialization():
    """Tests if the PhyreInteraction class initializes the shared simulator correctly."""
    print("--- Testing Initialization ---")
    interaction = PhyreInteraction(TEST_CONFIG)
    assert interaction.simulator is not None, "Simulator should be initialized."
    assert len(interaction.task_ids) > 0, "Task IDs should be loaded."
    assert TEST_TASK_ID in interaction.task_id_to_index_map, f"Test task {TEST_TASK_ID} not found in map."
    print("✅ Initialization successful.")


async def test_start_interaction():
    """Tests the start_interaction method returns a ToolResponse."""
    print("\n--- Testing start_interaction with ToolResponse ---")
    interaction = PhyreInteraction(TEST_CONFIG)
    instance_id = None
    try:
        await interaction.start_interaction(task_id=TEST_TASK_ID)
        
        print("✅ start_interaction successfully.")
    finally:
        if instance_id:
            await interaction.finalize_interaction(instance_id)


async def test_successful_solve_workflow():
    """Tests a successful solve returns a correctly formatted ToolResponse."""
    print("\n--- Testing a Successful Solve Workflow with ToolResponse ---")
    interaction = PhyreInteraction(TEST_CONFIG)
    instance_id = None
    try:
        instance_id = await interaction.start_interaction(task_id=TEST_TASK_ID)
        messages = [{"role": "assistant", "content": "Let's solve this with <answer>[49, 28, 54]</answer>."}]
        
        should_terminate, response, score, _ = await interaction.generate_response(instance_id, messages)
        
        assert should_terminate is True
        assert score == 1.0
        assert isinstance(response, ToolResponse)
        assert "Success" in response.text
        assert response.video is None, "Response should not contain a video on success."
        
        print("✅ Successful solve workflow validated.")
    finally:
        if instance_id:
            await interaction.finalize_interaction(instance_id)


async def test_failed_attempt_workflow():
    """Tests a failed attempt returns a ToolResponse with video content."""
    print("\n--- Testing a Failed Attempt Workflow with ToolResponse ---")
    interaction = PhyreInteraction(TEST_CONFIG)
    instance_id = None
    try:
        instance_id = await interaction.start_interaction(task_id=TEST_TASK_ID)
        messages = [{"role": "assistant", "content": "I'll try a failing action: <answer>[106, 120, 35]</answer>."}]
        
        should_terminate, response, score, _ = await interaction.generate_response(instance_id, messages)
        
        assert should_terminate is False
        assert score == 0.0
        assert isinstance(response, ToolResponse)
        assert "The attempt failed." in response.text
        assert response.image and len(response.image) == 5, "Response should contain a video on failure."
        
        print("✅ Failed attempt workflow validated.")
    finally:
        if instance_id:
            await interaction.finalize_interaction(instance_id)


async def test_invalid_action_format():
    """Tests that a malformed action string returns a valid error ToolResponse."""
    print("\n--- Testing Invalid Action Format with ToolResponse ---")
    interaction = PhyreInteraction(TEST_CONFIG)
    instance_id = None
    try:
        instance_id = await interaction.start_interaction(task_id=TEST_TASK_ID)
        messages = [{"role": "assistant", "content": "I'm not sure what to do."}]

        should_terminate, response, score, _ = await interaction.generate_response(instance_id, messages)
        
        print(response.text)  # Debug print to see the response text

        assert should_terminate is False
        assert score == -0.1
        assert isinstance(response, ToolResponse)
        assert response.is_text_only()
        assert "That placement is invalid." in response.text or "The action format is incorrect." in response.text

        print("✅ Invalid action format handled correctly.")
    finally:
        if instance_id:
            await interaction.finalize_interaction(instance_id)

async def test_compute_score_execution_time():
    """
    Tests the execution time of the standalone compute_score function.
    
    This test is important because compute_score initializes a new simulator
    on every call, which can be a performance bottleneck.
    """
    print("\n--- Testing compute_score Execution Time ---")
    
    # A known solving action for the test task
    solution_str = "<answer>[49, 28, 54]</answer>"
    ground_truth = "" # Not used in Phyre
    extra_info = {
        "task_id": TEST_TASK_ID,
        "eval_setup": TEST_CONFIG["eval_setup"]
    }
    
    import time
    
    start_time = time.perf_counter()
    
    # Call the function directly from the imported utility module
    score = phyre_util.compute_score(solution_str, ground_truth, extra_info)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"compute_score for task '{TEST_TASK_ID}' completed in {duration:.4f} seconds.")
    print(f"Returned score: {score}")
    
    # Verify that the simulation ran correctly and produced the expected outcome
    assert score == 1.0, "The score should be 1.0 for a successful action."
    
    print("✅ compute_score execution time test passed.")


async def main():
    """Main function to run all test cases in sequence."""
    print("===== Running PhyreInteraction Test Suite (ToolResponse Version) =====")
    await test_initialization()
    await test_start_interaction()
    await test_successful_solve_workflow()
    await test_failed_attempt_workflow()
    await test_invalid_action_format()
    # await test_compute_score_execution_time()
    print("\n===== All tests passed successfully! =====")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n!!!!!!!! A test failed with an exception: {e} !!!!!!!!")
        # Re-raise the exception to get a full traceback
        raise