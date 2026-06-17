import phyre
import json
import argparse

def find_and_save_solutions(eval_setup, fold_id, num_solutions, save_path, tier='ball'):
    """
    Finds a specified number of solved actions for a given task setup and saves them to a JSON file.

    Args:
        eval_setup (str): The name of the evaluation setup (e.g., 'ball_cross_template').
        fold_id (int): The fold ID for the evaluation setup.
        num_solutions (int): The number of solved actions to find for each task.
        save_path (str): The path to the output JSON file.
        tier (str): The PHYRE tier to use.
    """
    # 1. Get task IDs from the specified evaluation setup and fold.
    # We'll use the training split of the fold to find solutions.
    try:
        train_tasks, _, _ = phyre.get_fold(eval_setup, fold_id)
    except ValueError:
        print(f"Error: Evaluation setup '{eval_setup}' not found.")
        print("Available setups for 'ball' tier include: 'ball_cross_template', 'ball_within_template'")
        return

    cache = phyre.get_default_100k_cache(tier)
    solutions_data = {}

    print(f"Processing {len(train_tasks)} tasks from '{eval_setup}' fold {fold_id}...")
    for task_id in train_tasks:
        if task_id not in solutions_data:
            solutions_data[task_id] = []
        
        statuses = cache.load_simulation_states(task_id)
        
        # 2. Find the requested number of solved actions.
        for i, status in enumerate(statuses):
            if len(solutions_data[task_id]) >= num_solutions:
                break  # Stop once we have enough solutions for this task.
            if status == phyre.SimulationStatus.SOLVED:
                action = cache.action_array[i].tolist() # Convert numpy array to list for JSON.
                solutions_data[task_id].append(action)

        print(f"Found {len(solutions_data[task_id])} solution(s) for task {task_id}")

    # 3. Save the collected data to a JSON file.
    with open(save_path, 'w') as f:
        json.dump(solutions_data, f, indent=1)
    
    print(f"\nSuccessfully saved solved actions to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Find and save solved actions for PHYRE tasks based on their evaluation setup."
    )
    parser.add_argument(
        '--eval_setup', 
        type=str,
        default='ball_within_template',
        help="The evaluation setup to process (e.g., 'ball_cross_template', 'ball_within_template')."
    )
    parser.add_argument(
        '--fold_id',
        type=int,
        default=0,
        help="The fold ID for the chosen evaluation setup."
    )
    parser.add_argument(
        '--num_solutions', 
        type=int, 
        default=1,
        help="The number of solutions to find and save for each task."
    )
    parser.add_argument(
        '--save_path', 
        type=str, 
        default='solved_actions.json',
        help="The file path for the output JSON file."
    )
    
    args = parser.parse_args()
    
    find_and_save_solutions(args.eval_setup, args.fold_id, args.num_solutions, args.save_path)