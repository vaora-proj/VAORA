import phyre
import json
import argparse

def find_and_save_all_solutions(eval_setup, fold_id, save_path, tier='ball'):
    """
    Finds all recorded solved actions for a given task setup and saves them to a JSON file.

    Args:
        eval_setup (str): The name of the evaluation setup (e.g., 'ball_cross_template').
        fold_id (int): The fold ID for the evaluation setup.
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
        
        # 2. Find all solved actions.
        for i, status in enumerate(statuses):
            if status == phyre.SimulationStatus.SOLVED:
                action = cache.action_array[i].tolist() # Convert numpy array to list for JSON.
                solutions_data[task_id].append(action)

        print(f"Found {len(solutions_data[task_id])} solution(s) for task {task_id}")

    # 3. Save the collected data to a JSON file.
    with open(save_path, 'w') as f:
        json.dump(solutions_data, f, indent=1)
    
    print(f"\nSuccessfully saved all found solved actions to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Find and save all recorded solved actions for PHYRE tasks based on their evaluation setup."
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
        '--save_path', 
        type=str, 
        default='all_solved_actions.json', # Changed default save path to reflect saving all
        help="The file path for the output JSON file."
    )
    
    args = parser.parse_args()
    
    find_and_save_all_solutions(args.eval_setup, args.fold_id, args.save_path)