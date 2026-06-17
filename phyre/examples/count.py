import json
import argparse

def analyze_solutions(json_path):
    """
    Analyzes a JSON file containing solved actions to count the total solutions
    and find the minimum number of solutions per task.

    Args:
        json_path (str): The path to the JSON file containing solved actions.
    """
    try:
        with open(json_path, 'r') as f:
            solutions_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found. Please ensure the path is correct.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_path}'. Please check the file's content.")
        return

    total_solutions = 0
    min_solutions_per_task = float('inf') # Initialize with infinity to ensure any count is smaller
    
    task_counts = {}

    print(f"Analyzing solutions from: {json_path}")
    if not solutions_data:
        print("No tasks found in the JSON file.")
        return

    for task_id, solutions_list in solutions_data.items():
        num_solutions_for_task = len(solutions_list)
        total_solutions += num_solutions_for_task
        
        task_counts[task_id] = num_solutions_for_task

        if num_solutions_for_task < min_solutions_per_task:
            min_solutions_per_task = num_solutions_for_task
            
    print("-" * 30)
    print(f"Total tasks analyzed: {len(solutions_data)}")
    print(f"Total number of solutions found across all tasks: {total_solutions}")
    print(f"Minimum number of solutions found for any single task: {min_solutions_per_task}")
    print("-" * 30)

    # Optional: Print tasks with the minimum number of solutions
    if min_solutions_per_task > 0: # Only if tasks actually had solutions
        tasks_with_min = [task for task, count in task_counts.items() if count == min_solutions_per_task]
        print(f"Tasks with {min_solutions_per_task} solution(s): {tasks_with_min}")
    elif min_solutions_per_task == 0:
        tasks_with_zero = [task for task, count in task_counts.items() if count == 0]
        print(f"Tasks with 0 solutions: {tasks_with_zero}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze a JSON file of PHYRE solved actions to count totals and find minimums."
    )
    parser.add_argument(
        '--json_path', 
        type=str, 
        default='all_solved_actions.json', # Default to the output of the previous script
        help="The path to the JSON file containing the solved actions data."
    )
    
    args = parser.parse_args()
    
    analyze_solutions(args.json_path)