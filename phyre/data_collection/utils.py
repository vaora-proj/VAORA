import os, json, random
import phyre
from PIL import Image
import numpy as np
import cv2
from collections import defaultdict

def log_simulation_results(pred_action, task_index, tasks, simulation):
    # Three statuses could be returned.
    print('################# Simulation Statuses #################')
    print('Action solves task:', phyre.SimulationStatus.SOLVED)
    print('Action does not solve task:', phyre.SimulationStatus.NOT_SOLVED)
    print('Action is an invalid input on task (e.g., occludes a task object):',
        phyre.SimulationStatus.INVALID_INPUT)
    # May call is_* methods on the status to check the status.
    print()
    print('Result of taking action', pred_action, 'on task', tasks[task_index], 'is:',
        simulation.status)
    print('Does', pred_action, 'solve task', tasks[task_index], '?', simulation.status.is_solved())
    print('Is', pred_action, 'an invalid action on task', tasks[task_index], '?',
        simulation.status.is_invalid())
    print()
    return simulation.status.is_solved(), simulation.status.is_invalid()

def save_gif(phyre_images, path='./simulation.gif', duration=100, loop=0):
    frames = []
    for image in phyre_images:
        image = phyre.observations_to_float_rgb(image)
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        # Convert NumPy array to PIL image
        img_pil = Image.fromarray(img_uint8)
        frames.append(img_pil)
    frames[0].save(path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=loop)

def save_mp4(phyre_images, path='./simulation.mp4', fps=3.0):
    # Convert each frame to an image and save to a temporary video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = phyre_images[0].shape
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in phyre_images:
        out.write(frame)

    out.release()
    print(f"Video saved to {path}")

def save_json(data, path):
    """
    Save a dictionary to a JSON file.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {path}")

def save_image(image, path):
    """
    Save a NumPy array as an image file.
    """
    img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_pil.save(path)
    print(f"Image saved to {path}")

def convert_to_np(simulator_images):
    frames = []
    for image in simulator_images:
        image = phyre.observations_to_float_rgb(image)
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        frames.append(img_bgr)
    return np.array(frames)


def uniform_sample_dict(task_dict, total_samples=160000):
    tasks = list(task_dict.keys())
    num_tasks = len(tasks)
    target_per_task = total_samples // num_tasks  # 100 per task

    selected = defaultdict(list)

    # First pass
    for task_id in tasks:
        solutions = task_dict[task_id]
        if len(solutions) >= target_per_task:
            sampled = random.sample(solutions, target_per_task)
        else:
            sampled = solutions
        selected[task_id].extend(sampled)

    current_total = sum(len(sols) for sols in selected.values())
    remaining = total_samples - current_total

    if remaining > 0:
        # Second pass: fill from overflow
        overflow_pool = []
        for task_id in tasks:
            solutions = task_dict[task_id]
            already_sampled = set(tuple(s) for s in selected[task_id])
            extras = [s for s in solutions if tuple(s) not in already_sampled]
            overflow_pool.extend((task_id, s) for s in extras)

        additional_samples = random.sample(overflow_pool, min(remaining, len(overflow_pool)))
        for task_id, solution in additional_samples:
            selected[task_id].append(solution)

    # Convert defaultdict back to normal dict
    return dict(selected)

def load_solution(json_path):
    try:
        with open(json_path, 'r') as f:
            solutions_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found. Please ensure the path is correct.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_path}'. Please check the file's content.")
        return
    return solutions_data