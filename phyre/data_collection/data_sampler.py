import math
import json, os
import random
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook

import phyre
from utils import *

random.seed(0)

def main(args):
    eval_setups = args.eval_setups
    fold_id = args.fold_id
    output_root = args.output_root
    solution_json = args.solution_json
    
    solutions_data = load_solution(solution_json)
    solutions_data = uniform_sample_dict(solutions_data, total_samples=160000)
    
    with open('solutions_data_sampled.json', 'w') as f:
        json.dump(solutions_data, f, indent=4)
    
    description = "A video of a Phyre task simulation, gravity is applied to all objects, and the ball is dropped from the top of the scene. The video shows the interaction of the ball with the objects in the scene, and the ball's trajectory as it moves through the scene."
    splits = ['train', 'test']
    os.makedirs(output_root, exist_ok=True)
    expname = f'{eval_setups}_fold_{fold_id}_{time.strftime("%Y%m%d-%H%M%S")}'
    
    for split in splits:
        print('Experiment name:', expname)
        output_dir = os.path.join(output_root, expname, split)
        os.makedirs(output_dir, exist_ok=True)
        video_dir = os.path.join(output_dir, 'videos')
        input_image_dir = os.path.join(output_dir, 'input_images')
        helper_image_dir = os.path.join(output_dir, 'helper_images')
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(input_image_dir, exist_ok=True)
        os.makedirs(helper_image_dir, exist_ok=True)
        metadata = os.path.join(output_dir, 'metadata.jsonl')
        metadata_videos = os.path.join(output_dir, 'videos', 'metadata.jsonl')
        metadata_input_images = os.path.join(output_dir, 'input_images', 'metadata.jsonl')
        metadata_helper_images = os.path.join(output_dir, 'helper_images', 'metadata.jsonl')
        
        if eval_setups == 'all':
            eval_sets = ['ball_within_template', 'ball_cross_template', 'two_balls_within_template', 'two_balls_cross_template', 'ball_phyre_to_tools']
        else:
            eval_sets = eval_setups.split(',')

        for eval_setup in eval_sets:
            train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
            print('Size of resulting splits:\n train:', len(train_tasks), '\n dev:', len(dev_tasks), '\n test:', len(test_tasks))
            
            action_tier = phyre.eval_setup_to_action_tier(eval_setup)
            print('Action tier for', eval_setup, 'is', action_tier)
            
            if split == 'train':
                tasks = train_tasks
            else:
                tasks = dev_tasks

            # # Create the simulator from the tasks and tier.
            # simulator = phyre.initialize_simulator(tasks, action_tier)
            # Create a simulator for the task and tier.
            simulator = phyre.initialize_simulator(tasks, action_tier)
            # evaluator = phyre.Evaluator(tasks)
            
            # num_samples = phyre.MAX_TEST_ATTEMPTS
            num_samples =  1
            count_id = 0
            
            for task_index in range(len(tasks)):
                task_id = simulator.task_ids[task_index]
                print('Task ID:', task_id)
                initial_scene = simulator.initial_scenes[task_index]
                print('Initial scene shape=%s dtype=%s' % (initial_scene.shape, initial_scene.dtype))
                init_img = phyre.observations_to_float_rgb(initial_scene)
                
                for idx, action in enumerate(solutions_data[task_id]):
                    action = np.array(action, dtype=np.float32)
                    simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True)
                    
                    solved, invalid = log_simulation_results(action, task_index, tasks, simulation)
                    if invalid:
                        print(f'Invalid action {action} for task {task_id}, skipping...')
                        continue
                    if args.solved_only and not solved:
                        continue
                    
                    # evaluator.maybe_log_attempt(task_index, simulation.status)
                    
                    # Save the simulation results.
                    img = simulation.images[0]
                    first_img = phyre.observations_to_float_rgb(img)
                    task_id = task_id.replace(':', '_')
                    filename = f'{task_id}_{idx}'
                    
                    save_image(init_img, path=os.path.join(input_image_dir, f'{filename}.png'))
                    save_image(first_img, path=os.path.join(helper_image_dir, f'{filename}.png'))
                    save_mp4(convert_to_np(simulation.images), path=os.path.join(video_dir, f'{filename}.mp4'), fps=6.0)
                    
                    with open(metadata, 'a') as f:
                        f.write(json.dumps({'id': count_id, 'prompt': description, 'action': str(action)}) + '\n')
                    with open(metadata_videos, 'a') as f:
                        f.write(json.dumps({'file_name': f'{filename}.mp4', 'id': count_id}) + '\n')
                    with open(metadata_input_images, 'a') as f:
                        f.write(json.dumps({'file_name': f'{filename}.png', 'id': count_id}) + '\n')
                    with open(metadata_helper_images, 'a') as f:
                        f.write(json.dumps({'file_name': f'{filename}.png', 'id': count_id}) + '\n')
                        
                    count_id += 1
                    print(f"Saved {filename}.mp4 and {filename}.png")

                # while evaluator.get_attempts_for_task(task_index) < num_samples:
                #     # Sample a random valid action from the simulator for the given action space.
                #     action = simulator.sample()
                #     # Simulate the given action and add the status from taking the action to the evaluator.
                #     simulation = simulator.simulate_action(task_index,
                #                                         action,
                #                                         need_images=True, need_featurized_objects=True)
                #     solved, invalid = log_simulation_results(action, task_index, tasks, simulation)
                #     if invalid:
                #         continue
                #     if args.solved_only and not solved:
                #         continue
                #     evaluator.maybe_log_attempt(task_index, simulation.status)
                    
                #     # Save the simulation results.
                #     img = simulation.images[0]
                #     first_img = phyre.observations_to_float_rgb(img)
                #     task_id = task_id.replace(':', '_')
                #     filename = f'{task_id}_{evaluator.get_attempts_for_task(task_index)}'
                #     save_image(init_img, path=os.path.join(input_image_dir, f'{filename}.png'))
                #     save_image(first_img, path=os.path.join(helper_image_dir, f'{filename}.png'))
                #     # save_gif(simulation.images, './temp0.gif')
                #     save_mp4(convert_to_np(simulation.images), path=os.path.join(video_dir, f'{filename}.mp4'), fps=6.0)
                    
                #     with open(metadata, 'a') as f:
                #         f.write(json.dumps({'id': count_id, 'prompt': description, 'action': str(action)}) + '\n')
                #     with open(metadata_videos, 'a') as f:
                #         f.write(json.dumps({'file_name': f'{filename}.mp4', 'id': count_id}) + '\n')
                #     with open(metadata_input_images, 'a') as f:
                #         f.write(json.dumps({'file_name': f'{filename}.png', 'id': count_id}) + '\n')
                #     with open(metadata_helper_images, 'a') as f:
                #         f.write(json.dumps({'file_name': f'{filename}.png', 'id': count_id}) + '\n')
                #     count_id += 1
                #     print(f"Saved {filename}.mp4 and {filename}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze motion in videos using Gemini-2.5-Pro.")
    parser.add_argument('--eval_setups', type=str, default='ball_within_template') # ball_cross_template ball_within_template two_balls_cross_template two_balls_within_template ball_phyre_to_tools
    parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument('--output_root', type=str, default='/work/u5597173/Data/phyre')
    parser.add_argument('--solved_only', action='store_true', help='Only collect solved tasks.')
    parser.add_argument('--solution_json', type=str, default='all_solved_actions.json', help='Path to the solution JSON file to load solutions from.')

    args = parser.parse_args()
    main(args)