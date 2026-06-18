"""Utility helpers for running inference with trained DQN checkpoints."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np  # type: ignore
import torch  # type: ignore

import verl.utils.reward_score.neural_agent as neural_agent
import verl.utils.reward_score.nets as nets
import phyre

Observation = Union[np.ndarray, torch.Tensor]
ActionArray = Union[np.ndarray, torch.Tensor, Sequence[float]]


class DQNInferenceModel:
    """Light-weight wrapper around the trained DQN to run inference on demand.

    Example:
        inference = DQNInferenceModel("/path/to/ckpt.00010000", device="cpu")
        simulator = phyre.initialize_simulator(["task_id"], "ball_within_template")
        obs = simulator.initial_scenes[0]
        action = np.array([0.42, 0.8, 0.05], dtype=np.float32)
        prob = inference.predict_pair_probability(obs, action)
    """

    def __init__(self,
                 checkpoint_path: Union[str, Path],
                 device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device(device) if device is not None else nets.DEVICE
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.model_kwargs: Dict[str, Union[str, int, float]] = {}
        self.model: Optional[torch.nn.Module] = None
        self._load_model()

    def _resolve_checkpoint_path(self,
                                 checkpoint_path: Union[str,
                                                        Path]) -> Path:
        path = Path(checkpoint_path)
        if path.is_file():
            return path
        if path.is_dir():
            latest = neural_agent.get_latest_checkpoint(str(path))
            if latest is None:
                raise FileNotFoundError(
                    f"No checkpoints found under directory: {path}")
            return Path(latest)
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    def _load_model(self) -> None:
        load_kwargs = dict(map_location=self.device)
        try:
            checkpoint = torch.load(self.checkpoint_path,
                                    weights_only=False,
                                    **load_kwargs)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, **load_kwargs)
        missing = {'model_kwargs', 'model'} - set(checkpoint)
        if missing:
            raise KeyError(
                f"Checkpoint file {self.checkpoint_path} is missing keys: "
                f"{', '.join(sorted(missing))}")
        self.model_kwargs = checkpoint['model_kwargs']
        model = neural_agent.build_model(**self.model_kwargs)
        model.load_state_dict(checkpoint['model'])
        model.to(self.device)
        model.eval()
        self.model = model

    @property
    def action_space_dim(self) -> int:
        assert 'action_space_dim' in self.model_kwargs
        return int(self.model_kwargs['action_space_dim'])

    def preprocess_observation(self, observation: Observation) -> Dict[str,
                                                                       torch.Tensor]:
        obs_tensor = self._prepare_observation(observation)
        assert self.model is not None
        with torch.no_grad():
            features = self.model.preprocess(obs_tensor)
        return {k: v.detach() for k, v in features.items()}

    def predict_logits(self,
                       observation: Optional[Observation],
                       actions: ActionArray,
                       ) -> torch.Tensor:
        assert self.model is not None
        action_tensor = self._prepare_actions(actions)
        paired_logits = self._maybe_predict_paired_logits(observation,
                                                          action_tensor)
        return paired_logits

    def predict_proba(self,
                      observation: Optional[Observation],
                      actions: ActionArray,
                      ) -> np.ndarray:
        logits = self.predict_logits(observation,
                                     actions)
        return torch.sigmoid(logits).cpu().numpy()

    def predict_pair_probability(self,
                                 observation: Observation,
                                 action: ActionArray) -> float:
        probs = self.predict_proba(observation, action)
        return float(np.squeeze(probs))

    def _prepare_observation(self, observation: Observation) -> torch.Tensor:
        obs_tensor = torch.as_tensor(observation,
                                     dtype=torch.long,
                                     device=self.device)
        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.unsqueeze(0)
        elif obs_tensor.ndim != 3:
            raise ValueError(
                f"Observation tensor must have 2 or 3 dimensions, "
                f"got shape {tuple(obs_tensor.shape)}.")
        return obs_tensor

    def _prepare_actions(self, actions: ActionArray) -> torch.Tensor:
        action_tensor = torch.as_tensor(actions,
                                        dtype=torch.float64,
                                        device=self.device)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        elif action_tensor.ndim != 2:
            raise ValueError(
                f"Action tensor must have 1 or 2 dimensions, "
                f"got shape {tuple(action_tensor.shape)}.")
        return action_tensor

    def _maybe_predict_paired_logits(self, observation, actions):
        if observation is None:
            return None
        assert self.model is not None
        obs_tensor = self._prepare_observation(observation)
        if obs_tensor.shape[0] != actions.shape[0]:
            return None
        logits = []
        with torch.no_grad():
            logits = self.model(
                obs_tensor,
                actions,).detach().cpu()
        return logits


def _sample_task_ids(eval_setup: str, fold_id: int, split: str,
                     num_tasks: int) -> Sequence[str]:
    split_to_index = {'train': 0, 'dev': 1, 'test': 2}
    if split not in split_to_index:
        raise ValueError(
            f"Unknown split '{split}'. Expected one of "
            f"{', '.join(split_to_index)}.")
    train, dev, test = phyre.get_fold(eval_setup, fold_id)
    split_tasks = (train, dev, test)[split_to_index[split]]
    if num_tasks <= 0:
        raise ValueError('num_tasks must be positive.')
    if num_tasks > len(split_tasks):
        raise ValueError(
            f"Requested {num_tasks} tasks but split '{split}' "
            f"only has {len(split_tasks)} tasks.")
    return split_tasks[:num_tasks]


def _sample_actions(cache: phyre.SimulationCache, num_actions: int,
                    seed: int) -> np.ndarray:
    if num_actions <= 0:
        raise ValueError('num_actions must be positive.')
    action_array = cache.action_array
    if num_actions > len(action_array):
        raise ValueError(
            f'Requested {num_actions} actions but cache only has '
            f'{len(action_array)}.')
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(action_array), size=num_actions, replace=False)
    return action_array[indices]



'''
how to use the DQNInferenceModel to score actions on a task?
inference = DQNInferenceModel(checkpoint_path, device="cpu")
simulator = phyre.initialize_simulator(["task_id"], "ball_within_template")

Input:batch of obs (batch_size, 256, 256) and action (batch_size, 3)
obs = simulator.initial_scenes[0] (with shape(256, 256))
action = [x, y, r] in 0-1 range

prob = inference.predict_proba(obs, action)
Output:batch of prob (batch_size)
'''



def evaluate_checkpoint_on_random_actions(
        checkpoint_path: Union[str, Path],
        eval_setup: str = 'ball_within_template',
        fold_id: int = 0,
        split: str = 'test',
        num_tasks: int = 3,
        num_actions: int = 64,
        top_k: int = 5,
        batch_size: int = 128,
        seed: int = 0,
        device: Optional[Union[str, torch.device]] = None):
    """Loads a checkpoint and scores random actions on sampled tasks.

    Returns:
        Dictionary with metadata and per-task scoring summaries.
    """
    if top_k <= 0:
        raise ValueError('top_k must be positive.')
    if top_k > num_actions:
        raise ValueError('top_k cannot exceed num_actions.')
    action_tier = phyre.eval_setup_to_action_tier(eval_setup)
    inference = DQNInferenceModel(checkpoint_path, device=device)
    cache = phyre.get_default_100k_cache(action_tier)
    sampled_actions = _sample_actions(cache, num_actions, seed=seed)
    if sampled_actions.shape[1] != inference.action_space_dim:
        raise ValueError(
            'Action dimensionality mismatch between checkpoint '
            f'({inference.action_space_dim}) and cache '
            f'({sampled_actions.shape[1]}).')
    task_ids = _sample_task_ids(eval_setup, fold_id, split, num_tasks)
    simulator = phyre.initialize_simulator(task_ids, action_tier)
    summaries = []
    for local_idx, task_id in enumerate(simulator.task_ids):
        observation = simulator.initial_scenes[local_idx]
        # Make sure observations have shape (num_actions, ...) matching sampled_actions
        observations = np.stack([observation] * len(sampled_actions))
        probs = inference.predict_proba(observations,
                                        sampled_actions)
        top_indices = np.argsort(-probs)[:top_k]
        candidates = []
        for rank, action_idx in enumerate(top_indices, start=1):
            action = sampled_actions[action_idx]
            sim_result = simulator.simulate_action(local_idx,
                                                   action,
                                                   need_images=False)
            status = phyre.SimulationStatus(sim_result.status).name
            candidates.append(
                dict(rank=rank,
                     probability=float(probs[action_idx]),
                     status=status,
                     action=action.tolist()))
        summaries.append(dict(task_id=task_id, candidates=candidates))
    return dict(checkpoint=str(checkpoint_path),
                eval_setup=eval_setup,
                tier=action_tier,
                split=split,
                fold_id=fold_id,
                num_tasks=len(summaries),
                num_actions=num_actions,
                top_k=top_k,
                results=summaries)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Run a quick sanity check on a DQN checkpoint.')
    parser.add_argument('--checkpoint',
                        required=True,
                        help='Path to a checkpoint file or directory.')
    parser.add_argument('--eval-setup',
                        default='ball_within_template',
                        choices=phyre.MAIN_EVAL_SETUPS,
                        help='Evaluation setup to pull tasks from.')
    parser.add_argument('--fold-id',
                        type=int,
                        default=0,
                        help='Fold identifier to select tasks.')
    parser.add_argument('--split',
                        choices=('train', 'dev', 'test'),
                        default='test',
                        help='Which split of the fold to sample tasks from.')
    parser.add_argument('--num-tasks',
                        type=int,
                        default=3,
                        help='How many tasks to evaluate.')
    parser.add_argument('--num-actions',
                        type=int,
                        default=64,
                        help='Number of random actions to score.')
    parser.add_argument('--top-k',
                        type=int,
                        default=5,
                        help='How many top actions to simulate per task.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        help='Batch size for neural network inference.')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Seed for random action sampling.')
    parser.add_argument('--device',
                        type=str,
                        default=None,
                        help='Optional torch device override.')
    return parser


def _cli_main():
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        format=('%(asctime)s %(levelname)-8s {%(module)s:%(lineno)d} '
                '%(message)s'),
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    report = evaluate_checkpoint_on_random_actions(
        checkpoint_path=args.checkpoint,
        eval_setup=args.eval_setup,
        fold_id=args.fold_id,
        split=args.split,
        num_tasks=args.num_tasks,
        num_actions=args.num_actions,
        top_k=args.top_k,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device)
    for summary in report['results']:
        logging.info('Task %s', summary['task_id'])
        for candidate in summary['candidates']:
            logging.info('  #%d prob=%.3f status=%s action=%s',
                         candidate['rank'], candidate['probability'],
                         candidate['status'],
                         ','.join(f'{v:.3f}' for v in candidate['action']))
    logging.info('Finished scoring %d tasks with %d random actions.',
                 report['num_tasks'], report['num_actions'])


if __name__ == '__main__':
    _cli_main()