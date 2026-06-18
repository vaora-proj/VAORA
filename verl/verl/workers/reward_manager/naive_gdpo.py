# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        print(f"[NaiveRewardManager][__init__] compute_score: {compute_score.__class__.__name__}")
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""
        print(f"[NaiveRewardManager][__call__]")
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        score_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        placement_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        collision_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        grounding_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        prob_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        max_workers = int(os.environ.get("REWARD_THREADS", 8))
        max_workers = max(1, min(max_workers, len(data)))

        def _process_single(idx: int):
            data_item = data[idx]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            reward_extra = {}
            if isinstance(score, dict):
                # reward = score["score"]
                reward = score
                reward_extra = score
            else:
                reward = score

            return {
                "idx": idx,
                "reward": reward,
                "reward_extra": reward_extra,
                "valid_response_length": valid_response_length,
                "data_source": data_source,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "score": score,
            }

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_single, i) for i in range(len(data))]
            for fut in as_completed(futures):
                results.append(fut.result())

        results.sort(key=lambda x: x["idx"])

        already_print_data_sources: dict[str, int] = defaultdict(int)
        for res in results:
            i = res["idx"]
            reward = res["reward"]
            reward_extra = res["reward_extra"]
            valid_response_length = res["valid_response_length"]
            data_source = res["data_source"]
            prompt_str = res["prompt_str"]
            response_str = res["response_str"]
            ground_truth = res["ground_truth"]
            score = res["score"]

            if isinstance(reward_extra, dict):
                for key, value in reward_extra.items():
                    reward_extra_info[key].append(value)

            score_tensor[i, valid_response_length - 1] = reward["score"]
            placement_tensor[i, valid_response_length - 1] = reward["placement_reward"]
            collision_tensor[i, valid_response_length - 1] = reward["collision_reward"]
            grounding_tensor[i, valid_response_length - 1] = reward["grounding_reward"]
            prob_tensor[i, valid_response_length - 1] = reward["predicted_prob"]


            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "score_tensor": score_tensor,
                "placement_tensor": placement_tensor,
                "collision_tensor": collision_tensor,
                "grounding_tensor": grounding_tensor,
                "prob_tensor": prob_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return score_tensor, placement_tensor, collision_tensor, grounding_tensor, prob_tensor, reward_extra_info
