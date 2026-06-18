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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated

import threading
from typing import Any, Dict

import requests


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert common numpy/torch types to Python builtins for JSON."""
    # Fast-path for already serializable primitives
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Dict
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}

    # Sequence types
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(v) for v in obj]

    # Numpy scalars/arrays
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.generic):
            return _to_json_serializable(obj.item())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # Torch tensors
    try:
        import torch  # type: ignore

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    # Fallback: best-effort conversion via item() or string
    if hasattr(obj, "item"):
        try:
            return _to_json_serializable(obj.item())
        except Exception:
            pass

    return str(obj)

_EVALUATOR_CACHE: Dict[Any, Any] = {}
_CACHE_LOCK = threading.Lock()
_REQUEST_SESSION = None


def _get_requests_session():
    """Return a pooled requests.Session for HTTP-based reward calls."""
    global _REQUEST_SESSION
    if _REQUEST_SESSION is None:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
        _REQUEST_SESSION = sess
    return _REQUEST_SESSION


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
        
    # elif "phyre" in data_source:
    #     from . import phyre_util

    #     # res = phyre_util.compute_score(solution_str, ground_truth, extra_info)
        
    #     # --- MODIFIED SECTION FOR DECOUPLED PHYRE SCORING ---
    #     try:
    #         import requests
    #         # The URL of the Phyre server we just created
    #         phyre_server_url = "http://127.0.0.1:5001/score"
            
    #         payload = _to_json_serializable({
    #             "solution_str": solution_str,
    #             "extra_info": extra_info,
    #         })
            
    #         # Make a network request to the dedicated Phyre server
    #         response = requests.post(phyre_server_url, json=payload, timeout=60) # 60-second timeout
    #         response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
    #         res = response.json().get("score", 0.0)

    #     except requests.exceptions.RequestException as e:
    #         print(f"Error calling Phyre scoring server: {e}")
    #         res = 0.0 # Return a default score on failure
    
    elif "phyre" in data_source:
        from . import phyre_util
        # 1. Define the Strict Schema (Keys + Order)
        # This guarantees that 'reward_extra_keys' is identical for EVERY sample.
        REQUIRED_KEYS = [
            "score", 
            "placement_reward", 
            "collision_reward", 
            "grounding_reward", 
            "predicted_prob"
        ]
        import os
        phyre_port = os.environ.get("PHYRE_PORT", "5001")
        phyre_server_url = f"http://127.0.0.1:{phyre_port}/score"
        session = _get_requests_session()

        try:
            payload = _to_json_serializable({
                "solution_str": solution_str,
                "extra_info": extra_info,
            })
            
            # Increase timeout for safety
            response = session.post(phyre_server_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                print(f"[default_compute_score] data: {data}")
                # raw_score = data.get("score", 0.0)
                raw_score = data.get("score", 0.0)
                # Handle case where server returns a simple float (old version compatibility)
                if isinstance(raw_score, dict):
                    raw_res = raw_score
                else:
                    # If it's a float, we must build the dict manually
                    raw_res = {k: 0.0 for k in REQUIRED_KEYS}
                    raw_res["score"] = float(raw_score)
            else:
                # Server error (500, 404, etc)
                raw_res = {k: 0.0 for k in REQUIRED_KEYS}
                raw_res["score"] = -1.0

        except Exception as e:
            # Timeout or Connection Refused
            # print(f"⚠️ Phyre connection failed: {e}") 
            raw_res = {k: 0.0 for k in REQUIRED_KEYS}
            raw_res["score"] = -1.0

        # 2. THE ENFORCER: Rebuild the dictionary using the strict list.
        # This ensures keys are always present and always in the EXACT SAME ORDER.
        res = {k: raw_res.get(k, 0.0) for k in REQUIRED_KEYS}
        
        # Now 'res' is safe to return to verl

    elif "minigrid" in data_source:
        from . import minigrid_util

        REQUIRED_KEYS = [
            "score",
            "success",
            "env_reward",
            "steps_used",
            "format_valid",
            "terminated",
            "truncated",
            "action_count",
            "scene_reward",
            "scene_parse_valid",
            "scene_grid_correct",
            "scene_claim_count",
            "scene_claims_correct",
            "scene_object_accuracy",
            "plan_reward",
            "plan_parse_valid",
            "plan_subtask_count",
            "plan_subtasks_matched",
            "plan_subtask_accuracy",
            "plan_exact_match",
        ]
        raw_res = minigrid_util.compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        res = {k: raw_res.get(k, 0.0) for k in REQUIRED_KEYS}

        

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
