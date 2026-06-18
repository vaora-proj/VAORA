import aiohttp
import asyncio
from typing import Any, Dict

# Import the original sync function for fallback
# Adjust the import path if your file structure is different, 
# e.g. from verl.utils.reward_score import default_compute_score
try:
    from verl.utils.reward_score import default_compute_score as _sync_compute_score
except ImportError:
    # Fallback if circular import or path issue, though usually this works
    pass

# --- LAZY GLOBAL VARIABLES ---
# Do NOT instantiate objects here. Just set them to None.
_ASYNC_SESSION = None

async def async_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """
    Asynchronous version of compute_score.
    Initializes the aiohttp session lazily to avoid 'no running event loop' errors.
    """
    global _ASYNC_SESSION

    # 1. Lazy Initialization: Create session only when loop is running
    if _ASYNC_SESSION is None or _ASYNC_SESSION.closed:
        # We create the connector HERE, inside the async function
        connector = aiohttp.TCPConnector(limit=64, limit_per_host=64)
        _ASYNC_SESSION = aiohttp.ClientSession(connector=connector)

    if "phyre" in data_source:
        phyre_server_url = "http://127.0.0.1:5001/score"
        
        payload = {
            "solution_str": solution_str,
            "extra_info": extra_info,
        }
        
        REQUIRED_KEYS = [
            "score", "placement_reward", "collision_reward", 
            "grounding_reward", "predicted_prob"
        ]

        try:
            # Use the global persistent session
            async with _ASYNC_SESSION.post(phyre_server_url, json=payload, timeout=60) as response:
                if response.status == 200:
                    data = await response.json()
                    raw_score = data.get("score", 0.0)
                else:
                    raw_score = -1.0
                
                # Normalize result
                if isinstance(raw_score, dict):
                    raw_res = raw_score
                else:
                    raw_res = {k: 0.0 for k in REQUIRED_KEYS}
                    raw_res["score"] = float(raw_score)

        except Exception as e:
            # Handle Timeout or Connection Refused
            raw_res = {k: 0.0 for k in REQUIRED_KEYS}
            raw_res["score"] = -1.0

        return {k: raw_res.get(k, 0.0) for k in REQUIRED_KEYS}

    else:
        # --- Fallback for other datasets ---
        # Run legacy sync code in a thread to prevent blocking the loop
        loop = asyncio.get_running_loop()
        
        # We need to import the sync function inside here if not imported at top
        from verl.utils.reward_score import default_compute_score
        
        return await loop.run_in_executor(
            None, 
            lambda: default_compute_score(
                data_source, solution_str, ground_truth, extra_info, 
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, **kwargs
            )
        )