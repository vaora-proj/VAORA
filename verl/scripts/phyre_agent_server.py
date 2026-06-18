# phyre_server.py
import sys, os

# --- CRITICAL PERFORMANCE FIX: FORCE SINGLE THREADING ---
# We must set these BEFORE importing numpy or torch.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import threading
from flask import Flask, request, jsonify
from typing import Any, Dict, Tuple, List

# Import your PhyreEvaluator class
from verl.utils.reward_score.phyre_util import PhyreAgentEvaluator

print(f"!!! SERVER IS USING PYTHON FROM: {sys.executable} !!!")

app = Flask(__name__)

EVALUATOR_CACHE: Dict[Any, Any] = {}
CACHE_LOCK = threading.Lock()

# --- NEW: PRE-WARMING FUNCTION ---
def prewarm_cache():
    """Initializes common PhyreEvaluator instances at startup."""
    
    # IMPORTANT: Add the configurations you will actually use here
    configs_to_warmup: List[Tuple[str, int]] = [
        ("ball_within_template", 0),
        ("my_template_based_split", 1),
        ("my_template_based_split", 2),
        ("my_template_based_split", 3),
        # Add any other (eval_setup, fold_id) combos you expect to use
    ]
    
    print("🔥 Starting server cache pre-warming...")
    for phyre_key in configs_to_warmup:
        eval_setup, fold_id = phyre_key
        print(f"  -> Initializing PhyreEvaluator for {phyre_key}...")
        model_dir = "agent_checkpoints"
        model_path = os.path.join(model_dir, eval_setup, str(fold_id), "ckpt.00100000")
        no_action_cache_path = os.path.join(model_dir, 'no_action_feats.npy')
        device = "cpu"
        print("Using device:", device)
        try:
            with CACHE_LOCK:
                if phyre_key not in EVALUATOR_CACHE:
                    EVALUATOR_CACHE[phyre_key] = PhyreAgentEvaluator(
                        eval_setup=eval_setup,
                        fold_id=fold_id,
                        model_path=model_path,
                        device=device,
                        no_action_cache_path=no_action_cache_path
                    )
            print(f"  ✅ Finished initializing {phyre_key}.")
        except Exception as e:
            print(f"  ❌ Error pre-warming {phyre_key}: {e}")
    print("✅ Server cache pre-warming complete. Ready for requests!")

@app.route('/score', methods=['POST'])
def score_phyre_task():
    # Define a default failure dict that matches your success structure
    # This ensures 'collision_reward' ALWAYS exists
    default_failure_response = {
        "score": -1.0, 
        "placement_reward": 0.0, 
        "collision_reward": 0.0, 
        "grounding_reward": 0.0, 
        "predicted_prob": 0.0
    }

    data = request.get_json()
    if not data:
        return jsonify({"score": default_failure_response}), 400

    try:
        solution_str = data['solution_str']
        extra_info = data['extra_info']
        
        # ... (Your existing cache loading logic) ...
        phyre_key = (extra_info['eval_setup'], extra_info['fold_id'])
        # ... (rest of setup) ...
        
        with CACHE_LOCK:
             if phyre_key not in EVALUATOR_CACHE:
                 # ... (init logic) ...
                 pass # simplified for brevity
        
        evaluator = EVALUATOR_CACHE[phyre_key]
        score = evaluator.score(extra_info['task_id'], solution_str, extra_info['item_counter'])
        
        return jsonify({"score": score})

    except Exception as e:
        print(f"❌ CRITICAL ERROR during scoring: {e}")
        # IMPORTANT: Return the structure with all keys, even on error!
        # You can set the main score to a penalty (e.g. -1.0)
        return jsonify({"score": default_failure_response})

# Call this manually here so Gunicorn runs it when loading the file
print("🔥 Gunicorn is loading the app, starting pre-warm...")
prewarm_cache()

if __name__ == '__main__':
    # --- CALL THE PRE-WARMING FUNCTION BEFORE STARTING THE SERVER ---
    # prewarm_cache()
    
    # Run the server on localhost, port 5001
    app.run(host='0.0.0.0', port=5001)