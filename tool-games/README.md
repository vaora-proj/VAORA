# Tool Games (VAORA)

This directory reproduces our published results on the **Virtual-Tool** ([tool-games](https://k-r-allen.github.io/tool-games/)) environment. It contains the ToolPicker simulator, VLM runners, and a PHYRE DQN baseline for cross-dataset evaluation.

| Path | Contents |
|------|----------|
| [`environment/`](environment/) | ToolPicker simulator, inference scripts, trial JSONs |
| [`data/`](data/) | Paper human/model CSVs and DQN action cache |

## 1. Install the environment

Use the same Python environment as the rest of [VAORA](../README.md) (Python 3.10+ recommended).

**System dependency:** [Node.js](https://nodejs.org/) (required for the Chipmunk JS physics backend).

From `VAORA/tool-games/environment`:

```bash
cd /path/to/VAORA/tool-games/environment

pip install numpy scipy PyExecJS pymunk pillow tqdm torch

# Local Qwen3-VL inference
pip install transformers qwen-vl-utils

# Gemini API backend (optional)
pip install google-genai

python setup.py build
```

Quick smoke test:

```bash
python agent/vlm_toolpicker_agent.py \
  --backend qwen \
  --input_path ./unittest_files \
  --model_name Qwen/Qwen3-VL-8B-Instruct
```

## 2. Download checkpoints

Published weights are on Hugging Face: [`vaora-proj/vaora-checkpoints`](https://huggingface.co/vaora-proj/vaora-checkpoints).

```bash
huggingface-cli download vaora-proj/vaora-checkpoints --local-dir /path/to/vaora-checkpoints
```

Relevant directories:

```
vaora-checkpoints/
├── VAORA_VLM/          # VAORA-trained Qwen3-VL checkpoints (VLM inference)
│   ├── vaora_cross_template_testing_set_1/
│   ├── vaora_cross_template_testing_set_2/
│   ├── vaora_cross_template_testing_set_3/
│   └── within_template/
├── SFT_VLM/            # Supervised fine-tuned Qwen3-VL checkpoints (VLM inference)
│   ├── sft_cross_template_testing_set_1/
│   ├── sft_cross_template_testing_set_2/
│   ├── sft_cross_template_testing_set_3/
│   └── sft_within_template/
└── DQN-expert/         # PHYRE DQN baselines (DQN inference on tool-games)
    ├── dqn_cross_template_testing_set_1.ckpt
    ├── dqn_cross_template_testing_set_2.ckpt
    ├── dqn_cross_template_testing_set_3.ckpt
    └── dqn_within_template.ckpt
```

Set these for the commands below:

```bash
export VAORA_CKPT_ROOT=/path/to/vaora-checkpoints
```

## 3. Run inference

All commands below are run from `VAORA/tool-games/environment`.

### VLM (local checkpoint)

Evaluates on the full **Original** trial set by default (`Trials/Original`).

```bash
cd /path/to/VAORA/tool-games/environment

LOCAL_LOAD_PATH="${VAORA_CKPT_ROOT}/VAORA_VLM/within_template" \
  bash run_vlm_toolpicker_agent.sh
```

Use any model directory under `VAORA_VLM/` or `SFT_VLM/` for `LOCAL_LOAD_PATH`.

Unittest smoke test:

```bash
LOCAL_LOAD_PATH="${VAORA_CKPT_ROOT}/VAORA_VLM/within_template" \
  INPUT_PATH=./unittest_files \
  bash run_vlm_toolpicker_agent.sh
```

### VLM (Gemini API)

```bash
export GEMINI_API_KEY="YOUR_KEY"
BACKEND=gemini MODEL_NAME=gemini-3.1-flash bash run_vlm_toolpicker_agent.sh
```

### DQN-expert 

Run the PHYRE DQN baselines from `DQN-expert/` on Virtual-Tool levels (cross-dataset evaluation; no fine-tuning on tool-games).

**Prerequisites**
- Steps in [§1 Install](#1-install-the-environment) (including `torch` and `python setup.py build`).
- A downloaded `DQN-expert/*.ckpt` from [§2 Download checkpoints](#2-download-checkpoints).
- Action cache bundled in this repo: `data/action_array_ball_seed42_100k.npy` (used automatically).

From `VAORA/tool-games/environment`:

```bash
cd /path/to/VAORA/tool-games/environment

DQN_LOAD_FROM="${VAORA_CKPT_ROOT}/DQN-expert/dqn_within_template.ckpt" \
  bash run_dqn_toolgames_agent.sh
```

`DQN_LOAD_FROM` must point to either a single `.ckpt` file (Hugging Face layout) or a PHYRE training directory containing `ckpt.*` files.

By default the agent evaluates all levels in `Trials/Original`, scores up to 10,000 cached PHYRE actions, simulates the top 5 (`TOP_K=5`) on each tool, and writes pass@1 / pass@3 / pass@5 to `summary.json`.


**Optional flags** (environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_PATH` | `./Trials/Original` | ToolPicker JSON file or directory |
| `OUTPUT_ROOT` | `artifacts/dqn_toolgames` | Output root |
| `DQN_RANK_SIZE` | `10000` | Number of cached actions to score |
| `TOP_K` | `5` | Top actions to simulate per level |
| `ACTION_CACHE_PATH` | `../data/action_array_ball_seed42_100k.npy` | PHYRE action candidate cache |
| `SAVE_ATTEMPT_VIDEOS` | `1` | Write per-attempt GIFs (`0` to disable) |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU for DQN scoring |

Validation split example:

```bash
DQN_LOAD_FROM="${VAORA_CKPT_ROOT}/DQN-expert/dqn_cross_template_testing_set_1.ckpt" \
  INPUT_PATH=./Trials/Validation \
  bash run_dqn_toolgames_agent.sh
```

DQN-expert inference is supported only in `tool-games/` (not in `Batch_Inference/`).

### Outputs (VLM)

VLM and DQN runs write timestamped folders under `environment/artifacts/vlm_toolpicker/` and `environment/artifacts/dqn_toolgames/<timestamp>/`:

```text
<YYYY-MM-DD_HH-MM-SS>/
  ├── <level>.result.json
  ├── all_results.json
  └── summary.json
```

Override locations with `OUTPUT_ROOT` if needed.

### Evaluation

```bash
python eval_toolpicker_attempt_success.py \
  --run my_run artifacts/vlm_toolpicker/<timestamp>/all_results.json

python show_pass_at_k.py artifacts/vlm_toolpicker/<timestamp>/all_results.json
```
