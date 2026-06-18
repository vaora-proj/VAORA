# VAORA-VERL: Interactive GRPO Training on PHYRE

## Objective

This guide explains how to reproduce our **interactive GRPO training** experiments with **Qwen3-VL-8B** on the **PHYRE** physics puzzle environment using the VAORA fork of [verl](https://github.com/verl-project/verl).

The training loop is **multi-turn and interaction-based**: the VLM proposes actions in natural language, a PHYRE simulator scores placement / collision / grounding components, and a DQN expert provides `predicted_prob` for GDPO (Group Decomposed Policy Optimization). Reward scoring runs in a sidecar HTTP server (`scripts/phyre_agent_server.py`) started automatically by the training script.

---

## Prerequisites

1. **Environment** — follow [VAORA/README.md](../README.md):
   - Python 3.10+ conda env
   - `VAORA/verl` installed (`pip install --no-deps -e .` after `scripts/install_vllm_sglang_mcore.sh`)
   - `VAORA/phyre` installed (`pip install -e .`)
   - `gunicorn` for the reward server (`pip install gunicorn`)

2. **Hardware** — the reference script uses **2 GPUs** (`trainer.n_gpus_per_node=2`). Adjust if needed.

3. **Hugging Face CLI** — install and log in if the repos are gated:

```bash
pip install -U huggingface_hub
huggingface-cli login
```

---

## Step 1 — Download assets from Hugging Face

### Training dataset

Dataset repo: [`vaora-proj/vaora-dataset`](https://huggingface.co/datasets/vaora-proj/vaora-dataset)

```bash
huggingface-cli download vaora-proj/vaora-dataset \
  --repo-type dataset \
  --local-dir /path/to/vaora-dataset
```

GRPO training parquet files live under `train_data/VAORA_DATA/`:

```
vaora-dataset/
└── train_data/
    └── VAORA_DATA/
        ├── vaora_dataset_within_template/
        │   ├── train.parquet
        │   └── test.parquet
        ├── vaora_dataset_cross_template_testing_set_1/
        │   ├── train.parquet
        │   └── test.parquet
        ├── vaora_dataset_cross_template_testing_set_2/
        └── vaora_dataset_cross_template_testing_set_3/
```

### SFT VLM checkpoint (training initialization)

Checkpoint repo: [`vaora-proj/vaora-checkpoints`](https://huggingface.co/vaora-proj/vaora-checkpoints)

```bash
huggingface-cli download vaora-proj/vaora-checkpoints \
  --local-dir /path/to/vaora-checkpoints
```

Use a directory under `SFT_VLM/` as the **actor model** for GRPO (supervised fine-tuned Qwen3-VL before RL):

```
vaora-checkpoints/
└── SFT_VLM/
    ├── sft_within_template/
    ├── sft_cross_template_testing_set_1/
    ├── sft_cross_template_testing_set_2/
    └── sft_cross_template_testing_set_3/
```

Each folder is a standard Hugging Face model directory (`config.json`, `model-*.safetensors`, tokenizer files, etc.).

### DQN-expert checkpoint (reward server)

From the same [`vaora-proj/vaora-checkpoints`](https://huggingface.co/vaora-proj/vaora-checkpoints) repo:

```
vaora-checkpoints/
└── DQN-expert/
    ├── dqn_within_template.ckpt
    ├── dqn_cross_template_testing_set_1.ckpt
    ├── dqn_cross_template_testing_set_2.ckpt
    └── dqn_cross_template_testing_set_3.ckpt
```

These are **not** VLM weights. They are PHYRE DQN baselines used at reward time to compute `predicted_prob` and related signals.

---

## Step 2 — Place the DQN-expert checkpoint

The training script launches `scripts/phyre_agent_server.py`, which loads DQN weights from **`VAORA/verl/agent_checkpoints/`** (relative to the repo root; the script `cd`s there before starting).

Place checkpoints under one of these two directories:

### `VAORA/verl/agent_checkpoints/ball_within_template/`

For the **within-template** split (`eval_setup=ball_within_template`, `fold_id=0`):

```
VAORA/verl/agent_checkpoints/ball_within_template/
└── 0/
    └── ckpt.00100000
```

```bash
cd /path/to/VAORA/verl
mkdir -p agent_checkpoints/ball_within_template/0
cp /path/to/vaora-checkpoints/DQN-expert/dqn_within_template.ckpt \
   agent_checkpoints/ball_within_template/0/ckpt.00100000
```

### `VAORA/verl/agent_checkpoints/my_template_based_split/`

For **cross-template** testing sets (`eval_setup=my_template_based_split`, `fold_id=1/2/3`):

```
VAORA/verl/agent_checkpoints/my_template_based_split/
├── 1/
│   └── ckpt.00100000    # from dqn_cross_template_testing_set_1.ckpt
├── 2/
│   └── ckpt.00100000    # from dqn_cross_template_testing_set_2.ckpt
└── 3/
    └── ckpt.00100000    # from dqn_cross_template_testing_set_3.ckpt
```

```bash
mkdir -p agent_checkpoints/my_template_based_split/1
cp /path/to/vaora-checkpoints/DQN-expert/dqn_cross_template_testing_set_1.ckpt \
   agent_checkpoints/my_template_based_split/1/ckpt.00100000
# Repeat for fold 2 and 3 with the matching HF .ckpt files.
```

| Split | HF DQN file | Place under |
|-------|-------------|-------------|
| Within template | `dqn_within_template.ckpt` | `agent_checkpoints/ball_within_template/0/ckpt.00100000` |
| Cross template set 1 | `dqn_cross_template_testing_set_1.ckpt` | `agent_checkpoints/my_template_based_split/1/ckpt.00100000` |
| Cross template set 2 | `dqn_cross_template_testing_set_2.ckpt` | `agent_checkpoints/my_template_based_split/2/ckpt.00100000` |
| Cross template set 3 | `dqn_cross_template_testing_set_3.ckpt` | `agent_checkpoints/my_template_based_split/3/ckpt.00100000` |

`no_action_feats.npy` in `agent_checkpoints/` is optional; the reward server regenerates it if missing (slower first startup).

---

## Step 3 — Configure and run GRPO training

Open `examples/grpo_trainer/run_qwen3_vl-8b-2card-phyre.sh` and edit the three paths at the top of the script:

```bash
SFT_MODEL_PATH="/path/to/SFT_VLM/sft_within_template"
TRAIN_FILE="/path/to/vaora-dataset/train_data/VAORA_DATA/vaora_dataset_within_template/train.parquet"
VAL_FILE="/path/to/vaora-dataset/train_data/VAORA_DATA/vaora_dataset_within_template/test.parquet"
```

These feed directly into the verl trainer:

- `data.train_files` / `data.val_files` — training and validation parquet
- `actor_rollout_ref.model.path` — local SFT Qwen3-VL checkpoint

Then run:

```bash
cd /path/to/VAORA/verl
bash examples/grpo_trainer/run_qwen3_vl-8b-2card-phyre.sh
```

Optional: pass rollout engine or Hydra overrides as arguments:

```bash
bash examples/grpo_trainer/run_qwen3_vl-8b-2card-phyre.sh vllm trainer.total_epochs=50
```

---

## What the script does

1. Starts the PHYRE reward server (`gunicorn … scripts.phyre_agent_server:app`) on port `5001` (`PHYRE_PORT`).
2. Runs `python3 -m verl.trainer.main_ppo` with:
   - `algorithm.adv_estimator=grpo`
   - `+algorithm.gdpo=True` — decomposed advantages for placement / collision / grounding / score
   - Qwen3-VL-8B loaded from your SFT checkpoint
   - Multi-modal parquet data (`data.image_key=images`)

Logs go to console and WandB (`trainer.project_name=verl_sft_gdpo_phyre`). Set `export WANDB_MODE=online` to sync.

---

## Troubleshooting

| Symptom | Check |
|---------|--------|
| `SFT checkpoint not found` | `SFT_MODEL_PATH` in the script points to a directory with `config.json` |
| `Training/validation parquet files not found` | `TRAIN_FILE` / `VAL_FILE` in the script point to existing `.parquet` files |
| Reward scores always `-1.0` | DQN ckpt missing under `agent_checkpoints/ball_within_template/` or `agent_checkpoints/my_template_based_split/` |
| `Phyre is not installed` | `pip install -e /path/to/VAORA/phyre` in the active env |

---

## Related docs

- Hugging Face checkpoints overview: [tool-games/README.md](../tool-games/README.md)
- Dataset & inference layout: [Batch_Inference/README.md](../Batch_Inference/README.md)
- DQN training from scratch: [phyre/README.md](../phyre/README.md)
