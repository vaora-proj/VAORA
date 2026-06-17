# Batch Inference

Unified batch inference for **PHYRE** and **CRAFT** using API-based VLMs (ChatGPT, Claude, Gemini) or local checkpoints (Qwen3-VL, InternVL, CRAFT).

`run_vlm_inference.sh` is the single entry point: set `BACKEND` and `DATASET_PATH`, optionally point `LOCAL_LOAD_PATH` at a Hugging Face checkpoint, and run.

## Data and checkpoints (Huggingface)

Download the assets below, then set `VAORA_DATASET_ROOT` and `VAORA_CKPT_ROOT` to your local clone paths.

```bash
huggingface-cli download vaora-proj/vaora-dataset --repo-type dataset --local-dir /path/to/vaora-dataset
huggingface-cli download vaora-proj/vaora-checkpoints --local-dir /path/to/vaora-checkpoints
```

### Dataset — [`vaora-proj/vaora-dataset`](https://huggingface.co/datasets/vaora-proj/vaora-dataset)

Inference JSON files live under `test_data/`. Training data is under `train_data/` (not required for batch inference).

```
vaora-dataset/
├── test_data/
│   ├── phyre/
│   │   ├── cross_template_testing_set_1.json
│   │   ├── cross_template_testing_set_2.json
│   │   ├── cross_template_testing_set_3.json
│   │   └── within_template.json
│   └── craft/
│       ├── craft_dataset_1000_merged_infer.json
│       └── craft_frames_filtered/
│           ├── sid_1/          # frame PNGs per scene
│           ├── sid_2/
│           └── ...             # sid_3 … sid_20
└── train_data/
    ├── SFT_DATA/
    │   ├── sft_dataset_cross_template_testing_set_1.json
    │   ├── sft_dataset_cross_template_testing_set_2.json
    │   ├── sft_dataset_cross_template_testing_set_3.json
    │   └── sft_dataset_within_template.json
    └── VAORA_DATA/
        ├── vaora_dataset_cross_template_testing_set_1/
        │   ├── train.parquet
        │   ├── test.parquet
        │   ├── train_sample.json
        │   └── test_sample.json
        ├── vaora_dataset_cross_template_testing_set_2/
        ├── vaora_dataset_cross_template_testing_set_3/
        └── vaora_dataset_within_template/
```

### Checkpoints — [`vaora-proj/vaora-checkpoints`](https://huggingface.co/vaora-proj/vaora-checkpoints)

Use directories under `SFT_VLM/` for supervised fine-tuned Qwen3-VL weights, and `VAORA_VLM/` for VAORA-trained weights (including CRAFT).

```
vaora-checkpoints/
├── SFT_VLM/
│   ├── sft_cross_template_testing_set_1/   # Hugging Face model dir
│   ├── sft_cross_template_testing_set_2/
│   ├── sft_cross_template_testing_set_3/
│   └── sft_within_template/
├── VAORA_VLM/
│   ├── vaora_cross_template_testing_set_1/
│   ├── vaora_cross_template_testing_set_2/
│   ├── vaora_cross_template_testing_set_3/
│   └── within_template/
└── DQN-expert/                             # PHYRE DQN baselines (not VLM checkpoints)
    ├── dqn_cross_template_testing_set_1.ckpt
    ├── dqn_cross_template_testing_set_2.ckpt
    ├── dqn_cross_template_testing_set_3.ckpt
    ├── dqn_within_template.ckpt
    └── results_*.json
```

Each `SFT_VLM/*` and `VAORA_VLM/*` directory is a standard Hugging Face model folder (`config.json`, `model-*.safetensors`, tokenizer files, etc.). Pass the directory path to `LOCAL_LOAD_PATH`.

## Run inference

### Setup

```bash
cd /path/to/VAORA/Batch_Inference
chmod +x run_vlm_inference.sh

export VAORA_DATASET_ROOT=/path/to/vaora-dataset
export VAORA_CKPT_ROOT=/path/to/vaora-checkpoints
```

### Required inputs

| Variable | Description |
|----------|-------------|
| `BACKEND` | `chatgpt`, `claude`, `gemini`, `internvl`, `qwen3`, or `craft` |
| `DATASET_PATH` | JSON under `test_data/phyre/` or `test_data/craft/` |

Pass `BACKEND` as an environment variable or as the first positional argument:

```bash
BACKEND=claude DATASET_PATH=/path/to/file.json bash run_vlm_inference.sh
DATASET_PATH=/path/to/file.json bash run_vlm_inference.sh gemini
```

### API keys (API backends)

| Backend | Environment variable |
|---------|---------------------|
| `chatgpt` | `OPENAI_API_KEY` |
| `claude` | `ANTHROPIC_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |

### Local checkpoints

For `qwen3`, `internvl`, and `craft`, set `LOCAL_LOAD_PATH` to a model directory under `SFT_VLM/` or `VAORA_VLM/`:

```bash
LOCAL_LOAD_PATH="${VAORA_CKPT_ROOT}/SFT_VLM/sft_cross_template_testing_set_1"
LOCAL_LOAD_PATH="${VAORA_CKPT_ROOT}/VAORA_VLM/within_template"
```

### Examples

**ChatGPT on PHYRE**

```bash
export OPENAI_API_KEY=...
BACKEND=chatgpt \
  DATASET_PATH="${VAORA_DATASET_ROOT}/test_data/phyre/cross_template_testing_set_1.json" \
  bash run_vlm_inference.sh
```

**Claude on PHYRE (within-template split)**

```bash
export ANTHROPIC_API_KEY=...
BACKEND=claude \
  DATASET_PATH="${VAORA_DATASET_ROOT}/test_data/phyre/within_template.json" \
  bash run_vlm_inference.sh
```

**Qwen3-VL with an SFT checkpoint**

```bash
BACKEND=qwen3 \
  DATASET_PATH="${VAORA_DATASET_ROOT}/test_data/phyre/cross_template_testing_set_3.json" \
  LOCAL_LOAD_PATH="${VAORA_CKPT_ROOT}/SFT_VLM/sft_cross_template_testing_set_3" \
  LOG_MODEL_NAME=sft_cross_template_testing_set_3 \
  bash run_vlm_inference.sh
```

**CRAFT with a VAORA checkpoint**

```bash
BACKEND=craft \
  DATASET_PATH="${VAORA_DATASET_ROOT}/test_data/craft/craft_dataset_1000_merged_infer.json" \
  LOCAL_LOAD_PATH="${VAORA_CKPT_ROOT}/VAORA_VLM/within_template" \
  LOG_MODEL_NAME=within_template \
  bash run_vlm_inference.sh
```

### Optional flags

Useful environment variables: `MODEL_NAME`, `LOG_MODEL_NAME`, `BATCH_SIZE`, `TEMPERATURE`, `TOP_P`, `MAX_NEW_TOKENS`, `NUM_WORKERS`, `OUTPUT_ROOT`, `PYTHON_BIN`.

Extra CLI flags are forwarded to the Python agent:

```bash
BACKEND=qwen3 DATASET_PATH=/path/to/file.json bash run_vlm_inference.sh --eval_type test --no-save_images
```

### Outputs

Logs and artifacts are written under `batch_inference_output/<backend>/` by default. PHYRE explorer outputs go to `explorer_outputs/` (override with `OUTPUT_ROOT`).
