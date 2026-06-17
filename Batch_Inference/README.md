# Batch Inference

Unified batch inference for PHYRE, MiniGrid, and CRAFT using API-based VLMs or local checkpoints.

## Data and checkpoints source

- Dataset repo: [`vaora-proj/vaora-dataset`](https://huggingface.co/datasets/vaora-proj/vaora-dataset)
  - Use JSON files under its `test_data/` directory.
- Checkpoint repo: [`vaora-proj/vaora-checkpoints`](https://huggingface.co/vaora-proj/vaora-checkpoints)
  - Use model directories under `VAORA_VLM/` and `SFT_VLM/`.

`run_vlm_inference.sh` no longer hardcodes dataset paths. You must provide `DATASET_PATH`.

## Quick start

```bash
cd /path/to/VAORA/Batch_Inference
chmod +x run_vlm_inference.sh

# set where you cloned/downloaded Hugging Face assets
export VAORA_DATASET_ROOT=/path/to/vaora-dataset
export VAORA_CKPT_ROOT=/path/to/vaora-checkpoints
```

## Required inputs

- `BACKEND`: `chatgpt`, `claude`, `gemini`, `internvl`, `qwen3`, `craft`
- `DATASET_PATH`: JSON file in `vaora-proj/vaora-dataset/test_data/...`

Pass `BACKEND` as env var or first positional argument:

```bash
BACKEND=claude DATASET_PATH=/path/to/file.json bash run_vlm_inference.sh
DATASET_PATH=/path/to/file.json bash run_vlm_inference.sh gemini
```

## Environment and keys

- `ENV_TYPE=phyre|minigrid` for `chatgpt|claude|gemini|internvl|qwen3`
- `BACKEND=craft` uses CRAFT mode directly
- API keys:
  - `OPENAI_API_KEY` for ChatGPT
  - `ANTHROPIC_API_KEY` for Claude
  - `GEMINI_API_KEY` for Gemini

## Checkpoint usage

For local backends (`qwen3`, `internvl`, `craft`), point `LOCAL_LOAD_PATH` to a directory under:

- `${VAORA_CKPT_ROOT}/VAORA_VLM/...`
- `${VAORA_CKPT_ROOT}/SFT_VLM/...`

## Examples

### ChatGPT on PHYRE (HF dataset)

```bash
export OPENAI_API_KEY=...
BACKEND=chatgpt \
  DATASET_PATH="${VAORA_DATASET_ROOT}/test_data/qwen3_dataset_my_cross_fold_1.json" \
  bash run_vlm_inference.sh
```

### Claude on MiniGrid (HF dataset)

```bash
export ANTHROPIC_API_KEY=...
BACKEND=claude \
  ENV_TYPE=minigrid \
  DATASET_PATH="${VAORA_DATASET_ROOT}/test_data/qwen3_minigrid_dataset.json" \
  bash run_vlm_inference.sh
```

### Qwen3 checkpoint from `SFT_VLM`

```bash
conda activate vaora
BACKEND=qwen3 \
  DATASET_PATH="${VAORA_DATASET_ROOT}/test_data/qwen3_dataset_my_cross_fold_3.json" \
  LOCAL_LOAD_PATH="${VAORA_CKPT_ROOT}/SFT_VLM/qwen3_vl_8b/your_run" \
  LOG_MODEL_NAME=your_run \
  bash run_vlm_inference.sh
```

### CRAFT checkpoint from `VAORA_VLM`

```bash
conda activate vaora
BACKEND=craft \
  DATASET_PATH="${VAORA_DATASET_ROOT}/test_data/craft_dataset_1000_merged_infer.json" \
  LOCAL_LOAD_PATH="${VAORA_CKPT_ROOT}/VAORA_VLM/your_run" \
  LOG_MODEL_NAME=your_run \
  bash run_vlm_inference.sh
```

## Optional flags

Useful optional vars include `MODEL_NAME`, `LOG_MODEL_NAME`, `BATCH_SIZE`, `TEMPERATURE`, `TOP_P`, `MAX_NEW_TOKENS`, `NUM_WORKERS`, `OUTPUT_ROOT`, and `PYTHON_BIN`.

Extra CLI flags are forwarded to the Python agent:

```bash
BACKEND=qwen3 DATASET_PATH=/path/to/file.json bash run_vlm_inference.sh --eval_type test --no-save_images
```

## Outputs

Logs and artifacts are written under `batch_inference_output/<backend>/` by default. PHYRE/MiniGrid explorer outputs are written to `explorer_outputs/` (override with `OUTPUT_ROOT`).
