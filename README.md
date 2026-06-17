# VAORA

VAORA release repository for:

- `verl_new/`: training code (based on `verl`)
- `Batch_Inference/`: inference and dataset build code
- `phyre/`: physics benchmark used by our pipeline

## Installation

This project needs three parts installed:

1. `verl` (training framework)
2. `phyre` (physics environment)
3. remaining Python packages used by `Batch_Inference`

### 1) Install `verl` (`VAORA/verl_new`)

Follow upstream `verl` installation style (conda env + install script + editable install), then run our training scripts from `VAORA/verl_new`.

```bash
cd /path/to/VAORA

# recommended environment setup
conda create -n vaora python=3.10.13 -y
conda activate vaora

# install core dependencies (FSDP-only path)
cd verl_new
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# install verl itself
pip install --no-deps -e .
```

Notes:

- If you need Megatron support, use:
  `bash scripts/install_vllm_sglang_mcore.sh`
- Upstream installation reference:
  [verl install docs](https://verl.readthedocs.io/en/latest/start/install.html)
  and [verl repo](https://github.com/verl-project/verl).

### 2) Install `phyre` (`VAORA/phyre`)

We keep `phyre` as a local folder in this repo. The original PHYRE README recommends Python 3.6 for the pip package, but for this integrated VAORA stack, install from local source inside your active environment:

```bash
cd /path/to/VAORA/phyre
pip install -e .
```

Quick validation:

```bash
python -m phyre.server
# then open http://localhost:30303
```

Reference:
[PHYRE project](https://github.com/facebookresearch/phyre)

### 3) Install remaining packages for `Batch_Inference`

`Batch_Inference` uses additional libraries on top of `verl` + `phyre`:

- `numpy`, `pillow`, `tqdm`
- `pandas`, `matplotlib`
- `transformers`, `torch`, `qwen-vl-utils`
- optional API clients:
  - `openai` (for ChatGPT runner)
  - `anthropic` (for Claude runner)
  - `google-generativeai` (for Gemini runner)

Install command:

```bash
cd /path/to/VAORA
pip install numpy pillow tqdm pandas matplotlib transformers torch qwen-vl-utils openai anthropic google-generativeai
```

## Repository Layout

- `verl_new/`: training framework and project-specific training scripts
- `Batch_Inference/`: inference pipelines
- `phyre/`: local PHYRE source used by reward/inference pipeline
- `tool-games/`: tool-games environment and VLM inference (see `tool-games/README.md`)


