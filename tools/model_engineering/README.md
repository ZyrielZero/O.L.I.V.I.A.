# O.L.I.V.I.A. Model Engineering Tools

Scripts for fine-tuning, preference training, GGUF conversion, and model merging.

## Directory Structure

```
tools/model_engineering/
├── finetune_olivia.py      # QLoRA fine-tuning script
├── train_dpo.py            # DPO preference training
├── merge_lora.py           # Merge LoRA adapter + export GGUF
├── convert_hf_to_gguf.py   # Vendored from llama.cpp (excluded from ruff)
├── merge/                  # Model-merge experiment kit (see merge/README.md)
│   ├── olivia_dare_ties.yaml
│   ├── olivia_slerp_control.yaml
│   ├── run_merge.sh
│   ├── verify_merge.py     # Post-merge tokenizer + load smoke test
│   └── results/            # Committed eval result JSONs
└── README.md
```

## Prerequisites

Install model engineering dependencies:

```bash
pip install -e ".[model-engineering]"
```

Or install manually:

```bash
pip install unsloth transformers datasets trl gguf sentencepiece
```

## Workflow

### 1. Fine-tune the Model

Run QLoRA fine-tuning on the base model:

```bash
python tools/model_engineering/finetune_olivia.py
```

**Requirements:**
- Base model at `models/checkpoints/olivia-merged-v2/`
- Training data at `data/training/olivia_training_complete.jsonl`
- ~12GB VRAM (RTX 4080 SUPER recommended)

**Output:** LoRA adapter at `models/adapters/olivia-lora-output/`

### 2. (Optional) DPO Preference Training

```bash
python tools/model_engineering/train_dpo.py
```

### 3. Merge LoRA and Export GGUF

Merge the fine-tuned LoRA adapter back into the base model:

```bash
python tools/model_engineering/merge_lora.py
```

**Output:**
- Merged checkpoint at `models/checkpoints/olivia-finetuned/`
- GGUF file at `models/gguf/olivia-finetuned-q4_k_m.gguf`
- Modelfile at `models/ollama/Modelfile.olivia-finetuned`

GGUF conversion uses `convert_hf_to_gguf.py`, vendored from
[llama.cpp](https://github.com/ggerganov/llama.cpp) (not our code to lint —
excluded in `pyproject.toml`).

### 4. Create Ollama Model

```bash
cd models/ollama
ollama create olivia-finetuned -f Modelfile.olivia-finetuned
```

### 5. Test

```bash
ollama run olivia-finetuned "Hey"
```

## Model Merging

The merge experiment (DARE-TIES main arm + SLERP control) lives in
[`merge/`](merge/README.md) with its own configs, runner, and smoke test:

```bash
cd tools/model_engineering/merge
./run_merge.sh olivia_dare_ties.yaml        # WSL2, CPU-only
python verify_merge.py out/olivia_dare_ties # tokenizer + load check
```

## Model Locations

| Type | Location |
|------|----------|
| Base checkpoints | `models/checkpoints/` |
| LoRA adapters | `models/adapters/` |
| GGUF files | `models/gguf/` |
| Ollama Modelfiles | `models/ollama/` |
| Source models (for merge) | `models/source/` |

All of `models/` is gitignored — weights never enter the repo.

## Training Data Format

Training data uses ShareGPT format (JSONL):

```json
{"conversations": [
  {"from": "human", "value": "Hello!"},
  {"from": "gpt", "value": "Hey there. What's up?"}
]}
```
