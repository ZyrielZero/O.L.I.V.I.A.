# O.L.I.V.I.A. Model Engineering Tools

Scripts for model merging, fine-tuning, and GGUF conversion.

## Directory Structure

```
tools/model_engineering/
├── finetune_olivia.py      # QLoRA fine-tuning script
├── merge_lora.py           # Merge LoRA adapter + export GGUF
├── convert_to_gguf.py      # Manual GGUF conversion (Windows)
├── verify_merge.py         # Verify merged model quality
├── configs/
│   ├── olivia_della_merge.yaml      # DELLA merge config v1
│   └── olivia_della_merge_v2.yaml   # DELLA merge config v2
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

Run QLoRA fine-tuning on the merged base model:

```bash
python tools/model_engineering/finetune_olivia.py
```

**Requirements:**
- Base model at `models/checkpoints/olivia-merged-v2/`
- Training data at `data/training/olivia_training_complete.jsonl`
- ~12GB VRAM (RTX 4080 SUPER recommended)

**Output:** LoRA adapter at `models/adapters/olivia-lora-output/`

### 2. Merge LoRA and Export GGUF

Merge the fine-tuned LoRA adapter back into the base model:

```bash
python tools/model_engineering/merge_lora.py
```

**Output:**
- Merged checkpoint at `models/checkpoints/olivia-finetuned/`
- GGUF file at `models/gguf/olivia-finetuned-q4_k_m.gguf`
- Modelfile at `models/ollama/Modelfile.olivia-finetuned`

### 3. Create Ollama Model

```bash
cd models/ollama
ollama create olivia-finetuned -f Modelfile.olivia-finetuned
```

### 4. Test

```bash
ollama run olivia-finetuned "Hey"
```

## Alternative: Manual GGUF Conversion

If `merge_lora.py` fails to create GGUF, use the manual script:

```bash
python tools/model_engineering/convert_to_gguf.py
```

This downloads llama.cpp binaries and converts manually.

## Verify Merge Quality

Check tokenizer and generation:

```bash
python tools/model_engineering/verify_merge.py
```

## Model Locations

| Type | Location |
|------|----------|
| Base checkpoints | `models/checkpoints/` |
| LoRA adapters | `models/adapters/` |
| GGUF files | `models/gguf/` |
| Ollama Modelfiles | `models/ollama/` |
| Source models (for merge) | `models/source/` |

## Training Data Format

Training data uses ShareGPT format (JSONL):

```json
{"conversations": [
  {"from": "human", "value": "Hello!"},
  {"from": "gpt", "value": "Hey there. What's up?"}
]}
```

## DELLA Merge (Advanced)

To re-run the DELLA model merge:

1. Install mergekit: `pip install mergekit`
2. Download source models to `models/source/`
3. Run: `mergekit-yaml configs/olivia_della_merge_v2.yaml models/checkpoints/olivia-merged-v2`

Source models needed:
- meta-llama/Llama-3.1-8B
- Sao10K/Stheno-v3.4-Llama-3.1-8B
- NousResearch/Hermes-3-Llama-3.1-8B
- vicgalle/Humanish-LLama3.1-8B-Roleplay
- cognitivecomputations/dolphin-2.9.4-llama3.1-8b
- nvidia/Llama-3.1-Nemotron-Nano-8B-v1
