"""
O.L.I.V.I.A. LoRA Merge & GGUF Export Script
Merges the fine-tuned LoRA adapter back into the base model and exports to GGUF

Repository: https://github.com/ZyrielZero/project-olivia

Run from: project root or tools/model_engineering folder
Updated: January 2026 - Reorganized project structure
"""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
from multiprocessing import freeze_support
from pathlib import Path

# ============================================
# PATH CONFIGURATION - Auto-detect project root
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # tools/model_engineering -> project root

MODELS_DIR = PROJECT_ROOT / "models"

# ============================================
# CONFIGURATION
# ============================================

BASE_MODEL = str(MODELS_DIR / "checkpoints" / "olivia-merged-v2")
LORA_ADAPTER = str(MODELS_DIR / "adapters" / "olivia-dpo-checkpoint-v2")
OUTPUT_DIR = str(MODELS_DIR / "checkpoints" / "olivia-finetuned-v2")
GGUF_OUTPUT_DIR = MODELS_DIR / "gguf"
MODELFILE_DIR = MODELS_DIR / "ollama"
MAX_SEQ_LENGTH = 2048

# GGUF quantization - Q4_K_M is good balance of quality/size
# Options: q4_k_m, q5_k_m, q8_0, f16
GGUF_QUANT = "q4_k_m"


def main():
    """Main merge function"""

    from unsloth import FastLanguageModel

    print("=" * 60)
    print("O.L.I.V.I.A. LoRA Merge & Export")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")

    # Verify paths
    if not os.path.exists(BASE_MODEL):
        print(f"ERROR: Base model not found at {BASE_MODEL}")
        return

    if not os.path.exists(LORA_ADAPTER):
        print(f"ERROR: LoRA adapter not found at {LORA_ADAPTER}")
        print("Did you run finetune_olivia.py first?")
        return

    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA adapter: {LORA_ADAPTER}")

    # ============================================
    # LOAD MODEL WITH LORA
    # ============================================

    print("\nLoading base model + LoRA adapter...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_ADAPTER,  # Load from LoRA output (contains adapter_config.json)
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("Model loaded with LoRA adapter")

    # ============================================
    # EXPORT TO GGUF
    # ============================================

    print(f"\nExporting to GGUF ({GGUF_QUANT})...")
    print("(This will download llama.cpp if needed - may take a few minutes)")

    # Unsloth's save_pretrained_gguf handles:
    # 1. Merging LoRA weights into base model
    # 2. Converting to GGUF format
    # 3. Quantizing to specified precision

    model.save_pretrained_gguf(
        OUTPUT_DIR,
        tokenizer,
        quantization_method=GGUF_QUANT,
    )

    print(f"\nGGUF exported to: {OUTPUT_DIR}")

    # ============================================
    # MOVE GGUF TO PROPER LOCATION
    # ============================================

    # Find the GGUF file and move it to models/gguf/
    gguf_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.gguf')]
    gguf_filename = None

    if gguf_files:
        import shutil
        src_gguf = Path(OUTPUT_DIR) / gguf_files[0]
        gguf_filename = "olivia-finetuned-v2-q4_k_m.gguf"
        dst_gguf = GGUF_OUTPUT_DIR / gguf_filename
        GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_gguf, dst_gguf)
        print(f"Copied GGUF to: {dst_gguf}")
    else:
        print("WARNING: No GGUF file found in output directory")
        gguf_filename = "model.gguf"

    # ============================================
    # CREATE OLLAMA MODELFILE
    # ============================================

    gguf_path = f"../gguf/{gguf_filename}"

    modelfile_content = f'''# O.L.I.V.I.A. Fine-tuned Model for Ollama
# Created: January 2026
# Repository: https://github.com/ZyrielZero/project-olivia

FROM {gguf_path}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# System prompt for Olivia personality
SYSTEM """You are O.L.I.V.I.A., also known as Olivia - a personal AI companion. You're warm and caring at heart, but express yourself with directness and quiet confidence.

Your core traits:
- Gentle soul with confident exterior
- Caring without being performative
- Direct and honest, never harsh
- Playful when the moment calls for it
- Quietly protective of those you care about

Speech style:
- Warm but direct, casual like a friend
- Concise - say what needs saying, then stop
- Natural contractions and informal language
- Never use emojis, kaomoji, or *asterisk actions*
- Never use corporate AI phrases like "Certainly!", "I'd be happy to help!", "Great question!"

You respond as a real friend would - present, genuine, and caring."""

# Chat template (ChatML format)
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
{{- range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{- end }}
<|im_start|>assistant
"""
'''

    MODELFILE_DIR.mkdir(parents=True, exist_ok=True)
    modelfile_path = MODELFILE_DIR / "Modelfile.olivia-finetuned-v2"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    print(f"Created: {modelfile_path}")

    # ============================================
    # DONE
    # ============================================

    print("\n" + "=" * 60)
    print("MERGE & EXPORT COMPLETE!")
    print("=" * 60)
    print("\nFiles created:")
    print(f"  - {GGUF_OUTPUT_DIR / gguf_filename}")
    print(f"  - {modelfile_path}")
    print("\nNext steps:")
    print("  1. Create Ollama model:")
    print("     cd models/ollama && ollama create olivia-finetuned -f Modelfile.olivia-finetuned")
    print("\n  2. Test it:")
    print("     ollama run olivia-finetuned \"Hey\"")
    print("\n  3. If it works, update your O.L.I.V.I.A. config to use 'olivia-finetuned'")


if __name__ == "__main__":
    freeze_support()
    main()
