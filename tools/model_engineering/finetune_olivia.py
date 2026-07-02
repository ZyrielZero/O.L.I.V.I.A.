"""
O.L.I.V.I.A. QLoRA Fine-Tuning Script
Using Unsloth for efficient training on RTX 4080 SUPER

Repository: https://github.com/ZyrielZero/project-olivia

Run from: project root or tools/model_engineering folder
Updated: January 2026 - Reorganized project structure
"""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
from multiprocessing import freeze_support
from pathlib import Path

import torch

# ============================================
# PATH CONFIGURATION - Auto-detect project root
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # tools/model_engineering -> project root

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# ============================================
# CONFIGURATION - ADJUSTED FOR YOUR SETUP
# ============================================

# Your local DELLA-merged model
BASE_MODEL = str(MODELS_DIR / "checkpoints" / "olivia-merged-v2")

MAX_SEQ_LENGTH = 2048  # Context length
LOAD_IN_4BIT = True    # QLoRA 4-bit quantization

# LoRA Settings (optimized for personality fine-tuning)
LORA_R = 64            # LoRA rank - higher = more capacity, more VRAM
LORA_ALPHA = 128       # Usually 2x rank
LORA_DROPOUT = 0       # Set to 0 for Unsloth fast patching

# Training Settings
OUTPUT_DIR = str(MODELS_DIR / "adapters" / "olivia-sft-checkpoint-v2")
EPOCHS = 2             # 2 epochs to prevent overfitting (will do DPO after)
BATCH_SIZE = 2         # Per-device batch size (reduce to 1 if OOM)
GRADIENT_ACCUMULATION = 8  # Effective batch = 2 * 8 = 16
LEARNING_RATE = 2e-4   # Standard for LoRA
WARMUP_RATIO = 0.1     # 10% warmup for better convergence
SAVE_STEPS = 100       # Checkpoint frequency (must align with eval_steps)
LOGGING_STEPS = 10     # How often to log

# Data - use combined and validated training data
TRAIN_FILE = str(DATA_DIR / "training" / "olivia_sft_train.jsonl")
VAL_FILE = str(DATA_DIR / "training" / "olivia_sft_val.jsonl")


def main():
    """Main training function - must be wrapped for Windows multiprocessing"""

    # Import heavy libraries inside main() for Windows compatibility
    import json

    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    # ============================================
    # VERIFY SETUP
    # ============================================

    print("=" * 60)
    print("O.L.I.V.I.A. Fine-Tuning")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")

    # Check model exists
    if not os.path.exists(BASE_MODEL):
        print(f"ERROR: Model not found at {BASE_MODEL}")
        print("Make sure you have the merged model in models/checkpoints/olivia-merged-v2")
        return

    # Check training data exists
    if not os.path.exists(TRAIN_FILE):
        print(f"ERROR: Training data not found at {TRAIN_FILE}")
        print("\nPlease run tools/combine_training_data.py first!")
        return

    # Check validation data exists
    has_validation = os.path.exists(VAL_FILE)
    if not has_validation:
        print(f"WARNING: Validation data not found at {VAL_FILE}")
        print("Training will proceed without validation.")

    print(f"Model found: {BASE_MODEL}")
    print(f"Training data found: {TRAIN_FILE}")
    if has_validation:
        print(f"Validation data found: {VAL_FILE}")

    # ============================================
    # LOAD MODEL
    # ============================================

    print(f"\nLoading base model: {BASE_MODEL}")
    print("(This may take a minute for an 8B model...)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (float16/bfloat16)
        load_in_4bit=LOAD_IN_4BIT,
    )

    print("Model loaded")

    # Ensure padding token is set properly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ============================================
    # CONFIGURE LoRA
    # ============================================

    print(f"\nConfiguring LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory optimization
        random_state=42,
    )

    print("LoRA configured")

    # ============================================
    # SETUP CHAT TEMPLATE
    # ============================================

    # Use ChatML format (what your merged model likely uses)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )

    # Re-ensure padding after chat template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============================================
    # LOAD AND PROCESS DATASET - NO MULTIPROCESSING
    # Pure Python loops to avoid Windows issues
    # ============================================

    def load_and_process_data(filepath, name="data"):
        """Load and process a JSONL file."""
        print(f"\nLoading {name}: {filepath}")

        raw_data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                raw_data.append(json.loads(line.strip()))

        print(f"Loaded {len(raw_data)} examples")
        print(f"Processing {name} (pure Python, no multiprocessing)...")

        processed_data = []
        for i, example in enumerate(raw_data):
            # Convert ShareGPT format to messages
            convo = example["conversations"]
            messages = []
            for turn in convo:
                role = "user" if turn["from"] == "human" else "assistant"
                messages.append({"role": role, "content": turn["value"]})

            # Apply chat template to get text
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Tokenize
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                padding=False,
                return_tensors=None,
            )

            processed_data.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            })

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(raw_data)} examples...")

        print(f"All {len(processed_data)} {name} examples processed")
        return processed_data

    # Load training data
    processed_data = load_and_process_data(TRAIN_FILE, "training data")

    # Load validation data if available
    processed_val_data = None
    if has_validation:
        processed_val_data = load_and_process_data(VAL_FILE, "validation data")

    # Preview a sample
    print("\n--- Sample (first 50 tokens decoded) ---")
    sample_text = tokenizer.decode(processed_data[0]["input_ids"][:50])
    print(sample_text)
    print("...\n---")
    print(f"Sample input_ids length: {len(processed_data[0]['input_ids'])}")

    # Create Dataset from processed data (no .map() needed!)
    tokenized_dataset = Dataset.from_list(processed_data)
    print(f"Training dataset created: {len(tokenized_dataset)} examples")

    # Create validation dataset if available
    tokenized_val_dataset = None
    if processed_val_data:
        tokenized_val_dataset = Dataset.from_list(processed_val_data)
        print(f"Validation dataset created: {len(tokenized_val_dataset)} examples")

    # ============================================
    # DATA COLLATOR - handles labels and padding
    # ============================================

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # ============================================
    # TRAINING CONFIG (Modern SFTConfig API)
    # ============================================

    # Configure evaluation if validation data available
    eval_config = {}
    if tokenized_val_dataset is not None:
        eval_config = {
            "eval_strategy": "steps",
            "eval_steps": 100,
            "per_device_eval_batch_size": BATCH_SIZE,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,

        # Training parameters
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,

        # Logging and saving
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,

        # Precision
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        # Optimizer
        optim="adamw_8bit",

        # Dataset handling
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,  # CRITICAL: Disable padding-free mode that causes NaN!

        # Misc
        seed=42,
        report_to="none",

        # Evaluation settings (if validation data available)
        **eval_config,
    )

    # ============================================
    # TRAINER
    # ============================================

    print("Initializing trainer...")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_val_dataset,  # Add validation dataset
        data_collator=data_collator,
        args=sft_config,
    )

    # ============================================
    # PRE-TRAINING INFO
    # ============================================

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    max_memory = round(gpu_stats.total_memory / 1024**3, 2)

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Base Model:          {BASE_MODEL}")
    print(f"LoRA Rank:           {LORA_R}")
    print(f"LoRA Alpha:          {LORA_ALPHA}")
    print(f"Epochs:              {EPOCHS}")
    print(f"Batch Size:          {BATCH_SIZE}")
    print(f"Gradient Accum:      {GRADIENT_ACCUMULATION}")
    print(f"Effective Batch:     {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Learning Rate:       {LEARNING_RATE}")
    print(f"Dataset Size:        {len(tokenized_dataset)} examples")
    print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
    print("-" * 60)
    print(f"GPU:                 {gpu_stats.name}")
    print(f"VRAM Before:         {start_gpu_memory}GB / {max_memory}GB")
    print("=" * 60)

    # ============================================
    # TRAIN
    # ============================================

    print("\nStarting training...\n")

    trainer_stats = trainer.train()

    # ============================================
    # POST-TRAINING STATS
    # ============================================

    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    training_time = trainer_stats.metrics['train_runtime']

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training Time:       {training_time:.2f}s ({training_time/60:.1f} minutes)")
    print(f"Peak VRAM:           {used_memory}GB / {max_memory}GB")
    print(f"Final Loss:          {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")
    print("=" * 60)

    # ============================================
    # SAVE MODEL
    # ============================================

    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("SFT TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nLoRA adapter saved to: {OUTPUT_DIR}")
    print("\nNext steps (two-stage training):")
    print("  1. Run DPO training: python tools/model_engineering/train_dpo.py")
    print("  2. Merge and export: python tools/model_engineering/merge_lora.py")
    print("  3. Create Ollama model: ollama create olivia-finetuned -f models/ollama/Modelfile.olivia-finetuned")
    print("  4. Test: ollama run olivia-finetuned \"Hey\"")


# ============================================
# WINDOWS MULTIPROCESSING GUARD
# ============================================

if __name__ == "__main__":
    freeze_support()  # Required for Windows
    main()
