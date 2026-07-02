"""
O.L.I.V.I.A. DPO (Direct Preference Optimization) Training Script
Stage 2 of two-stage SFT->DPO personality training pipeline.

Uses Unsloth for efficient training on RTX 4080 SUPER.
Per Unsloth docs: with PEFT/LoRA, pass ref_model=None for memory efficiency.

WINDOWS COMPATIBILITY:
- Pre-tokenizes dataset completely in main process
- Subclasses DPOTrainer to skip internal tokenization
- Uses standard TRL DPOTrainer (NOT Unsloth's patched version)
- Disables all dataset multiprocessing

Repository: https://github.com/ZyrielZero/project-olivia
"""

import os

# Environment setup - MUST be before any other imports
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DATASETS_DISABLE_PARALLEL"] = "1"
os.environ["HF_DATASETS_DISABLE_CACHING"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from multiprocessing import freeze_support
from pathlib import Path

import torch

# ============================================
# PATH CONFIGURATION
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# ============================================
# CONFIGURATION
# ============================================

# Input: SFT checkpoint (output from finetune_olivia.py)
SFT_CHECKPOINT = str(MODELS_DIR / "adapters" / "olivia-sft-checkpoint-v2")

# If SFT checkpoint not found, use base model
BASE_MODEL = str(MODELS_DIR / "checkpoints" / "olivia-merged-v2")

# Output: DPO-trained checkpoint
OUTPUT_DIR = str(MODELS_DIR / "adapters" / "olivia-dpo-checkpoint-v2")

# Model settings
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA settings (should match SFT)
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0

# DPO-specific settings
BETA = 0.5           # KL penalty weight - increased from 0.1 to prevent mode collapse
LOSS_TYPE = "sigmoid"  # Standard DPO loss

# Training settings (lower than SFT)
EPOCHS = 1           # DPO typically needs fewer epochs
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4  # Effective batch = 8
LEARNING_RATE = 1e-5  # Reduced from 5e-5 for stability
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
SAVE_STEPS = 100

# DPO length settings
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512

# Data
DPO_FILE = str(DATA_DIR / "training" / "olivia_dpo_pairs.jsonl")


def pre_tokenize_dpo_dataset(
    raw_data: list,
    tokenizer,
    max_prompt_length: int,
    max_completion_length: int,
) -> list:
    """
    Pre-tokenize DPO dataset in the main process to avoid Windows multiprocessing issues.

    This replicates the logic from TRL's DPOTrainer.tokenize_row but runs in a simple loop.
    Returns data with the exact columns DPOTrainer expects after its internal tokenization.

    Args:
        raw_data: List of dicts with 'prompt', 'chosen', 'rejected' keys (formatted text)
        tokenizer: The tokenizer to use
        max_prompt_length: Maximum length for prompt tokens
        max_completion_length: Maximum length for completion tokens

    Returns:
        List of dicts with columns: prompt_input_ids, chosen_input_ids, rejected_input_ids
    """
    print(f"Pre-tokenizing {len(raw_data)} DPO pairs in main process...")

    processed = []

    for i, item in enumerate(raw_data):
        prompt_text = item["prompt"]
        chosen_text = item["chosen"]
        rejected_text = item["rejected"]

        # Tokenize prompt only (for the prompt portion)
        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors=None,
        )

        # Tokenize full chosen conversation
        chosen_tokens = tokenizer(
            chosen_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length + max_completion_length,
            return_tensors=None,
        )

        # Tokenize full rejected conversation
        rejected_tokens = tokenizer(
            rejected_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length + max_completion_length,
            return_tensors=None,
        )

        # DPOTrainer expects these exact column names
        processed.append({
            "prompt_input_ids": prompt_tokens["input_ids"],
            "prompt_attention_mask": prompt_tokens["attention_mask"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        })

        if (i + 1) % 200 == 0:
            print(f"  Tokenized {i + 1}/{len(raw_data)} pairs...")

    print(f"Pre-tokenization complete: {len(processed)} pairs")
    return processed


def create_pretokenized_dpo_trainer(
    model,
    tokenizer,
    train_dataset,
    dpo_config,
):
    """
    Create a DPOTrainer that skips internal tokenization.

    The trick is to subclass DPOTrainer and override _prepare_dataset to
    return the dataset unchanged if it already has the tokenized columns.
    This completely avoids any dataset.map() calls that would trigger
    multiprocessing pickle errors on Windows.
    """

    from trl import DPOTrainer

    class PreTokenizedDPOTrainer(DPOTrainer):
        """DPOTrainer subclass that skips tokenization for pre-tokenized datasets."""

        def _prepare_dataset(self, dataset, *args, **kwargs):
            """
            Override to skip tokenization if dataset already has tokenized columns.

            DPOTrainer expects: prompt_input_ids, chosen_input_ids, rejected_input_ids

            We accept *args, **kwargs for compatibility across TRL versions.
            """
            # Check if already tokenized
            required_cols = ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids"]
            if dataset is not None and all(col in dataset.column_names for col in required_cols):
                print("Dataset already tokenized, skipping DPOTrainer tokenization...")

                # Remove text columns if present (DPOTrainer normally does this)
                cols_to_remove = [c for c in ["prompt", "chosen", "rejected"]
                                  if c in dataset.column_names]
                if cols_to_remove:
                    dataset = dataset.remove_columns(cols_to_remove)

                # Return directly without any further processing
                # This avoids any .map() calls that could trigger pickle errors
                return dataset

            # Fall back to normal tokenization (shouldn't happen with our pre-tokenized data)
            print("WARNING: Dataset not pre-tokenized, falling back to DPOTrainer tokenization...")
            return super()._prepare_dataset(dataset, *args, **kwargs)

    trainer = PreTokenizedDPOTrainer(
        model=model,
        ref_model=None,  # For PEFT models
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=dpo_config,
    )

    return trainer


def main():
    """Main DPO training function."""

    import json
    import platform

    from datasets import Dataset
    from trl import DPOConfig
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    # IMPORTANT: Do NOT patch DPOTrainer on Windows - causes multiprocessing errors
    # The patched trainer has methods that can't be pickled for multiprocessing
    if platform.system() != "Windows":
        try:
            from unsloth import PatchDPOTrainer
            PatchDPOTrainer()
            print("Applied Unsloth DPO patches (non-Windows)")
        except ImportError:
            pass
    else:
        print("Windows detected: Using standard TRL DPOTrainer (no Unsloth patches)")

    # ============================================
    # VERIFY SETUP
    # ============================================

    print("=" * 60)
    print("O.L.I.V.I.A. DPO Training (Stage 2)")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Platform: {platform.system()}")

    # Determine model to load
    if os.path.exists(SFT_CHECKPOINT):
        model_path = SFT_CHECKPOINT
        print(f"Loading SFT checkpoint: {model_path}")
    elif os.path.exists(BASE_MODEL):
        model_path = BASE_MODEL
        print(f"WARNING: SFT checkpoint not found, using base model: {model_path}")
    else:
        print("ERROR: No model found!")
        print(f"  Expected SFT checkpoint: {SFT_CHECKPOINT}")
        print(f"  Or base model: {BASE_MODEL}")
        return

    # Check DPO data exists
    if not os.path.exists(DPO_FILE):
        print(f"ERROR: DPO pairs not found at {DPO_FILE}")
        print("Run tools/create_dpo_pairs.py first!")
        return

    print(f"DPO data found: {DPO_FILE}")

    # ============================================
    # LOAD MODEL
    # ============================================

    print(f"\nLoading model from: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    print("Model loaded")

    # Ensure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO typically uses left padding

    # ============================================
    # CONFIGURE LoRA (if not already applied)
    # ============================================

    # Check if model already has LoRA
    has_lora = False
    if hasattr(model, 'peft_config'):
        has_lora = True
    elif os.path.isdir(model_path):
        has_lora = 'adapter_config.json' in os.listdir(model_path)

    if not has_lora:
        print(f"\nConfiguring LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")

        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print("LoRA configured")
    else:
        print("Model already has LoRA adapter")

    # ============================================
    # SETUP CHAT TEMPLATE
    # ============================================

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============================================
    # LOAD AND FORMAT DPO DATASET
    # ============================================

    print(f"\nLoading DPO pairs: {DPO_FILE}")

    raw_data = []
    with open(DPO_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            raw_data.append(json.loads(line.strip()))

    print(f"Loaded {len(raw_data)} DPO pairs")

    # Format for DPO: create prompt, chosen, rejected as formatted text
    formatted_data = []

    for item in raw_data:
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Format as chat messages and apply template
        prompt_messages = [{"role": "user", "content": prompt}]
        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]
        rejected_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected}
        ]

        formatted_data.append({
            "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True),
            "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False),
            "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False),
        })

    print(f"Formatted {len(formatted_data)} DPO pairs")

    # Preview
    print("\n--- Sample DPO Pair ---")
    print(f"Prompt: {formatted_data[0]['prompt'][:100]}...")
    print(f"Chosen: {formatted_data[0]['chosen'][:100]}...")
    print(f"Rejected: {formatted_data[0]['rejected'][:100]}...")
    print("---\n")

    # ============================================
    # PRE-TOKENIZE ON WINDOWS
    # ============================================

    is_windows = platform.system() == "Windows"

    if is_windows:
        # Pre-tokenize to avoid multiprocessing pickle errors
        tokenized_data = pre_tokenize_dpo_dataset(
            formatted_data,
            tokenizer,
            MAX_PROMPT_LENGTH,
            MAX_COMPLETION_LENGTH,
        )

        # Merge formatted text with tokenized data
        final_data = []
        for fmt, tok in zip(formatted_data, tokenized_data):
            final_data.append({**fmt, **tok})

        dpo_dataset = Dataset.from_list(final_data)
        print(f"Created pre-tokenized dataset with {len(dpo_dataset)} pairs")
    else:
        # On Linux/Mac, let DPOTrainer handle tokenization normally
        dpo_dataset = Dataset.from_list(formatted_data)

    # ============================================
    # DPO TRAINING CONFIG
    # ============================================

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,

        # DPO-specific
        beta=BETA,
        loss_type=LOSS_TYPE,

        # Training
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.05,
        max_grad_norm=1.0,  # Gradient clipping for stability

        # Logging and saving
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,

        # Precision
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        # Optimizer
        optim="adamw_8bit",

        # DPO length settings
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,

        # CRITICAL: Disable multiprocessing for Windows compatibility
        dataset_num_proc=None,

        # Misc
        seed=42,
        report_to="none",

        # Disable precompute to avoid additional processing
        precompute_ref_log_probs=False,
    )

    # ============================================
    # CREATE TRAINER
    # ============================================

    print("Initializing DPO trainer...")

    if is_windows:
        # Use custom trainer that skips tokenization
        trainer = create_pretokenized_dpo_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dpo_dataset,
            dpo_config=dpo_config,
        )
    else:
        # Standard DPOTrainer on non-Windows
        from trl import DPOTrainer

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            processing_class=tokenizer,
            train_dataset=dpo_dataset,
            args=dpo_config,
        )

    # ============================================
    # PRE-TRAINING INFO
    # ============================================

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    max_memory = round(gpu_stats.total_memory / 1024**3, 2)

    print("\n" + "=" * 60)
    print("DPO TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model:               {model_path}")
    print(f"Beta (KL penalty):   {BETA}")
    print(f"Loss Type:           {LOSS_TYPE}")
    print(f"Epochs:              {EPOCHS}")
    print(f"Batch Size:          {BATCH_SIZE}")
    print(f"Gradient Accum:      {GRADIENT_ACCUMULATION}")
    print(f"Effective Batch:     {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Learning Rate:       {LEARNING_RATE}")
    print(f"DPO Pairs:           {len(dpo_dataset)}")
    print(f"Max Prompt Length:   {MAX_PROMPT_LENGTH}")
    print(f"Max Completion Len:  {MAX_COMPLETION_LENGTH}")
    print("-" * 60)
    print(f"GPU:                 {gpu_stats.name}")
    print(f"VRAM Before:         {start_gpu_memory}GB / {max_memory}GB")
    if is_windows:
        print("Mode:                Pre-tokenized (Windows)")
    print("=" * 60)

    # ============================================
    # TRAIN
    # ============================================

    print("\nStarting DPO training...\n")

    trainer_stats = trainer.train()

    # ============================================
    # POST-TRAINING STATS
    # ============================================

    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    training_time = trainer_stats.metrics['train_runtime']

    print("\n" + "=" * 60)
    print("DPO TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training Time:       {training_time:.2f}s ({training_time/60:.1f} minutes)")
    print(f"Peak VRAM:           {used_memory}GB / {max_memory}GB")
    print(f"Final Loss:          {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")

    # DPO-specific metrics
    for key in ['rewards/chosen', 'rewards/rejected', 'rewards/margins']:
        if key in trainer_stats.metrics:
            print(f"{key}:       {trainer_stats.metrics[key]:.4f}")

    print("=" * 60)

    # ============================================
    # SAVE MODEL
    # ============================================

    print(f"\nSaving DPO-trained adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("DPO TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nAdapter saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Merge and export: python tools/model_engineering/merge_lora.py")
    print("     (Update merge_lora.py to use olivia-dpo-checkpoint)")
    print("  2. Create Ollama model: ollama create olivia-finetuned -f models/ollama/Modelfile.olivia-finetuned")
    print("  3. Test: ollama run olivia-finetuned \"Hey\"")
    print("  4. Evaluate: python tools/evaluate_personality.py")


# ============================================
# WINDOWS MULTIPROCESSING GUARD
# ============================================

if __name__ == "__main__":
    freeze_support()
    main()
