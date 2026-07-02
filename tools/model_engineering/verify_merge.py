"""
O.L.I.V.I.A. Merged Model Verification Script
Verifies that the merged model has correct tokenizer and can generate text

Repository: https://github.com/ZyrielZero/project-olivia

Run from: project root or tools/model_engineering folder
Updated: January 2026 - Reorganized project structure
"""

from pathlib import Path

# ============================================
# PATH CONFIGURATION - Auto-detect project root
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # tools/model_engineering -> project root

MODELS_DIR = PROJECT_ROOT / "models"

# Model to verify (default: olivia-merged-v2)
MODEL_PATH = str(MODELS_DIR / "checkpoints" / "olivia-merged-v2")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("O.L.I.V.I.A. Model Verification")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model path: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("\n=== Tokenizer Verification ===")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Check special tokens are single IDs (not split)
    test_tokens = [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|begin_of_text|>",
    ]

    print("\nSpecial Token IDs (should be single integers):")
    for token in test_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        status = "OK" if len(ids) == 1 else "SPLIT - WARNING"
        print(f"  {token}: {ids} {status}")

    # Check chat template exists
    print(f"\nChat template present: {tokenizer.chat_template is not None}")

    # Quick generation test
    print("\n=== Quick Generation Test ===")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    messages = [
        {"role": "system", "content": "You are Olivia, a warm and caring companion."},
        {"role": "user", "content": "Hello!"}
    ]

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    output = model.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"Response: {response}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
