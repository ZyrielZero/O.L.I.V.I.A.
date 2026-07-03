"""O.L.I.V.I.A. merged-model verification script.

Post-merge smoke test for the merge kit: checks that the tokenizer survived
the merge (Llama-3 special tokens must encode to single IDs), that a chat
template is present, and that the model loads and generates.

Usage:
    python verify_merge.py <model_path> [--max-new-tokens N] [--skip-generation]

Exits nonzero if any special token is split, so it can gate scripts.
"""

import argparse
import sys

# Llama-3 special tokens — each must encode to a single ID after a merge.
# Correct for the all-L3.1 lineup this kit merges; if the base family ever
# changes, this list changes with it.
LLAMA3_SPECIAL_TOKENS = [
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|begin_of_text|>",
]


def main() -> int:
    """Run tokenizer checks and an optional generation smoke test.

    Returns:
        Process exit code: 0 if all special tokens encode to single IDs,
        1 if any token is split by the tokenizer.
    """
    parser = argparse.ArgumentParser(
        description="Verify a merged model's tokenizer is intact and the model loads."
    )
    parser.add_argument("model_path", help="Path to the merged model directory")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Tokens to generate in the smoke test (default: 50)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Tokenizer-only check; skips loading the model (useful on a busy GPU)",
    )
    args = parser.parse_args()

    # Heavy imports stay inside main() so --help works without ML deps installed
    from transformers import AutoTokenizer

    print("=" * 60)
    print("O.L.I.V.I.A. Model Verification")
    print("=" * 60)
    print(f"Model path: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("\n=== Tokenizer Verification ===")
    print(f"Vocab size: {tokenizer.vocab_size}")

    print("\nSpecial Token IDs (should be single integers):")
    split_detected = False
    for token in LLAMA3_SPECIAL_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            status = "OK"
        else:
            status = "SPLIT - WARNING"
            split_detected = True
        print(f"  {token}: {ids} {status}")

    print(f"\nChat template present: {tokenizer.chat_template is not None}")

    if not args.skip_generation:
        import torch
        from transformers import AutoModelForCausalLM

        print("\n=== Quick Generation Test ===")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

        messages = [
            {"role": "system", "content": "You are Olivia, a warm and caring companion."},
            {"role": "user", "content": "Hello!"},
        ]

        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
            model.device
        )
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Response: {response}")

    print("\n" + "=" * 60)
    if split_detected:
        print("VERIFICATION FAILED — tokenizer split a special token")
    else:
        print("VERIFICATION COMPLETE")
    print("=" * 60)
    return 1 if split_detected else 0


if __name__ == "__main__":
    sys.exit(main())
