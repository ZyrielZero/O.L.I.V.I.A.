"""
Generate DPO preference pairs for O.L.I.V.I.A. personality training.

Hybrid approach:
- 800 programmatic pairs (string manipulation)
- 200 model-generated pairs (persona contrast via Ollama)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama not installed. Model-generated pairs will be skipped.")

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "training"
INPUT_FILE = DATA_DIR / "olivia_sft_combined.jsonl"
OUTPUT_FILE = DATA_DIR / "olivia_dpo_pairs.jsonl"

# Generation config
TOTAL_PAIRS = 1000
PROGRAMMATIC_PAIRS = 800
MODEL_PAIRS = 200

# Programmatic strategy distribution
STRATEGY_DISTRIBUTION = {
    "corporate": 200,
    "verbose": 200,
    "questions": 150,
    "emoji": 150,
    "i_prefix": 100,
}

# Model strategy distribution
MODEL_DISTRIBUTION = {
    "persona_contrast": 150,
    "verbose_rewrite": 50,
}


class ProgrammaticGenerator:
    """Generate rejected responses via string manipulation."""

    CORPORATE_PREFIXES = [
        "Certainly! ",
        "Absolutely! ",
        "Of course! ",
        "Great question! ",
        "I'd be happy to help! ",
        "That's a great question! ",
        "I'd be delighted to assist! ",
    ]

    CORPORATE_SUFFIXES = [
        " I hope that helps!",
        " Let me know if you need anything else!",
        " Feel free to ask more questions!",
        " Is there anything else I can help with?",
        " I'm here to help!",
    ]

    VERBOSE_FILLERS = [
        "That's a really interesting thought! ",
        "I think it's important to consider that ",
        "There are many aspects to this. ",
        "Let me explain in more detail. ",
        "Well, to be honest with you, ",
        "I've been thinking about this, and ",
    ]

    VERBOSE_PADDING = [
        " This is something that many people wonder about.",
        " It's a complex topic with many facets to explore.",
        " There's a lot more to say about this, but that's the gist.",
        " I could elaborate further if you'd like.",
        " Of course, there are always more nuances to consider.",
    ]

    EXTRA_QUESTIONS = [
        " What do you think?",
        " Does that make sense?",
        " How does that sound?",
        " Would you like me to elaborate?",
        " Is that helpful?",
        " What are your thoughts?",
        " Do you have any other questions?",
    ]

    EMOJIS = ["😊", "👍", "❤️", "✨", "🎉", "😄", "🙂", "💜", "🌟", "😃"]

    ASTERISK_ACTIONS = [
        "*smiles*",
        "*nods*",
        "*thinks*",
        "*laughs*",
        "*tilts head*",
        "*giggles*",
    ]

    I_PREFIXES = [
        "I think ",
        "I believe ",
        "I would say ",
        "I feel like ",
        "I personally think ",
        "I'd say ",
    ]

    def corporatize(self, response: str) -> str:
        """Add corporate AI patterns."""
        prefix = random.choice(self.CORPORATE_PREFIXES)
        suffix = random.choice(self.CORPORATE_SUFFIXES)
        return prefix + response + suffix

    def verbosify(self, response: str) -> str:
        """Pad with verbose filler text."""
        filler = random.choice(self.VERBOSE_FILLERS)
        padding = random.choice(self.VERBOSE_PADDING)
        return filler + response + padding

    def add_questions(self, response: str) -> str:
        """Add multiple follow-up questions."""
        questions = random.sample(self.EXTRA_QUESTIONS, 3)
        return response + "".join(questions)

    def add_emoji(self, response: str) -> str:
        """Add emojis and/or asterisk actions."""
        choices = []

        # Add emoji
        emoji = random.choice(self.EMOJIS)
        if random.random() < 0.7:
            choices.append(("emoji_end", emoji))
        if random.random() < 0.3:
            choices.append(("emoji_mid", emoji))

        # Add asterisk action
        if random.random() < 0.5:
            action = random.choice(self.ASTERISK_ACTIONS)
            choices.append(("action", action))

        if not choices:
            choices.append(("emoji_end", emoji))

        result = response
        for choice_type, value in choices:
            if choice_type == "emoji_end":
                result = result.rstrip(".!?") + "! " + value
            elif choice_type == "emoji_mid":
                words = result.split()
                if len(words) > 3:
                    pos = len(words) // 2
                    words.insert(pos, value)
                    result = " ".join(words)
            elif choice_type == "action":
                result = value + " " + result

        return result

    def add_i_prefix(self, response: str) -> str:
        """Force response to start with 'I'."""
        # If already starts with I, transform differently
        if response.strip().startswith("I"):
            prefix = random.choice(self.I_PREFIXES)
            # Lowercase first letter and prepend
            if response[0].isupper():
                response = response[0].lower() + response[1:]
            return prefix + response
        else:
            prefix = random.choice(self.I_PREFIXES)
            # Lowercase first letter
            if response[0].isupper():
                response = response[0].lower() + response[1:]
            return prefix + response

    def generate(self, response: str, strategy: str) -> str:
        """Generate a rejected response using the specified strategy."""
        if strategy == "corporate":
            return self.corporatize(response)
        elif strategy == "verbose":
            return self.verbosify(response)
        elif strategy == "questions":
            return self.add_questions(response)
        elif strategy == "emoji":
            return self.add_emoji(response)
        elif strategy == "i_prefix":
            return self.add_i_prefix(response)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class ModelGenerator:
    """Generate rejected responses via Ollama model calls."""

    GENERIC_SYSTEM_PROMPT = """You are a helpful AI assistant. Respond naturally and helpfully to the user.
Be thorough and complete in your responses. Use professional language."""

    VERBOSE_SYSTEM_PROMPT = """You are a detailed and thorough AI assistant.
Expand on the following response to make it more comprehensive.
Add context, explanations, and elaborate on the points.
Make the response at least 2-3 times longer while keeping it relevant."""

    def __init__(self, model: str = "olivia-merged"):
        self.model = model

    def generate_persona_contrast(self, prompt: str, chosen: str) -> Optional[str]:
        """Generate a response without the Olivia persona."""
        if not OLLAMA_AVAILABLE:
            return None

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.GENERIC_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.8, "num_predict": 150}
            )
            rejected = response["message"]["content"].strip()

            # Ensure it's actually different from chosen
            if rejected.lower() == chosen.lower():
                return None

            return rejected
        except Exception as e:
            print(f"Ollama error: {e}")
            return None

    def generate_verbose_rewrite(self, prompt: str, chosen: str) -> Optional[str]:
        """Have model expand a concise response verbosely."""
        if not OLLAMA_AVAILABLE:
            return None

        try:
            expand_prompt = f"""The user asked: "{prompt}"

A concise response was: "{chosen}"

Rewrite this response to be much more detailed, thorough, and comprehensive.
Add explanations, context, and be more elaborate. Make it at least 80 words."""

            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.VERBOSE_SYSTEM_PROMPT},
                    {"role": "user", "content": expand_prompt}
                ],
                options={"temperature": 0.7, "num_predict": 200}
            )
            rejected = response["message"]["content"].strip()
            return rejected
        except Exception as e:
            print(f"Ollama error: {e}")
            return None


def load_sft_data(filepath: Path) -> List[Dict]:
    """Load SFT data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)
    return data


def extract_qa_pair(entry: Dict) -> Tuple[str, str]:
    """Extract question and answer from ShareGPT format."""
    prompt = ""
    response = ""

    for turn in entry.get("conversations", []):
        if turn.get("from") == "human":
            prompt = turn.get("value", "")
        elif turn.get("from") == "gpt":
            response = turn.get("value", "")

    return prompt, response


def create_dpo_pairs(use_model: bool = True, verbose: bool = True):
    """Generate DPO preference pairs."""
    print("=" * 60)
    print("O.L.I.V.I.A. DPO Pair Generator")
    print("=" * 60)

    # Load SFT data
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Run combine_training_data.py first!")
        return

    sft_data = load_sft_data(INPUT_FILE)
    print(f"\nLoaded {len(sft_data)} SFT examples")

    # Initialize generators
    prog_gen = ProgrammaticGenerator()
    model_gen = ModelGenerator() if use_model and OLLAMA_AVAILABLE else None

    # Shuffle data
    random.seed(42)
    shuffled_data = sft_data.copy()
    random.shuffle(shuffled_data)

    pairs = []
    stats = {strategy: 0 for strategy in list(STRATEGY_DISTRIBUTION.keys()) + list(MODEL_DISTRIBUTION.keys())}
    data_idx = 0

    # Generate programmatic pairs
    print(f"\n{'=' * 60}")
    print("Generating programmatic pairs...")
    print("=" * 60)

    for strategy, count in STRATEGY_DISTRIBUTION.items():
        if verbose:
            print(f"\n  Strategy: {strategy} (target: {count})")

        generated = 0
        while generated < count and data_idx < len(shuffled_data):
            prompt, chosen = extract_qa_pair(shuffled_data[data_idx])
            data_idx += 1

            if not prompt or not chosen:
                continue

            rejected = prog_gen.generate(chosen, strategy)

            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "strategy": strategy
            })
            generated += 1
            stats[strategy] += 1

        if verbose:
            print(f"    Generated: {generated}")

    # Generate model pairs (if available)
    if model_gen and use_model:
        print(f"\n{'=' * 60}")
        print("Generating model-based pairs...")
        print("=" * 60)

        for strategy, count in MODEL_DISTRIBUTION.items():
            if verbose:
                print(f"\n  Strategy: {strategy} (target: {count})")

            generated = 0
            attempts = 0
            max_attempts = count * 3  # Allow retries

            while generated < count and data_idx < len(shuffled_data) and attempts < max_attempts:
                prompt, chosen = extract_qa_pair(shuffled_data[data_idx])
                data_idx += 1
                attempts += 1

                if not prompt or not chosen:
                    continue

                if strategy == "persona_contrast":
                    rejected = model_gen.generate_persona_contrast(prompt, chosen)
                elif strategy == "verbose_rewrite":
                    rejected = model_gen.generate_verbose_rewrite(prompt, chosen)
                else:
                    continue

                if rejected:
                    pairs.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "strategy": strategy
                    })
                    generated += 1
                    stats[strategy] += 1

                    if verbose and generated % 10 == 0:
                        print(f"    Progress: {generated}/{count}")

            if verbose:
                print(f"    Generated: {generated}")
    else:
        print("\nSkipping model-based pairs (Ollama not available or disabled)")

        # Fill with more programmatic pairs instead
        remaining = MODEL_PAIRS
        strategies = list(STRATEGY_DISTRIBUTION.keys())

        while remaining > 0 and data_idx < len(shuffled_data):
            strategy = random.choice(strategies)
            prompt, chosen = extract_qa_pair(shuffled_data[data_idx])
            data_idx += 1

            if not prompt or not chosen:
                continue

            rejected = prog_gen.generate(chosen, strategy)
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "strategy": strategy + "_fallback"
            })
            stats[strategy] += 1
            remaining -= 1

    # Shuffle final pairs
    random.shuffle(pairs)

    # Write output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Total pairs generated: {len(pairs)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nBy strategy:")
    for strategy, count in stats.items():
        if count > 0:
            print(f"  {strategy}: {count}")

    # Sample output
    print(f"\n{'=' * 60}")
    print("SAMPLE PAIRS")
    print("=" * 60)

    for i, pair in enumerate(pairs[:3]):
        print(f"\n--- Pair {i+1} ({pair['strategy']}) ---")
        print(f"Prompt: {pair['prompt'][:80]}...")
        print(f"Chosen: {pair['chosen'][:80]}...")
        print(f"Rejected: {pair['rejected'][:80]}...")

    return len(pairs)


def validate_dpo_pairs():
    """Validate generated DPO pairs."""
    if not OUTPUT_FILE.exists():
        print("DPO pairs file not found. Run create_dpo_pairs() first.")
        return

    print("\n" + "=" * 60)
    print("Validating DPO pairs...")
    print("=" * 60)

    # Import validation from combine script
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from combine_training_data import validate_response

    pairs = load_sft_data(OUTPUT_FILE)

    issues = {
        "chosen_invalid": 0,
        "rejected_valid": 0,  # Rejected should NOT be valid
        "no_contrast": 0,
    }

    for i, pair in enumerate(pairs):
        chosen_result = validate_response(pair["chosen"])
        rejected_result = validate_response(pair["rejected"])

        if not chosen_result.valid:
            issues["chosen_invalid"] += 1

        if rejected_result.valid:
            # Rejected should have at least one issue
            issues["rejected_valid"] += 1

        if pair["chosen"] == pair["rejected"]:
            issues["no_contrast"] += 1

    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Chosen responses invalid: {issues['chosen_invalid']}")
    print(f"Rejected responses accidentally valid: {issues['rejected_valid']}")
    print(f"No contrast (identical): {issues['no_contrast']}")

    valid_pairs = len(pairs) - issues["chosen_invalid"] - issues["no_contrast"]
    print(f"\nEffective valid pairs: {valid_pairs} ({100*valid_pairs/len(pairs):.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate DPO pairs for O.L.I.V.I.A.")
    parser.add_argument("--no-model", action="store_true", help="Skip model-generated pairs")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing pairs")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    if args.validate_only:
        validate_dpo_pairs()
    else:
        create_dpo_pairs(use_model=not args.no_model, verbose=not args.quiet)
        validate_dpo_pairs()
