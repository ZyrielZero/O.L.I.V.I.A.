"""Combine, validate, and split training data for O.L.I.V.I.A. fine-tuning."""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "training"
INPUT_FILES = [
    DATA_DIR / "olivia_training_complete.jsonl",
    DATA_DIR / "olivia_training_hobbies.jsonl",
    DATA_DIR / "olivia_training_companion.jsonl",
]
OUTPUT_COMBINED = DATA_DIR / "olivia_sft_combined.jsonl"
OUTPUT_TRAIN = DATA_DIR / "olivia_sft_train.jsonl"
OUTPUT_VAL = DATA_DIR / "olivia_sft_val.jsonl"

# Validation config
FORBIDDEN_PHRASES = [
    "Certainly!", "Absolutely!", "Of course!", "Great question!",
    "That's a great question!", "I'd be happy to help!", "I'd be delighted to!",
    "I hope that helps!", "Hope this helps!", "Please let me know if you need anything else!",
    "Feel free to ask!", "Is there anything else I can help with?",
    "Thank you for sharing!", "That's very interesting!", "How fascinating!",
    "Wonderful!", "Fantastic!", "Amazing!", "Excellent!", "I apologize for any confusion",
    "As an AI", "As a language model", "I don't have feelings", "I'm just an AI"
]

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U0001F004-\U0001F0CF"
    "]+"
)

ASTERISK_ACTION_PATTERN = re.compile(r'\*[^*]+\*')
KAOMOJI_PATTERN = re.compile(r'[\(\)><\^_\-\*°]+[oOvV>\<\^_\-\*°ω・｡\(\)]+[\(\)><\^_\-\*°]+|[:;=][-]?[\(\)DPpOo3\|\\\/\[\]]+')


@dataclass
class ValidationResult:
    valid: bool
    hard_issues: List[str]  # Automatic rejection
    soft_issues: List[str]  # Warning only


def validate_response(response: str) -> ValidationResult:
    """Validate a single response against Olivia personality rules."""
    hard_issues = []
    soft_issues = []

    # Hard failures - automatic rejection
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in response.lower():
            hard_issues.append(f"Forbidden phrase: '{phrase}'")

    if EMOJI_PATTERN.search(response):
        hard_issues.append("Contains emoji")

    if ASTERISK_ACTION_PATTERN.search(response):
        hard_issues.append("Contains asterisk action (*action*)")

    if KAOMOJI_PATTERN.search(response):
        hard_issues.append("Contains kaomoji")

    # Soft failures - warning only
    word_count = len(response.split())
    if word_count > 60:
        hard_issues.append(f"Too long: {word_count} words (max 60)")
    elif word_count > 50:
        soft_issues.append(f"Slightly long: {word_count} words (target <50)")

    question_count = response.count('?')
    if question_count > 1:
        hard_issues.append(f"Multiple questions: {question_count} (max 1)")

    if response.strip().startswith("I "):
        soft_issues.append("Starts with 'I'")

    return ValidationResult(
        valid=len(hard_issues) == 0,
        hard_issues=hard_issues,
        soft_issues=soft_issues
    )


def load_and_validate(filepath: Path) -> Tuple[List[dict], List[dict], int, int]:
    """Load JSONL file and validate each entry.

    Returns: (valid_entries, invalid_entries, soft_warning_count, total_count)
    """
    valid = []
    invalid = []
    soft_warnings = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                # Get the assistant response
                response = None
                for turn in data.get('conversations', []):
                    if turn.get('from') == 'gpt':
                        response = turn.get('value', '')
                        break

                if response is None:
                    invalid.append({
                        'line': i,
                        'file': filepath.name,
                        'data': data,
                        'issues': ['No gpt response found']
                    })
                    continue

                result = validate_response(response)

                if result.valid:
                    valid.append(data)
                    if result.soft_issues:
                        soft_warnings += 1
                else:
                    invalid.append({
                        'line': i,
                        'file': filepath.name,
                        'data': data,
                        'issues': result.hard_issues
                    })

            except json.JSONDecodeError:
                invalid.append({
                    'line': i,
                    'file': filepath.name,
                    'data': None,
                    'issues': ['Invalid JSON']
                })

    return valid, invalid, soft_warnings, i


def combine_and_split(val_ratio: float = 0.1, seed: int = 42):
    """Combine all training data, validate, shuffle, and split."""
    print("=" * 60)
    print("O.L.I.V.I.A. Training Data Combiner")
    print("=" * 60)

    all_valid = []
    all_invalid = []
    total_soft_warnings = 0

    # Load and validate each file
    for filepath in INPUT_FILES:
        if not filepath.exists():
            print(f"\nWARNING: File not found: {filepath}")
            continue

        print(f"\nProcessing: {filepath.name}")
        valid, invalid, soft_warnings, total = load_and_validate(filepath)

        print(f"  Total: {total}")
        print(f"  Valid: {len(valid)}")
        print(f"  Invalid: {len(invalid)}")
        print(f"  Soft warnings: {soft_warnings}")

        all_valid.extend(valid)
        all_invalid.extend(invalid)
        total_soft_warnings += soft_warnings

    # Report invalid entries
    if all_invalid:
        print(f"\n{'=' * 60}")
        print(f"HARD FAILURES ({len(all_invalid)} total)")
        print("=" * 60)
        for entry in all_invalid[:20]:  # Show first 20
            print(f"\n  File: {entry['file']}, Line: {entry['line']}")
            print(f"  Issues: {', '.join(entry['issues'])}")
            if entry['data']:
                resp = entry['data'].get('conversations', [{}])[-1].get('value', '')[:100]
                print(f"  Response: {resp}...")
        if len(all_invalid) > 20:
            print(f"\n  ... and {len(all_invalid) - 20} more")

    # Shuffle
    print(f"\n{'=' * 60}")
    print("SHUFFLING AND SPLITTING")
    print("=" * 60)

    random.seed(seed)
    random.shuffle(all_valid)

    # Split
    val_size = int(len(all_valid) * val_ratio)
    train_size = len(all_valid) - val_size

    train_data = all_valid[:train_size]
    val_data = all_valid[train_size:]

    print(f"\nTotal valid samples: {len(all_valid)}")
    print(f"Training set: {len(train_data)} ({100 * (1 - val_ratio):.0f}%)")
    print(f"Validation set: {len(val_data)} ({100 * val_ratio:.0f}%)")
    print(f"Soft warnings: {total_soft_warnings}")

    # Write combined file
    with open(OUTPUT_COMBINED, 'w', encoding='utf-8') as f:
        for entry in all_valid:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"\nWritten: {OUTPUT_COMBINED}")

    # Write train file
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Written: {OUTPUT_TRAIN}")

    # Write validation file
    with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Written: {OUTPUT_VAL}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Input files: {len(INPUT_FILES)}")
    print(f"Total samples processed: {len(all_valid) + len(all_invalid)}")
    print(f"Valid samples: {len(all_valid)}")
    print(f"Invalid samples (removed): {len(all_invalid)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    return len(all_valid), len(all_invalid), len(train_data), len(val_data)


if __name__ == "__main__":
    combine_and_split()
