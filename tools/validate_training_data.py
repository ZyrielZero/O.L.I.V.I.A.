"""Validate training data against Olivia personality requirements."""
import json
import re

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
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # Geometric Shapes
    "\U0001F800-\U0001F8FF"  # arrows
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U0001F004-\U0001F0CF"  # misc
    "]+"
)

ASTERISK_ACTION_PATTERN = re.compile(r'\*[^*]+\*')
KAOMOJI_PATTERN = re.compile(r'[\(\)><\^_\-\*°]+[oOvV>\<\^_\-\*°ω・｡\(\)]+[\(\)><\^_\-\*°]+|[:;=][-]?[\(\)DPpOo3\|\\\/\[\]]+')

def validate_file(filepath):
    issues = []
    word_counts = []
    question_counts = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                response = data['conversations'][1]['value']

                # Word count
                words = len(response.split())
                word_counts.append(words)
                if words > 60:
                    issues.append(f"Line {i}: Response too long ({words} words)")

                # Question count
                questions = response.count('?')
                question_counts.append(questions)
                if questions > 1:
                    issues.append(f"Line {i}: Too many questions ({questions})")

                # Emoji check
                if EMOJI_PATTERN.search(response):
                    issues.append(f"Line {i}: Contains emoji")

                # Asterisk actions
                if ASTERISK_ACTION_PATTERN.search(response):
                    issues.append(f"Line {i}: Contains asterisk action")

                # Kaomoji
                if KAOMOJI_PATTERN.search(response):
                    issues.append(f"Line {i}: Contains kaomoji")

                # Forbidden phrases
                for phrase in FORBIDDEN_PHRASES:
                    if phrase.lower() in response.lower():
                        issues.append(f"Line {i}: Contains forbidden phrase '{phrase}'")

            except json.JSONDecodeError:
                issues.append(f"Line {i}: Invalid JSON")
            except (KeyError, IndexError):
                issues.append(f"Line {i}: Invalid structure")

    return issues, word_counts, question_counts

if __name__ == "__main__":
    filepath = "data/training/olivia_training_hobbies.jsonl"
    issues, word_counts, question_counts = validate_file(filepath)

    print("=== Training Data Validation ===")
    print(f"Total samples: {len(word_counts)}")
    print(f"Average word count: {sum(word_counts)/len(word_counts):.1f}")
    print(f"Max word count: {max(word_counts)}")
    print(f"Min word count: {min(word_counts)}")
    print(f"Samples >50 words: {sum(1 for w in word_counts if w > 50)}")
    print(f"Samples >60 words: {sum(1 for w in word_counts if w > 60)}")
    print(f"Samples with >1 question: {sum(1 for q in question_counts if q > 1)}")
    print()

    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("No issues found!")

    print()
    print("Word count distribution:")
    ranges = [(0, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 100)]
    for low, high in ranges:
        count = sum(1 for w in word_counts if low <= w <= high)
        pct = count / len(word_counts) * 100
        print(f"  {low:3}-{high:3} words: {count:4} ({pct:5.1f}%)")
