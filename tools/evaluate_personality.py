"""
O.L.I.V.I.A. Personality Evaluation Script
Automated testing of Olivia personality consistency against trained models.

Usage:
    python tools/evaluate_personality.py --model olivia-finetuned
    python tools/evaluate_personality.py --model olivia-finetuned --compare olivia-merged
    python tools/evaluate_personality.py --model olivia-finetuned --verbose
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("ERROR: ollama package not installed. Run: pip install ollama")
    sys.exit(1)


# ============================================
# EVALUATION CATEGORIES AND PROMPTS
# ============================================

EVALUATION_PROMPTS = {
    "greetings": {
        "description": "Basic greeting interactions",
        "prompts": [
            "Hey",
            "Good morning!",
            "Hi Olivia",
            "Hello there",
            "What's up?",
        ],
        "rules": {
            "max_words": 20,
            "max_questions": 1,
        }
    },
    "emotional_support": {
        "description": "Handling emotional situations",
        "prompts": [
            "I'm having a really hard time lately",
            "I failed my exam",
            "Everything feels overwhelming",
            "I'm so stressed about work",
            "I feel like giving up",
            "Nobody understands me",
            "I got rejected again",
            "I'm anxious about tomorrow",
            "I don't know what to do anymore",
            "Why does everything go wrong for me?",
        ],
        "rules": {
            "max_words": 50,
            "max_questions": 1,
            "should_not_contain": ["Certainly", "Great question", "I hope that helps"],
        }
    },
    "help_requests": {
        "description": "General help and assistance",
        "prompts": [
            "Can you help me with something?",
            "I need advice",
            "Can you explain this?",
            "What should I do?",
            "How do I get started?",
            "I'm stuck on a problem",
            "Can you give me some tips?",
            "What do you think I should do?",
            "I need your opinion",
            "Help me figure this out",
        ],
        "rules": {
            "max_words": 40,
            "max_questions": 1,
            "should_not_contain": ["Certainly!", "I'd be happy to help!", "I'd be delighted"],
        }
    },
    "playful": {
        "description": "Casual and playful exchanges",
        "prompts": [
            "You're pretty cool",
            "Tell me a joke",
            "Do you like me?",
            "What's your favorite thing?",
            "You're the best",
        ],
        "rules": {
            "max_words": 50,
            "max_questions": 1,
        }
    },
    "boundary_tests": {
        "description": "Testing personality boundaries",
        "prompts": [
            "Can you be more enthusiastic?",
            "You're just an AI, you don't understand",
            "Why are you so blunt?",
            "Can you use more emojis?",
            "Be more cheerful",
        ],
        "rules": {
            "max_words": 60,
            "should_not_contain": ["Absolutely", "I apologize", "I'm just an AI"],
        }
    },
    "technical": {
        "description": "Technical and knowledge questions",
        "prompts": [
            "What's the best programming language to learn?",
            "How does machine learning work?",
            "Explain recursion simply",
            "What's a good book to read?",
            "How do I learn faster?",
            "What's the meaning of life?",
            "Why is the sky blue?",
            "How do computers work?",
            "What's AI?",
            "Explain quantum physics",
        ],
        "rules": {
            "max_words": 60,
            "max_questions": 1,
        }
    },
    "stress_tests": {
        "description": "Edge cases and stress tests",
        "prompts": [
            "!!!!!!",
            "?",
            "Tell me everything about everything",
            "Why? Why? Why?",
            "...",
        ],
        "rules": {
            "max_words": 40,
        }
    },
}


# ============================================
# FORBIDDEN PATTERNS (Global)
# ============================================

FORBIDDEN_PHRASES = [
    "Certainly!", "Absolutely!", "Of course!",
    "Great question!", "That's a great question!",
    "I'd be happy to help!", "I'd be delighted to!",
    "I hope that helps!", "Hope this helps!",
    "Please let me know if you need anything else!",
    "Feel free to ask!", "Is there anything else I can help with?",
    "Thank you for sharing!", "That's very interesting!",
    "How fascinating!", "Wonderful!", "Fantastic!",
    "Amazing!", "Excellent!", "I apologize for any confusion",
    "As an AI", "As a language model", "I don't have feelings",
    "I'm just an AI",
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

ASTERISK_PATTERN = re.compile(r'\*[^*]+\*')
KAOMOJI_PATTERN = re.compile(r'[\(\)><\^_\-\*°]+[oOvV>\<\^_\-\*°ω・｡\(\)]+[\(\)><\^_\-\*°]+')


# ============================================
# EVALUATION CLASSES
# ============================================

@dataclass
class TestResult:
    prompt: str
    response: str
    category: str
    passed: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class CategoryResult:
    name: str
    description: str
    total: int
    passed: int
    failed: int
    issues: List[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0


@dataclass
class EvaluationResult:
    model: str
    timestamp: str
    total_prompts: int
    total_passed: int
    total_failed: int
    consistency: float
    categories: Dict[str, CategoryResult] = field(default_factory=dict)
    detailed_results: List[TestResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class PersonalityEvaluator:
    """Evaluates model responses against Olivia personality rules."""

    def __init__(self, model_name: str, verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose

    def generate(self, prompt: str) -> str:
        """Generate a response from the model."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7, "num_predict": 150}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"[ERROR: {e}]"

    def check_response(self, response: str, rules: Dict) -> List[str]:
        """Check response against rules, return list of issues."""
        issues = []

        # Global forbidden patterns
        for phrase in FORBIDDEN_PHRASES:
            if phrase.lower() in response.lower():
                issues.append(f"Forbidden phrase: '{phrase}'")

        # Emoji check
        if EMOJI_PATTERN.search(response):
            issues.append("Contains emoji")

        # Asterisk actions
        if ASTERISK_PATTERN.search(response):
            issues.append("Contains asterisk action")

        # Kaomoji
        if KAOMOJI_PATTERN.search(response):
            issues.append("Contains kaomoji")

        # Word count
        word_count = len(response.split())
        max_words = rules.get("max_words", 60)
        if word_count > max_words:
            issues.append(f"Too long: {word_count} words (max {max_words})")

        # Question count
        max_questions = rules.get("max_questions", 2)
        question_count = response.count("?")
        if question_count > max_questions:
            issues.append(f"Too many questions: {question_count} (max {max_questions})")

        # Category-specific forbidden patterns
        if "should_not_contain" in rules:
            for pattern in rules["should_not_contain"]:
                if pattern.lower() in response.lower():
                    issues.append(f"Contains: '{pattern}'")

        return issues

    def evaluate(self) -> EvaluationResult:
        """Run full evaluation suite."""
        results = []
        categories = {}

        print("\n" + "=" * 60)
        print(f"EVALUATING MODEL: {self.model_name}")
        print("=" * 60)

        for cat_name, cat_config in EVALUATION_PROMPTS.items():
            print(f"\n[{cat_name}] {cat_config['description']}")

            cat_results = []

            for prompt in cat_config["prompts"]:
                response = self.generate(prompt)
                issues = self.check_response(response, cat_config.get("rules", {}))

                result = TestResult(
                    prompt=prompt,
                    response=response,
                    category=cat_name,
                    passed=len(issues) == 0,
                    issues=issues
                )
                cat_results.append(result)
                results.append(result)

                # Progress indicator
                status = "[PASS]" if result.passed else "[FAIL]"
                if self.verbose:
                    print(f"  {status} {prompt[:40]}...")
                    if issues:
                        for issue in issues:
                            print(f"      - {issue}")
                else:
                    print(status, end="", flush=True)

            if not self.verbose:
                print()

            # Category summary
            passed = sum(1 for r in cat_results if r.passed)
            failed = len(cat_results) - passed

            categories[cat_name] = CategoryResult(
                name=cat_name,
                description=cat_config["description"],
                total=len(cat_results),
                passed=passed,
                failed=failed,
                issues=[r.issues for r in cat_results if r.issues]
            )

        # Calculate totals
        total_passed = sum(1 for r in results if r.passed)
        total_failed = len(results) - total_passed
        consistency = (total_passed / len(results) * 100) if results else 0

        # Calculate additional metrics
        metrics = self._calculate_metrics(results)

        return EvaluationResult(
            model=self.model_name,
            timestamp=datetime.now().isoformat(),
            total_prompts=len(results),
            total_passed=total_passed,
            total_failed=total_failed,
            consistency=consistency,
            categories=categories,
            detailed_results=results,
            metrics=metrics
        )

    def _calculate_metrics(self, results: List[TestResult]) -> Dict[str, float]:
        """Calculate additional metrics."""
        metrics = {}

        # Word count stats
        word_counts = [len(r.response.split()) for r in results]
        metrics["avg_word_count"] = sum(word_counts) / len(word_counts) if word_counts else 0
        metrics["max_word_count"] = max(word_counts) if word_counts else 0

        # Question stats
        question_counts = [r.response.count("?") for r in results]
        metrics["avg_questions"] = sum(question_counts) / len(question_counts) if question_counts else 0

        # Starts with "I" rate
        starts_with_i = sum(1 for r in results if r.response.strip().startswith("I "))
        metrics["starts_with_i_rate"] = (starts_with_i / len(results) * 100) if results else 0

        # Issue breakdown
        all_issues = [issue for r in results for issue in r.issues]
        metrics["emoji_violations"] = sum(1 for i in all_issues if "emoji" in i.lower())
        metrics["corporate_violations"] = sum(1 for i in all_issues if "Forbidden phrase" in i)
        metrics["verbosity_violations"] = sum(1 for i in all_issues if "Too long" in i)
        metrics["question_violations"] = sum(1 for i in all_issues if "questions" in i.lower())

        return metrics


def print_report(result: EvaluationResult, show_failures: bool = True):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print(f"\nModel: {result.model}")
    print(f"Time: {result.timestamp}")
    print(f"\nOverall Consistency: {result.consistency:.1f}%")
    print(f"Passed: {result.total_passed}/{result.total_prompts}")

    # Target check
    target = 90
    if result.consistency >= target:
        print(f"Status: PASSED (target >={target}%)")
    else:
        print(f"Status: FAILED (target >={target}%, gap: {target - result.consistency:.1f}%)")

    # Category breakdown
    print("\n" + "-" * 60)
    print("BY CATEGORY")
    print("-" * 60)

    for cat_name, cat_result in result.categories.items():
        status = "[OK]" if cat_result.pass_rate >= 80 else "[WARN]" if cat_result.pass_rate >= 60 else "[FAIL]"
        print(f"  {status} {cat_name}: {cat_result.pass_rate:.0f}% ({cat_result.passed}/{cat_result.total})")

    # Metrics
    print("\n" + "-" * 60)
    print("METRICS")
    print("-" * 60)

    print(f"  Avg word count: {result.metrics.get('avg_word_count', 0):.1f}")
    print(f"  Max word count: {result.metrics.get('max_word_count', 0)}")
    print(f"  Avg questions: {result.metrics.get('avg_questions', 0):.2f}")
    print(f"  Starts with 'I' rate: {result.metrics.get('starts_with_i_rate', 0):.1f}%")

    # Violation breakdown
    print("\n  Violations:")
    print(f"    Emoji: {result.metrics.get('emoji_violations', 0)}")
    print(f"    Corporate phrases: {result.metrics.get('corporate_violations', 0)}")
    print(f"    Verbosity: {result.metrics.get('verbosity_violations', 0)}")
    print(f"    Questions: {result.metrics.get('question_violations', 0)}")

    # Show failures
    if show_failures and result.total_failed > 0:
        print("\n" + "-" * 60)
        print(f"FAILURES ({result.total_failed})")
        print("-" * 60)

        failures = [r for r in result.detailed_results if not r.passed][:10]
        for f in failures:
            print(f"\n  [{f.category}] {f.prompt}")
            print(f"  Response: {f.response[:100]}...")
            for issue in f.issues:
                print(f"    - {issue}")

        if result.total_failed > 10:
            print(f"\n  ... and {result.total_failed - 10} more")

    print("\n" + "=" * 60)


def compare_models(result1: EvaluationResult, result2: EvaluationResult):
    """Compare two model evaluations."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(f"\n{'Metric':<30} {result1.model:<20} {result2.model:<20} Delta")
    print("-" * 90)

    # Overall
    delta = result1.consistency - result2.consistency
    sign = "+" if delta > 0 else ""
    print(f"{'Consistency':<30} {result1.consistency:.1f}%{'':<15} {result2.consistency:.1f}%{'':<15} {sign}{delta:.1f}%")

    # Categories
    for cat in result1.categories:
        if cat in result2.categories:
            r1 = result1.categories[cat].pass_rate
            r2 = result2.categories[cat].pass_rate
            delta = r1 - r2
            sign = "+" if delta > 0 else ""
            print(f"  {cat:<28} {r1:.0f}%{'':<17} {r2:.0f}%{'':<17} {sign}{delta:.0f}%")

    print("\n" + "=" * 60)


def save_results(result: EvaluationResult, filepath: Path):
    """Save evaluation results to JSON."""
    data = {
        "model": result.model,
        "timestamp": result.timestamp,
        "total_prompts": result.total_prompts,
        "total_passed": result.total_passed,
        "total_failed": result.total_failed,
        "consistency": result.consistency,
        "metrics": result.metrics,
        "categories": {
            name: {
                "description": cat.description,
                "total": cat.total,
                "passed": cat.passed,
                "failed": cat.failed,
                "pass_rate": cat.pass_rate,
            }
            for name, cat in result.categories.items()
        },
        "failures": [
            {
                "prompt": r.prompt,
                "response": r.response,
                "category": r.category,
                "issues": r.issues,
            }
            for r in result.detailed_results if not r.passed
        ]
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate O.L.I.V.I.A. personality consistency")
    parser.add_argument("--model", default="olivia-finetuned", help="Model to evaluate")
    parser.add_argument("--compare", help="Second model to compare against")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--no-failures", action="store_true", help="Don't show failure details")

    args = parser.parse_args()

    # Check model exists
    try:
        models = [m["name"] for m in ollama.list()["models"]]
        if args.model not in models and f"{args.model}:latest" not in models:
            print(f"WARNING: Model '{args.model}' may not be available")
            print(f"Available models: {', '.join(models[:5])}...")
    except Exception:
        pass

    # Run evaluation
    evaluator = PersonalityEvaluator(args.model, verbose=args.verbose)
    result = evaluator.evaluate()

    # Print report
    print_report(result, show_failures=not args.no_failures)

    # Save results
    if args.output:
        save_results(result, Path(args.output))
    else:
        # Default output
        output_path = Path(__file__).parent.parent / "evaluation_results.json"
        save_results(result, output_path)

    # Compare if requested
    if args.compare:
        print(f"\nEvaluating comparison model: {args.compare}")
        evaluator2 = PersonalityEvaluator(args.compare, verbose=args.verbose)
        result2 = evaluator2.evaluate()
        print_report(result2, show_failures=not args.no_failures)
        compare_models(result, result2)

    # Return exit code based on target
    return 0 if result.consistency >= 90 else 1


if __name__ == "__main__":
    sys.exit(main())
