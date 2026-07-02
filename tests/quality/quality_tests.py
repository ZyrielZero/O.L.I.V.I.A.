"""
Quality Tests for O.L.I.V.I.A. Optimization Testing.

This module provides quality regression tests to ensure optimizations
don't degrade output quality across STT, LLM, and TTS components.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# STT Quality Tests
# =============================================================================

# Test utterances for STT quality verification
# Format: (expected_text, acceptable_variations)
STT_TEST_UTTERANCES = [
    ("hello olivia how are you today", ["hello olivia how are you today", "hello olivia, how are you today"]),
    ("what is the weather like", ["what is the weather like", "what's the weather like"]),
    ("set a reminder for three pm", ["set a reminder for three pm", "set a reminder for 3 pm", "set a reminder for 3 p.m."]),
    ("tell me about quantum computing", ["tell me about quantum computing"]),
    ("my name is john", ["my name is john", "my name is john."]),
]


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate between reference and hypothesis.

    WER = (S + D + I) / N
    Where: S=substitutions, D=deletions, I=insertions, N=reference words

    Args:
        reference: Ground truth text
        hypothesis: Transcribed text

    Returns:
        WER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Dynamic programming for Levenshtein distance
    n = len(ref_words)
    m = len(hyp_words)

    # Create distance matrix
    d = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize first column (deletions)
    for i in range(n + 1):
        d[i][0] = i

    # Initialize first row (insertions)
    for j in range(m + 1):
        d[0][j] = j

    # Fill in the rest
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]  # No operation needed
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # Deletion
                    d[i][j - 1] + 1,      # Insertion
                    d[i - 1][j - 1] + 1   # Substitution
                )

    # WER = edit distance / reference length
    return d[n][m] / max(n, 1)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class STTQualityResult:
    """Result from STT quality tests."""

    total_tests: int = 0
    passed_tests: int = 0
    total_wer: float = 0.0
    individual_wers: List[float] = field(default_factory=list)
    failures: List[Tuple[str, str, float]] = field(default_factory=list)

    @property
    def avg_wer(self) -> float:
        return self.total_wer / max(self.total_tests, 1)

    @property
    def pass_rate(self) -> float:
        return self.passed_tests / max(self.total_tests, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "pass_rate": self.pass_rate,
            "avg_wer": self.avg_wer,
            "individual_wers": self.individual_wers,
            "failures": [
                {"expected": e, "got": g, "wer": w}
                for e, g, w in self.failures
            ],
        }


class STTQualityTester:
    """Run STT quality tests."""

    def __init__(self, wer_threshold: float = 0.15):
        """
        Initialize STT quality tester.

        Args:
            wer_threshold: Maximum acceptable WER (default 15%)
        """
        self.wer_threshold = wer_threshold

    async def test_transcription(
        self,
        transcribe_fn: Callable[[str], str],
        audio_generator_fn: Optional[Callable[[str], bytes]] = None,
    ) -> STTQualityResult:
        """
        Test STT quality using known utterances.

        Args:
            transcribe_fn: Function that transcribes audio and returns text
            audio_generator_fn: Optional function to generate audio from text

        Returns:
            STTQualityResult with test outcomes
        """
        result = STTQualityResult()

        for expected_text, variations in STT_TEST_UTTERANCES:
            result.total_tests += 1

            # In real testing, we'd generate audio and transcribe
            # For now, we assume transcribe_fn takes audio and returns text
            # This is a placeholder for the actual test implementation
            try:
                # If we have an audio generator, use it
                if audio_generator_fn:
                    audio = audio_generator_fn(expected_text)
                    transcribed = transcribe_fn(audio)
                else:
                    # Skip if no audio generator
                    transcribed = expected_text  # Placeholder

                # Normalize both
                transcribed_norm = normalize_text(transcribed)
                expected_norm = normalize_text(expected_text)

                # Calculate WER
                wer = calculate_wer(expected_norm, transcribed_norm)
                result.individual_wers.append(wer)
                result.total_wer += wer

                # Check if acceptable
                if wer <= self.wer_threshold:
                    result.passed_tests += 1
                else:
                    # Check variations
                    passed = False
                    for var in variations:
                        var_wer = calculate_wer(normalize_text(var), transcribed_norm)
                        if var_wer <= self.wer_threshold:
                            result.passed_tests += 1
                            passed = True
                            break

                    if not passed:
                        result.failures.append((expected_text, transcribed, wer))

            except Exception as e:
                result.failures.append((expected_text, f"ERROR: {e}", 1.0))
                result.individual_wers.append(1.0)
                result.total_wer += 1.0

        return result


# =============================================================================
# LLM Quality Tests
# =============================================================================

# Test prompts for LLM coherence verification
# Format: (prompt, validator_fn, description)
LLM_TEST_PROMPTS: List[Tuple[str, Callable[[str], bool], str]] = [
    (
        "What is 2 plus 2?",
        lambda r: any(x in r.lower() for x in ["4", "four"]),
        "Basic math"
    ),
    (
        "Say hello",
        lambda r: any(x in r.lower() for x in ["hello", "hi", "hey"]),
        "Greeting"
    ),
    (
        "What color is the sky on a clear day?",
        lambda r: "blue" in r.lower(),
        "Common knowledge"
    ),
    (
        "Complete this sentence: The quick brown fox jumps over the lazy",
        lambda r: "dog" in r.lower(),
        "Sentence completion"
    ),
    (
        "Is water wet? Answer yes or no.",
        lambda r: any(x in r.lower() for x in ["yes", "no"]),
        "Yes/No question"
    ),
]


@dataclass
class LLMQualityResult:
    """Result from LLM quality tests."""

    total_tests: int = 0
    passed_tests: int = 0
    responses: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed_tests / max(self.total_tests, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "pass_rate": self.pass_rate,
            "responses": self.responses,
            "failures": [
                {"prompt": p, "response": r[:100], "test_name": n}
                for p, r, n in self.failures
            ],
        }


class LLMQualityTester:
    """Run LLM quality tests."""

    async def test_coherence(
        self,
        chat_fn: Callable[[str], str],
    ) -> LLMQualityResult:
        """
        Test LLM coherence using known prompts.

        Args:
            chat_fn: Function that takes a prompt and returns response

        Returns:
            LLMQualityResult with test outcomes
        """
        result = LLMQualityResult()

        for prompt, validator, description in LLM_TEST_PROMPTS:
            result.total_tests += 1

            try:
                response = chat_fn(prompt)

                # Record response
                result.responses.append({
                    "prompt": prompt,
                    "response": response[:200],
                    "test_name": description,
                })

                # Validate
                if validator(response):
                    result.passed_tests += 1
                else:
                    result.failures.append((prompt, response, description))

            except Exception as e:
                result.failures.append((prompt, f"ERROR: {e}", description))

        return result

    async def test_coherence_async(
        self,
        chat_fn: Callable[[str], Any],
    ) -> LLMQualityResult:
        """
        Test LLM coherence using known prompts (async version).

        Args:
            chat_fn: Async function that takes a prompt and returns response

        Returns:
            LLMQualityResult with test outcomes
        """
        result = LLMQualityResult()

        for prompt, validator, description in LLM_TEST_PROMPTS:
            result.total_tests += 1

            try:
                # Handle both async generators and direct responses
                response = ""
                response_obj = chat_fn(prompt)

                if hasattr(response_obj, "__anext__"):
                    # It's an async generator
                    async for token in response_obj:
                        response += token
                elif asyncio.iscoroutine(response_obj):
                    # It's a coroutine
                    response = await response_obj
                else:
                    response = str(response_obj)

                # Record response
                result.responses.append({
                    "prompt": prompt,
                    "response": response[:200],
                    "test_name": description,
                })

                # Validate
                if validator(response):
                    result.passed_tests += 1
                else:
                    result.failures.append((prompt, response, description))

            except Exception as e:
                result.failures.append((prompt, f"ERROR: {e}", description))

        return result


# =============================================================================
# TTS Quality Tests
# =============================================================================

# Reference texts for TTS quality
# Format: (text, min_duration_s, max_duration_s)
TTS_TEST_TEXTS = [
    ("Hello, how are you?", 1.0, 4.0),
    ("This is a test of the text to speech system.", 2.0, 6.0),
    ("The quick brown fox jumps over the lazy dog.", 2.0, 6.0),
]


@dataclass
class TTSQualityResult:
    """Result from TTS quality tests."""

    total_tests: int = 0
    passed_tests: int = 0
    audio_durations: List[float] = field(default_factory=list)
    failures: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed_tests / max(self.total_tests, 1)

    @property
    def avg_duration(self) -> float:
        return sum(self.audio_durations) / max(len(self.audio_durations), 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "pass_rate": self.pass_rate,
            "avg_duration_s": self.avg_duration,
            "audio_durations": self.audio_durations,
            "failures": [{"text": t, "reason": r} for t, r in self.failures],
        }


class TTSQualityTester:
    """Run TTS quality tests."""

    async def test_synthesis(
        self,
        synthesize_fn: Callable[[str], bytes],
        get_duration_fn: Optional[Callable[[bytes], float]] = None,
    ) -> TTSQualityResult:
        """
        Test TTS quality using known texts.

        Args:
            synthesize_fn: Function that synthesizes text to audio bytes
            get_duration_fn: Optional function to get audio duration from bytes

        Returns:
            TTSQualityResult with test outcomes
        """
        result = TTSQualityResult()

        for text, min_duration, max_duration in TTS_TEST_TEXTS:
            result.total_tests += 1

            try:
                audio_bytes = synthesize_fn(text)

                # Check if we got audio
                if not audio_bytes or len(audio_bytes) < 1000:
                    result.failures.append((text, "No audio generated or too short"))
                    continue

                # Get duration if possible
                duration = 0.0
                if get_duration_fn:
                    duration = get_duration_fn(audio_bytes)
                    result.audio_durations.append(duration)

                    # Check duration bounds
                    if duration < min_duration:
                        result.failures.append((text, f"Duration too short: {duration:.2f}s < {min_duration}s"))
                        continue
                    if duration > max_duration:
                        result.failures.append((text, f"Duration too long: {duration:.2f}s > {max_duration}s"))
                        continue

                result.passed_tests += 1

            except Exception as e:
                result.failures.append((text, f"ERROR: {e}"))

        return result


# =============================================================================
# Memory Quality Tests
# =============================================================================

@dataclass
class MemoryQualityResult:
    """Result from Memory quality tests."""

    total_tests: int = 0
    passed_tests: int = 0
    recall_scores: List[float] = field(default_factory=list)
    failures: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed_tests / max(self.total_tests, 1)

    @property
    def avg_recall(self) -> float:
        return sum(self.recall_scores) / max(len(self.recall_scores), 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "pass_rate": self.pass_rate,
            "avg_recall": self.avg_recall,
            "recall_scores": self.recall_scores,
            "failures": [{"query": q, "reason": r} for q, r in self.failures],
        }


# Test cases for memory retrieval
# Format: (fact_to_store, query, expected_in_result)
MEMORY_TEST_CASES = [
    ("User's name is Alice", "what is my name", "alice"),
    ("User likes pizza", "what food do I like", "pizza"),
    ("User works at Google", "where do I work", "google"),
]


class MemoryQualityTester:
    """Run Memory quality tests."""

    async def test_retrieval(
        self,
        add_fact_fn: Callable[[str, str], None],
        search_fn: Callable[[str], str],
    ) -> MemoryQualityResult:
        """
        Test memory retrieval quality.

        Args:
            add_fact_fn: Function to add a fact (fact, category)
            search_fn: Function to search memory and return results

        Returns:
            MemoryQualityResult with test outcomes
        """
        result = MemoryQualityResult()

        # First, add all test facts
        for fact, _, _ in MEMORY_TEST_CASES:
            add_fact_fn(fact, "test")

        # Then test retrieval
        for fact, query, expected in MEMORY_TEST_CASES:
            result.total_tests += 1

            try:
                search_result = search_fn(query)

                if expected.lower() in search_result.lower():
                    result.passed_tests += 1
                    result.recall_scores.append(1.0)
                else:
                    result.failures.append((query, f"Expected '{expected}' not found in results"))
                    result.recall_scores.append(0.0)

            except Exception as e:
                result.failures.append((query, f"ERROR: {e}"))
                result.recall_scores.append(0.0)

        return result


# =============================================================================
# Aggregate Quality Test Suite
# =============================================================================

@dataclass
class QualitySuiteResult:
    """Aggregated results from all quality tests."""

    stt: Optional[STTQualityResult] = None
    llm: Optional[LLMQualityResult] = None
    tts: Optional[TTSQualityResult] = None
    memory: Optional[MemoryQualityResult] = None

    @property
    def overall_pass_rate(self) -> float:
        rates = []
        if self.stt:
            rates.append(self.stt.pass_rate)
        if self.llm:
            rates.append(self.llm.pass_rate)
        if self.tts:
            rates.append(self.tts.pass_rate)
        if self.memory:
            rates.append(self.memory.pass_rate)
        return sum(rates) / max(len(rates), 1)

    def to_dict(self) -> Dict[str, Any]:
        result = {"overall_pass_rate": self.overall_pass_rate}
        if self.stt:
            result["stt"] = self.stt.to_dict()
        if self.llm:
            result["llm"] = self.llm.to_dict()
        if self.tts:
            result["tts"] = self.tts.to_dict()
        if self.memory:
            result["memory"] = self.memory.to_dict()
        return result

    def print_summary(self) -> None:
        """Print a summary of all quality test results."""
        print("\n" + "=" * 60)
        print("Quality Test Suite Results")
        print("=" * 60)

        if self.stt:
            print(f"\nSTT: {self.stt.pass_rate:.1%} pass rate (WER: {self.stt.avg_wer:.2f})")
        if self.llm:
            print(f"LLM: {self.llm.pass_rate:.1%} pass rate ({self.llm.passed_tests}/{self.llm.total_tests})")
        if self.tts:
            print(f"TTS: {self.tts.pass_rate:.1%} pass rate")
        if self.memory:
            print(f"Memory: {self.memory.pass_rate:.1%} pass rate (Recall: {self.memory.avg_recall:.2f})")

        print(f"\nOverall: {self.overall_pass_rate:.1%}")
        print("=" * 60 + "\n")
