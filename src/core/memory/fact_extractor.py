"""LLM-based fact extraction for O.L.I.V.I.A.

Background extraction using LLM with regex fallback.

Performance Optimizations:
- Pre-compiled regex patterns at module level
- frozenset for O(1) hallucination lookups
- Compiled regex for hallucination detection
- Batch ChromaDB duplicate queries
- Set-based model lookup instead of linear search
"""

import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.utils.logger import get_logger

    log = get_logger("facts")
except ImportError:
    import logging

    log = logging.getLogger("facts")

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# =============================================================================
# PRE-COMPILED REGEX PATTERNS (module-level, compiled once)
# O(n) compile per call -> O(1) reuse
# =============================================================================
_QUICK_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"(?:my name is|i'm called|call me) ([a-zA-Z]+)", re.IGNORECASE), "personal", "User's name is {0}"),
    (re.compile(r"(?:i am|i'm) (\d+)(?: years old)?", re.IGNORECASE), "personal", "User is {0} years old"),
    (re.compile(r"(?:i live in|i'm from) ([^.,!]+)", re.IGNORECASE), "personal", "User lives in {0}"),
    (re.compile(r"(?:i work (?:at|as|for)) ([^.,!]+)", re.IGNORECASE), "work", "User works {0}"),
    (
        re.compile(r"(?:my (?:wife|husband|partner|spouse)'s name is) ([a-zA-Z]+)", re.IGNORECASE),
        "relationship",
        "User's spouse is named {0}",
    ),
    (
        re.compile(r"(?:my (?:dog|cat|pet)'s name is) ([a-zA-Z]+)", re.IGNORECASE),
        "relationship",
        "User has a pet named {0}",
    ),
    (re.compile(r"(?:i (?:really )?(?:like|love|enjoy)) ([^.,!]+)", re.IGNORECASE), "preference", "User likes {0}"),
    (re.compile(r"(?:i (?:hate|dislike|can't stand)) ([^.,!]+)", re.IGNORECASE), "preference", "User dislikes {0}"),
    (re.compile(r"(?:my birthday is|i was born on) ([^.,!]+)", re.IGNORECASE), "personal", "User's birthday is {0}"),
    (
        re.compile(r"(?:i have (?:a )?(?:meeting|appointment)) (?:on|at) ([^.,!]+)", re.IGNORECASE),
        "schedule",
        "User has appointment {0}",
    ),
]

# =============================================================================
# HALLUCINATION DETECTION (frozenset for O(1) lookup + compiled regex)
# O(n) list iteration -> O(1) set lookup + O(1) regex match
# =============================================================================
_HALLUCINATED_EXACT: frozenset = frozenset({
    "user's sister's birthday is march 15th",
    "user's sister's birthday is march 15",
    "user likes...",
    "user's name is john",
})

# Single compiled regex for pattern-based hallucination detection
# Matches any of the hallucinated patterns in one pass
_HALLUCINATED_PATTERN: re.Pattern = re.compile(
    r"sister's birthday|march 15|example fact|placeholder|john doe|jane doe",
    re.IGNORECASE
)

# JSON extraction pattern (compiled once)
_JSON_PATTERN: re.Pattern = re.compile(r"\{[\s\S]*\}")


@dataclass
class FactExtractorConfig:
    """Configuration for fact extraction."""

    # Model settings - use a small model for efficiency
    model: str = "olivia-finetuned"  # Tiny, fast model for extraction
    fallback_model: str = "olivia-finetuned"  # Fall back to main model

    # Processing
    max_queue_size: int = 100
    batch_delay_seconds: float = 1.0  # Delay between extractions
    min_confidence: float = 0.7  # Minimum confidence to store fact

    # Duplicate detection
    similarity_threshold: float = 0.3  # ChromaDB distance threshold


@dataclass
class ExtractedFact:
    """A fact extracted from conversation."""

    fact: str
    category: str
    confidence: float
    source_user_msg: str
    source_ai_msg: str
    timestamp: datetime = field(default_factory=datetime.now)


EXTRACTION_PROMPT = """Extract factual information about the user from this conversation exchange.

USER: {user_msg}
ASSISTANT: {ai_msg}

Return ONLY valid JSON in this exact format:
{{
  "facts": [
    {{
      "category": "personal|preference|work|schedule|relationship|health|hobby",
      "fact": "User's sister's birthday is March 15th",
      "confidence": 0.9
    }}
  ]
}}

Categories:
- personal: Name, age, location, occupation
- preference: Likes, dislikes, favorites
- work: Job, projects, colleagues, deadlines
- schedule: Appointments, plans, routines
- relationship: Family, friends, pets
- health: Conditions, medications, fitness
- hobby: Games, activities, interests

Rules:
- Only extract facts EXPLICITLY stated or STRONGLY implied
- Use third person ("User likes..." not "I like...")
- Include confidence score (0.0-1.0)
- Return empty array if no facts found
- DO NOT infer or assume beyond text
- Be conservative - when in doubt, don't extract"""


class LLMFactExtractor:
    """Background fact extraction using LLM.

    Processes conversation pairs asynchronously and stores
    extracted facts to memory.
    """

    def __init__(
        self,
        memory_db: Any,  # SmartMemoryDB
        config: Optional[FactExtractorConfig] = None,
    ):
        self.memory = memory_db
        self.config = config or FactExtractorConfig()

        # Queue for background processing
        self._queue: queue.Queue = queue.Queue(maxsize=self.config.max_queue_size)

        # Worker thread
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Stats
        self.facts_extracted = 0
        self.conversations_processed = 0
        self.errors = 0

        # Model availability (cached)
        self._model_checked = False
        self._model_available = False
        self._available_models_set: Optional[frozenset] = None  # O(1) lookup cache

        log.info(f"LLMFactExtractor initialized (model: {self.config.model})")

    def start(self):
        """Start the background worker."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        log.info("Fact extraction worker started")

    def stop(self):
        """Stop the background worker."""
        self._running = False
        self._queue.put(None)  # Signal to stop
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        log.info("Fact extraction worker stopped")

    def queue_extraction(self, user_msg: str, ai_msg: str):
        """Queue a conversation for background fact extraction.

        Non-blocking - returns immediately.
        """
        if not self._running:
            self.start()

        # Skip very short messages
        if len(user_msg) < 10:
            return

        try:
            self._queue.put_nowait((user_msg, ai_msg, datetime.now()))
            log.debug(f"Queued for extraction: {user_msg[:50]}...")
        except queue.Full:
            log.warning("Extraction queue full, skipping")

    def _worker_loop(self):
        """Main worker loop - processes queue items."""
        log.info("Fact extraction worker running...")

        while self._running:
            try:
                # Get item with timeout
                item = self._queue.get(timeout=1.0)

                if item is None:
                    break

                user_msg, ai_msg, timestamp = item

                # Process
                facts = self._extract_facts(user_msg, ai_msg)

                # Batch duplicate check for all valid facts
                # O(n) individual queries -> O(1) batch query
                valid_facts = [f for f in facts if f.confidence >= self.config.min_confidence]
                if valid_facts:
                    fact_texts = [f.fact for f in valid_facts]
                    is_duplicate = self._batch_check_duplicates(fact_texts)

                    for i, fact in enumerate(valid_facts):
                        if not is_duplicate[i]:
                            self._store_fact(fact)
                            self.facts_extracted += 1
                            log.info(f"Extracted: {fact.fact[:60]}...")

                self.conversations_processed += 1

                # Rate limiting
                time.sleep(self.config.batch_delay_seconds)

            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Extraction worker error: {e}")
                self.errors += 1
                time.sleep(1.0)

    def _check_model(self) -> bool:
        """Check if extraction model is available."""
        if self._model_checked:
            return self._model_available

        self._model_checked = True

        try:
            model_names = self._get_available_models()
            # Cache as frozenset for O(1) lookup
            self._available_models_set = frozenset(model_names)
            log.info(f"Found models: {model_names}")

            fallback_base = self.config.fallback_model.split(":")[0]

            if self._model_matches_fast(self.config.model):
                self._model_available = True
            elif self._model_matches_fast(fallback_base):
                self.config.model = self.config.fallback_model
                self._model_available = True
                log.info(f"Using fallback model: {self.config.fallback_model}")
            else:
                log.warning(f"Extraction model '{self.config.model}' not found")
                self._model_available = False

            if self._model_available:
                log.info(f"Fact extraction active using: {self.config.model}")
            return self._model_available

        except Exception as e:
            log.error(f"Model check failed: {e}")
            self._model_available = True
            return True

    def _get_available_models(self) -> List[str]:
        """Get list of available model names from Ollama."""
        resp = ollama.list()
        model_list = (
            getattr(resp, "models", None)
            or (resp.get("models", []) if isinstance(resp, dict) else resp)
            or []
        )
        names = []
        for m in model_list:
            name = (
                getattr(m, "model", None)
                or getattr(m, "name", None)
                or (m.get("name") or m.get("model") if isinstance(m, dict) else None)
                or (m if isinstance(m, str) else None)
            )
            if name:
                names.append(name)
        return names

    def _model_matches_fast(self, target: str) -> bool:
        """Check if target model exists in available models.

        O(n) linear search -> O(1) set lookup
        """
        if self._available_models_set is None:
            return False

        # Direct O(1) lookup
        if target in self._available_models_set:
            return True

        # Check base name match (still needed for partial matches)
        target_base = target.split(":")[0]
        return any(target_base in name for name in self._available_models_set)

    def _extract_facts(self, user_msg: str, ai_msg: str) -> List[ExtractedFact]:
        """Extract facts using LLM."""
        if not OLLAMA_AVAILABLE:
            return []

        if not self._check_model():
            return []

        try:
            prompt = EXTRACTION_PROMPT.format(user_msg=user_msg, ai_msg=ai_msg)

            response = ollama.chat(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": 300,
                    "temperature": 0.1,  # Low temp for consistent JSON
                },
            )

            content = response["message"]["content"].strip()

            # Parse JSON from response
            facts = self._parse_json_response(content, user_msg, ai_msg)
            return facts

        except Exception as e:
            log.debug(f"Extraction failed: {e}")
            return []

    def _parse_json_response(self, content: str, user_msg: str, ai_msg: str) -> List[ExtractedFact]:
        """Parse JSON response from LLM."""
        facts = []

        try:
            # Find JSON in response (pre-compiled pattern)
            json_match = _JSON_PATTERN.search(content)
            if not json_match:
                return []

            data = json.loads(json_match.group())

            for fact_data in data.get("facts", []):
                if not isinstance(fact_data, dict):
                    continue

                fact_text = fact_data.get("fact", "").strip()
                if not fact_text or len(fact_text) < 5:
                    continue

                # Check for hallucinations using optimized detection
                # O(n) list iteration -> O(1) set + regex lookup
                if self._is_hallucinated_fact(fact_text):
                    log.debug(f"Ignored hallucinated example fact: {fact_text}")
                    continue

                facts.append(
                    ExtractedFact(
                        fact=fact_text,
                        category=fact_data.get("category", "general"),
                        confidence=float(fact_data.get("confidence", 0.5)),
                        source_user_msg=user_msg,
                        source_ai_msg=ai_msg,
                    )
                )

        except json.JSONDecodeError:
            log.debug("Failed to parse JSON from LLM response")
        except Exception as e:
            log.debug(f"Parse error: {e}")

        return facts

    def _is_hallucinated_fact(self, fact: str) -> bool:
        """Check if fact is a hallucinated example from the prompt.

        O(n) any() iteration -> O(1) frozenset lookup + O(1) regex search
        """
        fact_lower = fact.lower()

        # O(1) exact match check
        if fact_lower in _HALLUCINATED_EXACT:
            return True

        # O(1) pattern match (single compiled regex)
        return bool(_HALLUCINATED_PATTERN.search(fact_lower))

    def _batch_check_duplicates(self, facts: List[str]) -> List[bool]:
        """Batch duplicate check for multiple facts.

        O(n) individual queries -> O(1) batch query
        Uses ChromaDB's batch query capability.
        """
        if not facts:
            return []

        try:
            if self.memory.facts.count() == 0:
                return [False] * len(facts)

            # Single batch query instead of n individual queries
            results = self.memory.facts.query(query_texts=facts, n_results=1)

            duplicates = []
            if results and results.get("distances"):
                for fact_distances in results["distances"]:
                    if fact_distances and len(fact_distances) > 0:
                        duplicates.append(fact_distances[0] < self.config.similarity_threshold)
                    else:
                        duplicates.append(False)
            else:
                duplicates = [False] * len(facts)

            return duplicates

        except Exception as e:
            log.debug(f"Batch duplicate check failed: {e}")
            return [False] * len(facts)

    def _is_duplicate(self, fact: str) -> bool:
        """Check if fact already exists (semantic similarity)."""
        try:
            if self.memory.facts.count() == 0:
                return False

            results = self.memory.facts.query(query_texts=[fact], n_results=1)

            if results and results.get("distances"):
                distances = results["distances"]
                if distances and len(distances) > 0 and len(distances[0]) > 0:
                    if distances[0][0] < self.config.similarity_threshold:
                        log.debug(f"Duplicate detected: {fact[:40]}...")
                        return True

            return False

        except Exception as e:
            log.debug(f"Duplicate check failed: {e}")
            return False

    def _store_fact(self, fact: ExtractedFact):
        """Store extracted fact in memory."""
        try:
            self.memory.add_fact(fact.fact, fact.category)
        except Exception as e:
            log.error(f"Failed to store fact: {e}")
            self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            "facts_extracted": self.facts_extracted,
            "conversations_processed": self.conversations_processed,
            "errors": self.errors,
            "queue_size": self._queue.qsize(),
            "model": self.config.model,
            "running": self._running,
        }

    def extract_now(self, user_msg: str, ai_msg: str) -> List[ExtractedFact]:
        """Synchronous extraction (for testing/debugging).

        Bypasses queue and extracts immediately.
        """
        return self._extract_facts(user_msg, ai_msg)


class HybridFactExtractor:
    """Combines LLM extraction with regex fallback.

    Uses regex for quick extraction, LLM for deeper analysis.
    """

    def __init__(self, memory_db: Any, config: Optional[FactExtractorConfig] = None):
        self.memory = memory_db
        self.config = config or FactExtractorConfig()
        self.llm_extractor = LLMFactExtractor(memory_db, config)

    def start(self):
        """Start the LLM extractor worker."""
        self.llm_extractor.start()

    def stop(self):
        """Stop the LLM extractor worker."""
        self.llm_extractor.stop()

    def extract(self, user_msg: str, ai_msg: str):
        """Extract facts using hybrid approach.

        1. Quick regex extraction (immediate)
        2. Queue for LLM extraction (background)
        """
        # Quick regex extraction
        quick_facts = self._regex_extract(user_msg)

        if quick_facts:
            # Batch duplicate check instead of per-fact queries
            # O(n) queries -> O(1) batch query
            facts_only = [fact for fact, _ in quick_facts]

            # Use memory's batch method if available, otherwise fall back
            if hasattr(self.memory, "batch_check_duplicates"):
                is_duplicate = self.memory.batch_check_duplicates(facts_only)
            else:
                is_duplicate = [self.memory.is_duplicate_fact(f) for f in facts_only]

            for i, (fact, category) in enumerate(quick_facts):
                if not is_duplicate[i]:
                    self.memory.add_fact(fact, category)
                    log.info(f"Quick extract: {fact[:50]}...")

        # Queue for deeper LLM extraction
        self.llm_extractor.queue_extraction(user_msg, ai_msg)

    def _regex_extract(self, text: str) -> List[Tuple[str, str]]:
        """Quick regex-based extraction.

        Uses pre-compiled patterns for O(1) pattern access.
        """
        facts = []
        text_lower = text.lower()

        # Use pre-compiled patterns (O(1) access vs O(n) compile per call)
        for pattern, category, template in _QUICK_PATTERNS:
            matches = pattern.findall(text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                match = match.strip()
                if len(match) > 2 and len(match) < 100:
                    fact = template.format(match.title())
                    facts.append((fact, category))

        return facts

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = self.llm_extractor.get_stats()
        stats["type"] = "hybrid"
        return stats


_extractor: Optional[HybridFactExtractor] = None


def get_fact_extractor() -> Optional[HybridFactExtractor]:
    """Get the global fact extractor."""
    return _extractor


def create_fact_extractor(
    memory_db: Any, config: Optional[FactExtractorConfig] = None
) -> HybridFactExtractor:
    """Create and configure the fact extractor."""
    global _extractor
    _extractor = HybridFactExtractor(memory_db, config)
    return _extractor


if __name__ == "__main__":
    print("Testing Fact Extractor regex patterns...")

    class MockMemory:
        """Minimal SmartMemory stand-in for the regex smoke test."""

        def is_duplicate_fact(self, f):
            """Always report the fact as new."""
            return False

        def batch_check_duplicates(self, facts):
            """Report every fact as new."""
            return [False] * len(facts)

        def add_fact(self, f, c):
            """Print the fact instead of storing it."""
            print(f"  [{c}] {f}")

    extractor = HybridFactExtractor(MockMemory())
    test_msgs = [
        "My name is John and I work at Google.",
        "I really love playing chess on weekends.",
        "My dog's name is Max and he's a golden retriever.",
    ]
    for msg in test_msgs:
        print(f"\nInput: {msg}")
        for fact, cat in extractor._regex_extract(msg):
            print(f"  [{cat}] {fact}")
