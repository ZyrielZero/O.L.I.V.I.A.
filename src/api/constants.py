"""API constants - centralized configuration values."""

import re
from dataclasses import dataclass
from typing import FrozenSet, List, Pattern

# =============================================================================
# TIMEOUTS (seconds)
# =============================================================================

@dataclass(frozen=True)
class Timeouts:
    """Timeout constants for service operations."""

    # Health checks
    HEALTH_CHECK: float = 2.0

    # Memory service
    MEMORY_OPERATION: float = 10.0
    MEMORY_INIT: float = 30.0
    MEMORY_HEALTH: float = 2.0

    # STT service
    STT_TRANSCRIBE: float = 30.0
    STT_INIT: float = 60.0

    # TTS service
    TTS_QUEUE_WAIT: float = 0.1
    TTS_SENTENCE_QUEUE: float = 2.0
    TTS_SYNTH_BASE: float = 20.0
    TTS_PLAYBACK: float = 60.0
    TTS_WORDS_PER_SECOND_FACTOR: float = 0.3


TIMEOUTS = Timeouts()


# =============================================================================
# WEB SEARCH CONFIG
# =============================================================================

@dataclass(frozen=True)
class WebSearchConfig:
    """Web search configuration constants."""

    MAX_RESULTS: int = 10
    MAX_SCRAPE_RESULTS: int = 5
    MAX_CONTENT_LENGTH: int = 8000
    MAX_TOTAL_CONTEXT: int = 30000
    SEARCH_TIMEOUT: int = 15
    SCRAPE_TIMEOUT: int = 12
    REQUEST_DELAY: float = 0.5


WEB_SEARCH = WebSearchConfig()


# =============================================================================
# CHAT PATTERNS
# =============================================================================

# Greeting patterns for O(1) lookup
GREETING_PATTERNS: FrozenSet[str] = frozenset([
    "hi", "hello", "hey", "thanks", "thank you"
])

# Pre-compiled greeting regex
GREETING_REGEX: Pattern = re.compile(
    r"^(hi|hello|hey|thanks|thank you)\s*[!?.]?$",
    re.IGNORECASE
)

# Pre-compiled search detection patterns
SEARCH_PATTERNS_COMPILED: List[Pattern] = [
    re.compile(r"(?:can you |please )?search (?:for |the web for )?(.+?)(?:\?|$)", re.IGNORECASE),
    re.compile(r"(?:can you |please )?look up (.+?)(?:\?|$)", re.IGNORECASE),
    re.compile(r"what (?:is|are|was|were) (.+?)(?:\?|$)", re.IGNORECASE),
    re.compile(r"who (?:is|are|was|were) (.+?)(?:\?|$)", re.IGNORECASE),
    re.compile(r"when (?:is|are|was|were|did) (.+?)(?:\?|$)", re.IGNORECASE),
    re.compile(r"where (?:is|are|was|were|can) (.+?)(?:\?|$)", re.IGNORECASE),
    re.compile(r"how (?:do|does|did|can|to) (.+?)(?:\?|$)", re.IGNORECASE),
]

# Search mode keywords
DEEP_SEARCH_WORDS: FrozenSet[str] = frozenset(["detailed", "comprehensive", "thorough"])
QUICK_SEARCH_WORDS: FrozenSet[str] = frozenset(["quick", "briefly"])


# =============================================================================
# SSE TEMPLATES
# =============================================================================

# Pre-built JSON templates for streaming responses
# Avoids json.dumps() overhead per token (~1.5us -> ~0.3us)
SSE_TOKEN_TEMPLATE: str = '{{"token": {}, "done": false}}'
SSE_DONE: str = '{"token": "", "done": true}'
SSE_ERROR_TEMPLATE: str = '{{"error": {}, "done": true}}'


# =============================================================================
# SERVICE KEYS
# =============================================================================

class ServiceKey:
    """Service registry keys (avoids string typos)."""

    LLM: str = "llm"
    MEMORY: str = "memory"
    STT: str = "stt"
    TTS: str = "tts"
    STATE: str = "state"
    WEB: str = "web"


# =============================================================================
# SENTENCE BUFFER
# =============================================================================

@dataclass(frozen=True)
class SentenceBufferConfig:
    """Sentence buffer configuration for TTS streaming."""

    MIN_LENGTH: int = 1  # allow single-word responses
    MAX_LENGTH: int = 500
    MIN_WORDS: int = 6
    MAX_WORDS: int = 30


SENTENCE_BUFFER = SentenceBufferConfig()


# =============================================================================
# OLLAMA DEFAULTS
# =============================================================================

@dataclass(frozen=True)
class OllamaDefaults:
    """Default values for Ollama LLM service."""

    MAX_WORKERS: int = 20
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 150


OLLAMA = OllamaDefaults()
