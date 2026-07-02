"""Experimental memory features for O.L.I.V.I.A.

Contains:
- DreamingSystem: Memory consolidation during idle/shutdown
- LLMFactExtractor: Background fact extraction using LLM

These features are fully implemented but not yet integrated into the main application.
To use them, import directly from this module.
"""

from .dreaming import (
    DreamConfig,
    DreamingEngine,
    DreamReport,
    IdleDetector,
    create_dreaming_engine,
    get_dreaming_engine,
)
from .fact_extractor import (
    ExtractedFact,
    FactExtractorConfig,
    HybridFactExtractor,
    LLMFactExtractor,
    create_fact_extractor,
    get_fact_extractor,
)

__all__ = [
    # Dreaming
    "DreamingEngine",
    "DreamConfig",
    "DreamReport",
    "IdleDetector",
    "create_dreaming_engine",
    "get_dreaming_engine",
    # Fact extraction
    "LLMFactExtractor",
    "HybridFactExtractor",
    "FactExtractorConfig",
    "ExtractedFact",
    "create_fact_extractor",
    "get_fact_extractor",
]
