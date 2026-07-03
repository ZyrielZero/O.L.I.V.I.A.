"""Typed service container for dependency injection.

Replaces the global `services = {}` dict with a typed dataclass container
while maintaining backward compatibility via dict-like access methods.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.api.services.llm_service import LLMService
    from src.api.services.memory_service import MemoryService
    from src.api.services.state_manager import StateManager
    from src.api.services.stt_service import STTService
    from src.api.services.tts_service import TTSService
    from src.core.memory.dreaming import DreamingEngine
    from src.core.memory.fact_extractor import HybridFactExtractor


@dataclass
class ServiceContainer:
    """Typed service container.

    Provides typed access to services while supporting lazy initialization.
    All services are Optional since they may not be loaded yet (lazy-loading).

    Usage:
        container = get_container()
        container.llm = LLMService(...)
        if container.llm:
            await container.llm.chat_stream(...)

    Backward compatibility:
        container["llm"] = service  # dict-like assignment
        service = container.get("llm")  # dict-like access
    """

    llm: Optional["LLMService"] = None
    memory: Optional["MemoryService"] = None
    stt: Optional["STTService"] = None
    tts: Optional["TTSService"] = None
    state: Optional["StateManager"] = None
    dreaming: Optional["DreamingEngine"] = None
    fact_extractor: Optional["HybridFactExtractor"] = None

    def get(self, name: str):
        """Get service by name (dict-like access for backward compatibility).

        Args:
            name: Service key ("llm", "memory", "stt", "tts", "state")

        Returns:
            Service instance or None if not loaded.
        """
        return getattr(self, name, None)

    def __setitem__(self, name: str, value):
        """Dict-like assignment for backward compatibility."""
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            raise KeyError(f"Unknown service: {name}")

    def __getitem__(self, name: str):
        """Dict-like access for backward compatibility."""
        if hasattr(self, name):
            return getattr(self, name)
        raise KeyError(f"Unknown service: {name}")

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator."""
        return hasattr(self, name) and getattr(self, name) is not None

    def clear(self):
        """Clear all services (useful for testing)."""
        self.llm = None
        self.memory = None
        self.stt = None
        self.tts = None
        self.state = None
        self.dreaming = None
        self.fact_extractor = None

    def is_critical_ready(self) -> bool:
        """Check if critical services (LLM, Memory) are initialized."""
        return (
            self.llm is not None
            and self.llm.is_initialized()
            and self.memory is not None
            and self.memory.is_initialized()
        )


# Module-level singleton
_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """Get or create the service container singleton."""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def reset_container() -> None:
    """Reset container to fresh state (for testing only)."""
    global _container
    if _container:
        _container.clear()
    _container = None
