"""DI for FastAPI routes.

Provides dependency injection for services with proper type hints.
Uses ServiceContainer internally but maintains backward compatibility
with dict-like `services` access.
"""

from typing import Annotated

from fastapi import Depends, HTTPException

from src.api.container import get_container

# Import concrete types for proper type hints
from src.api.services.llm_service import LLMService
from src.api.services.memory_service import MemoryService
from src.api.services.state_manager import StateManager
from src.api.services.stt_service import STTService
from src.api.services.tts_service import TTSService


class _ServicesProxy:
    """Proxy that provides dict-like access to container for backward compatibility.

    Allows existing code using `services["llm"]` or `services.get("llm")` to
    continue working without modification.
    """

    def get(self, name: str):
        """Get service by name."""
        return get_container().get(name)

    def __getitem__(self, name: str):
        """Dict-like read access."""
        return get_container()[name]

    def __setitem__(self, name: str, value):
        """Dict-like write access."""
        get_container()[name] = value

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator."""
        return name in get_container()

    def clear(self):
        """Clear all services."""
        get_container().clear()


# Backward compatibility: routes using `dependencies.services` still work
services = _ServicesProxy()


def get_service(name: str):
    """Get service by name, returns None if unavailable."""
    return get_container().get(name)


# =============================================================================
# DEPENDENCY PROVIDERS
# =============================================================================


async def get_llm_service() -> LLMService:
    """Dependency: LLM service."""
    svc = get_container().llm
    if not svc:
        raise HTTPException(status_code=503, detail="LLM service unavailable")
    return svc


async def get_memory_service() -> MemoryService:
    """Dependency: Memory service."""
    svc = get_container().memory
    if not svc:
        raise HTTPException(status_code=503, detail="Memory service unavailable")
    return svc


async def get_stt_service() -> STTService:
    """Dependency: STT service."""
    svc = get_container().stt
    if not svc:
        raise HTTPException(status_code=503, detail="STT service unavailable")
    return svc


async def get_tts_service() -> TTSService:
    """Dependency: TTS service."""
    svc = get_container().tts
    if not svc:
        raise HTTPException(status_code=503, detail="TTS service unavailable")
    return svc


async def get_state_manager() -> StateManager:
    """Dependency: State manager."""
    svc = get_container().state
    if not svc:
        raise HTTPException(status_code=503, detail="State manager unavailable")
    return svc


# =============================================================================
# TYPE ALIASES
# =============================================================================
# Use concrete types instead of `object` for proper IDE support and type checking

LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]
STTServiceDep = Annotated[STTService, Depends(get_stt_service)]
TTSServiceDep = Annotated[TTSService, Depends(get_tts_service)]
StateManagerDep = Annotated[StateManager, Depends(get_state_manager)]
