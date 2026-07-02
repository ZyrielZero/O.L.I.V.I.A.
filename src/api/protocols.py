"""Service protocols for type checking.

Defines formal Protocol interfaces that services implement.
Using Protocol (structural subtyping) allows existing classes to conform
without explicit inheritance - they just need to have the right methods.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    import numpy as np

    from src.api.services.state_manager import Session


# =============================================================================
# BASE SERVICE PROTOCOLS
# =============================================================================


@runtime_checkable
class ServiceProtocol(Protocol):
    """Base protocol all services should implement."""

    async def initialize(self) -> None:
        """Initialize the service. May raise ServiceInitializationError."""
        ...

    async def health_check(self) -> bool:
        """Return True if service is healthy, False otherwise."""
        ...

    def is_initialized(self) -> bool:
        """Return True if service has been initialized."""
        ...


@runtime_checkable
class CleanupServiceProtocol(ServiceProtocol, Protocol):
    """Protocol for services that require cleanup on shutdown."""

    async def cleanup(self) -> None:
        """Release resources held by the service."""
        ...


# =============================================================================
# LLM SERVICE PROTOCOL
# =============================================================================


@runtime_checkable
class LLMServiceProtocol(CleanupServiceProtocol, Protocol):
    """Protocol for LLM service."""

    model: str
    host: str
    system_prompt: str

    async def chat_stream(
        self,
        message: str,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response tokens."""
        ...

    async def clear_history(self) -> None:
        """Clear conversation history."""
        ...

    async def update_system_prompt(self, prompt: str) -> None:
        """Update the system prompt."""
        ...


# =============================================================================
# MEMORY SERVICE PROTOCOL
# =============================================================================


@runtime_checkable
class MemoryServiceProtocol(ServiceProtocol, Protocol):
    """Protocol for memory service (ChromaDB)."""

    persist_directory: str

    async def add_conversation(
        self, user_msg: str, ai_msg: str, auto_extract: bool = True
    ) -> Optional[List[str]]:
        """Store a conversation exchange, optionally extract facts."""
        ...

    async def get_relevant_context(self, query: str, n_results: int = 3) -> str:
        """Retrieve relevant context for a query."""
        ...

    async def query_memory(
        self, query: str, n_results: int = 3, mem_type: str = "all"
    ) -> List[str]:
        """Query the memory database."""
        ...

    async def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        ...


# =============================================================================
# STT SERVICE PROTOCOL
# =============================================================================


@runtime_checkable
class STTServiceProtocol(ServiceProtocol, Protocol):
    """Protocol for speech-to-text service (Whisper)."""

    model_size: str
    device: str
    compute_type: str

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe PCM audio bytes (16-bit, 16kHz, mono) to text."""
        ...

    async def transcribe_numpy(self, arr: "np.ndarray") -> str:
        """Transcribe float32 numpy array to text."""
        ...


# =============================================================================
# TTS SERVICE PROTOCOL
# =============================================================================


@runtime_checkable
class TTSServiceProtocol(CleanupServiceProtocol, Protocol):
    """Protocol for text-to-speech service (ChatterBox)."""

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to PCM audio bytes (16-bit LE)."""
        ...

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream synthesized audio chunks."""
        ...

    async def speak(self, text: str) -> None:
        """Synthesize and play audio through speakers."""
        ...

    async def stop(self) -> None:
        """Stop current speech (barge-in support)."""
        ...

    async def play_audio(self, audio: bytes) -> None:
        """Play PCM audio bytes."""
        ...

    async def get_status(self) -> Dict[str, Any]:
        """Get TTS status and metrics."""
        ...


# =============================================================================
# STATE MANAGER PROTOCOL
# =============================================================================


@runtime_checkable
class StateManagerProtocol(Protocol):
    """Protocol for state manager (no init/health_check needed - always available)."""

    async def create_session(self, session_id: Optional[str] = None) -> "Session":
        """Create a new session."""
        ...

    async def get_session(self, sid: str) -> Optional["Session"]:
        """Get session by ID."""
        ...

    async def update_session_state(self, sid: str, state: str) -> None:
        """Update session state."""
        ...

    async def cleanup_session(self, sid: str) -> None:
        """Clean up a session."""
        ...

    async def cleanup_stale_sessions(self, timeout_seconds: int = 3600) -> None:
        """Remove sessions inactive longer than timeout."""
        ...

    def get_active_sessions(self) -> int:
        """Get count of active sessions."""
        ...

    def get_all_session_ids(self) -> List[str]:
        """Get all session IDs."""
        ...
