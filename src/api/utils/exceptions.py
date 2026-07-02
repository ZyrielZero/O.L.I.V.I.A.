"""API exceptions."""


class ServiceUnavailableError(Exception):
    """Service not available."""

    pass


class ServiceInitializationError(Exception):
    """Service failed to init."""

    pass


class TranscriptionError(Exception):
    """STT failed."""

    pass


class SynthesisError(Exception):
    """TTS failed."""

    pass


class LLMError(Exception):
    """LLM generation failed."""

    pass


class MemoryServiceError(Exception):
    """Memory op failed."""

    pass
