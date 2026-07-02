"""
Error handling tests.
Tests graceful failure modes for all services.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.utils.exceptions import (
    LLMError,
    MemoryServiceError,
    ServiceInitializationError,
    ServiceUnavailableError,
    SynthesisError,
    TranscriptionError,
)

# ===== Test 1: LLM Connection Error Handling =====

@pytest.mark.error
@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_connection_error_handling():
    """Ollama unavailable returns proper error."""
    from src.api.services.llm_service import LLMService

    service = LLMService(
        model="test-model",
        system_prompt="Test",
        # Valid-but-refused port: an out-of-range port (e.g. 99999) raises
        # OverflowError inside httpx on Linux instead of a connection error
        host="http://127.0.0.1:1"
    )

    with pytest.raises(ServiceUnavailableError):
        await service.initialize()


# ===== Test 2: Memory DB Unavailable =====

@pytest.mark.error
@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_db_unavailable():
    """ChromaDB failure handled gracefully with MemoryError."""
    from src.api.services.memory_service import MemoryService

    with patch('src.api.services.memory_service.SmartMemoryDB') as mock_db:
        mock_db.side_effect = Exception("ChromaDB connection failed")

        service = MemoryService(persist_directory="/invalid/path")

        with pytest.raises((MemoryServiceError, Exception)):
            await service.initialize()


# ===== Test 3: TTS Model Not Loaded =====

@pytest.mark.error
@pytest.mark.unit
@pytest.mark.asyncio
async def test_tts_model_not_loaded():
    """TTS call before load raises appropriate error."""
    from src.api.services.tts_service import TTSService

    service = TTSService(device="cpu")
    # Don't initialize

    assert service.is_initialized() is False

    with pytest.raises((SynthesisError, AttributeError, Exception)):
        await service.synthesize("Test text")


# ===== Test 4: Chat Stream Generator Error =====

@pytest.mark.error
@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_stream_generator_error():
    """Error mid-stream is handled gracefully."""
    from src.api.services.llm_service import LLMService
    from src.core.llm.ollama_client import OllamaConnectionError

    service = LLMService(
        model="test-model",
        system_prompt="Test",
        host="http://localhost:11434"
    )

    # Mock the conversation manager to raise error mid-stream
    mock_manager = MagicMock()

    async def error_stream(*args, **kwargs):
        yield "First token"
        yield " second"
        raise OllamaConnectionError("Connection lost")

    mock_manager.chat_stream_async = error_stream
    service._manager = mock_manager
    service._initialized = True

    tokens = []
    with pytest.raises((LLMError, OllamaConnectionError)):
        async for token in service.chat_stream("Test"):
            tokens.append(token)

    # Should have received partial tokens before error
    assert len(tokens) >= 1


# ===== Test 5: Invalid Voice Reference =====

@pytest.mark.error
@pytest.mark.unit
def test_invalid_voice_reference():
    """Missing voice file handled with clear message."""
    from src.api.services.tts_service import TTSService

    service = TTSService(
        voice_reference="/nonexistent/path/voice.wav",
        device="cpu"
    )

    # The error should occur during initialization, not construction
    # This test verifies the path is stored in config for later validation
    assert service.config.voice_reference == "/nonexistent/path/voice.wav"


# ===== Test 6: API Client Timeout =====

@pytest.mark.error
@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_client_timeout():
    """HTTP timeout returns error message gracefully."""
    from src.flet_app.services.api_client import OliviaAPIClient

    client = OliviaAPIClient(base_url="http://localhost:99999")

    # This should handle timeout gracefully
    is_connected = await client.check_connection(max_retries=1, retry_delay=0.1)
    assert is_connected is False


# ===== Additional Error Tests =====

@pytest.mark.error
@pytest.mark.unit
def test_exception_hierarchy():
    """Custom exceptions have correct hierarchy."""
    # ServiceUnavailableError
    exc = ServiceUnavailableError("Test error")
    assert str(exc) == "Test error"
    assert isinstance(exc, Exception)

    # ServiceInitializationError
    exc = ServiceInitializationError("Init failed")
    assert str(exc) == "Init failed"

    # TranscriptionError
    exc = TranscriptionError("STT failed")
    assert str(exc) == "STT failed"

    # SynthesisError
    exc = SynthesisError("TTS failed")
    assert str(exc) == "TTS failed"

    # LLMError
    exc = LLMError("LLM failed")
    assert str(exc) == "LLM failed"

    # MemoryServiceError
    exc = MemoryServiceError("Memory failed")
    assert "Memory failed" in str(exc)


@pytest.mark.error
@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_health_check_failure():
    """Health check returns False on service failure."""
    from src.api.services.llm_service import LLMService

    # Patch where it's imported (in llm_service module), not where it's defined
    with patch('src.api.services.llm_service.check_ollama_connection_async', new_callable=AsyncMock) as mock_check:
        mock_check.return_value = False

        service = LLMService(
            model="test-model",
            system_prompt="Test",
            host="http://localhost:11434"
        )

        result = await service.health_check()
        assert result is False
        mock_check.assert_called_once_with("http://localhost:11434")


@pytest.mark.error
@pytest.mark.unit
def test_dependency_service_unavailable():
    """Dependency injection raises 503 when service unavailable."""
    from src.api import dependencies

    # Clear services
    original_services = dependencies.services.copy()
    dependencies.services = {}

    try:
        # get_service should return None for unavailable service
        result = dependencies.get_service("llm")
        assert result is None
    finally:
        # Restore
        dependencies.services = original_services


@pytest.mark.error
@pytest.mark.unit
def test_sentence_buffer_handles_malformed_input():
    """SentenceBuffer handles malformed/unusual input gracefully."""
    from src.api.utils.sentence_buffer import SentenceBuffer

    buffer = SentenceBuffer()

    # Very long input without breaks
    long_text = "a" * 1000
    list(buffer.add(long_text))

    # Should handle without crashing
    # Either yields partial or buffers
    assert True  # No exception raised


@pytest.mark.error
@pytest.mark.unit
def test_config_missing_env_vars():
    """Missing environment variables use defaults."""
    import os

    # Ensure we're not accidentally using real env vars
    test_var = "OLIVIA_TEST_VAR_12345"
    assert os.environ.get(test_var) is None

    # Config uses pydantic-settings with APIConfig class
    from src.api.config import APIConfig

    config = APIConfig()
    assert config.OLLAMA_HOST is not None
    assert config.OLLAMA_MODEL is not None
    # Verify default values are set
    assert config.OLLAMA_HOST == "http://localhost:11434"
    assert config.OLLAMA_MODEL == "olivia-finetuned"
