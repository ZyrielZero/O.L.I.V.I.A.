"""
Unit tests for service wrappers.
Tests LLM, Memory, STT, and TTS service classes with mocked dependencies.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestLLMService:
    """Tests for LLMService class."""

    @pytest.mark.unit
    def test_llm_service_initialization(self):
        """Test LLM service initialization."""
        from src.api.services.llm_service import LLMService

        service = LLMService(
            model="test-model",
            system_prompt="You are a test assistant.",
            host="http://localhost:11434",
        )

        assert service.model == "test-model"
        assert service.system_prompt == "You are a test assistant."
        assert service.host == "http://localhost:11434"
        assert service._manager is None

    @pytest.mark.unit
    def test_llm_service_is_initialized_false_initially(self):
        """Test that service reports not initialized initially."""
        from src.api.services.llm_service import LLMService

        service = LLMService(
            model="test-model", system_prompt="Test prompt", host="http://localhost:11434"
        )

        assert service.is_initialized() is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_service_initialize_fails_without_ollama(self):
        """Test that initialization fails when Ollama is not running."""
        from src.api.services.llm_service import LLMService
        from src.api.utils.exceptions import ServiceUnavailableError

        with patch(
            "src.api.services.llm_service.check_ollama_connection_async",
            new_callable=AsyncMock,
            return_value=False,
        ):
            service = LLMService(
                model="test-model", system_prompt="Test prompt", host="http://localhost:11434"
            )

            with pytest.raises(ServiceUnavailableError):
                await service.initialize()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_service_chat_stream_not_initialized(self):
        """Test that chat_stream raises error when not initialized."""
        from src.api.services.llm_service import LLMService
        from src.api.utils.exceptions import LLMError

        service = LLMService(
            model="test-model", system_prompt="Test prompt", host="http://localhost:11434"
        )

        with pytest.raises(LLMError):
            async for _ in service.chat_stream("Hello"):
                pass

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_service_health_check_returns_false_on_error(self):
        """Test health check returns False on connection error."""
        from src.api.services.llm_service import LLMService

        with patch(
            "src.api.services.llm_service.check_ollama_connection_async",
            new_callable=AsyncMock,
            side_effect=Exception("Connection failed"),
        ):
            service = LLMService(
                model="test-model", system_prompt="Test prompt", host="http://localhost:11434"
            )

            result = await service.health_check()
            assert result is False


class TestMemoryService:
    """Tests for MemoryService class."""

    @pytest.mark.unit
    def test_memory_service_initialization(self):
        """Test Memory service initialization."""
        from src.api.services.memory_service import MemoryService

        service = MemoryService(persist_directory="test_memory_db")

        assert service.persist_directory == "test_memory_db"
        assert service._db is None

    @pytest.mark.unit
    def test_memory_service_is_initialized_false_initially(self):
        """Test that service reports not initialized initially."""
        from src.api.services.memory_service import MemoryService

        service = MemoryService(persist_directory="test_memory_db")

        assert service.is_initialized() is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_service_add_conversation_not_initialized(self):
        """Test that add_conversation raises error when not initialized."""
        from src.api.services.memory_service import MemoryService
        from src.api.utils.exceptions import MemoryServiceError

        service = MemoryService(persist_directory="test_memory_db")

        with pytest.raises(MemoryServiceError):
            await service.add_conversation("Hello", "Hi there!")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_service_get_relevant_context_not_initialized(self):
        """Test that get_relevant_context raises error when not initialized."""
        from src.api.services.memory_service import MemoryService
        from src.api.utils.exceptions import MemoryServiceError

        service = MemoryService(persist_directory="test_memory_db")

        with pytest.raises(MemoryServiceError):
            await service.get_relevant_context("test query")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_service_health_check_false_when_not_initialized(self):
        """Test health check returns False when not initialized."""
        from src.api.services.memory_service import MemoryService

        service = MemoryService(persist_directory="test_memory_db")

        result = await service.health_check()
        assert result is False


class TestSTTService:
    """Tests for STTService class."""

    @pytest.mark.unit
    def test_stt_service_initialization(self):
        """Test STT service initialization."""
        from src.api.services.stt_service import STTService

        service = STTService(model_size="small.en", device="cuda", compute_type="float16")

        assert service.model_size == "small.en"
        assert service.device == "cuda"
        assert service.compute_type == "float16"
        assert service._engine is None

    @pytest.mark.unit
    def test_stt_service_default_values(self):
        """Test STT service default values."""
        from src.api.services.stt_service import STTService

        service = STTService()

        assert service.model_size == "small.en"
        assert service.device == "cuda"
        assert service.compute_type == "float16"

    @pytest.mark.unit
    def test_stt_service_is_initialized_false_initially(self):
        """Test that service reports not initialized initially."""
        from src.api.services.stt_service import STTService

        service = STTService()

        assert service.is_initialized() is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stt_service_transcribe_not_initialized(self):
        """Test that transcribe raises error when not initialized."""
        from src.api.services.stt_service import STTService
        from src.api.utils.exceptions import TranscriptionError

        service = STTService()

        with pytest.raises(TranscriptionError):
            await service.transcribe(b"\x00\x00")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stt_service_transcribe_numpy_not_initialized(self):
        """Test that transcribe_numpy raises error when not initialized."""
        from src.api.services.stt_service import STTService
        from src.api.utils.exceptions import TranscriptionError

        service = STTService()

        with pytest.raises(TranscriptionError):
            await service.transcribe_numpy(np.zeros(100, dtype=np.float32))

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stt_service_health_check_false_when_not_initialized(self):
        """Test health check returns False when not initialized."""
        from src.api.services.stt_service import STTService

        service = STTService()

        result = await service.health_check()
        assert result is False


class TestTTSService:
    """Tests for TTSService class."""

    @pytest.mark.unit
    def test_tts_service_initialization(self):
        """Test TTS service initialization."""
        from src.api.services.tts_service import TTSService

        service = TTSService(
            voice_reference="test/reference.wav", device="cuda", cfg_weight=0.6, exaggeration=0.7
        )

        assert service.config.voice_reference == "test/reference.wav"
        assert service.config.device == "cuda"
        assert service.config.cfg_weight == 0.6
        assert service.config.exaggeration == 0.7

    @pytest.mark.unit
    def test_tts_service_default_values(self):
        """Test TTS service default values."""
        from src.api.services.tts_service import TTSService

        service = TTSService()

        assert service.config.device == "cuda"
        assert service.config.cfg_weight == 0.5
        assert service.config.exaggeration == 0.5

    @pytest.mark.unit
    def test_tts_service_is_initialized_false_initially(self):
        """Test that service reports not initialized initially."""
        from src.api.services.tts_service import TTSService

        service = TTSService()

        assert service.is_initialized() is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_tts_service_synthesize_not_initialized(self):
        """Test that synthesize raises error when not initialized."""
        from src.api.services.tts_service import TTSService
        from src.api.utils.exceptions import SynthesisError

        service = TTSService()

        with pytest.raises(SynthesisError):
            await service.synthesize("Hello world")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_tts_service_synthesize_stream_not_initialized(self):
        """Test that synthesize_stream raises error when not initialized."""
        from src.api.services.tts_service import TTSService
        from src.api.utils.exceptions import SynthesisError

        service = TTSService()

        with pytest.raises(SynthesisError):
            async for _ in service.synthesize_stream("Hello"):
                pass

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_tts_service_health_check_false_when_not_initialized(self):
        """Test health check returns False when not initialized."""
        from src.api.services.tts_service import TTSService

        service = TTSService()

        result = await service.health_check()
        assert result is False


class TestDependencies:
    """Tests for dependency injection functions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_llm_service_unavailable(self):
        """Test get_llm_service raises HTTPException when unavailable."""
        from fastapi import HTTPException

        from src.api.dependencies import get_llm_service, services

        # Clear services
        services.clear()

        with pytest.raises(HTTPException) as exc_info:
            await get_llm_service()

        assert exc_info.value.status_code == 503

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_memory_service_unavailable(self):
        """Test get_memory_service raises HTTPException when unavailable."""
        from fastapi import HTTPException

        from src.api.dependencies import get_memory_service, services

        services.clear()

        with pytest.raises(HTTPException) as exc_info:
            await get_memory_service()

        assert exc_info.value.status_code == 503

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_stt_service_unavailable(self):
        """Test get_stt_service raises HTTPException when unavailable."""
        from fastapi import HTTPException

        from src.api.dependencies import get_stt_service, services

        services.clear()

        with pytest.raises(HTTPException) as exc_info:
            await get_stt_service()

        assert exc_info.value.status_code == 503

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_tts_service_unavailable(self):
        """Test get_tts_service raises HTTPException when unavailable."""
        from fastapi import HTTPException

        from src.api.dependencies import get_tts_service, services

        services.clear()

        with pytest.raises(HTTPException) as exc_info:
            await get_tts_service()

        assert exc_info.value.status_code == 503

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_state_manager_unavailable(self):
        """Test get_state_manager raises HTTPException when unavailable."""
        from fastapi import HTTPException

        from src.api.dependencies import get_state_manager, services

        services.clear()

        with pytest.raises(HTTPException) as exc_info:
            await get_state_manager()

        assert exc_info.value.status_code == 503

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_llm_service_when_available(self):
        """Test get_llm_service returns service when available."""
        from src.api.container import get_container
        from src.api.dependencies import get_llm_service

        container = get_container()
        original_llm = container.llm
        try:
            mock_service = MagicMock()
            container.llm = mock_service

            result = await get_llm_service()
            assert result is mock_service
        finally:
            container.llm = original_llm


class TestExceptions:
    """Tests for custom exception classes."""

    @pytest.mark.unit
    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError exception."""
        from src.api.utils.exceptions import ServiceUnavailableError

        error = ServiceUnavailableError("Ollama not running")
        assert str(error) == "Ollama not running"

    @pytest.mark.unit
    def test_service_initialization_error(self):
        """Test ServiceInitializationError exception."""
        from src.api.utils.exceptions import ServiceInitializationError

        error = ServiceInitializationError("Failed to load model")
        assert str(error) == "Failed to load model"

    @pytest.mark.unit
    def test_transcription_error(self):
        """Test TranscriptionError exception."""
        from src.api.utils.exceptions import TranscriptionError

        error = TranscriptionError("Audio too short")
        assert str(error) == "Audio too short"

    @pytest.mark.unit
    def test_synthesis_error(self):
        """Test SynthesisError exception."""
        from src.api.utils.exceptions import SynthesisError

        error = SynthesisError("Voice reference not found")
        assert str(error) == "Voice reference not found"

    @pytest.mark.unit
    def test_llm_error(self):
        """Test LLMError exception."""
        from src.api.utils.exceptions import LLMError

        error = LLMError("Generation timeout")
        assert str(error) == "Generation timeout"

    @pytest.mark.unit
    def test_memory_error(self):
        """Test MemoryServiceError exception."""
        from src.api.utils.exceptions import MemoryServiceError

        error = MemoryServiceError("ChromaDB connection failed")
        assert str(error) == "ChromaDB connection failed"
