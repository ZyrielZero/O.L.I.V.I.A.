"""
Tests for LLM streaming optimization.
Verifies that tokens are yielded immediately (true streaming) rather than batched.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLLMStreamingOptimization:
    """Test suite for LLM streaming behavior."""

    @pytest.fixture
    def mock_conversation_manager(self):
        """Create a mock ConversationManager that simulates token streaming."""
        manager = MagicMock()

        async def chat_stream_async_generator(*args, **kwargs):
            """Simulate streaming tokens with small delays (async)."""
            tokens = ["Hello", " ", "there", "!", " How", " are", " you", "?"]
            for token in tokens:
                await asyncio.sleep(0.05)  # 50ms delay between tokens
                yield token

        manager.chat_stream_async = chat_stream_async_generator
        return manager

    @pytest.fixture
    def mock_slow_conversation_manager(self):
        """Create a mock that simulates slow LLM generation."""
        manager = MagicMock()

        async def slow_chat_stream_async(*args, **kwargs):
            """Simulate slow token generation (100ms per token)."""
            tokens = ["This", " is", " a", " slow", " response", "."]
            for token in tokens:
                await asyncio.sleep(0.1)  # 100ms delay
                yield token

        manager.chat_stream_async = slow_chat_stream_async
        return manager

    @pytest.mark.asyncio
    async def test_chat_stream_yields_tokens_immediately(self, mock_conversation_manager):
        """
        Test that tokens are yielded immediately as they arrive.
        TTFT (Time To First Token) should be much less than total generation time.
        """
        from src.api.services.llm_service import LLMService

        service = LLMService(
            model="test-model",
            system_prompt="Test prompt",
            host="http://localhost:11434"
        )
        service._manager = mock_conversation_manager

        start_time = time.time()
        first_token_time = None
        all_tokens = []

        async for token in service.chat_stream("Hello"):
            if first_token_time is None:
                first_token_time = time.time()
            all_tokens.append(token)

        end_time = time.time()

        # TTFT should be less than 200ms (first token arrives quickly)
        ttft = first_token_time - start_time
        total_time = end_time - start_time

        assert ttft < 0.2, f"TTFT too slow: {ttft:.3f}s (expected < 0.2s)"
        assert len(all_tokens) == 8, f"Expected 8 tokens, got {len(all_tokens)}"

        # Total time should be much greater than TTFT (tokens stream over time)
        assert total_time > ttft * 2, "Tokens should stream over time, not arrive all at once"

    @pytest.mark.asyncio
    async def test_chat_stream_not_batched(self, mock_slow_conversation_manager):
        """
        Test that streaming doesn't batch all tokens before yielding.
        This catches the old bug where list() collected all tokens first.
        """
        from src.api.services.llm_service import LLMService

        service = LLMService(
            model="test-model",
            system_prompt="Test prompt",
            host="http://localhost:11434"
        )
        service._manager = mock_slow_conversation_manager

        token_times = []
        start_time = time.time()

        async for token in service.chat_stream("Test"):
            token_times.append(time.time() - start_time)

        # Tokens should arrive with delays between them
        # If batched, all tokens would arrive at nearly the same time
        assert len(token_times) == 6, f"Expected 6 tokens, got {len(token_times)}"

        # Check that tokens arrived progressively
        for i in range(1, len(token_times)):
            time_diff = token_times[i] - token_times[i - 1]
            # Each token should have some delay (at least 50ms)
            assert time_diff > 0.05, f"Token {i} arrived too quickly after token {i-1}"

    @pytest.mark.asyncio
    async def test_chat_stream_handles_errors(self):
        """Test that errors in the generator are properly propagated."""
        from src.api.services.llm_service import LLMService
        from src.api.utils.exceptions import LLMError

        manager = MagicMock()

        async def error_generator_async(*args, **kwargs):
            yield "First"
            raise ValueError("Simulated error")

        manager.chat_stream_async = error_generator_async

        service = LLMService(
            model="test-model",
            system_prompt="Test prompt",
            host="http://localhost:11434"
        )
        service._manager = manager

        tokens = []
        with pytest.raises(LLMError) as exc_info:
            async for token in service.chat_stream("Test"):
                tokens.append(token)

        assert "Simulated error" in str(exc_info.value)
        assert len(tokens) == 1  # First token should have been yielded

    @pytest.mark.asyncio
    async def test_chat_stream_empty_response(self):
        """Test handling of empty responses from LLM."""
        from src.api.services.llm_service import LLMService

        manager = MagicMock()

        async def empty_generator_async(*args, **kwargs):
            return
            yield  # Make it an async generator

        manager.chat_stream_async = empty_generator_async

        service = LLMService(
            model="test-model",
            system_prompt="Test prompt",
            host="http://localhost:11434"
        )
        service._manager = manager

        tokens = []
        async for token in service.chat_stream("Test"):
            tokens.append(token)

        assert len(tokens) == 0

    @pytest.mark.asyncio
    async def test_chat_stream_not_initialized(self):
        """Test that calling chat_stream without initialization raises error."""
        from src.api.services.llm_service import LLMService
        from src.api.utils.exceptions import LLMError

        service = LLMService(
            model="test-model",
            system_prompt="Test prompt",
            host="http://localhost:11434"
        )
        # Don't initialize - _manager is None

        with pytest.raises(LLMError) as exc_info:
            async for token in service.chat_stream("Test"):
                pass

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_chat_stream_with_parameters(self):
        """Test that parameters are passed correctly to the underlying manager."""
        from src.api.services.llm_service import LLMService

        manager = MagicMock()
        call_args = {}

        async def capture_args_generator(*args, **kwargs):
            """Capture the arguments passed to chat_stream_async."""
            call_args.update(kwargs)
            yield "Token"

        manager.chat_stream_async = capture_args_generator

        service = LLMService(
            model="test-model",
            system_prompt="Test prompt",
            host="http://localhost:11434"
        )
        service._manager = manager

        tokens = []
        async for token in service.chat_stream(
            message="Hello",
            context="Some context",
            temperature=0.8,
            max_tokens=100
        ):
            tokens.append(token)

        # Verify chat_stream_async was called with correct parameters
        assert call_args.get("user_input") == "Hello"
        assert call_args.get("context") == "Some context"
        assert call_args.get("temperature") == 0.8
        assert call_args.get("max_tokens") == 100


class TestLLMServiceInitialization:
    """Test LLM service initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        from src.api.services.llm_service import LLMService

        with patch('src.api.services.llm_service.check_ollama_connection_async', new_callable=AsyncMock, return_value=True):
            with patch('src.api.services.llm_service.ConversationManager') as mock_manager_class:
                service = LLMService(
                    model="test-model",
                    system_prompt="Test prompt",
                    host="http://localhost:11434"
                )
                await service.initialize()

                assert service.is_initialized()
                mock_manager_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_ollama_not_running(self):
        """Test initialization when Ollama is not running."""
        from src.api.services.llm_service import LLMService
        from src.api.utils.exceptions import ServiceUnavailableError

        with patch('src.api.services.llm_service.check_ollama_connection_async', new_callable=AsyncMock, return_value=False):
            service = LLMService(
                model="test-model",
                system_prompt="Test prompt",
                host="http://localhost:11434"
            )

            with pytest.raises(ServiceUnavailableError):
                await service.initialize()
