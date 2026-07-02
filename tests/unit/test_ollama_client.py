"""
Unit tests for Ollama ConversationManager.
Tests history management, payload building, and connection handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.core.llm.ollama_client import (
    GEN_PARAMS,
    STOP_TOKENS,
    ConversationManager,
    check_ollama_connection,
    check_ollama_connection_async,
)

# ===== Test 1: History Management =====


@pytest.mark.unit
def test_conversation_manager_history_management():
    """History is properly maintained and trimmed."""
    manager = ConversationManager(model="test-model", system_prompt="Test prompt")

    # Initial state: only system prompt
    assert len(manager.history) == 1
    assert manager.history[0]["role"] == "system"

    # Add messages to history by building payload
    for i in range(25):
        manager._build_payload(f"Message {i}")

    # History should have system + 25 user messages
    assert len(manager.history) == 26

    # Trim to last 20
    manager.trim_history(keep=20)

    # Should have system + 20 messages = 21 total
    assert len(manager.history) == 21
    assert manager.history[0]["role"] == "system"
    assert manager.history[0]["content"] == "Test prompt"


# ===== Test 2: System Prompt Update =====


@pytest.mark.unit
def test_conversation_manager_system_prompt_update():
    """System prompt can be updated mid-conversation."""
    manager = ConversationManager(model="test-model", system_prompt="Original prompt")

    # Verify initial prompt
    assert manager.system_prompt == "Original prompt"
    assert manager.history[0]["content"] == "Original prompt"

    # Update prompt
    manager.update_system_prompt("New updated prompt")

    # Verify update
    assert manager.system_prompt == "New updated prompt"
    assert manager.history[0]["content"] == "New updated prompt"


# ===== Test 3: Payload Building =====


@pytest.mark.unit
def test_conversation_manager_payload_building():
    """Payload includes all parameters correctly."""
    manager = ConversationManager(model="test-model", system_prompt="Test prompt")

    # Build payload with custom parameters
    payload = manager._build_payload(
        user_input="Test message", context="Some context", temperature=0.8, max_tokens=200
    )

    # Verify structure
    assert payload["model"] == "test-model"
    assert payload["stream"] is True

    # Verify messages
    messages = payload["messages"]
    assert messages[0]["role"] == "system"

    # Context injection should be present
    context_msgs = [m for m in messages if "[Background:" in m.get("content", "")]
    assert len(context_msgs) == 1

    # Verify options
    options = payload["options"]
    assert options["temperature"] == 0.8
    assert options["num_predict"] == 200
    assert options["stop"] == STOP_TOKENS
    assert options["top_p"] == GEN_PARAMS["top_p"]
    assert options["top_k"] == GEN_PARAMS["top_k"]


# ===== Test 4: Connection Timeout Handling =====


@pytest.mark.unit
def test_check_ollama_connection_timeout():
    """Connection check fails gracefully on timeout."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("Connection timeout")
        mock_client_class.return_value = mock_client

        # Should return False, not raise exception
        result = check_ollama_connection("http://localhost:11434")
        assert result is False


# ===== Additional Tests =====


@pytest.mark.unit
def test_conversation_manager_clear_history():
    """Clear history resets to just system prompt."""
    manager = ConversationManager(model="test-model", system_prompt="Test prompt")

    # Add some history
    manager._build_payload("Message 1")
    manager._build_payload("Message 2")

    assert len(manager.history) > 1

    # Clear history
    manager.clear_history()

    # Should only have system prompt
    assert len(manager.history) == 1
    assert manager.history[0]["role"] == "system"


@pytest.mark.unit
def test_conversation_manager_default_params():
    """Default generation parameters are sensible."""
    # Verify defaults match expected values
    assert GEN_PARAMS["temperature"] == 0.3
    assert GEN_PARAMS["top_p"] == 0.7
    assert GEN_PARAMS["top_k"] == 15
    assert GEN_PARAMS["repeat_penalty"] == 1.3
    assert GEN_PARAMS["num_ctx"] == 4096
    assert GEN_PARAMS["num_predict"] == 100


@pytest.mark.unit
def test_conversation_manager_stop_tokens():
    """Stop tokens are configured to prevent template bleeding."""
    expected_stops = [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]

    for stop in expected_stops:
        assert stop in STOP_TOKENS


@pytest.mark.unit
@pytest.mark.asyncio
async def test_check_ollama_connection_async_success():
    """Async connection check returns True when Ollama is available."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await check_ollama_connection_async("http://localhost:11434")
        assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_check_ollama_connection_async_failure():
    """Async connection check returns False when Ollama is unavailable."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await check_ollama_connection_async("http://localhost:11434")
        assert result is False


@pytest.mark.unit
def test_payload_without_context():
    """Payload built correctly without context."""
    manager = ConversationManager(model="test-model", system_prompt="Test prompt")

    payload = manager._build_payload(
        user_input="Test message", context=None, temperature=None, max_tokens=None
    )

    # No context injection
    messages = payload["messages"]
    context_msgs = [m for m in messages if "[Background:" in m.get("content", "")]
    assert len(context_msgs) == 0

    # Default values used
    options = payload["options"]
    assert options["temperature"] == GEN_PARAMS["temperature"]
    assert options["num_predict"] == GEN_PARAMS["num_predict"]
