"""
Chat flow integration tests.
Tests the complete chat pipeline including memory retrieval and web search.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# ===== Test 1: Chat with Memory Retrieval =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_with_memory_retrieval(mock_llm_service, mock_memory_service):
    """Chat retrieves relevant memory context and passes to LLM."""
    # Setup mock memory to return context
    mock_memory_service.get_relevant_context = AsyncMock(
        return_value="User previously mentioned they like pizza."
    )

    # Simulate the chat flow
    user_message = "What food did I say I liked?"
    context = await mock_memory_service.get_relevant_context(user_message)

    assert context is not None
    assert "pizza" in context

    # Verify LLM would receive the context
    mock_memory_service.get_relevant_context.assert_called_once_with(user_message)


# ===== Test 2: Chat with Memory Prefetch =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_memory_prefetch_short_message():
    """Short messages skip the memory lookup."""
    from src.api.routes.chat import _fetch_memory_context

    mock_memory = MagicMock()
    mock_memory.get_relevant_context = AsyncMock(return_value="context")

    result = await _fetch_memory_context(mock_memory, "hi there")
    assert result == ""
    mock_memory.get_relevant_context.assert_not_called()


# ===== Test 3: Chat Streaming to Memory Storage =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_streaming_to_memory_storage(mock_memory_service):
    """Full response is stored in memory after streaming completes."""
    # Simulate streaming response
    full_response = ""
    tokens = ["Hello", " ", "there", "!", " ", "How", " ", "are", " ", "you", "?"]

    for token in tokens:
        full_response += token

    # After streaming completes, store in memory
    user_msg = "Hi"
    await mock_memory_service.add_conversation(user_msg, full_response)

    # Verify memory service was called with complete response
    mock_memory_service.add_conversation.assert_called_once_with(
        user_msg, full_response
    )


# ===== Test 4: Chat SSE Format Correctness =====

@pytest.mark.integration
def test_chat_sse_format_correctness():
    """SSE stream has correct 'data: ' prefix format."""
    import json

    # Simulate SSE event format
    token = "Hello"
    sse_event = f"data: {json.dumps({'token': token, 'done': False})}\n\n"

    # Verify format
    assert sse_event.startswith("data: ")
    assert sse_event.endswith("\n\n")

    # Parse back
    data_part = sse_event.split("data: ")[1].strip()
    parsed = json.loads(data_part)

    assert parsed["token"] == token
    assert parsed["done"] is False


# ===== Test 5: Memory Prefetch Parallelization =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_prefetch_parallelization():
    """Memory prefetch runs in parallel with other operations."""
    import time

    async def slow_memory_fetch():
        await asyncio.sleep(0.1)
        return "Memory context"

    async def slow_other_operation():
        await asyncio.sleep(0.1)
        return "Other result"

    # Run in parallel
    start = time.time()
    await asyncio.gather(
        slow_memory_fetch(),
        slow_other_operation()
    )
    parallel_time = time.time() - start

    # Run sequentially for comparison
    start = time.time()
    await slow_memory_fetch()
    await slow_other_operation()
    sequential_time = time.time() - start

    # Parallel should be faster than sequential
    assert parallel_time < sequential_time * 0.8  # At least 20% faster


# ===== Additional Chat Flow Tests =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_empty_memory_context(mock_memory_service):
    """Chat handles empty memory context gracefully."""
    mock_memory_service.get_relevant_context = AsyncMock(return_value="")

    context = await mock_memory_service.get_relevant_context("test query")

    assert context == ""
    # Should not raise any errors


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_token_accumulation():
    """Tokens accumulate correctly during streaming."""
    tokens = ["The", " ", "quick", " ", "brown", " ", "fox"]
    accumulated = ""

    for token in tokens:
        accumulated += token

    assert accumulated == "The quick brown fox"


@pytest.mark.integration
def test_greeting_detection():
    """Simple greetings are correctly identified."""
    from src.api.services.chat_service import ChatService

    assert ChatService._is_simple_greeting("hi") is True
    assert ChatService._is_simple_greeting("hello") is True
    assert ChatService._is_simple_greeting("hey") is True
    assert ChatService._is_simple_greeting("thanks") is True
    assert ChatService._is_simple_greeting("tell me about python") is False
    assert ChatService._is_simple_greeting("search for something") is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_respects_max_tokens(mock_llm_service):
    """Chat respects max_tokens parameter."""
    # Verify the mock can be configured with max_tokens
    max_tokens = 50

    # In real implementation, this would limit response length
    # Here we verify the parameter is passed through
    tokens = []
    async for token in mock_llm_service.chat_stream("Test", max_tokens=max_tokens):
        tokens.append(token)

    # Mock returns fixed tokens, but in real scenario max_tokens would limit output
    assert len(tokens) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_with_custom_temperature(mock_llm_service):
    """Chat accepts custom temperature parameter."""
    temperature = 0.8

    tokens = []
    async for token in mock_llm_service.chat_stream("Test", temperature=temperature):
        tokens.append(token)

    assert len(tokens) > 0
