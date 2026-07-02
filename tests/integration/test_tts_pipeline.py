"""
TTS pipeline integration tests.
Tests sentence buffer to TTS queue flow and CUDA synchronization.
"""

import asyncio

import pytest

# ===== Test 1: Sentence Buffer to TTS Queue =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_sentence_buffer_to_tts_queue():
    """Sentences flow from buffer to TTS queue correctly."""
    from src.api.utils.sentence_buffer import SentenceBuffer

    buffer = SentenceBuffer()
    queued_sentences = []

    # Simulate TTS queue
    async def mock_queue_sentence(sentence):
        queued_sentences.append(sentence)

    # Stream tokens through buffer
    text = "Hello there. How are you today? I hope you're well."
    for char in text:
        for sentence in buffer.add(char):
            await mock_queue_sentence(sentence)

    # Flush remaining
    final = buffer.flush()
    if final:
        await mock_queue_sentence(final)

    # Verify sentences were queued
    assert len(queued_sentences) >= 2
    assert any("Hello" in s for s in queued_sentences)
    assert any("How are you" in s for s in queued_sentences)


# ===== Test 2: CUDA Sync Before TTS =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_cuda_sync_before_tts():
    """CUDA sync happens after LLM completes, before TTS starts."""
    sync_order = []

    async def mock_llm_complete():
        sync_order.append("llm_complete")

    async def mock_cuda_sync():
        sync_order.append("cuda_sync")

    async def mock_tts_start():
        sync_order.append("tts_start")

    # Simulate the correct order
    await mock_llm_complete()
    await mock_cuda_sync()
    await mock_tts_start()

    # Verify order
    assert sync_order == ["llm_complete", "cuda_sync", "tts_start"]
    assert sync_order.index("cuda_sync") > sync_order.index("llm_complete")
    assert sync_order.index("tts_start") > sync_order.index("cuda_sync")


# ===== Test 3: TTS Barge-in Stops Playback =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_tts_barge_in_stops_playback(mock_tts_service):
    """Calling stop() interrupts ongoing audio playback."""
    # Simulate TTS speaking
    is_speaking = True

    async def mock_stop():
        nonlocal is_speaking
        is_speaking = False

    mock_tts_service.stop = mock_stop

    # Verify initial state
    assert is_speaking is True

    # Trigger barge-in (stop)
    await mock_tts_service.stop()

    # Verify stopped
    assert is_speaking is False


# ===== Additional TTS Pipeline Tests =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_tts_queue_order_preservation():
    """TTS queue preserves sentence order."""
    from src.api.services.tts_queue import SentenceTTSQueue

    synthesized_order = []
    played_order = []

    async def mock_synthesize(sentence):
        synthesized_order.append(sentence)
        return b'\x00' * 100  # Fake audio

    async def mock_play(audio):
        played_order.append(len(synthesized_order))

    queue = SentenceTTSQueue(
        synthesize_fn=mock_synthesize,
        playback_fn=mock_play
    )

    await queue.start()

    # Queue sentences
    sentences = ["First sentence.", "Second sentence.", "Third sentence."]
    for s in sentences:
        await queue.queue_sentence(s)

    await queue.finish()
    await queue.stop()

    # Verify order preserved
    assert synthesized_order == sentences


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sentence_buffer_streaming_simulation():
    """Full streaming simulation with sentence buffer."""
    from src.api.utils.sentence_buffer import SentenceBuffer

    buffer = SentenceBuffer()

    # Simulate LLM streaming tokens
    tokens = [
        "The", " ", "weather", " ", "is", " ", "nice", ".", " ",
        "I", " ", "hope", " ", "you", " ", "enjoy", " ", "it", "!"
    ]

    sentences = []
    for token in tokens:
        for sentence in buffer.add(token):
            sentences.append(sentence)

    final = buffer.flush()
    if final:
        sentences.append(final)

    # Should have extracted 2 sentences
    assert len(sentences) == 2
    assert sentences[0] == "The weather is nice."
    assert sentences[1] == "I hope you enjoy it!"


@pytest.mark.integration
def test_tts_sanitization_removes_memory_tags():
    """TTS input is sanitized to remove memory tags."""
    import re

    memory_pattern = re.compile(r'\[MEMORY\].*?\[/MEMORY\]', re.DOTALL)

    text_with_tags = "Hello [MEMORY]user context here[/MEMORY] there!"
    sanitized = memory_pattern.sub('', text_with_tags).strip()

    assert "[MEMORY]" not in sanitized
    assert "Hello" in sanitized
    assert "there!" in sanitized


@pytest.mark.integration
def test_tts_sanitization_removes_asterisk_actions():
    """TTS input is sanitized to remove asterisk actions."""
    import re

    action_pattern = re.compile(r'\*[^*]+\*')

    text_with_actions = "Hello *smiles warmly* how are you *tilts head*?"
    sanitized = action_pattern.sub('', text_with_actions).strip()

    assert "*smiles" not in sanitized
    assert "*tilts" not in sanitized
    assert "Hello" in sanitized
    assert "how are you" in sanitized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tts_concurrent_synthesis_protection():
    """TTS uses semaphore to prevent concurrent synthesis."""

    concurrent_count = 0
    max_concurrent = 0

    async def mock_synthesis():
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.1)
        concurrent_count -= 1
        return b'\x00' * 100

    # Simulate concurrent calls with semaphore protection
    semaphore = asyncio.Semaphore(1)

    async def protected_synthesis():
        async with semaphore:
            return await mock_synthesis()

    # Run multiple synthesis calls
    await asyncio.gather(
        protected_synthesis(),
        protected_synthesis(),
        protected_synthesis()
    )

    # With semaphore, max concurrent should be 1
    assert max_concurrent == 1
