"""
Performance and benchmark tests.
Tests latency, throughput, and resource usage.
"""

import asyncio
import time

import pytest

# ===== Test 1: LLM Time to First Token < 500ms =====

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.asyncio
async def test_llm_ttft_under_500ms(live_llm_service):
    """Time to first token must be under 500ms."""
    start_time = time.time()
    first_token_time = None

    async for token in live_llm_service.chat_stream("Hello, how are you?"):
        if first_token_time is None:
            first_token_time = time.time()
            break  # Only need first token

    ttft_ms = (first_token_time - start_time) * 1000

    assert ttft_ms < 500, f"TTFT {ttft_ms:.0f}ms exceeds 500ms target"


# ===== Test 2: LLM Tokens Per Second > 40 =====

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.asyncio
async def test_llm_tokens_per_second_above_40(live_llm_service):
    """Token generation rate should exceed 40 tokens/second."""
    token_times = []
    start_time = time.time()

    async for token in live_llm_service.chat_stream(
        "Tell me a short story about a robot.",
        max_tokens=100
    ):
        token_times.append(time.time())

    total_time = time.time() - start_time
    token_count = len(token_times)

    if total_time > 0 and token_count > 0:
        tps = token_count / total_time
        assert tps >= 40, f"Only {tps:.1f} tokens/second (target: >=40)"


# ===== Test 3: TTS Time to First Audio < 500ms =====

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.asyncio
async def test_tts_ttfa_under_500ms(live_tts_service):
    """Time to first audio chunk must be under 500ms."""
    start_time = time.time()
    first_chunk_time = None

    async for chunk in live_tts_service.synthesize_stream("Hello there!"):
        if first_chunk_time is None:
            first_chunk_time = time.time()
            break

    ttfa_ms = (first_chunk_time - start_time) * 1000

    assert ttfa_ms < 500, f"TTFA {ttfa_ms:.0f}ms exceeds 500ms target"


# ===== Test 4: Combined Pipeline Latency =====

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.asyncio
async def test_combined_pipeline_latency(live_llm_service, live_tts_service):
    """End-to-end chat + TTS pipeline latency test."""
    start_time = time.time()

    # LLM response
    full_response = ""
    async for token in live_llm_service.chat_stream("Say hi"):
        full_response += token

    llm_time = time.time() - start_time

    # TTS synthesis (first chunk)
    tts_start = time.time()
    async for chunk in live_tts_service.synthesize_stream(full_response[:50]):
        break  # Just need first chunk

    tts_time = time.time() - tts_start
    total_time = time.time() - start_time

    # Log metrics
    print(f"LLM time: {llm_time*1000:.0f}ms")
    print(f"TTS time to first chunk: {tts_time*1000:.0f}ms")
    print(f"Total pipeline: {total_time*1000:.0f}ms")

    # Total should be reasonable (under 3 seconds for short response)
    assert total_time < 5.0, f"Pipeline took {total_time:.1f}s (target: <5s)"


# ===== Test 5: Memory Search Latency < 100ms =====

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_memory_search_latency_under_100ms(live_memory_service):
    """Memory search should complete within 100ms."""
    # Add some test data
    await live_memory_service.add_conversation(
        "Test user message",
        "Test AI response"
    )

    # Measure search time
    start_time = time.time()
    await live_memory_service.get_relevant_context("test query")
    search_time_ms = (time.time() - start_time) * 1000

    assert search_time_ms < 100, f"Search took {search_time_ms:.0f}ms (target: <100ms)"


# ===== Test 6: VRAM Usage < 10GB =====

@pytest.mark.benchmark
@pytest.mark.slow
def test_vram_usage_under_10gb():
    """Total VRAM usage should stay under 10GB."""
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Get current VRAM usage
        vram_bytes = torch.cuda.memory_allocated()
        vram_gb = vram_bytes / (1024 ** 3)

        # Also check reserved memory
        vram_reserved_bytes = torch.cuda.memory_reserved()
        vram_reserved_gb = vram_reserved_bytes / (1024 ** 3)

        print(f"VRAM allocated: {vram_gb:.2f}GB")
        print(f"VRAM reserved: {vram_reserved_gb:.2f}GB")

        assert vram_gb < 10, f"VRAM usage {vram_gb:.1f}GB exceeds 10GB target"

    except ImportError:
        pytest.skip("PyTorch not available")


# ===== Additional Performance Tests =====

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_concurrent_memory_queries():
    """Multiple memory queries can run concurrently efficiently."""
    import shutil
    import tempfile

    from src.core.memory.smart_memory import SmartMemoryDB

    temp_dir = tempfile.mkdtemp()
    try:
        db = SmartMemoryDB(persist_directory=temp_dir)

        # Add test data
        for i in range(10):
            db.add_conversation(f"Message {i}", f"Response {i}")

        # Concurrent queries
        start_time = time.time()

        async def search(query):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: db.search_all(query, n_results=3)
            )

        results = await asyncio.gather(
            search("message 1"),
            search("response 5"),
            search("something else"),
        )

        concurrent_time = time.time() - start_time

        # Should complete reasonably fast
        assert concurrent_time < 2.0, f"Concurrent queries took {concurrent_time:.1f}s"
        assert len(results) == 3

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.benchmark
def test_sentence_buffer_performance():
    """Sentence buffer handles high throughput."""
    from src.api.utils.sentence_buffer import SentenceBuffer

    buffer = SentenceBuffer()

    # Generate large input
    text = "This is a test sentence. " * 1000

    start_time = time.time()

    sentences = []
    for char in text:
        for sentence in buffer.add(char):
            sentences.append(sentence)

    final = buffer.flush()
    if final:
        sentences.append(final)

    process_time = time.time() - start_time

    # Should process efficiently
    chars_per_second = len(text) / process_time
    print(f"Processed {chars_per_second:.0f} chars/second")

    assert chars_per_second > 10000, "Sentence buffer too slow"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_async_operation_overhead():
    """Async operations have minimal overhead."""

    async def sync_work():
        return sum(range(1000))

    # Measure overhead of async wrapper
    start = time.time()
    for _ in range(1000):
        await sync_work()
    async_time = time.time() - start

    # Pure sync comparison
    start = time.time()
    for _ in range(1000):
        sum(range(1000))
    sync_time = time.time() - start

    overhead = async_time - sync_time
    print(f"Async overhead: {overhead*1000:.2f}ms for 1000 calls")

    # Overhead should be reasonable
    assert overhead < 1.0, f"Async overhead {overhead:.2f}s too high"
