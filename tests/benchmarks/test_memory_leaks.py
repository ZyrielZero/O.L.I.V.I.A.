"""Memory leak detection tests."""
from __future__ import annotations

import gc
import tracemalloc

import pytest


@pytest.mark.benchmark
@pytest.mark.slow
async def test_no_memory_leak_on_repeated_searches(live_memory_service):
    """
    Verify memory doesn't grow linearly with repeated queries.

    Complexity: O(n) where n = number of iterations
    Memory: Should be O(1) - constant regardless of iterations
    """
    gc.collect()
    tracemalloc.start()

    baseline = tracemalloc.get_traced_memory()[0]

    iterations = 100
    gc_interval = 25

    for i in range(iterations):
        await live_memory_service.get_relevant_context(f"query {i}")

        # Periodic GC to simulate real usage
        if i % gc_interval == 0:
            gc.collect()

    final = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    growth_total = final - baseline
    growth_per_query = growth_total / iterations

    # Should be <10KB per query (allow some overhead)
    max_growth_per_query = 10_000
    assert growth_per_query < max_growth_per_query, (
        f"Memory leak detected: {growth_per_query / 1024:.1f}KB/query "
        f"(total growth: {growth_total / 1024 / 1024:.1f}MB)"
    )


@pytest.mark.benchmark
@pytest.mark.slow
async def test_no_memory_leak_on_repeated_llm_calls(live_llm_service):
    """
    Verify LLM service doesn't leak memory on repeated calls.

    Complexity: O(n) iterations, O(1) memory
    """
    gc.collect()
    tracemalloc.start()

    baseline = tracemalloc.get_traced_memory()[0]

    iterations = 20
    gc_interval = 5

    for i in range(iterations):
        # Collect tokens into list - O(1) amortized append
        tokens = []
        async for token in live_llm_service.chat_stream(f"Say hello {i}"):
            tokens.append(token)

        if i % gc_interval == 0:
            gc.collect()

    final = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    growth_per_call = (final - baseline) / iterations

    # LLM calls may have more overhead, allow 50KB per call
    max_growth_per_call = 50_000
    assert growth_per_call < max_growth_per_call, (
        f"Memory leak in LLM: {growth_per_call / 1024:.1f}KB/call"
    )


@pytest.mark.benchmark
@pytest.mark.slow
async def test_no_memory_leak_on_repeated_tts_calls(live_tts_service):
    """
    Verify TTS service doesn't leak memory on repeated synthesis.

    Complexity: O(n) iterations, O(1) memory
    """
    gc.collect()
    tracemalloc.start()

    baseline = tracemalloc.get_traced_memory()[0]

    iterations = 10
    gc_interval = 3
    test_phrases = [
        "Hello, how are you today?",
        "The weather is nice.",
        "I am doing well, thank you.",
    ]

    for i in range(iterations):
        phrase = test_phrases[i % len(test_phrases)]

        # Consume audio chunks
        async for _ in live_tts_service.synthesize_stream(phrase):
            pass

        if i % gc_interval == 0:
            gc.collect()

    final = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    growth_per_call = (final - baseline) / iterations

    # TTS may buffer audio, allow 100KB per call
    max_growth_per_call = 100_000
    assert growth_per_call < max_growth_per_call, (
        f"Memory leak in TTS: {growth_per_call / 1024:.1f}KB/call"
    )
