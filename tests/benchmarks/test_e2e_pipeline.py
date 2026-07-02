"""End-to-end pipeline benchmark tests."""
from __future__ import annotations

import time
from typing import List

import pytest


@pytest.mark.benchmark
@pytest.mark.slow
async def test_full_pipeline_latency(live_llm_service, live_tts_service, live_memory_service):
    """
    Full Memory->LLM->TTS pipeline benchmark.

    Measures:
    - TTFT (Time To First Token): Target <500ms
    - Total pipeline time: Target <3s

    Complexity: O(n) where n = response length
    """
    # Pre-populate memory for realistic test - O(n) insertions
    for i in range(5):
        await live_memory_service.add_conversation(
            f"User message {i}",
            f"Assistant response {i}",
        )

    # Inject memory context - O(log n) similarity search
    context = await live_memory_service.get_relevant_context("test query")

    start = time.perf_counter()

    # LLM streaming - O(n) tokens
    response_parts: List[str] = []  # List append is O(1) amortized
    ttft = None

    async for token in live_llm_service.chat_stream("Hello, how are you?", context=context):
        if ttft is None:
            ttft = time.perf_counter() - start
        response_parts.append(token)

    response = "".join(response_parts)  # O(n) join vs O(n^2) concat
    llm_done = time.perf_counter() - start

    # TTS first chunk - measure time to first audio
    ttfa = None
    tts_text = response[:100] if response else "Hello"

    async for _ in live_tts_service.synthesize_stream(tts_text):
        if ttfa is None:
            ttfa = time.perf_counter() - start
        break  # Only need first chunk for TTFA

    total = time.perf_counter() - start

    # Assertions
    assert ttft is not None, "No tokens received from LLM"
    assert ttft < 0.5, f"TTFT {ttft * 1000:.0f}ms exceeds 500ms target"
    assert total < 3.0, f"Total pipeline {total:.1f}s exceeds 3s target"

    print("\n=== Pipeline Benchmark ===")
    print(f"TTFT: {ttft * 1000:.0f}ms")
    print(f"LLM complete: {llm_done * 1000:.0f}ms")
    print(f"TTFA: {ttfa * 1000:.0f}ms" if ttfa else "TTFA: N/A")
    print(f"Total: {total * 1000:.0f}ms")


@pytest.mark.benchmark
async def test_memory_search_latency(live_memory_service):
    """
    Memory search latency benchmark.

    Target: <100ms per search
    Complexity: O(log n) for vector similarity search
    """
    # Populate with realistic data - O(n) insertions
    for i in range(50):
        await live_memory_service.add_conversation(
            f"User talked about topic {i}",
            f"Assistant discussed topic {i} in detail",
        )

    latencies: List[float] = []
    num_queries = 20

    for i in range(num_queries):
        start = time.perf_counter()
        await live_memory_service.get_relevant_context(f"topic {i % 10}")
        latencies.append(time.perf_counter() - start)

    # O(n log n) sort for percentile calculation
    sorted_latencies = sorted(latencies)
    avg_ms = sum(latencies) / len(latencies) * 1000
    p95_idx = int(len(sorted_latencies) * 0.95)
    p95_ms = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)] * 1000

    assert avg_ms < 100, f"Average search {avg_ms:.0f}ms exceeds 100ms target"

    print("\n=== Memory Search Benchmark ===")
    print(f"Average: {avg_ms:.0f}ms")
    print(f"P95: {p95_ms:.0f}ms")


@pytest.mark.benchmark
async def test_llm_streaming_throughput(live_llm_service):
    """
    LLM streaming throughput benchmark.

    Target: >40 tokens/second
    Complexity: O(n) where n = token count
    """
    start = time.perf_counter()

    token_count = 0
    ttft = None

    async for token in live_llm_service.chat_stream(
        "Write a short paragraph about the weather."
    ):
        if ttft is None:
            ttft = time.perf_counter() - start
        token_count += 1

    total_time = time.perf_counter() - start

    # Exclude TTFT from throughput calculation
    generation_time = total_time - (ttft or 0)
    tokens_per_second = token_count / generation_time if generation_time > 0 else 0

    assert tokens_per_second > 40, (
        f"Throughput {tokens_per_second:.1f} tok/s below 40 tok/s target"
    )

    print("\n=== LLM Throughput Benchmark ===")
    print(f"TTFT: {(ttft or 0) * 1000:.0f}ms")
    print(f"Tokens: {token_count}")
    print(f"Throughput: {tokens_per_second:.1f} tok/s")
