"""
Latency benchmark tests for O.L.I.V.I.A.
Measures TTFT (Time To First Token) and TTFB (Time To First Byte) for LLM and TTS.

Run with: pytest tests/benchmarks/test_latency.py -v --benchmark-only
Requires: pytest-benchmark (pip install pytest-benchmark)
"""

import asyncio

# Skip if pytest-benchmark not available
import importlib.util
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest_benchmark_available = importlib.util.find_spec("pytest_benchmark") is not None


class TestLLMLatencyBenchmarks:
    """Benchmark tests for LLM streaming latency."""

    @pytest.fixture
    def realistic_llm_mock(self):
        """Create a mock that simulates realistic LLM token generation."""
        manager = MagicMock()

        async def realistic_chat_stream_async(*args, **kwargs):
            """Simulate realistic token streaming with variable delays."""
            tokens = [
                "Hello", "!", " ", "I", "'m", " ", "doing", " ", "well", ",",
                " ", "thank", " ", "you", " ", "for", " ", "asking", "."
            ]
            for token in tokens:
                # Simulate variable token generation time (10-50ms)
                delay = 0.01 + (np.random.random() * 0.04)
                await asyncio.sleep(delay)
                yield token

        manager.chat_stream_async = realistic_chat_stream_async
        return manager

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_llm_ttft_benchmark(self, realistic_llm_mock):
        """
        Benchmark Time To First Token (TTFT) for LLM streaming.

        Target: < 200ms TTFT with the new streaming implementation.
        """
        from src.api.services.llm_service import LLMService

        service = LLMService(
            model="test-model",
            system_prompt="Test prompt",
            host="http://localhost:11434"
        )
        service._manager = realistic_llm_mock

        # Measure TTFT multiple times
        ttft_measurements = []

        for _ in range(5):
            start_time = time.time()
            first_token_received = False

            async for token in service.chat_stream("How are you?"):
                if not first_token_received:
                    ttft = time.time() - start_time
                    ttft_measurements.append(ttft)
                    first_token_received = True
                # Continue to drain the generator
                continue

        avg_ttft = sum(ttft_measurements) / len(ttft_measurements)
        max_ttft = max(ttft_measurements)

        print("\nLLM TTFT Benchmark Results:")
        print(f"  Average TTFT: {avg_ttft * 1000:.2f}ms")
        print(f"  Max TTFT: {max_ttft * 1000:.2f}ms")
        print(f"  Samples: {len(ttft_measurements)}")

        # Assert reasonable TTFT (should be under 200ms for mock)
        assert avg_ttft < 0.2, f"Average TTFT too high: {avg_ttft * 1000:.2f}ms"
        assert max_ttft < 0.3, f"Max TTFT too high: {max_ttft * 1000:.2f}ms"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_llm_full_response_latency(self, realistic_llm_mock):
        """
        Benchmark full response latency for LLM streaming.
        """
        from src.api.services.llm_service import LLMService

        service = LLMService(
            model="test-model",
            system_prompt="Test prompt",
            host="http://localhost:11434"
        )
        service._manager = realistic_llm_mock

        latency_measurements = []

        for _ in range(3):
            start_time = time.time()
            tokens = []

            async for token in service.chat_stream("Hello"):
                tokens.append(token)

            total_latency = time.time() - start_time
            latency_measurements.append(total_latency)

        avg_latency = sum(latency_measurements) / len(latency_measurements)

        print("\nLLM Full Response Latency:")
        print(f"  Average: {avg_latency * 1000:.2f}ms")
        print(f"  Token count: {len(tokens)}")

        # Full response should complete in reasonable time
        assert avg_latency < 2.0, f"Full response too slow: {avg_latency * 1000:.2f}ms"


class TestTTSLatencyBenchmarks:
    """Benchmark tests for TTS synthesis latency."""

    @pytest.fixture
    def realistic_tts_mock(self):
        """Create a mock that simulates realistic TTS chunk generation."""
        engine = MagicMock()
        engine._loaded = True
        engine._model = MagicMock()
        engine._processed_reference = "test.wav"
        engine._generation_count = 0
        engine.get_metrics = MagicMock(return_value=None)
        return engine

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_tts_ttfb_benchmark(self, realistic_tts_mock):
        """
        Benchmark Time To First Byte (TTFB) for TTS streaming.

        Target: < 500ms TTFB with optimizations.
        """
        from src.api.services.tts_service import TTSService

        service = TTSService(voice_reference="test.wav", device="cpu")
        service._engine_speaker_mode = realistic_tts_mock

        def mock_synthesize(text):
            # Simulate realistic synthesis latency before audio is available
            time.sleep(0.05)
            return np.random.randn(24000).astype(np.float32), 24000

        realistic_tts_mock.synthesize_to_numpy = mock_synthesize

        ttfb_measurements = []

        for _ in range(3):
            start_time = time.time()
            first_chunk_received = False

            async for chunk in service.synthesize_stream("Hello world"):
                if not first_chunk_received:
                    ttfb = time.time() - start_time
                    ttfb_measurements.append(ttfb)
                    first_chunk_received = True
                continue

        if ttfb_measurements:
            avg_ttfb = sum(ttfb_measurements) / len(ttfb_measurements)
            print("\nTTS TTFB Benchmark Results:")
            print(f"  Average TTFB: {avg_ttfb * 1000:.2f}ms")
            print(f"  Samples: {len(ttfb_measurements)}")

            # Assert reasonable TTFB
            assert avg_ttfb < 0.5, f"Average TTFB too high: {avg_ttfb * 1000:.2f}ms"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_tts_synthesis_throughput(self, realistic_tts_mock):
        """
        Benchmark TTS synthesis throughput.
        """
        from src.api.services.tts_service import TTSService

        service = TTSService(voice_reference="test.wav", device="cpu")
        service._engine_speaker_mode = realistic_tts_mock

        def mock_synthesize(text):
            # Simulate synthesis of ~1s of audio at 24kHz in ~200ms
            time.sleep(0.2)
            return np.random.randn(24000).astype(np.float32), 24000

        realistic_tts_mock.synthesize_to_numpy = mock_synthesize

        start_time = time.time()
        total_bytes = 0

        async for chunk in service.synthesize_stream("Test synthesis throughput"):
            total_bytes += len(chunk)

        elapsed = time.time() - start_time

        if elapsed > 0:
            throughput = total_bytes / elapsed / 1024  # KB/s

            print("\nTTS Synthesis Throughput:")
            print(f"  Total bytes: {total_bytes}")
            print(f"  Elapsed: {elapsed * 1000:.2f}ms")
            print(f"  Throughput: {throughput:.2f} KB/s")


class TestEndToEndLatency:
    """End-to-end latency benchmarks."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_combined_llm_tts_latency(self):
        """
        Benchmark combined LLM + TTS pipeline latency.
        Simulates a full chat response with audio generation.
        """
        from src.api.services.llm_service import LLMService

        # Create mock LLM
        llm_manager = MagicMock()

        async def llm_stream_async(*args, **kwargs):
            tokens = ["Hello", "!", " ", "How", " ", "can", " ", "I", " ", "help", "?"]
            for token in tokens:
                await asyncio.sleep(0.02)
                yield token

        llm_manager.chat_stream_async = llm_stream_async

        llm_service = LLMService(
            model="test-model",
            system_prompt="Test",
            host="http://localhost:11434"
        )
        llm_service._manager = llm_manager

        # Measure LLM response collection
        start_time = time.time()
        response_tokens = []
        first_token_time = None

        async for token in llm_service.chat_stream("Hello"):
            if first_token_time is None:
                first_token_time = time.time()
            response_tokens.append(token)

        llm_complete_time = time.time()
        full_response = "".join(response_tokens)

        print("\nCombined Pipeline Benchmark:")
        print(f"  LLM TTFT: {(first_token_time - start_time) * 1000:.2f}ms")
        print(f"  LLM Total: {(llm_complete_time - start_time) * 1000:.2f}ms")
        print(f"  Response: '{full_response}'")

        # Verify reasonable performance
        assert (first_token_time - start_time) < 0.2, "LLM TTFT too high"


# Utility for running benchmarks with pytest-benchmark
@pytest.mark.skipif(not pytest_benchmark_available, reason="pytest-benchmark not installed")
class TestWithPytestBenchmark:
    """Tests that use pytest-benchmark for detailed performance analysis."""

    def test_sync_token_processing(self, benchmark):
        """Benchmark synchronous token processing overhead."""

        def process_tokens():
            tokens = ["Hello", " ", "world", "!"]
            result = []
            for token in tokens:
                result.append(token)
            return "".join(result)

        result = benchmark(process_tokens)
        assert result == "Hello world!"
