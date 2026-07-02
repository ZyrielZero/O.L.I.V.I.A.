#!/usr/bin/env python
"""
Baseline Metrics Collection Script for O.L.I.V.I.A.

This script collects baseline performance metrics for all components
before applying optimizations. Run this BEFORE making any changes.

Usage:
    python tests/benchmarks/run_baseline.py
    python tests/benchmarks/run_baseline.py --tag phase1
    python tests/benchmarks/run_baseline.py --iterations 50
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path (must precede the project imports below)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.benchmarks.baseline_metrics import (  # noqa: E402
    ComponentMetrics,
    LatencyTimer,
    LLMMetrics,
    PipelineMetrics,
    TTSMetrics,
    get_gpu_memory_mb,
    get_gpu_memory_peak_mb,
    get_system_metrics,
    reset_gpu_peak_stats,
)
from tests.benchmarks.vram_tracker import print_vram_summary  # noqa: E402


async def measure_stt(stt_engine, iterations: int = 10) -> ComponentMetrics:
    """Measure STT latency and VRAM usage."""
    import numpy as np

    metrics = ComponentMetrics("stt")

    print(f"\n[STT] Running {iterations} iterations...")

    # Get initial VRAM
    metrics.vram_mb_before = get_gpu_memory_mb()
    reset_gpu_peak_stats()

    # Test audio: 2 seconds of speech-like audio (simulated)
    sample_rate = 16000
    duration = 2.0
    test_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

    for i in range(iterations):
        timer = LatencyTimer()
        with timer:
            try:
                _ = stt_engine.transcribe_audio(test_audio)
            except Exception as e:
                print(f"  [STT] Iteration {i+1} error: {e}")
                continue

        metrics.add_sample(timer.elapsed)

        if (i + 1) % 5 == 0:
            print(f"  [STT] Completed {i+1}/{iterations} (last: {timer.elapsed*1000:.1f}ms)")

    metrics.vram_mb_after = get_gpu_memory_mb()
    metrics.vram_mb_peak = get_gpu_memory_peak_mb()

    print(f"  [STT] Average: {metrics.latency_avg*1000:.1f}ms, P95: {metrics.latency_p95*1000:.1f}ms")
    print(f"  [STT] VRAM: {metrics.vram_mb_before:.1f}MB -> {metrics.vram_mb_after:.1f}MB (peak: {metrics.vram_mb_peak:.1f}MB)")

    return metrics


async def measure_llm(llm_client, iterations: int = 10) -> LLMMetrics:
    """Measure LLM latency, TTFT, and tokens/second."""
    metrics = LLMMetrics("llm")

    print(f"\n[LLM] Running {iterations} iterations...")

    # Get initial VRAM
    metrics.vram_mb_before = get_gpu_memory_mb()
    reset_gpu_peak_stats()

    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me a short joke.",
        "What is 2 plus 2?",
        "Describe the weather today in one sentence.",
    ]

    for i in range(iterations):
        prompt = test_prompts[i % len(test_prompts)]

        total_timer = LatencyTimer()
        first_token_time = None
        token_count = 0

        try:
            with total_timer:
                async for token in llm_client.chat_stream_async(prompt):
                    if first_token_time is None:
                        first_token_time = total_timer.split()
                    token_count += 1

            ttft = first_token_time or total_timer.elapsed
            total_time = total_timer.elapsed
            tokens_per_second = token_count / total_time if total_time > 0 else 0

            metrics.add_llm_sample(ttft, total_time, tokens_per_second)

            if (i + 1) % 5 == 0:
                print(f"  [LLM] Completed {i+1}/{iterations} (TTFT: {ttft*1000:.1f}ms, total: {total_time*1000:.1f}ms)")

        except Exception as e:
            print(f"  [LLM] Iteration {i+1} error: {e}")
            continue

    metrics.vram_mb_after = get_gpu_memory_mb()
    metrics.vram_mb_peak = get_gpu_memory_peak_mb()

    print(f"  [LLM] Average TTFT: {metrics.ttft_avg*1000:.1f}ms, Total: {metrics.latency_avg*1000:.1f}ms")
    print(f"  [LLM] Tokens/sec: {metrics.tokens_per_second_avg:.1f}")

    return metrics


async def measure_tts(tts_engine, iterations: int = 10) -> TTSMetrics:
    """Measure TTS latency, TTFB, and RTF."""
    metrics = TTSMetrics("tts")

    print(f"\n[TTS] Running {iterations} iterations...")

    # Get initial VRAM
    metrics.vram_mb_before = get_gpu_memory_mb()
    reset_gpu_peak_stats()

    test_texts = [
        "Hello, how are you today?",
        "The weather is nice outside.",
        "I hope you're having a great day.",
        "Let me help you with that.",
        "That's an interesting question.",
    ]

    for i in range(iterations):
        text = test_texts[i % len(test_texts)]

        try:
            # Get metrics from TTS engine if available
            if hasattr(tts_engine, '_last_metrics') and tts_engine._last_metrics:
                # Use ChatterBox's built-in metrics
                timer = LatencyTimer()
                with timer:
                    await tts_engine.speak_blocking(text)

                tts_metrics = tts_engine._last_metrics
                ttfb = tts_metrics.ttfb_ms / 1000 if tts_metrics.ttfb_ms else timer.elapsed
                rtf = tts_metrics.rtf if tts_metrics.rtf else 0.1
                total_time = timer.elapsed

            else:
                # Fallback: just measure total time
                timer = LatencyTimer()
                with timer:
                    if hasattr(tts_engine, 'synthesize'):
                        _ = await tts_engine.synthesize(text)
                    elif hasattr(tts_engine, 'speak_blocking'):
                        await tts_engine.speak_blocking(text)
                    else:
                        await tts_engine.speak(text)

                total_time = timer.elapsed
                ttfb = total_time * 0.3  # Estimate TTFB as 30% of total
                rtf = 0.1  # Estimate RTF

            metrics.add_tts_sample(ttfb, total_time, rtf)

            if (i + 1) % 5 == 0:
                print(f"  [TTS] Completed {i+1}/{iterations} (TTFB: {ttfb*1000:.1f}ms, total: {total_time*1000:.1f}ms)")

        except Exception as e:
            print(f"  [TTS] Iteration {i+1} error: {e}")
            continue

    metrics.vram_mb_after = get_gpu_memory_mb()
    metrics.vram_mb_peak = get_gpu_memory_peak_mb()

    print(f"  [TTS] Average TTFB: {metrics.ttfb_avg*1000:.1f}ms, Total: {metrics.latency_avg*1000:.1f}ms")
    print(f"  [TTS] RTF: {metrics.rtf_avg:.3f}")

    return metrics


async def measure_memory(memory_db, iterations: int = 10) -> ComponentMetrics:
    """Measure Memory (ChromaDB) query latency."""
    metrics = ComponentMetrics("memory")

    print(f"\n[Memory] Running {iterations} iterations...")

    test_queries = [
        "What is my name?",
        "What do I like to do?",
        "Where do I work?",
        "What is my favorite food?",
        "Tell me about my hobbies.",
    ]

    for i in range(iterations):
        query = test_queries[i % len(test_queries)]

        timer = LatencyTimer()
        with timer:
            try:
                _ = memory_db.search_all(query, n_results=3)
            except Exception as e:
                print(f"  [Memory] Iteration {i+1} error: {e}")
                continue

        metrics.add_sample(timer.elapsed)

        if (i + 1) % 5 == 0:
            print(f"  [Memory] Completed {i+1}/{iterations} (last: {timer.elapsed*1000:.1f}ms)")

    print(f"  [Memory] Average: {metrics.latency_avg*1000:.1f}ms, P95: {metrics.latency_p95*1000:.1f}ms")

    return metrics


async def run_baseline_collection(
    iterations: int = 10,
    tag: str = "baseline",
    skip_components: list = None,
) -> PipelineMetrics:
    """
    Collect baseline metrics for all components.

    Args:
        iterations: Number of iterations per component
        tag: Tag for this measurement (e.g., "baseline", "phase1")
        skip_components: List of components to skip

    Returns:
        PipelineMetrics with all measurements
    """
    skip_components = skip_components or []
    metrics = PipelineMetrics(tag=tag)

    print("=" * 60)
    print("O.L.I.V.I.A. Baseline Metrics Collection")
    print(f"Tag: {tag}")
    print(f"Iterations: {iterations}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Print system info
    system_info = get_system_metrics()
    print("\nSystem Info:")
    print(f"  Platform: {system_info.get('platform', 'Unknown')}")
    print(f"  Python: {system_info.get('python_version', 'Unknown')}")
    if system_info.get('cuda_available'):
        print(f"  GPU: {system_info.get('gpu_name', 'Unknown')}")
        print(f"  CUDA: {system_info.get('cuda_version', 'Unknown')}")
        print(f"  VRAM: {system_info.get('gpu_memory_total_mb', 0):.0f}MB")

    print_vram_summary()

    # Initialize components
    try:
        # STT
        if "stt" not in skip_components:
            print("\n[INIT] Loading STT engine...")
            from src.core.speech.stt import STTEngine
            stt_engine = STTEngine(model_size="small.en", device="cuda", compute_type="float16")
            stt_engine.load_model()
            metrics.stt = await measure_stt(stt_engine, iterations)
        else:
            print("\n[SKIP] STT")

        # LLM
        if "llm" not in skip_components:
            print("\n[INIT] Loading LLM client...")
            from src.core.llm.ollama_client import ConversationManager
            llm_client = ConversationManager()
            metrics.llm = await measure_llm(llm_client, iterations)
        else:
            print("\n[SKIP] LLM")

        # TTS
        if "tts" not in skip_components:
            print("\n[INIT] Loading TTS engine...")
            try:
                from src.core.speech.chatterbox_tts import ChatterBoxConfig, ChatterBoxTTS
                tts_config = ChatterBoxConfig(device="cuda")
                tts_engine = ChatterBoxTTS(tts_config)
                tts_engine.load_model()
                metrics.tts = await measure_tts(tts_engine, iterations)
            except Exception as e:
                print(f"  [TTS] Failed to load: {e}")
        else:
            print("\n[SKIP] TTS")

        # Memory
        if "memory" not in skip_components:
            print("\n[INIT] Loading Memory DB...")
            from src.core.memory.smart_memory import SmartMemoryDB
            memory_db = SmartMemoryDB(persist_directory="data/memory_db_test")
            metrics.memory = await measure_memory(memory_db, iterations)
        else:
            print("\n[SKIP] Memory")

    except Exception as e:
        print(f"\n[ERROR] Component initialization failed: {e}")
        import traceback
        traceback.print_exc()

    return metrics


def save_metrics(metrics: PipelineMetrics, output_dir: str = "tests/results") -> Path:
    """Save metrics to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{metrics.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_path / filename

    with open(filepath, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    return filepath


def print_summary(metrics: PipelineMetrics) -> None:
    """Print summary of collected metrics."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    data = metrics.to_dict()

    for component in ["stt", "llm", "tts", "memory"]:
        if component in data and data[component]["sample_count"] > 0:
            c = data[component]
            print(f"\n{component.upper()}:")
            print(f"  Latency (avg): {c['latency_avg_ms']:.1f}ms")
            print(f"  Latency (p95): {c['latency_p95_ms']:.1f}ms")

            if component == "llm":
                print(f"  TTFT (avg): {c.get('ttft_avg_ms', 0):.1f}ms")
                print(f"  Tokens/sec: {c.get('tokens_per_second_avg', 0):.1f}")

            if component == "tts":
                print(f"  TTFB (avg): {c.get('ttfb_avg_ms', 0):.1f}ms")
                print(f"  RTF: {c.get('rtf_avg', 0):.3f}")

            if c.get('vram_mb_after', 0) > 0:
                print(f"  VRAM: {c['vram_mb_after']:.1f}MB (peak: {c['vram_mb_peak']:.1f}MB)")

    print("\n" + "=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Collect O.L.I.V.I.A. baseline metrics")
    parser.add_argument("--iterations", "-n", type=int, default=10, help="Iterations per component")
    parser.add_argument("--tag", "-t", type=str, default="baseline", help="Tag for this measurement")
    parser.add_argument("--output", "-o", type=str, default="tests/results", help="Output directory")
    parser.add_argument("--skip", type=str, nargs="*", default=[], help="Components to skip (stt, llm, tts, memory)")

    args = parser.parse_args()

    # Run collection
    metrics = await run_baseline_collection(
        iterations=args.iterations,
        tag=args.tag,
        skip_components=args.skip,
    )

    # Save results
    filepath = save_metrics(metrics, args.output)
    print(f"\n[SAVED] Results saved to: {filepath}")

    # Print summary
    print_summary(metrics)

    return metrics


if __name__ == "__main__":
    asyncio.run(main())
