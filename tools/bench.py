#!/usr/bin/env python
"""Full-pipeline latency benchmark harness (dev machine, GPU).

Runs the real voice pipeline stages against ~10 canned utterances and writes
benchmarks/results/{YYYY-MM-DD}_{git-sha}.json with per-stage timings and
peak VRAM. Committed results are the optimization record: bench before a
change, bench after, commit both JSONs with the change.

Usage:
    python tools/bench.py                     # all stages, 5 iterations
    python tools/bench.py --iterations 10
    python tools/bench.py --skip-tts          # e.g. when audio device is busy
    python tools/bench.py --tag phase1-baseline

Compare two results:
    python tools/bench_compare.py benchmarks/results/old.json benchmarks/results/new.json
"""

import argparse
import asyncio
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

UTTERANCES = [
    "Hey, how are you doing today?",
    "What's the weather looking like?",
    "Tell me something interesting about space.",
    "Can you remind me what we talked about yesterday?",
    "I'm feeling a bit tired this afternoon.",
    "What do you think I should have for dinner?",
    "Explain how a neural network learns.",
    "Do you remember my favorite game?",
    "I finished that project I was working on.",
    "Good night, see you tomorrow.",
]


def _stats(samples_ms):
    """Summarize a list of millisecond samples."""
    if not samples_ms:
        return None
    ordered = sorted(samples_ms)
    p95_idx = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "mean_ms": round(statistics.mean(ordered), 2),
        "median_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[p95_idx], 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
        "samples": len(ordered),
    }


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _env_info(model_tag: str) -> dict:
    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ollama_model": model_tag,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        info["driver"] = out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return info


def _vram_peak_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    except Exception:
        pass
    return 0.0


def _reset_vram_peak():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def bench_vad(iterations: int) -> dict:
    """Silero VAD over 512-sample chunks of synthetic audio."""
    import numpy as np
    import torch

    from src.core.speech.stt import _get_silero_vad

    model = _get_silero_vad()
    audio = (np.random.randn(16000 * 2) * 0.05).astype(np.float32)  # 2s
    chunks = [audio[i : i + 512] for i in range(0, len(audio) - 512, 512)]

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        for chunk in chunks:
            with torch.no_grad():
                model(torch.from_numpy(chunk.copy()), 16000)
        samples.append((time.perf_counter() - start) * 1000 / len(chunks))

    return {"per_chunk": _stats(samples)}


def bench_stt(iterations: int) -> dict:
    """faster-whisper transcription of 2s of synthetic audio."""
    import numpy as np

    from src.core.speech.stt import STTEngine

    engine = STTEngine()
    engine.load_model(warmup=True)
    audio = (np.random.randn(16000 * 2) * 0.05).astype(np.float32)

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        engine.transcribe_audio(audio)
        samples.append((time.perf_counter() - start) * 1000)

    return {"transcribe_2s": _stats(samples), "vram_peak_mb": _vram_peak_mb()}


async def bench_llm(model_tag: str, host: str) -> dict:
    """LLM TTFT + total time per canned utterance (fresh history each turn)."""
    from src.core.llm.ollama_client import ConversationManager

    ttft_samples, total_samples, tps_samples = [], [], []

    for utterance in UTTERANCES:
        manager = ConversationManager(
            system_prompt="You are a concise assistant.", model=model_tag, host=host
        )
        start = time.perf_counter()
        first = None
        n_tokens = 0
        async for _tok in manager.chat_stream_async(utterance):
            if first is None:
                first = time.perf_counter() - start
            n_tokens += 1
        total = time.perf_counter() - start
        await manager.close()

        if first is not None:
            ttft_samples.append(first * 1000)
        total_samples.append(total * 1000)
        if total > 0:
            tps_samples.append(n_tokens / total)

    return {
        "ttft": _stats(ttft_samples),
        "total": _stats(total_samples),
        "tokens_per_second_mean": round(statistics.mean(tps_samples), 1) if tps_samples else None,
        "vram_peak_mb": _vram_peak_mb(),
    }


async def bench_tts(iterations: int) -> dict:
    """ChatterBox TTFB (first streamed chunk) + full synthesis time."""
    from src.api.config import APIConfig
    from src.api.services.tts_service import TTSService

    cfg = APIConfig()
    tts = TTSService(
        voice_reference=cfg.TTS_VOICE_REFERENCE,
        device=cfg.TTS_DEVICE,
        cfg_weight=cfg.TTS_CFG_WEIGHT,
        exaggeration=cfg.TTS_EXAGGERATION,
    )
    await tts.initialize()

    ttfb_samples, total_samples = [], []
    texts = UTTERANCES[:iterations] if iterations < len(UTTERANCES) else UTTERANCES

    for text in texts:
        start = time.perf_counter()
        first = None
        async for _chunk in tts.synthesize_stream(text):
            if first is None:
                first = time.perf_counter() - start
        total = time.perf_counter() - start

        if first is not None:
            ttfb_samples.append(first * 1000)
        total_samples.append(total * 1000)

    return {
        "ttfb": _stats(ttfb_samples),
        "total": _stats(total_samples),
        "vram_peak_mb": _vram_peak_mb(),
    }


def _estimate_voice_to_voice(stages: dict) -> dict:
    """Component-sum estimate of voice-to-voice time-to-first-audio.

    stt(transcribe) + llm(ttft) + tts(ttfb). Honest label: this is a sum of
    independently measured stages, not one chained run — chaining lands with
    the /ws/voice pipeline (Phase 1).
    """
    parts = {
        "stt": stages.get("stt", {}).get("transcribe_2s"),
        "llm_ttft": stages.get("llm", {}).get("ttft"),
        "tts_ttfb": stages.get("tts", {}).get("ttfb"),
    }
    if any(v is None for v in parts.values()):
        return {"error": "missing stage data", "available": {k: bool(v) for k, v in parts.items()}}
    return {
        "estimated_ttfa_mean_ms": round(sum(v["mean_ms"] for v in parts.values()), 2),
        "estimated_ttfa_p95_ms": round(sum(v["p95_ms"] for v in parts.values()), 2),
        "method": "component sum (stt + llm_ttft + tts_ttfb), not a chained run",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="O.L.I.V.I.A. full-pipeline benchmark")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--model", default=None, help="Ollama model tag (default: APIConfig)")
    parser.add_argument("--tag", default="", help="Suffix for the result filename")
    parser.add_argument("--skip-vad", action="store_true")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    args = parser.parse_args()

    from src.api.config import APIConfig

    cfg = APIConfig()
    model_tag = args.model or cfg.OLLAMA_MODEL

    stages: dict = {}

    def run_stage(name, skip, fn):
        if skip:
            stages[name] = {"skipped": True}
            return
        print(f"[bench] {name}...", flush=True)
        _reset_vram_peak()
        start = time.perf_counter()
        try:
            stages[name] = fn()
        except Exception as e:
            print(f"[bench] {name} FAILED: {e}", flush=True)
            stages[name] = {"error": str(e)}
        print(f"[bench] {name} done in {time.perf_counter() - start:.1f}s", flush=True)

    run_stage("vad", args.skip_vad, lambda: bench_vad(args.iterations))
    run_stage("stt", args.skip_stt, lambda: bench_stt(args.iterations))
    run_stage("llm", args.skip_llm, lambda: asyncio.run(bench_llm(model_tag, cfg.OLLAMA_HOST)))
    run_stage("tts", args.skip_tts, lambda: asyncio.run(bench_tts(args.iterations)))

    stages["voice_to_voice"] = _estimate_voice_to_voice(stages)

    result = {"meta": _env_info(model_tag), "stages": stages}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS_DIR / (
        f"{datetime.now().strftime('%Y-%m-%d')}_{_git_sha()}{suffix}.json"
    )
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n[bench] Results written to {out_path}")
    print(json.dumps(stages.get("voice_to_voice", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
