"""Rolling latency metrics for the voice pipeline (Phase 1.6).

One process-wide collector; stages record milliseconds as they happen and
/health exposes the rolling averages. tools/bench.py is the per-commit
historical view; this is the runtime view — same stages, same names.
"""

import math
import statistics
import threading
from collections import deque
from typing import Deque, Dict, Optional

_WINDOW = 50  # rolling samples kept per stage

STAGES = (
    "stt_ms",            # speech-end -> transcript
    "llm_ttft_ms",       # request -> first token
    "llm_total_ms",      # request -> last token
    "tts_ttfb_ms",       # sentence queued -> first audio chunk
    "voice_to_voice_ms",  # user speech end -> first audio to client
)


class LatencyMetrics:
    """Thread-safe rolling latency windows keyed by stage name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: Dict[str, Deque[float]] = {s: deque(maxlen=_WINDOW) for s in STAGES}

    def record(self, stage: str, ms: float) -> None:
        """Record one sample; unknown stages are ignored (typo safety)."""
        if stage not in self._samples or ms < 0:
            return
        with self._lock:
            self._samples[stage].append(ms)

    def averages(self) -> Dict[str, Dict[str, float]]:
        """Rolling mean + p95 + count per stage that has data."""
        out: Dict[str, Dict[str, float]] = {}
        with self._lock:
            for stage, samples in self._samples.items():
                if not samples:
                    continue
                ordered = sorted(samples)
                p95_idx = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
                out[stage] = {
                    "avg_ms": round(statistics.mean(ordered), 1),
                    "p95_ms": round(ordered[p95_idx], 1),
                    "n": len(ordered),
                }
        return out

    def reset(self) -> None:
        """Clear all windows (tests)."""
        with self._lock:
            for samples in self._samples.values():
                samples.clear()


_metrics: Optional[LatencyMetrics] = None
_metrics_lock = threading.Lock()


def get_metrics() -> LatencyMetrics:
    """Process-wide metrics singleton."""
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = LatencyMetrics()
    return _metrics
