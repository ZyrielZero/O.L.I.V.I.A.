"""
Baseline Metrics Collection for O.L.I.V.I.A. Optimization Testing.

This module provides classes and utilities for measuring and tracking
performance metrics across all components of the voice assistant pipeline.
"""

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ComponentMetrics:
    """Metrics for a single component (STT, LLM, TTS, Memory)."""

    component: str
    latency_samples: List[float] = field(default_factory=list)
    vram_mb_before: float = 0.0
    vram_mb_after: float = 0.0
    vram_mb_peak: float = 0.0

    @property
    def latency_avg(self) -> float:
        """Average latency in seconds."""
        return statistics.mean(self.latency_samples) if self.latency_samples else 0.0

    @property
    def latency_p50(self) -> float:
        """Median latency in seconds."""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        mid = len(sorted_samples) // 2
        return sorted_samples[mid]

    @property
    def latency_p95(self) -> float:
        """95th percentile latency in seconds."""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def latency_p99(self) -> float:
        """
        99th percentile latency in seconds.

        Complexity: O(n log n) for sorting
        """
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def latency_stddev(self) -> float:
        """
        Standard deviation of latency in seconds.

        Complexity: O(n)
        """
        if len(self.latency_samples) < 2:
            return 0.0
        return statistics.stdev(self.latency_samples)

    @property
    def latency_min(self) -> float:
        """Minimum latency in seconds."""
        return min(self.latency_samples) if self.latency_samples else 0.0

    @property
    def latency_max(self) -> float:
        """Maximum latency in seconds."""
        return max(self.latency_samples) if self.latency_samples else 0.0

    @property
    def vram_delta(self) -> float:
        """VRAM change in MB."""
        return self.vram_mb_after - self.vram_mb_before

    @property
    def sample_count(self) -> int:
        """Number of latency samples collected."""
        return len(self.latency_samples)

    def add_sample(self, latency: float) -> None:
        """Add a latency sample."""
        self.latency_samples.append(latency)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "component": self.component,
            "latency_avg_ms": self.latency_avg * 1000,
            "latency_p50_ms": self.latency_p50 * 1000,
            "latency_p95_ms": self.latency_p95 * 1000,
            "latency_p99_ms": self.latency_p99 * 1000,
            "latency_stddev_ms": self.latency_stddev * 1000,
            "latency_min_ms": self.latency_min * 1000,
            "latency_max_ms": self.latency_max * 1000,
            "sample_count": self.sample_count,
            "vram_mb_before": self.vram_mb_before,
            "vram_mb_after": self.vram_mb_after,
            "vram_mb_peak": self.vram_mb_peak,
            "vram_delta_mb": self.vram_delta,
        }


@dataclass
class TTSMetrics(ComponentMetrics):
    """Extended metrics for TTS component."""

    ttfb_samples: List[float] = field(default_factory=list)
    rtf_samples: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not hasattr(self, 'component'):
            self.component = "tts"

    @property
    def ttfb_avg(self) -> float:
        """Average Time-To-First-Byte in seconds."""
        return statistics.mean(self.ttfb_samples) if self.ttfb_samples else 0.0

    @property
    def ttfb_p95(self) -> float:
        """95th percentile TTFB in seconds."""
        if not self.ttfb_samples:
            return 0.0
        sorted_samples = sorted(self.ttfb_samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def rtf_avg(self) -> float:
        """Average Real-Time Factor (generation_time / audio_duration)."""
        return statistics.mean(self.rtf_samples) if self.rtf_samples else 0.0

    def add_tts_sample(self, ttfb: float, total_time: float, rtf: float) -> None:
        """Add a TTS sample with all metrics."""
        self.ttfb_samples.append(ttfb)
        self.latency_samples.append(total_time)
        self.rtf_samples.append(rtf)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        base = super().to_dict()
        base.update({
            "ttfb_avg_ms": self.ttfb_avg * 1000,
            "ttfb_p95_ms": self.ttfb_p95 * 1000,
            "rtf_avg": self.rtf_avg,
        })
        return base


@dataclass
class LLMMetrics(ComponentMetrics):
    """Extended metrics for LLM component."""

    ttft_samples: List[float] = field(default_factory=list)
    tokens_per_second_samples: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not hasattr(self, 'component'):
            self.component = "llm"

    @property
    def ttft_avg(self) -> float:
        """Average Time-To-First-Token in seconds."""
        return statistics.mean(self.ttft_samples) if self.ttft_samples else 0.0

    @property
    def ttft_p95(self) -> float:
        """95th percentile TTFT in seconds."""
        if not self.ttft_samples:
            return 0.0
        sorted_samples = sorted(self.ttft_samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def tokens_per_second_avg(self) -> float:
        """Average tokens per second."""
        return statistics.mean(self.tokens_per_second_samples) if self.tokens_per_second_samples else 0.0

    def add_llm_sample(self, ttft: float, total_time: float, tokens_per_second: float) -> None:
        """Add an LLM sample with all metrics."""
        self.ttft_samples.append(ttft)
        self.latency_samples.append(total_time)
        self.tokens_per_second_samples.append(tokens_per_second)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        base = super().to_dict()
        base.update({
            "ttft_avg_ms": self.ttft_avg * 1000,
            "ttft_p95_ms": self.ttft_p95 * 1000,
            "tokens_per_second_avg": self.tokens_per_second_avg,
        })
        return base


@dataclass
class PipelineMetrics:
    """End-to-end pipeline metrics aggregating all components."""

    stt: ComponentMetrics = field(default_factory=lambda: ComponentMetrics("stt"))
    llm: LLMMetrics = field(default_factory=lambda: LLMMetrics("llm"))
    tts: TTSMetrics = field(default_factory=lambda: TTSMetrics("tts"))
    memory: ComponentMetrics = field(default_factory=lambda: ComponentMetrics("memory"))
    e2e_latency_samples: List[float] = field(default_factory=list)
    timestamp: str = ""
    tag: str = "baseline"

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()

    @property
    def e2e_latency_avg(self) -> float:
        """Average end-to-end latency in seconds."""
        return statistics.mean(self.e2e_latency_samples) if self.e2e_latency_samples else 0.0

    @property
    def e2e_latency_p95(self) -> float:
        """95th percentile E2E latency in seconds."""
        if not self.e2e_latency_samples:
            return 0.0
        sorted_samples = sorted(self.e2e_latency_samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def add_e2e_sample(self, latency: float) -> None:
        """Add an end-to-end latency sample."""
        self.e2e_latency_samples.append(latency)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "timestamp": self.timestamp,
            "tag": self.tag,
            "stt": self.stt.to_dict(),
            "llm": self.llm.to_dict(),
            "tts": self.tts.to_dict(),
            "memory": self.memory.to_dict(),
            "e2e": {
                "latency_avg_ms": self.e2e_latency_avg * 1000,
                "latency_p95_ms": self.e2e_latency_p95 * 1000,
                "sample_count": len(self.e2e_latency_samples),
            }
        }


class LatencyTimer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "LatencyTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    def split(self) -> float:
        """Get elapsed time without stopping the timer."""
        return time.perf_counter() - self.start_time


@contextmanager
def measure_latency():
    """Context manager that yields elapsed time in seconds."""
    timer = LatencyTimer()
    with timer:
        yield timer


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return 0.0


def get_gpu_memory_reserved_mb() -> float:
    """Get current GPU memory reserved in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1024 / 1024
    except ImportError:
        pass
    return 0.0


def get_gpu_memory_peak_mb() -> float:
    """Get peak GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return 0.0


def reset_gpu_peak_stats() -> None:
    """Reset GPU peak memory statistics."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics including GPU and CPU info."""
    import platform

    metrics = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
    }

    try:
        import torch
        metrics["torch_version"] = torch.__version__
        metrics["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            metrics["cuda_version"] = torch.version.cuda
            metrics["gpu_name"] = torch.cuda.get_device_name(0)
            metrics["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    except ImportError:
        metrics["torch_available"] = False

    try:
        import psutil
        metrics["cpu_count"] = psutil.cpu_count()
        metrics["ram_total_gb"] = psutil.virtual_memory().total / 1024 / 1024 / 1024
    except ImportError:
        pass

    return metrics
