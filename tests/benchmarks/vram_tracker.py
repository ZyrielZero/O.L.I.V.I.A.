"""
VRAM Tracker for O.L.I.V.I.A. Optimization Testing.

This module provides utilities for tracking GPU memory usage
across operations to measure the impact of optimizations.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, List


@dataclass
class VRAMSnapshot:
    """A snapshot of VRAM usage at a point in time."""

    timestamp: float
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    label: str = ""

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "allocated_mb": self.allocated_mb,
            "reserved_mb": self.reserved_mb,
            "peak_mb": self.peak_mb,
            "label": self.label,
        }


class VRAMTracker:
    """Track VRAM usage across operations."""

    def __init__(self, auto_reset_peak: bool = True):
        """
        Initialize VRAM tracker.

        Args:
            auto_reset_peak: If True, reset peak stats before each tracked operation
        """
        self.snapshots: List[VRAMSnapshot] = []
        self.auto_reset_peak = auto_reset_peak
        self._cuda_available = self._check_cuda()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_allocated(self) -> float:
        """Get allocated VRAM in MB."""
        if not self._cuda_available:
            return 0.0
        try:
            import torch
            return torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            return 0.0

    def _get_reserved(self) -> float:
        """Get reserved VRAM in MB."""
        if not self._cuda_available:
            return 0.0
        try:
            import torch
            return torch.cuda.memory_reserved() / 1024 / 1024
        except Exception:
            return 0.0

    def _get_peak(self) -> float:
        """Get peak allocated VRAM in MB."""
        if not self._cuda_available:
            return 0.0
        try:
            import torch
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        except Exception:
            return 0.0

    def _reset_peak(self) -> None:
        """Reset peak memory statistics."""
        if not self._cuda_available:
            return
        try:
            import torch
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def snapshot(self, label: str = "") -> VRAMSnapshot:
        """
        Take a VRAM snapshot.

        Args:
            label: Optional label for this snapshot

        Returns:
            VRAMSnapshot with current memory state
        """
        snap = VRAMSnapshot(
            timestamp=time.time(),
            allocated_mb=self._get_allocated(),
            reserved_mb=self._get_reserved(),
            peak_mb=self._get_peak(),
            label=label,
        )
        self.snapshots.append(snap)
        return snap

    @contextmanager
    def track(self, label: str) -> Generator[None, None, None]:
        """
        Context manager to track VRAM delta for an operation.

        Args:
            label: Label for this operation

        Usage:
            tracker = VRAMTracker()
            with tracker.track("model_load"):
                model = load_model()
        """
        if self.auto_reset_peak:
            self._reset_peak()

        before = self.snapshot(f"{label}_before")
        yield
        after = self.snapshot(f"{label}_after")

        # Log the change
        delta = after.allocated_mb - before.allocated_mb
        peak = after.peak_mb
        print(
            f"[VRAM] {label}: {before.allocated_mb:.1f}MB -> {after.allocated_mb:.1f}MB "
            f"(delta: {delta:+.1f}MB, peak: {peak:.1f}MB)"
        )

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics from all snapshots."""
        if not self.snapshots:
            return {}

        allocated_values = [s.allocated_mb for s in self.snapshots]
        peak_values = [s.peak_mb for s in self.snapshots]

        return {
            "initial_mb": self.snapshots[0].allocated_mb,
            "final_mb": self.snapshots[-1].allocated_mb,
            "min_mb": min(allocated_values),
            "max_mb": max(allocated_values),
            "peak_mb": max(peak_values),
            "total_delta_mb": self.snapshots[-1].allocated_mb - self.snapshots[0].allocated_mb,
            "snapshot_count": len(self.snapshots),
        }

    def get_operation_deltas(self) -> Dict[str, float]:
        """
        Get VRAM deltas for each tracked operation.

        Returns:
            Dictionary mapping operation labels to VRAM deltas in MB
        """
        deltas = {}

        # Find pairs of before/after snapshots
        i = 0
        while i < len(self.snapshots) - 1:
            current = self.snapshots[i]
            if current.label.endswith("_before"):
                # Look for matching _after
                base_label = current.label[:-7]  # Remove "_before"
                for j in range(i + 1, len(self.snapshots)):
                    if self.snapshots[j].label == f"{base_label}_after":
                        deltas[base_label] = (
                            self.snapshots[j].allocated_mb - current.allocated_mb
                        )
                        break
            i += 1

        return deltas

    def clear(self) -> None:
        """Clear all snapshots."""
        self.snapshots.clear()

    def to_dict(self) -> Dict:
        """Convert all data to dictionary for JSON export."""
        return {
            "summary": self.get_summary(),
            "operation_deltas": self.get_operation_deltas(),
            "snapshots": [s.to_dict() for s in self.snapshots],
        }


class ComponentVRAMTracker:
    """
    Track VRAM usage per component with labeled measurements.

    Provides a higher-level interface for tracking VRAM across
    multiple components (STT, LLM, TTS, Memory).
    """

    def __init__(self):
        self.trackers: Dict[str, VRAMTracker] = {
            "stt": VRAMTracker(),
            "llm": VRAMTracker(),
            "tts": VRAMTracker(),
            "memory": VRAMTracker(),
            "total": VRAMTracker(),
        }

    def track_component(self, component: str, operation: str):
        """
        Context manager to track a specific component operation.

        Args:
            component: One of "stt", "llm", "tts", "memory", "total"
            operation: Label for the operation
        """
        if component not in self.trackers:
            self.trackers[component] = VRAMTracker()
        return self.trackers[component].track(operation)

    def get_component_summary(self, component: str) -> Dict[str, float]:
        """Get summary for a specific component."""
        if component not in self.trackers:
            return {}
        return self.trackers[component].get_summary()

    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """Get summaries for all components."""
        return {
            component: tracker.get_summary()
            for component, tracker in self.trackers.items()
        }

    def to_dict(self) -> Dict:
        """Convert all data to dictionary for JSON export."""
        return {
            component: tracker.to_dict()
            for component, tracker in self.trackers.items()
        }


def measure_model_vram(load_fn, label: str = "model") -> Dict[str, float]:
    """
    Measure VRAM usage of loading a model.

    Args:
        load_fn: Callable that loads the model
        label: Label for this measurement

    Returns:
        Dictionary with VRAM measurements
    """
    tracker = VRAMTracker()

    with tracker.track(label):
        load_fn()

    summary = tracker.get_summary()
    summary["label"] = label
    return summary


def get_current_vram_state() -> Dict[str, float]:
    """Get current VRAM state."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda_available": False}

        return {
            "cuda_available": True,
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "peak_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
        }
    except ImportError:
        return {"cuda_available": False, "torch_installed": False}
    except Exception as e:
        return {"cuda_available": False, "error": str(e)}


def print_vram_summary():
    """Print current VRAM state to console."""
    state = get_current_vram_state()

    if not state.get("cuda_available"):
        print("[VRAM] CUDA not available")
        return

    total = state.get("total_mb", 0)
    allocated = state.get("allocated_mb", 0)
    reserved = state.get("reserved_mb", 0)
    peak = state.get("peak_mb", 0)

    print(f"[VRAM] Allocated: {allocated:.1f}MB / {total:.0f}MB ({allocated/total*100:.1f}%)")
    print(f"[VRAM] Reserved:  {reserved:.1f}MB / {total:.0f}MB ({reserved/total*100:.1f}%)")
    print(f"[VRAM] Peak:      {peak:.1f}MB / {total:.0f}MB ({peak/total*100:.1f}%)")
