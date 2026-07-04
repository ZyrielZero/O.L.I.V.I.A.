"""Persistent audio output: one OutputStream + ring buffer (Phase 1.2).

Replaces per-sentence sd.play()/sd.wait(), which reopened the audio device
for every sentence and produced inter-sentence gaps and clicks. The stream
opens on first write, feeds the device from a ring buffer in the audio
callback, plays silence while idle, and closes itself after an idle timeout.
flush() drops all pending audio — the barge-in primitive.
"""

import logging
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np
import sounddevice as sd

log = logging.getLogger("api.audio_out")

_IDLE_CLOSE_S = 30.0  # close the device after this much silence


class AudioOutput:
    """Persistent playback sink for float32 mono audio at a fixed sample rate."""

    def __init__(self, sample_rate: int, idle_close_s: float = _IDLE_CLOSE_S):
        self.sample_rate = sample_rate
        self._idle_close_s = idle_close_s
        self._lock = threading.Lock()
        self._chunks: Deque[np.ndarray] = deque()
        self._offset = 0  # read position within the head chunk
        self._stream: Optional[sd.OutputStream] = None
        self._drained = threading.Event()
        self._drained.set()
        self._last_active = time.monotonic()

    # -- audio callback (device thread) --------------------------------------

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.debug(f"Output stream status: {status}")
        filled = 0
        with self._lock:
            while filled < frames and self._chunks:
                head = self._chunks[0]
                take = min(frames - filled, len(head) - self._offset)
                outdata[filled : filled + take, 0] = head[self._offset : self._offset + take]
                filled += take
                self._offset += take
                if self._offset >= len(head):
                    self._chunks.popleft()
                    self._offset = 0
            if not self._chunks:
                self._drained.set()
            if filled:
                self._last_active = time.monotonic()
        if filled < frames:
            outdata[filled:, 0] = 0.0  # silence while idle

    # -- public API -----------------------------------------------------------

    def write(self, audio: np.ndarray) -> None:
        """Queue float32 mono audio for playback; opens the device if needed."""
        if audio is None or len(audio) == 0:
            return
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        with self._lock:
            self._chunks.append(arr)
            self._drained.clear()
            self._last_active = time.monotonic()
        self._ensure_stream()

    def flush(self) -> None:
        """Drop all pending audio immediately (barge-in)."""
        with self._lock:
            self._chunks.clear()
            self._offset = 0
            self._drained.set()

    def wait_drained(self, timeout: Optional[float] = None) -> bool:
        """Block until everything queued has been handed to the device."""
        return self._drained.wait(timeout)

    @property
    def pending_samples(self) -> int:
        """Number of queued samples not yet handed to the device."""
        with self._lock:
            total = sum(len(c) for c in self._chunks)
            return max(0, total - self._offset)

    def maybe_close_idle(self) -> bool:
        """Close the device if it has been silent past the idle timeout."""
        with self._lock:
            idle = not self._chunks and (
                time.monotonic() - self._last_active > self._idle_close_s
            )
        if idle:
            self.close()
        return idle

    def close(self) -> None:
        """Stop and release the audio device."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.warning("Output stream close failed", exc_info=True)
            log.debug("Audio output closed")

    # -- internals --------------------------------------------------------------

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self._callback,
            )
            self._stream.start()
            log.info(f"Audio output opened ({self.sample_rate} Hz)")
        except Exception:
            self._stream = None
            log.error("Failed to open audio output", exc_info=True)
            raise


_output: Optional[AudioOutput] = None
_output_lock = threading.Lock()


def get_audio_output(sample_rate: int) -> AudioOutput:
    """Process-wide output; recreated if the sample rate changes."""
    global _output
    with _output_lock:
        if _output is None or _output.sample_rate != sample_rate:
            if _output is not None:
                _output.close()
            _output = AudioOutput(sample_rate)
        return _output


def close_audio_output() -> None:
    """Shutdown hook."""
    global _output
    with _output_lock:
        if _output is not None:
            _output.close()
            _output = None
