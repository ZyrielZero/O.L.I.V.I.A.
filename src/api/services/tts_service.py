"""TTS service wrapper."""

import asyncio
import concurrent.futures
import logging
import threading
from typing import TYPE_CHECKING, AsyncGenerator, Optional

if TYPE_CHECKING:
    from src.api.services.tts_queue import SentenceTTSQueue

import numpy as np

from src.api.utils.exceptions import ServiceInitializationError, SynthesisError
from src.core.speech.chatterbox_tts import ChatterBoxConfig, ChatterBoxEngine

log = logging.getLogger("api.tts")

# Single GPU thread: CUDA inference must be serialized on one thread
_gpu_exec = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-gpu")

_INT16_MAX = np.float32(32767.0)
_INT16_DTYPE = np.int16
_FLOAT32_DTYPE = np.float32

# Active TTS queue reference for barge-in cancellation
_active_tts_queue: Optional["SentenceTTSQueue"] = None
_active_tts_queue_lock = threading.Lock()


def set_active_tts_queue(q):
    """Set the currently active TTS queue for barge-in."""
    global _active_tts_queue
    with _active_tts_queue_lock:
        _active_tts_queue = q


def get_active_tts_queue():
    """Get the currently active TTS queue."""
    with _active_tts_queue_lock:
        return _active_tts_queue


async def barge_in():
    """Stop active TTS playback (called when user starts speaking).

    Cancels pending synthesis AND drops audio already queued on the device.
    """
    q = get_active_tts_queue()
    if q:
        await q.stop()
    from src.api.services.audio_output import _output  # current instance, if any

    if _output is not None:
        _output.flush()


async def get_session_tts_queue(tts: "TTSService") -> "SentenceTTSQueue":
    """Long-lived TTS queue owned by the service layer, not the request.

    The route feeds sentences and returns; a client closing its SSE stream
    no longer cancels playback mid-sentence (Phase 1.1). Recreated on demand
    after a barge-in/stop killed the workers.
    """
    from src.api.services.tts_queue import SentenceTTSQueue

    q = get_active_tts_queue()
    if q is not None and q.is_running():
        return q

    q = SentenceTTSQueue(tts.synthesize_f32, tts.play_f32)
    await q.start()
    set_active_tts_queue(q)
    return q


class TTSService:
    """Async wrapper for ChatterBox TTS with streaming support."""

    def __init__(
        self,
        voice_reference: str = "assets/voice/reference.wav",
        device: str = "cuda",
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
    ):
        self.config = ChatterBoxConfig(
            device=device,
            voice_reference=voice_reference,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
        )
        self._engine: Optional[ChatterBoxEngine] = None
        self._engine_speaker_mode: Optional[ChatterBoxEngine] = None
        # OPT: Semaphore(2) allows limited parallelism for non-conflicting operations
        # Safe because ChatterBox model inference is thread-safe, only GPU memory is shared
        # This allows synthesis prep to overlap with previous playback
        self._synth_sem = asyncio.Semaphore(2)

    async def initialize(self):
        """Load ChatterBox model."""
        loop = asyncio.get_event_loop()

        def _load():
            try:
                eng = ChatterBoxEngine(self.config)
                eng.load_model()
                return eng
            except Exception as e:
                log.error(f"ChatterBox load failed: {e}")
                raise ServiceInitializationError(f"TTS init failed: {e}")

        try:
            self._engine_speaker_mode = await loop.run_in_executor(_gpu_exec, _load)
            log.info("TTS ready (ChatterBox)")
        except Exception as e:
            raise ServiceInitializationError(f"TTS init failed: {e}")

    async def synthesize(self, text: str) -> bytes:
        """Synthesize to PCM bytes (16-bit LE)."""
        if not self._engine_speaker_mode:
            raise SynthesisError("TTS not initialized")

        async with self._synth_sem:
            loop = asyncio.get_event_loop()
            chunks: list[np.ndarray] = []
            err = None

            def cb(chunk: np.ndarray, sr: int):
                nonlocal err
                try:
                    chunks.append((chunk * _INT16_MAX).astype(_INT16_DTYPE))
                except Exception as e:
                    err = e

            def _synth():
                nonlocal err
                try:
                    audio, sr = self._engine_speaker_mode.synthesize_to_numpy(text)
                    if len(audio) > 0:
                        cb(audio, sr)
                except Exception as e:
                    err = e

            await loop.run_in_executor(_gpu_exec, _synth)

            if err:
                raise SynthesisError(f"Synthesis failed: {err}")

            return np.concatenate(chunks).tobytes() if chunks else b""

    async def synthesize_f32(self, text: str) -> np.ndarray:
        """Synthesize to float32 ndarray — no int16 round-trip."""
        if not self._engine_speaker_mode:
            raise SynthesisError("TTS not initialized")

        async with self._synth_sem:
            loop = asyncio.get_event_loop()
            chunks: list[np.ndarray] = []
            err = None

            def cb(chunk: np.ndarray, sr: int):
                nonlocal err
                try:
                    chunks.append(chunk.copy())
                except Exception as e:
                    err = e

            def _synth():
                nonlocal err
                try:
                    audio, sr = self._engine_speaker_mode.synthesize_to_numpy(text)
                    if len(audio) > 0:
                        cb(audio, sr)
                except Exception as e:
                    err = e

            await loop.run_in_executor(_gpu_exec, _synth)

            if err:
                raise SynthesisError(f"Synthesis failed: {err}")

            return np.concatenate(chunks) if chunks else np.array([], dtype=_FLOAT32_DTYPE)

    async def play_audio(self, audio: bytes) -> None:
        """Play PCM (16-bit LE) audio via the persistent output stream."""
        if not audio:
            return
        arr = np.frombuffer(audio, dtype=_INT16_DTYPE).astype(_FLOAT32_DTYPE)
        arr /= _INT16_MAX
        await self.play_f32(arr)

    async def play_f32(self, audio: np.ndarray) -> None:
        """Play float32 audio via the persistent output stream (Phase 1.2).

        One long-lived OutputStream + ring buffer replaces per-sentence
        sd.play()/sd.wait() — no device reopen, no inter-sentence clicks.
        Returns once this audio has been handed to the device.
        """
        if audio is None or len(audio) == 0:
            return

        from src.api.services.audio_output import get_audio_output

        loop = asyncio.get_event_loop()
        out = get_audio_output(self.config.sample_rate)
        out.write(audio)

        # Audio duration + generous margin; barge-in flush() releases early
        timeout = len(audio) / self.config.sample_rate + 30.0

        def _wait():
            try:
                out.wait_drained(timeout=timeout)
            except Exception:
                log.error("Playback wait error", exc_info=True)

        await loop.run_in_executor(None, _wait)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks (16-bit PCM LE)."""
        if not self._engine_speaker_mode:
            raise SynthesisError("TTS not initialized")

        async with self._synth_sem:
            loop = asyncio.get_event_loop()
            q: asyncio.Queue = asyncio.Queue(maxsize=20)
            done = asyncio.Event()
            err = None

            def cb(chunk: np.ndarray, sr: int):
                nonlocal err
                try:
                    audio_bytes = (chunk * _INT16_MAX).astype(_INT16_DTYPE).tobytes()
                    asyncio.run_coroutine_threadsafe(q.put(audio_bytes), loop)
                except Exception as e:
                    err = e
                    log.error(f"Audio callback error: {e}")

            def _synth():
                nonlocal err
                try:
                    audio, sr = self._engine_speaker_mode.synthesize_to_numpy(text)
                    if len(audio) > 0:
                        cb(audio, sr)
                    asyncio.run_coroutine_threadsafe(q.put(None), loop)
                    done.set()
                except Exception as e:
                    err = e
                    asyncio.run_coroutine_threadsafe(q.put(None), loop)
                    done.set()

            fut = loop.run_in_executor(_gpu_exec, _synth)

            try:
                while True:
                    chunk = await q.get()
                    if chunk is None:
                        break
                    if err:
                        raise SynthesisError(f"Synthesis failed: {err}")
                    yield chunk
            finally:
                try:
                    await asyncio.wait_for(asyncio.shield(fut), timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    log.warning("Synthesis executor didn't finish in 5s")

            if err:
                raise SynthesisError(f"Synthesis failed: {err}")

    async def stop(self):
        """Stop current speech (barge-in)."""
        if self._engine_speaker_mode:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._engine_speaker_mode.stop)

    async def cleanup(self):
        """Cleanup TTS resources."""
        if self._engine_speaker_mode:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._engine_speaker_mode.close)
            except Exception as e:
                log.warning(f"TTS close: {e}")

        global _gpu_exec
        if _gpu_exec:
            _gpu_exec.shutdown(wait=True, cancel_futures=True)

    async def speak(self, text: str) -> None:
        """Synthesize and play via speakers."""
        if not self._engine_speaker_mode:
            return
        if not text or not text.strip():
            return

        audio = await self.synthesize_f32(text)
        await self.play_f32(audio)

    async def health_check(self) -> bool:
        """Check that the speaker-mode engine is loaded."""
        return self._engine_speaker_mode is not None and self._engine_speaker_mode._loaded

    def is_initialized(self) -> bool:
        """Return True if the speaker-mode engine is loaded."""
        return self._engine_speaker_mode is not None and self._engine_speaker_mode._loaded

    async def get_status(self) -> dict:
        """Get TTS status and metrics."""
        status = {
            "initialized": self.is_initialized(),
            "model_loaded": False,
            "device": self.config.device,
            "sample_rate": self.config.sample_rate,
            "optimizations": {
                "torch_compile": self.config.use_torch_compile,
                "compile_mode": self.config.compile_mode,
                "inference_mode": self.config.enable_inference_mode,
                "cudnn_benchmark": self.config.enable_cudnn_benchmark,
                "tf32": self.config.enable_tf32,
                "adaptive_chunking": self.config.adaptive_chunking,
            },
            "vram": None,
            "last_metrics": None,
            "generation_count": 0,
        }

        if self._engine_speaker_mode:
            status["model_loaded"] = self._engine_speaker_mode._loaded
            status["generation_count"] = self._engine_speaker_mode._generation_count

            if self.config.device == "cuda":
                try:
                    import torch

                    if torch.cuda.is_available():
                        status["vram"] = {
                            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                        }
                except Exception:
                    pass

            if m := self._engine_speaker_mode.get_metrics():
                status["last_metrics"] = {
                    "ttfb_ms": m.ttfb_ms,
                    "total_generation_ms": m.total_generation_ms,
                    "audio_duration_s": m.audio_duration_s,
                    "rtf": m.rtf,
                    "chunks_generated": m.chunks_generated,
                    "text_length": m.text_length,
                }

        return status

    async def cleanup_memory(self):
        """Manual CUDA memory cleanup."""
        if self._engine_speaker_mode:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._engine_speaker_mode._cleanup_cuda_memory)
