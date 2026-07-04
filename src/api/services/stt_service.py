"""STT service wrapper."""

import asyncio
import logging
from typing import Optional

import numpy as np

from src.api.utils.audio_utils import bytes_to_numpy_audio, numpy_audio_to_float
from src.api.utils.exceptions import ServiceInitializationError, TranscriptionError
from src.core.speech.stt import STTEngine

log = logging.getLogger("api.stt")

# OPT: Timeout constants - prevents zombie requests from blocking the executor
# 30s is generous for most audio clips; longer clips should be chunked
_TRANSCRIBE_TIMEOUT_S = 30.0
_INIT_TIMEOUT_S = 60.0  # Model loading can take up to 60s on slow disks


class STTService:
    """Async wrapper for faster-whisper STT."""

    def __init__(
        self, model_size: str = "small.en", device: str = "cuda", compute_type: str = "float16"
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._engine: Optional[STTEngine] = None

    async def initialize(self) -> None:
        """Load Whisper model with timeout protection.

        OPT: Timeout prevents hanging during model download or disk I/O issues.
        Uses wait_for() which raises TimeoutError on expiration.
        """
        loop = asyncio.get_event_loop()

        def _load() -> STTEngine:
            try:
                eng = STTEngine(
                    model_size=self.model_size, device=self.device, compute_type=self.compute_type
                )
                eng.load_model()
                return eng
            except Exception as e:
                log.error(f"Whisper load failed: {e}")
                raise ServiceInitializationError(f"STT init failed: {e}")

        try:
            # OPT: Timeout protection - prevents zombie initialization
            self._engine = await asyncio.wait_for(
                loop.run_in_executor(None, _load),
                timeout=_INIT_TIMEOUT_S,
            )
            log.info(f"STT ready: {self.model_size}")
        except asyncio.TimeoutError:
            raise ServiceInitializationError(f"STT init timed out after {_INIT_TIMEOUT_S}s")
        except Exception as e:
            raise ServiceInitializationError(f"STT init failed: {e}")

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe PCM bytes (16-bit, 16kHz, mono).

        OPT: Timeout protection prevents zombie requests from blocking executor.
        30s is sufficient for ~5 minutes of audio at normal speech rate.
        """
        if not self._engine:
            raise TranscriptionError("STT not initialized")

        loop = asyncio.get_event_loop()

        def _run() -> str:
            try:
                arr = bytes_to_numpy_audio(audio, dtype=np.int16)
                arr_f = numpy_audio_to_float(arr)
                return self._engine.transcribe_audio(arr_f)
            except Exception as e:
                log.error(f"Transcription failed: {e}")
                raise TranscriptionError(f"Transcription failed: {e}")

        try:
            # OPT: Timeout protection - prevents zombie transcription requests
            return await asyncio.wait_for(
                loop.run_in_executor(None, _run),
                timeout=_TRANSCRIBE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise TranscriptionError(f"Transcription timed out after {_TRANSCRIBE_TIMEOUT_S}s")

    async def transcribe_numpy(self, arr: np.ndarray) -> str:
        """Transcribe float32 numpy array.

        OPT: Timeout protection prevents zombie requests from blocking executor.
        """
        if not self._engine:
            raise TranscriptionError("STT not initialized")

        loop = asyncio.get_event_loop()

        def _run() -> str:
            try:
                return self._engine.transcribe_audio(arr)
            except Exception as e:
                log.error(f"Transcription failed: {e}")
                raise TranscriptionError(f"Transcription failed: {e}")

        try:
            # OPT: Timeout protection - prevents zombie transcription requests
            return await asyncio.wait_for(
                loop.run_in_executor(None, _run),
                timeout=_TRANSCRIBE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise TranscriptionError(f"Transcription timed out after {_TRANSCRIBE_TIMEOUT_S}s")

    async def health_check(self) -> bool:
        """Check that the engine and model are loaded."""
        return self._engine is not None and self._engine.model is not None

    def is_initialized(self) -> bool:
        """Return True if the engine and model are loaded."""
        return self._engine is not None and self._engine.model is not None
