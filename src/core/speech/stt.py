"""Speech-to-Text Module for O.L.I.V.I.A.
Whisper-based STT with push-to-talk and continuous listening modes.
"""

import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np
import sounddevice as sd
import torch

try:
    from src.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name):
        """Fallback logger factory when the project logger is unavailable."""
        return logging.getLogger(name)


# Module-level singleton for Silero VAD — shared across ContinuousSTT instances
_silero_vad_model = None
_silero_vad_lock = threading.Lock()


def _get_silero_vad():
    """Load Silero VAD model (CPU, singleton). Thread-safe."""
    global _silero_vad_model
    if _silero_vad_model is not None:
        return _silero_vad_model

    with _silero_vad_lock:
        if _silero_vad_model is not None:
            return _silero_vad_model

        log = get_logger("stt")
        log.info("Loading Silero VAD model...")
        from silero_vad import load_silero_vad  # model ships in the wheel — no network needed

        model = load_silero_vad()
        model.to("cpu")
        model.eval()
        _silero_vad_model = model
        log.info("Silero VAD loaded (CPU)")
        return _silero_vad_model


class STTEngine:
    """Core speech-to-text engine using faster-whisper."""

    def __init__(
        self, model_size: str = "small.en", device: str = "cuda", compute_type: str = "float16"
    ):
        self.log = get_logger("stt")
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Any = None
        self.sample_rate = 16000
        self.mic_index: Optional[int] = None
        self.log.info(f"STT Engine initialized (model: {model_size}, device: {device})")

    def load_model(self, warmup: bool = True) -> None:
        """Load the Whisper model with optional warmup.

        Args:
            warmup: If True, run a dummy inference to pre-compile CUDA kernels.
                    This eliminates the first-inference latency spike.
        """
        from faster_whisper import WhisperModel

        self.log.info(f"Loading Whisper model '{self.model_size}'...")
        self.model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        self.log.info("Whisper model loaded")

        # Warmup: Run dummy inference to compile CUDA kernels
        # This eliminates the ~500ms+ first-inference latency spike
        if warmup and self.device == "cuda":
            self._warmup_model()

    def _warmup_model(self) -> None:
        """Run dummy inference to warm up CUDA kernels.

        This pre-compiles the computation graph, eliminating the
        latency spike on the first real transcription.

        OPT: Uses beam_size=5 to match actual inference parameters.
        Different beam sizes compile different CUDA kernels, so warmup
        must match inference settings to be effective.
        """
        self.log.info("Warming up STT model...")
        try:
            # 1 second of silence at 16kHz
            warmup_audio = np.zeros(16000, dtype=np.float32)

            # OPT: beam_size=5 matches transcribe_audio() for proper kernel warmup
            # Using beam_size=1 would warm up different code paths
            _ = list(self.model.transcribe(warmup_audio, beam_size=5)[0])

            self.log.info("STT model warmed up")
        except Exception as e:
            # Warmup failure is not critical - log and continue
            self.log.warning(f"STT warmup failed (non-fatal): {e}")

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text."""
        if self.model is None:
            self.load_model()

        model = self.model
        if model is None:
            raise RuntimeError("Failed to load Whisper model")

        # Ensure correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / 32768.0

        segments, _ = model.transcribe(audio_data, beam_size=5)
        return " ".join(seg.text for seg in segments).strip()

    def list_microphones(self) -> List[Dict[str, Any]]:
        """List available microphone devices."""
        devices = sd.query_devices()
        mics: List[Dict[str, Any]] = []

        for i in range(len(devices)):  # type: ignore
            dev = devices[i]  # type: ignore
            dev_dict: Dict[str, Any] = dict(dev) if hasattr(dev, "keys") else {"name": str(dev)}
            if dev_dict.get("max_input_channels", 0) > 0:
                mics.append(
                    {
                        "index": i,
                        "name": dev_dict.get("name", f"Device {i}"),
                        "channels": dev_dict.get("max_input_channels"),
                    }
                )

        return mics


class PushToTalkSTT:
    """Push-to-talk speech recognition - hold button to record."""

    # Max recording duration: 30 seconds at 16kHz with 100ms chunks = 300 chunks
    MAX_BUFFER_CHUNKS = 300

    def __init__(self, stt_engine: STTEngine):
        self.stt = stt_engine
        self.log = get_logger("stt")
        self.is_recording = False
        # Optimization: Use bounded deque instead of unbounded list
        # deque(maxlen=N) provides O(1) append and automatic eviction of old data
        # Prevents memory growth for long recordings and pre-allocates capacity
        self.audio_buffer: Deque[np.ndarray] = deque(maxlen=self.MAX_BUFFER_CHUNKS)
        self.stream: Optional[sd.InputStream] = None
        self.sample_rate = 16000

    def start_recording(self) -> None:
        """Start recording audio."""
        self.is_recording = True
        self.audio_buffer.clear()

        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if self.is_recording:
                self.audio_buffer.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=callback,
            device=self.stt.mic_index,
        )
        self.stream.start()
        self.log.debug("PTT Recording started")

    def stop_recording(self) -> str:
        """Stop recording and transcribe."""
        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_buffer:
            return ""

        self.log.debug("Transcribing PTT audio...")
        # Optimization: Convert deque to list for np.concatenate (required for proper stacking)
        audio_data = np.concatenate(list(self.audio_buffer)).flatten()
        text = self.stt.transcribe_audio(audio_data)
        self.log.info(f"Heard: {text}")
        return text


class ContinuousSTT:
    """Always-on listening with Voice Activity Detection."""

    BASE_THRESHOLD = 0.005
    TTS_ACTIVE_THRESHOLD = 0.01  # Allows barge-in during TTS while filtering echo

    # Max speech duration: 30 seconds at 16kHz with 100ms chunks = 300 chunks
    MAX_SPEECH_CHUNKS = 300

    def __init__(
        self,
        stt_engine: STTEngine,
        vad_threshold: float = 0.5,
        silence_duration: float = 0.4,
        min_speech_duration: float = 0.3,
    ):
        self.stt = stt_engine
        self.log = get_logger("stt")
        self.sample_rate = 16000

        # Silero VAD settings
        self.vad_threshold = vad_threshold  # Silero confidence threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration

        # Amplitude pre-filter threshold (skip Silero for obvious silence)
        self._amp_threshold = 0.001

        # Load Silero VAD (CPU singleton)
        self._vad_model = _get_silero_vad()

        # Store base threshold for restoration
        self._base_threshold = vad_threshold

        # State
        self.is_running = False
        self._stop_event = threading.Event()
        self._stream: Optional[sd.InputStream] = None
        self._listen_thread: Optional[threading.Thread] = None

        # Optimization: Use bounded deque for audio buffer
        # deque(maxlen=N) provides O(1) append with automatic old data eviction
        # Prevents unbounded memory growth for long speech segments
        self._audio_buffer: Deque[np.ndarray] = deque(maxlen=self.MAX_SPEECH_CHUNKS)
        self._is_speaking = False
        self._silence_frames = 0

        # Transcription queue
        self._transcribe_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._transcribe_thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_speech_start: Optional[Callable[[], None]] = None
        self.on_speech_end: Optional[Callable[[str], None]] = None

        # TTS active flag - for echo cancellation
        self._tts_active = False

        # Optimization: Cache debug timing with perf_counter for precision
        self._last_debug_log = 0.0

    def start(self) -> None:
        """Start continuous listening."""
        if self.is_running:
            return

        self.is_running = True
        self._stop_event.clear()

        # Start transcription worker
        self._transcribe_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self._transcribe_thread.start()

        # Start listening thread
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

        self.log.info("Continuous listening started")

    def stop(self) -> None:
        """Stop continuous listening."""
        self.is_running = False
        self._stop_event.set()

        # Close stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        # Signal transcription worker to stop
        self._transcribe_queue.put(None)

        self.log.info("Continuous listening stopped")

    def set_tts_active(self, active: bool) -> None:
        """Set TTS active state for echo cancellation.

        During TTS, amplitude pre-filter is raised and Silero threshold
        is raised to require clearer speech for barge-in.
        """
        self._tts_active = active
        if active:
            self.log.debug("TTS active - thresholds raised for echo cancellation")
        else:
            self.log.debug("TTS inactive - thresholds restored")

    def _get_amp_threshold(self) -> float:
        """Amplitude pre-filter threshold based on TTS state."""
        if self._tts_active:
            return self.TTS_ACTIVE_THRESHOLD
        return self._amp_threshold

    def _get_vad_threshold(self) -> float:
        """Silero VAD confidence threshold based on TTS state."""
        if self._tts_active:
            return min(self._base_threshold + 0.2, 0.9)  # stricter during TTS
        return self._base_threshold

    def _listen_loop(self) -> None:
        """Main listening loop with Silero VAD."""
        frames_per_chunk = int(self.sample_rate * 0.1)  # 100ms chunks
        silence_chunks_needed = int(self.silence_duration / 0.1)
        min_speech_chunks = int(self.min_speech_duration / 0.1)

        # Cache frequently accessed attributes
        audio_buffer = self._audio_buffer
        get_amp_threshold = self._get_amp_threshold
        get_vad_threshold = self._get_vad_threshold
        vad_model = self._vad_model
        on_speech_start = self.on_speech_start
        transcribe_queue = self._transcribe_queue
        log = self.log

        def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            nonlocal on_speech_start

            volume = float(np.abs(indata).mean())
            amp_thresh = get_amp_threshold()

            current_time = time.perf_counter()
            if current_time - self._last_debug_log > 5.0:
                log.debug(f"Vol: {volume:.5f} | AmpT: {amp_thresh:.5f} | TTS: {self._tts_active}")
                self._last_debug_log = current_time

            # Fast amplitude pre-filter: skip Silero for obvious silence
            if volume < amp_thresh:
                if self._is_speaking:
                    audio_buffer.append(indata.copy())
                    self._silence_frames += 1

                    if self._silence_frames >= silence_chunks_needed:
                        log.debug(f"Speech END ({len(audio_buffer)} chunks)")
                        self._is_speaking = False
                        self._silence_frames = 0

                        if len(audio_buffer) >= min_speech_chunks:
                            audio_data = np.concatenate(list(audio_buffer)).flatten()
                            audio_buffer.clear()
                            transcribe_queue.put(audio_data)
                        else:
                            audio_buffer.clear()
                return

            # Run Silero VAD on chunk
            chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            chunk_tensor = torch.from_numpy(chunk.copy())
            try:
                speech_prob = float(vad_model(chunk_tensor, self.sample_rate))
            except Exception:
                speech_prob = 0.0

            vad_thresh = get_vad_threshold()
            is_speech = speech_prob > vad_thresh

            if is_speech:
                if not self._is_speaking:
                    log.debug(f"Speech START (prob: {speech_prob:.3f})")
                    self._is_speaking = True
                    audio_buffer.clear()

                    on_speech_start = self.on_speech_start
                    if on_speech_start:
                        try:
                            on_speech_start()
                        except Exception:
                            pass

                audio_buffer.append(indata.copy())
                self._silence_frames = 0

            elif self._is_speaking:
                audio_buffer.append(indata.copy())
                self._silence_frames += 1

                if self._silence_frames >= silence_chunks_needed:
                    log.debug(f"Speech END ({len(audio_buffer)} chunks)")
                    self._is_speaking = False
                    self._silence_frames = 0

                    if len(audio_buffer) >= min_speech_chunks:
                        audio_data = np.concatenate(list(audio_buffer)).flatten()
                        audio_buffer.clear()
                        transcribe_queue.put(audio_data)
                    else:
                        audio_buffer.clear()

        # Start audio stream
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=frames_per_chunk,
                callback=audio_callback,
                device=self.stt.mic_index,
            )
            self._stream.start()
            self.log.info(
                f"Mic stream started (Silero VAD, threshold: {self.vad_threshold}, "
                f"silence: {self.silence_duration}s)"
            )

            # Wait until stopped
            while not self._stop_event.is_set():
                time.sleep(0.1)

        except Exception as e:
            self.log.error(f"Mic error: {e}")

    def _transcription_worker(self) -> None:
        """Background worker that transcribes audio."""
        while True:
            try:
                audio_data = self._transcribe_queue.get(timeout=1.0)

                if audio_data is None:
                    break  # Shutdown signal

                text = self.stt.transcribe_audio(audio_data)

                if text.strip():
                    self.log.info(f"Transcribed: {text}")
                    if self.on_speech_end:
                        self.on_speech_end(text.strip())

            except queue.Empty:
                continue
            except Exception as e:
                self.log.error(f"Transcription error: {e}")


class HybridSTT:
    """Unified interface for PTT and continuous listening modes.

    Provides seamless switching between:
    - Push-to-Talk (PTT): User holds button to record
    - Continuous: Always-on VAD-based detection
    """

    def __init__(self, stt_engine: STTEngine):
        self.stt = stt_engine
        self.ptt = PushToTalkSTT(stt_engine)
        self.continuous = ContinuousSTT(stt_engine)
        self.mode = "ptt"
        self.log = get_logger("stt")

    def set_mode(self, mode: str) -> None:
        """Switch between 'ptt' and 'continuous' modes."""
        if mode not in ("ptt", "continuous"):
            raise ValueError("Mode must be 'ptt' or 'continuous'")

        # Stop current mode if switching
        if self.mode == "continuous" and mode == "ptt":
            self.continuous.stop()

        if self.mode == "ptt" and mode == "continuous":
            self.continuous.start()

        self.mode = mode
        self.log.info(f"STT mode set to: {mode.upper()}")

    def set_speech_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for when speech is transcribed (continuous mode)."""
        self.continuous.on_speech_end = callback

    def set_speech_start_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for when speech starts (continuous mode)."""
        self.continuous.on_speech_start = callback

    def set_tts_active(self, active: bool) -> None:
        """Set TTS active state for echo cancellation."""
        self.continuous.set_tts_active(active)

    def start_recording(self) -> None:
        """Start PTT recording."""
        if self.mode == "ptt":
            self.ptt.start_recording()

    def stop_recording(self) -> str:
        """Stop PTT recording and get transcription."""
        if self.mode == "ptt":
            return self.ptt.stop_recording()
        return ""

    def start_continuous(self) -> None:
        """Start continuous listening."""
        self.continuous.start()
        self.mode = "continuous"

    def stop_continuous(self) -> None:
        """Stop continuous listening."""
        self.continuous.stop()
        self.mode = "ptt"

    def stop(self) -> None:
        """Stop all listening."""
        if self.mode == "continuous":
            self.continuous.stop()


if __name__ == "__main__":
    engine = STTEngine(model_size="tiny.en", device="cpu", compute_type="int8")
    print("Available Microphones:")
    for mic in engine.list_microphones():
        print(f"  [{mic['index']}] {mic['name']} ({mic['channels']} ch)")
    print(f"Base threshold: {ContinuousSTT.BASE_THRESHOLD}")
    print(f"TTS-active threshold: {ContinuousSTT.TTS_ACTIVE_THRESHOLD}")
