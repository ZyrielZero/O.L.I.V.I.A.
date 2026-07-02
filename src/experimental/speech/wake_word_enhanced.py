"""Enhanced Wake Word Detection for O.L.I.V.I.A.
Uses openWakeWord with "hey_jarvis" as proxy for "Hey Olivia".
"""

import random
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from src.utils.logger import get_logger

    log = get_logger("wake")
except ImportError:
    import logging

    log = logging.getLogger("wake")

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    from openwakeword.model import Model as OWWModel

    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""

    # Detection settings
    wake_words: List[str] = field(default_factory=lambda: ["hey_jarvis"])
    threshold: float = 0.5
    cooldown_seconds: float = 2.0

    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 1280  # 80ms at 16kHz

    # Behavior
    pause_during_tts: bool = True
    resume_delay_seconds: float = 0.5

    # Custom model path (for future custom training)
    custom_model_path: Optional[str] = None


class EnhancedWakeWordDetector:
    """Enhanced wake word detection with better integration.

    Features:
    - Smooth TTS integration (pauses during speech)
    - Configurable callbacks
    - Statistics tracking
    - Ready for custom model integration
    """

    # Pre-trained models
    AVAILABLE_MODELS = ["hey_jarvis", "hey_mycroft", "alexa", "timer", "weather"]

    def __init__(self, config: Optional[WakeWordConfig] = None):
        self.config = config or WakeWordConfig()

        self.model: Optional[Any] = None
        self.is_running = False
        self._stop_event = threading.Event()
        self._listen_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

        # TTS integration
        self._is_paused = False
        self._pause_until: float = 0

        # Stats
        self.detections = 0
        self.false_positives_rejected = 0
        self._last_detection_time: float = 0
        self._detection_history: List[Tuple[str, float, float]] = []  # (word, score, time)

        # Callbacks
        self.on_wake_word: Optional[Callable[[str, float], None]] = None
        self.on_listening_start: Optional[Callable[[], None]] = None
        self.on_listening_stop: Optional[Callable[[], None]] = None
        self.on_detection_rejected: Optional[Callable[[str, float, str], None]] = None

        log.info("🎯 EnhancedWakeWordDetector initialized")
        log.info(f"   Wake words: {self.config.wake_words}")
        log.info(f"   Threshold: {self.config.threshold}")

    def load_model(self) -> bool:
        """Load wake word model(s)."""
        if not OPENWAKEWORD_AVAILABLE:
            log.error("❌ openwakeword not available")
            return False

        try:
            # Check for custom model
            model_paths = []
            if self.config.custom_model_path:
                custom_path = Path(self.config.custom_model_path)
                if custom_path.exists():
                    model_paths.append(str(custom_path))
                    log.info(f"📥 Loading custom model: {custom_path}")

            # Add pre-trained models
            for wake_word in self.config.wake_words:
                if wake_word in self.AVAILABLE_MODELS:
                    model_paths.append(wake_word)

            if not model_paths:
                log.error("❌ No valid wake word models specified")
                return False

            self.model = OWWModel(wakeword_models=model_paths, inference_framework="onnx")

            log.info(f"✅ Wake word models loaded: {model_paths}")
            return True

        except Exception as e:
            log.error(f"❌ Failed to load wake word model: {e}")
            return False

    def start(self) -> bool:
        """Start listening for wake words."""
        if not SOUNDDEVICE_AVAILABLE:
            log.error("❌ sounddevice not available")
            return False

        if self.model is None:
            if not self.load_model():
                return False

        if self.is_running:
            return True

        self.is_running = True
        self._stop_event.clear()

        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

        if self.on_listening_start:
            self.on_listening_start()

        log.info("🎤 Wake word detection started")
        return True

    def stop(self):
        """Stop listening."""
        self.is_running = False
        self._stop_event.set()

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)

        if self.on_listening_stop:
            self.on_listening_stop()

        log.info("🛑 Wake word detection stopped")

    def pause(self, duration: Optional[float] = None):
        """Pause detection temporarily.

        Args:
            duration: Seconds to pause. None = indefinite until resume()
        """
        self._is_paused = True
        if duration:
            self._pause_until = time.time() + duration
        else:
            self._pause_until = float("inf")
        log.debug("⏸️ Wake word paused" + (f" for {duration}s" if duration else ""))

    def resume(self):
        """Resume detection after pause."""
        self._is_paused = False
        self._pause_until = 0
        log.debug("▶️ Wake word resumed")

    def set_tts_active(self, active: bool):
        """Set TTS active state for automatic pausing.

        When TTS is active, detection is paused to avoid
        false triggers from the AI's voice.
        """
        if self.config.pause_during_tts:
            if active:
                self.pause()
            else:
                # Brief delay before resuming
                time.sleep(self.config.resume_delay_seconds)
                self.resume()

    def set_threshold(self, threshold: float):
        """Update detection threshold (0.0-1.0)."""
        self.config.threshold = max(0.0, min(1.0, threshold))
        log.info(f"🎯 Threshold updated: {self.config.threshold}")

    def _listen_loop(self):
        """Main listening loop."""

        def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any):
            # Check pause state
            if self._is_paused:
                if time.time() < self._pause_until:
                    return
                else:
                    self._is_paused = False

            if not self.is_running:
                return

            try:
                audio_data = indata.flatten().astype(np.float32)
                predictions = self.model.predict(audio_data)

                for wake_word, score in predictions.items():
                    if score >= self.config.threshold:
                        current_time = time.time()

                        # Check cooldown
                        if current_time - self._last_detection_time < self.config.cooldown_seconds:
                            self.false_positives_rejected += 1
                            if self.on_detection_rejected:
                                self.on_detection_rejected(wake_word, score, "cooldown")
                            continue

                        # Valid detection
                        self._last_detection_time = current_time
                        self.detections += 1
                        self._detection_history.append((wake_word, score, current_time))

                        # Keep history limited
                        if len(self._detection_history) > 100:
                            self._detection_history = self._detection_history[-100:]

                        log.info(f"🔔 Wake word: '{wake_word}' (score: {score:.2f})")

                        if self.on_wake_word:
                            threading.Thread(
                                target=self.on_wake_word, args=(wake_word, score), daemon=True
                            ).start()

                        # Reset model state
                        self.model.reset()
                        break

            except Exception as e:
                log.error(f"Detection error: {e}")

        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.config.chunk_size,
                callback=audio_callback,
            )
            self._stream.start()

            log.info("🎤 Wake word audio stream active")

            while not self._stop_event.is_set():
                time.sleep(0.1)

        except Exception as e:
            log.error(f"❌ Audio stream error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "detections": self.detections,
            "false_positives_rejected": self.false_positives_rejected,
            "is_running": self.is_running,
            "is_paused": self._is_paused,
            "threshold": self.config.threshold,
            "recent_detections": self._detection_history[-5:] if self._detection_history else [],
        }


class OliviaWakeWord(EnhancedWakeWordDetector):
    """Pre-configured wake word detector for O.L.I.V.I.A.

    Uses "hey_jarvis" (phonetically similar to "Hey Olivia")
    with optimized settings.
    """

    def __init__(self, threshold: float = 0.55, cooldown: float = 2.0):
        config = WakeWordConfig(
            wake_words=["hey_jarvis"],
            threshold=threshold,
            cooldown_seconds=cooldown,
            pause_during_tts=True,
        )
        super().__init__(config)

        log.info("🎯 OliviaWakeWord ready (say 'Hey Jarvis' to activate)")


class WakeWordTrainingGenerator:
    """Generates training data for custom wake word.

    Creates synthetic audio samples for training openWakeWord.
    """

    def __init__(
        self, wake_phrase: str = "Hey Olivia", output_dir: str = "training_data/wake_word"
    ):
        self.wake_phrase = wake_phrase
        self.output_dir = Path(output_dir)

        # Variations to generate
        self.variations = [
            wake_phrase,
            f"{wake_phrase}!",
            f"{wake_phrase}?",
            wake_phrase.lower(),
            wake_phrase.upper(),
        ]

    def generate_with_tts(
        self, tts_engine: Any, num_samples: int = 100, speed_range: Tuple[float, float] = (0.9, 1.1)
    ) -> int:
        """Generate training samples using TTS.

        Args:
            tts_engine: TTSEngine instance
            num_samples: Number of samples to generate
            speed_range: Speed variation range

        Returns:
            Number of samples generated
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        positive_dir = self.output_dir / "positive"
        positive_dir.mkdir(exist_ok=True)

        generated = 0

        log.info(f"🎵 Generating {num_samples} training samples...")

        for i in range(num_samples):
            try:
                # Random variation
                phrase = random.choice(self.variations)

                # Generate audio
                audio = tts_engine.generate_audio(phrase)

                # Save as WAV
                filename = f"wake_{i:04d}.wav"
                filepath = positive_dir / filename

                self._save_wav(audio, filepath, tts_engine.sample_rate or 22050)
                generated += 1

                if (i + 1) % 10 == 0:
                    log.info(f"   Generated {i + 1}/{num_samples}")

            except Exception as e:
                log.error(f"Failed to generate sample {i}: {e}")

        log.info(f"✅ Generated {generated} training samples in {positive_dir}")
        return generated

    def generate_negative_samples(
        self, tts_engine: Any, phrases: List[str], num_per_phrase: int = 10
    ) -> int:
        """Generate negative (non-wake-word) samples."""
        negative_dir = self.output_dir / "negative"
        negative_dir.mkdir(parents=True, exist_ok=True)

        generated = 0

        for phrase in phrases:
            for i in range(num_per_phrase):
                try:
                    audio = tts_engine.generate_audio(phrase)
                    filename = f"negative_{generated:04d}.wav"
                    filepath = negative_dir / filename
                    self._save_wav(audio, filepath, tts_engine.sample_rate or 22050)
                    generated += 1
                except Exception:
                    pass

        log.info(f"✅ Generated {generated} negative samples in {negative_dir}")
        return generated

    def _save_wav(self, audio: np.ndarray, path: Path, sample_rate: int):
        """Save audio as WAV file."""
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.95

        # Convert to 16-bit int
        audio_int = (audio * 32767).astype(np.int16)

        with wave.open(str(path), "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int.tobytes())

    def get_training_instructions(self) -> str:
        """Get instructions for training custom wake word."""
        return f"""
═══ CUSTOM WAKE WORD TRAINING INSTRUCTIONS ═══

1. Generate training data:
   - Run generate_with_tts() to create positive samples
   - Run generate_negative_samples() for negative samples

2. Install openWakeWord training:
   pip install openwakeword[training]

3. Train the model:
   python -m openwakeword.train \\
       --positive_dir {self.output_dir}/positive \\
       --negative_dir {self.output_dir}/negative \\
       --output_dir models/hey_olivia \\
       --model_name hey_olivia

4. Use the trained model:
   config = WakeWordConfig(
       custom_model_path="models/hey_olivia/hey_olivia.onnx"
   )
   detector = EnhancedWakeWordDetector(config)

═══════════════════════════════════════════════
"""


_detector: Optional[EnhancedWakeWordDetector] = None


def get_wake_detector() -> Optional[EnhancedWakeWordDetector]:
    """Get the global wake word detector."""
    return _detector


def create_wake_detector(
    wake_words: Optional[List[str]] = None,
    threshold: float = 0.5,
    on_wake: Optional[Callable[[str, float], None]] = None,
) -> Optional[EnhancedWakeWordDetector]:
    """Create and configure wake word detector."""
    global _detector

    if not OPENWAKEWORD_AVAILABLE:
        log.warning("Wake word unavailable (openwakeword not installed)")
        return None

    config = WakeWordConfig(wake_words=wake_words or ["hey_jarvis"], threshold=threshold)

    _detector = EnhancedWakeWordDetector(config)

    if on_wake:
        _detector.on_wake_word = on_wake

    return _detector


def is_wake_available() -> bool:
    """Check if wake word detection is available."""
    return OPENWAKEWORD_AVAILABLE and SOUNDDEVICE_AVAILABLE


if __name__ == "__main__":
    if not is_wake_available():
        print("Wake word detection not available - install openwakeword sounddevice")
        exit(1)

    detector = OliviaWakeWord(threshold=0.5)
    detector.on_wake_word = lambda word, score: print(f"DETECTED: '{word}' ({score:.2f})")

    print("Say 'Hey Jarvis' to trigger (Ctrl+C to stop)")
    if detector.start():
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            detector.stop()
            print(f"Stats: {detector.get_stats()}")
