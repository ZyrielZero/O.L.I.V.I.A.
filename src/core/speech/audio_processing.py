"""Audio preprocessing and post-processing for speech synthesis.

Reference audio validation, preprocessing, and streaming chunk crossfading.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from scipy.signal import resample

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

log = logging.getLogger("AudioProcessing")


@dataclass
class AudioQualityConfig:
    """Configuration for audio quality requirements and processing parameters."""

    # Reference audio requirements
    min_duration_sec: float = 4.0  # Lowered from 5.0 to accept 4.9s reference
    recommended_duration_sec: float = 10.0
    target_sample_rate: int = 24000

    # Preprocessing
    normalize_peak_db: float = -3.0
    remove_dc_offset: bool = True
    trim_silence_db: float = -40.0

    # Post-processing (conservative approach)
    crossfade_duration_ms: int = 10
    micro_fade_ms: int = 5

    # Caching
    cache_processed_reference: bool = True


class ReferenceAudioValidator:
    """Validates and preprocesses reference audio files for voice cloning."""

    def __init__(self, config: Optional[AudioQualityConfig] = None):
        """Initialize the validator.

        Args:
            config: Audio quality configuration. Uses defaults if not provided.
        """
        self.config = config or AudioQualityConfig()
        self._cache: Dict[str, str] = {}  # Maps original path to processed path

    def validate(self, audio_path: str) -> Tuple[bool, List[str]]:
        """Validate a reference audio file.

        Checks file existence, duration, sample rate, channels, and format.

        Args:
            audio_path: Path to the audio file to validate.

        Returns:
            Tuple of (is_valid, warnings_list). is_valid is False only for
            critical failures (file not found, too short).
        """
        warnings = []
        is_valid = True

        # Check file exists
        if not os.path.exists(audio_path):
            log.error(f"Reference audio file not found: {audio_path}")
            return False, ["File not found"]

        if sf is None:
            log.warning("soundfile not available, skipping detailed validation")
            return True, ["soundfile not available for validation"]

        try:
            info = sf.info(audio_path)
        except Exception as e:
            log.error(f"Failed to read audio file info: {e}")
            return False, [f"Cannot read file: {e}"]

        # Check duration
        duration = info.duration
        if duration < self.config.min_duration_sec:
            log.error(
                f"Reference audio too short: {duration:.1f}s "
                f"(minimum: {self.config.min_duration_sec}s)"
            )
            is_valid = False
            warnings.append(
                f"Duration {duration:.1f}s is below minimum {self.config.min_duration_sec}s"
            )
        elif duration < self.config.recommended_duration_sec:
            log.warning(
                f"Reference audio shorter than recommended: {duration:.1f}s "
                f"(recommended: {self.config.recommended_duration_sec}s)"
            )
            warnings.append(
                f"Duration {duration:.1f}s is below recommended {self.config.recommended_duration_sec}s"
            )

        # Check sample rate
        if info.samplerate < self.config.target_sample_rate:
            log.warning(
                f"Reference audio sample rate {info.samplerate}Hz is below "
                f"target {self.config.target_sample_rate}Hz"
            )
            warnings.append(
                f"Sample rate {info.samplerate}Hz is below target {self.config.target_sample_rate}Hz"
            )

        # Check channels
        if info.channels > 1:
            log.warning(
                f"Reference audio is stereo ({info.channels} channels), will be converted to mono"
            )
            warnings.append(f"Audio has {info.channels} channels, will convert to mono")

        # Check format
        path = Path(audio_path)
        if path.suffix.lower() not in [".wav", ".wave"]:
            log.warning(
                f"Reference audio format {path.suffix} is not WAV, may have quality implications"
            )
            warnings.append(f"Format {path.suffix} is not WAV")

        return is_valid, warnings

    def preprocess(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """Preprocess a reference audio file.

        Processing steps:
        1. Load audio
        2. Convert stereo to mono
        3. Remove DC offset
        4. Resample to target rate (if scipy available)
        5. Trim silence from start/end
        6. Normalize peak amplitude
        7. Save processed audio

        Args:
            audio_path: Path to the input audio file.
            output_path: Path for processed output. If None, uses input.processed.wav.

        Returns:
            Path to the processed audio file.

        Raises:
            RuntimeError: If soundfile is not available or processing fails.
        """
        if sf is None:
            raise RuntimeError("soundfile library is required for audio preprocessing")

        # Check cache
        if self.config.cache_processed_reference and audio_path in self._cache:
            cached_path = self._cache[audio_path]
            if os.path.exists(cached_path):
                log.debug(f"Using cached processed audio: {cached_path}")
                return cached_path

        # Determine output path
        if output_path is None:
            path = Path(audio_path)
            output_path = str(path.parent / f"{path.stem}.processed.wav")

        log.info(f"Preprocessing reference audio: {audio_path}")

        try:
            # Load audio
            audio, sample_rate = sf.read(audio_path, dtype="float32")
            log.debug(f"Loaded audio: shape={audio.shape}, sr={sample_rate}")

            # Convert stereo to mono
            if audio.ndim > 1:
                log.debug("Converting stereo to mono")
                audio = np.mean(audio, axis=1)

            # Remove DC offset
            if self.config.remove_dc_offset:
                dc_offset = np.mean(audio)
                if abs(dc_offset) > 1e-6:
                    log.debug(f"Removing DC offset: {dc_offset:.6f}")
                    audio = audio - dc_offset

            # Resample if needed
            if sample_rate != self.config.target_sample_rate:
                if SCIPY_AVAILABLE:
                    log.debug(
                        f"Resampling from {sample_rate}Hz to {self.config.target_sample_rate}Hz"
                    )
                    num_samples = int(len(audio) * self.config.target_sample_rate / sample_rate)
                    audio = resample(audio, num_samples)
                    sample_rate = self.config.target_sample_rate
                else:
                    log.warning(
                        "scipy not available, skipping resampling. "
                        "Audio may not be at optimal sample rate."
                    )

            # Trim silence from start/end
            audio = self._trim_silence(audio, sample_rate)

            # Normalize peak
            audio = self._normalize_peak(audio)

            # Save processed audio
            sf.write(output_path, audio, sample_rate)
            log.info(f"Saved processed audio: {output_path}")

            # Cache result
            if self.config.cache_processed_reference:
                self._cache[audio_path] = output_path

            return output_path

        except Exception as e:
            log.error(f"Failed to preprocess audio: {e}")
            raise RuntimeError(f"Audio preprocessing failed: {e}") from e

    def _trim_silence(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Trim silence from start and end of audio.

        Args:
            audio: Audio samples as numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            Trimmed audio array.
        """
        threshold_db = self.config.trim_silence_db
        threshold_linear = 10 ** (threshold_db / 20)

        # Find start (first sample above threshold)
        abs_audio = np.abs(audio)
        above_threshold = abs_audio > threshold_linear

        if not np.any(above_threshold):
            log.warning("Audio appears to be all silence")
            return audio

        start_idx = np.argmax(above_threshold)

        # Find end (last sample above threshold)
        end_idx = len(audio) - np.argmax(above_threshold[::-1])

        # Add small buffer (10ms)
        buffer_samples = int(0.01 * sample_rate)
        start_idx = max(0, start_idx - buffer_samples)
        end_idx = min(len(audio), end_idx + buffer_samples)

        trimmed = audio[start_idx:end_idx]

        trim_start_ms = start_idx / sample_rate * 1000
        trim_end_ms = (len(audio) - end_idx) / sample_rate * 1000

        if trim_start_ms > 10 or trim_end_ms > 10:
            log.debug(f"Trimmed {trim_start_ms:.0f}ms from start, {trim_end_ms:.0f}ms from end")

        return trimmed

    def _normalize_peak(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target peak level.

        Args:
            audio: Audio samples as numpy array.

        Returns:
            Normalized audio array.
        """
        peak = np.max(np.abs(audio))
        if peak < 1e-6:
            log.warning("Audio has very low amplitude, skipping normalization")
            return audio

        target_linear = 10 ** (self.config.normalize_peak_db / 20)
        gain = target_linear / peak

        log.debug(
            f"Normalizing: peak={20 * np.log10(peak):.1f}dB, gain={20 * np.log10(gain):.1f}dB"
        )

        return audio * gain

    def clear_cache(self) -> None:
        """Clear the processed audio cache."""
        self._cache.clear()
        log.debug("Cleared audio processing cache")


class OutputAudioProcessor:
    """Post-processor for streaming audio chunks.

    Handles crossfading between chunks and applying fades at utterance
    boundaries to eliminate clicks and discontinuities.
    """

    def __init__(self, config: Optional[AudioQualityConfig] = None):
        """Initialize the processor.

        Args:
            config: Audio quality configuration. Uses defaults if not provided.
        """
        self.config = config or AudioQualityConfig()
        self._previous_tail: Optional[np.ndarray] = None
        self._previous_sample_rate: Optional[int] = None

    def reset(self) -> None:
        """Reset state for a new utterance."""
        self._previous_tail = None
        self._previous_sample_rate = None
        log.debug("Reset audio processor state")

    def process_chunk(
        self, audio: np.ndarray, sample_rate: int, is_first: bool, is_final: bool
    ) -> np.ndarray:
        """Process an audio chunk with crossfading and fades.

        Args:
            audio: Audio samples as numpy array.
            sample_rate: Sample rate of the audio.
            is_first: True if this is the first chunk of an utterance.
            is_final: True if this is the last chunk of an utterance.

        Returns:
            Processed audio chunk.
        """
        if len(audio) == 0:
            return audio

        # Work with a copy to avoid modifying input
        processed = audio.astype(np.float32, copy=True)

        # Apply crossfade with previous chunk (if not first)
        if not is_first and self._previous_tail is not None:
            processed = self._apply_crossfade(processed, sample_rate)

        # Store tail for next chunk's crossfade (if not final)
        if not is_final:
            crossfade_samples = int(self.config.crossfade_duration_ms * sample_rate / 1000)
            if len(processed) > crossfade_samples:
                self._previous_tail = processed[-crossfade_samples:].copy()
                self._previous_sample_rate = sample_rate
                # Return audio without the tail (will be blended with next chunk)
                processed = processed[:-crossfade_samples]
            else:
                # Chunk too small for crossfading, just store it
                self._previous_tail = processed.copy()
                self._previous_sample_rate = sample_rate
                return np.array([], dtype=np.float32)
        else:
            # Final chunk - include all audio and clear state
            if self._previous_tail is not None:
                # Prepend any remaining tail from previous chunk
                processed = np.concatenate([self._previous_tail, processed])
            self._previous_tail = None
            self._previous_sample_rate = None

        # Apply micro fade-in on first chunk
        if is_first:
            processed = self._apply_fade_in(processed, sample_rate)

        # Apply micro fade-out on final chunk
        if is_final:
            processed = self._apply_fade_out(processed, sample_rate)

        return processed

    def _apply_crossfade(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply linear crossfade with the previous chunk's tail.

        Args:
            audio: Current chunk audio.
            sample_rate: Sample rate of the audio.

        Returns:
            Audio with crossfade applied at the start.
        """
        if self._previous_tail is None:
            return audio

        tail = self._previous_tail
        crossfade_len = len(tail)

        if len(audio) < crossfade_len:
            # Current chunk smaller than crossfade - blend what we can
            crossfade_len = len(audio)
            tail = tail[-crossfade_len:]

        # Create linear crossfade
        fade_out = np.linspace(1.0, 0.0, crossfade_len, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, crossfade_len, dtype=np.float32)

        # Apply crossfade to the overlap region
        crossfaded = tail * fade_out + audio[:crossfade_len] * fade_in

        # Concatenate: crossfaded region + rest of current chunk
        result = np.concatenate([crossfaded, audio[crossfade_len:]])

        return result

    def _apply_fade_in(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply micro fade-in at the start of audio.

        Args:
            audio: Audio samples.
            sample_rate: Sample rate.

        Returns:
            Audio with fade-in applied.
        """
        fade_samples = int(self.config.micro_fade_ms * sample_rate / 1000)
        fade_samples = min(fade_samples, len(audio))

        if fade_samples < 2:
            return audio

        fade_curve = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        audio[:fade_samples] *= fade_curve

        return audio

    def _apply_fade_out(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply micro fade-out at the end of audio.

        Args:
            audio: Audio samples.
            sample_rate: Sample rate.

        Returns:
            Audio with fade-out applied.
        """
        fade_samples = int(self.config.micro_fade_ms * sample_rate / 1000)
        fade_samples = min(fade_samples, len(audio))

        if fade_samples < 2:
            return audio

        fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        audio[-fade_samples:] *= fade_curve

        return audio
