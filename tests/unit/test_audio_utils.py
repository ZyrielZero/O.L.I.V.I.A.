"""
Unit tests for audio utility functions.
Tests encoding, decoding, and format conversions.
"""

import base64
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.utils.audio_utils import (
    bytes_to_numpy_audio,
    decode_base64_to_audio,
    encode_audio_to_base64,
    numpy_audio_to_float,
)


class TestEncodeAudioToBase64:
    """Tests for base64 audio encoding."""

    @pytest.mark.unit
    def test_encode_simple_bytes(self):
        """Test encoding simple byte data."""
        audio_bytes = b'\x00\x01\x02\x03\x04'
        result = encode_audio_to_base64(audio_bytes)
        assert isinstance(result, str)
        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert decoded == audio_bytes

    @pytest.mark.unit
    def test_encode_empty_bytes(self):
        """Test encoding empty byte data."""
        audio_bytes = b''
        result = encode_audio_to_base64(audio_bytes)
        assert result == ''

    @pytest.mark.unit
    def test_encode_audio_pcm_data(self):
        """Test encoding realistic PCM audio data."""
        # Generate 100 samples of 16-bit audio
        audio_array = np.random.randint(-32768, 32767, size=100, dtype=np.int16)
        audio_bytes = audio_array.tobytes()

        result = encode_audio_to_base64(audio_bytes)

        # Verify roundtrip
        decoded = base64.b64decode(result)
        assert decoded == audio_bytes

    @pytest.mark.unit
    def test_encode_large_audio(self):
        """Test encoding larger audio data (1 second at 16kHz)."""
        # 16000 samples = 1 second at 16kHz, 16-bit = 32000 bytes
        audio_array = np.zeros(16000, dtype=np.int16)
        audio_bytes = audio_array.tobytes()

        result = encode_audio_to_base64(audio_bytes)

        assert len(result) > 0
        # Base64 encoding increases size by ~33%
        assert len(result) > len(audio_bytes)


class TestDecodeBase64ToAudio:
    """Tests for base64 audio decoding."""

    @pytest.mark.unit
    def test_decode_simple_base64(self):
        """Test decoding simple base64 string."""
        original = b'\x00\x01\x02\x03\x04'
        encoded = base64.b64encode(original).decode('utf-8')

        result = decode_base64_to_audio(encoded)
        assert result == original

    @pytest.mark.unit
    def test_decode_empty_string(self):
        """Test decoding empty base64 string."""
        result = decode_base64_to_audio('')
        assert result == b''

    @pytest.mark.unit
    def test_decode_invalid_base64(self):
        """Test decoding invalid base64 raises error."""
        with pytest.raises(Exception):  # base64.binascii.Error
            decode_base64_to_audio('not-valid-base64!!!')

    @pytest.mark.unit
    def test_encode_decode_roundtrip(self):
        """Test that encode/decode roundtrip preserves data."""
        original = np.random.bytes(1000)

        encoded = encode_audio_to_base64(original)
        decoded = decode_base64_to_audio(encoded)

        assert decoded == original


class TestBytesToNumpyAudio:
    """Tests for bytes to numpy conversion."""

    @pytest.mark.unit
    def test_convert_int16_audio(self):
        """Test converting 16-bit PCM bytes to numpy array."""
        # Create known int16 values
        original_array = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
        audio_bytes = original_array.tobytes()

        result = bytes_to_numpy_audio(audio_bytes, dtype=np.int16)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, original_array)

    @pytest.mark.unit
    def test_convert_empty_bytes(self):
        """Test converting empty bytes."""
        result = bytes_to_numpy_audio(b'', dtype=np.int16)
        assert len(result) == 0
        assert result.dtype == np.int16

    @pytest.mark.unit
    def test_convert_float32_audio(self):
        """Test converting float32 audio bytes."""
        original_array = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        audio_bytes = original_array.tobytes()

        result = bytes_to_numpy_audio(audio_bytes, dtype=np.float32)

        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, original_array)

    @pytest.mark.unit
    def test_sample_count_calculation(self):
        """Test that sample count is calculated correctly."""
        # 100 int16 samples = 200 bytes
        audio_bytes = np.zeros(100, dtype=np.int16).tobytes()
        result = bytes_to_numpy_audio(audio_bytes, dtype=np.int16)
        assert len(result) == 100


class TestNumpyAudioToFloat:
    """Tests for numpy int to float conversion."""

    @pytest.mark.unit
    def test_convert_silence(self):
        """Test converting silence (zeros)."""
        audio_int = np.zeros(100, dtype=np.int16)
        result = numpy_audio_to_float(audio_int)

        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.zeros(100, dtype=np.float32))

    @pytest.mark.unit
    def test_convert_max_positive(self):
        """Test converting maximum positive value."""
        audio_int = np.array([32767], dtype=np.int16)
        result = numpy_audio_to_float(audio_int)

        # Should be close to but not exceed 1.0
        assert result[0] < 1.0
        assert result[0] > 0.99  # 32767/32768 = 0.99997

    @pytest.mark.unit
    def test_convert_max_negative(self):
        """Test converting maximum negative value."""
        audio_int = np.array([-32768], dtype=np.int16)
        result = numpy_audio_to_float(audio_int)

        # Should be exactly -1.0
        assert result[0] == -1.0

    @pytest.mark.unit
    def test_convert_range(self):
        """Test that output is in range [-1.0, 1.0]."""
        # Random int16 values
        audio_int = np.random.randint(-32768, 32768, size=1000, dtype=np.int16)
        result = numpy_audio_to_float(audio_int)

        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    @pytest.mark.unit
    def test_preserve_shape(self):
        """Test that shape is preserved."""
        audio_int = np.zeros((10, 2), dtype=np.int16)  # Stereo-like
        result = numpy_audio_to_float(audio_int)
        assert result.shape == (10, 2)

    @pytest.mark.unit
    def test_dtype_is_float32(self):
        """Test that output dtype is float32."""
        audio_int = np.array([0], dtype=np.int16)
        result = numpy_audio_to_float(audio_int)
        assert result.dtype == np.float32


class TestAudioUtilsIntegration:
    """Integration tests for audio utility functions together."""

    @pytest.mark.unit
    def test_full_audio_pipeline(self):
        """Test complete pipeline: bytes -> numpy -> float -> back."""
        # Create realistic audio signal (sine wave)
        sample_rate = 16000
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        frequency = 440  # A4 note
        sine_wave = np.sin(2 * np.pi * frequency * t)

        # Convert to int16 (simulating recording)
        audio_int = (sine_wave * 32767).astype(np.int16)
        audio_bytes = audio_int.tobytes()

        # Pipeline: base64 encode -> decode -> numpy -> float
        encoded = encode_audio_to_base64(audio_bytes)
        decoded = decode_base64_to_audio(encoded)
        numpy_audio = bytes_to_numpy_audio(decoded, dtype=np.int16)
        float_audio = numpy_audio_to_float(numpy_audio)

        # Verify final result
        assert float_audio.dtype == np.float32
        assert len(float_audio) == len(audio_int)
        assert np.all(float_audio >= -1.0)
        assert np.all(float_audio <= 1.0)

    @pytest.mark.unit
    def test_websocket_audio_simulation(self):
        """Simulate WebSocket audio transmission scenario."""
        # Client sends audio as base64
        original_audio = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
        client_payload = encode_audio_to_base64(original_audio.tobytes())

        # Server receives and processes
        server_bytes = decode_base64_to_audio(client_payload)
        server_audio = bytes_to_numpy_audio(server_bytes, dtype=np.int16)
        float_for_whisper = numpy_audio_to_float(server_audio)

        # Verify integrity
        np.testing.assert_array_equal(server_audio, original_audio)
        assert float_for_whisper.dtype == np.float32
