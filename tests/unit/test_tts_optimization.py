"""
Tests for TTS service optimization.
Verifies removal of blocking operations and proper streaming behavior.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestTTSStreamingOptimization:
    """Test suite for TTS streaming optimization."""

    @pytest.fixture
    def mock_chatterbox_engine(self):
        """Create a mock ChatterBoxEngine."""
        engine = MagicMock()
        engine._loaded = True
        engine._model = MagicMock()
        engine._processed_reference = "test_reference.wav"

        def speak_blocking(text):
            """Simulate TTS synthesis that calls callback."""
            # Simulate generating audio chunks
            for i in range(5):
                time.sleep(0.02)  # 20ms per chunk
                # This would normally call the callback
                pass

        engine.speak_blocking = speak_blocking
        return engine

    @pytest.fixture
    def mock_tts_config(self):
        """Create a mock ChatterBoxConfig."""
        config = MagicMock()
        config.device = "cpu"
        config.voice_reference = "test.wav"
        config.cfg_weight = 0.5
        config.exaggeration = 0.5
        config.sample_rate = 24000
        config.use_torch_compile = False
        config.compile_mode = "default"
        config.enable_inference_mode = True
        config.enable_cudnn_benchmark = False
        config.enable_tf32 = False
        config.adaptive_chunking = False
        return config

    @pytest.mark.asyncio
    async def test_synthesize_no_blocking_sleep(self, mock_tts_config, mock_chatterbox_engine):
        """
        Test that synthesize() doesn't have unnecessary delays.
        The 100ms sleep should have been removed.
        """
        from src.api.services.tts_service import TTSService

        service = TTSService(voice_reference="test.wav", device="cpu")
        service.config = mock_tts_config

        # Mock synthesize_to_numpy on the engine
        audio = np.random.randn(3000).astype(np.float32)
        mock_chatterbox_engine.synthesize_to_numpy = MagicMock(
            return_value=(audio, 24000)
        )
        service._engine_speaker_mode = mock_chatterbox_engine

        start_time = time.time()
        result = await service.synthesize("Hello world")
        elapsed = time.time() - start_time

        # Should complete quickly - no 100ms sleep
        # Allow some overhead but should be well under 200ms for mock
        assert elapsed < 0.3, f"Synthesis took too long: {elapsed:.3f}s (possible blocking sleep?)"
        assert len(result) > 0, "Expected non-empty PCM bytes"

    @pytest.mark.asyncio
    async def test_synthesize_stream_yields_chunks(self):
        """
        Test that synthesize_stream yields audio chunks from synthesize_to_numpy.
        """
        from src.api.services.tts_service import TTSService

        service = TTSService(voice_reference="test.wav", device="cpu")

        mock_engine = MagicMock()
        mock_engine._loaded = True
        mock_engine._model = MagicMock()
        mock_engine._processed_reference = "test.wav"

        # Mock synthesize_to_numpy to return audio
        audio = np.random.randn(5000).astype(np.float32)
        mock_engine.synthesize_to_numpy = MagicMock(return_value=(audio, 24000))
        service._engine_speaker_mode = mock_engine

        chunks = []
        async for chunk in service.synthesize_stream("Hello"):
            chunks.append(chunk)

        assert len(chunks) >= 1, f"Expected at least 1 chunk, got {len(chunks)}"
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes > 0, "Expected non-empty audio output"

    @pytest.mark.asyncio
    async def test_speak_method_no_lock_requirement(self):
        """
        Test that speak() method works without blocking other operations.
        """
        from src.api.services.tts_service import TTSService

        service = TTSService(voice_reference="test.wav", device="cpu")

        mock_engine = MagicMock()
        mock_engine._loaded = True
        mock_engine.synthesize_to_numpy = MagicMock(
            return_value=(np.zeros(1000, dtype=np.float32), 24000)
        )

        service._engine_speaker_mode = mock_engine

        # Playback goes through the persistent AudioOutput now (Phase 1.2)
        mock_out = MagicMock()
        mock_out.wait_drained = MagicMock(return_value=True)
        with patch("src.api.services.audio_output.get_audio_output", return_value=mock_out):
            await service.speak("Hello world")

        mock_engine.synthesize_to_numpy.assert_called_once()
        mock_out.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_speak_empty_text_skipped(self):
        """Test that empty text is handled gracefully."""
        from src.api.services.tts_service import TTSService

        service = TTSService(voice_reference="test.wav", device="cpu")

        mock_engine = MagicMock()
        mock_engine._loaded = True

        service._engine_speaker_mode = mock_engine

        # Should not call speak_blocking for empty text
        await service.speak("")
        await service.speak("   ")

        mock_engine.speak_blocking.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_not_initialized(self):
        """Test that synthesize raises error when not initialized."""
        from src.api.services.tts_service import TTSService
        from src.api.utils.exceptions import SynthesisError

        service = TTSService(voice_reference="test.wav", device="cpu")
        # Don't initialize

        with pytest.raises(SynthesisError) as exc_info:
            await service.synthesize("Hello")

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_synthesize_stream_not_initialized(self):
        """Test that synthesize_stream raises error when not initialized."""
        from src.api.services.tts_service import TTSService
        from src.api.utils.exceptions import SynthesisError

        service = TTSService(voice_reference="test.wav", device="cpu")
        # Don't initialize

        with pytest.raises(SynthesisError) as exc_info:
            async for chunk in service.synthesize_stream("Hello"):
                pass

        assert "not initialized" in str(exc_info.value).lower()


class TestTTSServiceHealth:
    """Test TTS service health and status methods."""

    @pytest.mark.asyncio
    async def test_health_check_initialized(self):
        """Test health check when service is initialized."""
        from src.api.services.tts_service import TTSService

        service = TTSService(voice_reference="test.wav", device="cpu")

        mock_engine = MagicMock()
        mock_engine._loaded = True

        service._engine_speaker_mode = mock_engine

        result = await service.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Test health check when service is not initialized."""
        from src.api.services.tts_service import TTSService

        service = TTSService(voice_reference="test.wav", device="cpu")
        # Don't initialize

        result = await service.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test get_status returns expected information."""
        from src.api.services.tts_service import TTSService

        service = TTSService(voice_reference="test.wav", device="cpu")

        mock_engine = MagicMock()
        mock_engine._loaded = True
        mock_engine._generation_count = 5
        mock_engine.get_metrics.return_value = None

        service._engine_speaker_mode = mock_engine

        status = await service.get_status()

        assert status["initialized"] is True
        assert status["model_loaded"] is True
        assert status["generation_count"] == 5
        assert "optimizations" in status

    @pytest.mark.asyncio
    async def test_stop_method(self):
        """Test stop method for barge-in support."""
        from src.api.services.tts_service import TTSService

        service = TTSService(voice_reference="test.wav", device="cpu")

        mock_engine = MagicMock()
        mock_engine._loaded = True
        mock_engine.stop = MagicMock()

        service._engine_speaker_mode = mock_engine

        await service.stop()

        mock_engine.stop.assert_called_once()


class TestTTSAudioConversion:
    """Test audio format conversion in TTS service."""

    def test_float32_to_int16_conversion(self):
        """Test that float32 audio is correctly converted to int16."""
        # Simulate the conversion that happens in the callback
        float_audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        int16_audio = (float_audio * 32767).astype(np.int16)

        expected = np.array([0, 16383, -16383, 32767, -32767], dtype=np.int16)
        np.testing.assert_array_almost_equal(int16_audio, expected, decimal=0)

    def test_audio_bytes_conversion(self):
        """Test that int16 audio is correctly converted to bytes."""
        int16_audio = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
        audio_bytes = int16_audio.tobytes()

        # Verify we can convert back
        recovered = np.frombuffer(audio_bytes, dtype=np.int16)
        np.testing.assert_array_equal(int16_audio, recovered)
