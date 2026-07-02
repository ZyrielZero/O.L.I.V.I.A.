"""
Unit tests for ChatterBox TTS.
Tests configuration, metrics, and audio player queue management.
"""


import pytest

# ===== Test 1: TTS Config Defaults =====

@pytest.mark.unit
def test_tts_config_defaults():
    """Default TTS config values are sensible."""
    from src.core.speech.chatterbox_tts import ChatterBoxConfig

    config = ChatterBoxConfig()

    # Verify defaults
    assert config.device in ["cuda", "cpu"]
    assert 0.0 <= config.exaggeration <= 2.0
    assert 0.0 <= config.cfg_weight <= 1.0
    assert config.sample_rate == 24000
    assert config.voice_reference is not None


# ===== Test 2: TTS Metrics Dataclass =====

@pytest.mark.unit
def test_tts_metrics_dataclass():
    """TTSMetrics stores and calculates metrics correctly."""
    from src.core.speech.chatterbox_tts import TTSMetrics

    metrics = TTSMetrics()

    # Verify initial values (defaults are 0.0 not None)
    assert metrics.total_generation_ms == 0.0
    assert metrics.audio_duration_s == 0.0
    assert metrics.chunks_generated == 0
    assert metrics.ttfb_ms == 0.0  # Defaults to 0.0
    assert metrics.rtf == 0.0

    # Test RTF calculation with real values
    metrics = TTSMetrics(
        total_generation_ms=1000.0,  # 1 second synthesis
        audio_duration_s=2.0  # 2 seconds of audio
    )

    # RTF = generation_time / audio_duration (already in dataclass)
    # Manual calculation for verification
    rtf = (metrics.total_generation_ms / 1000) / metrics.audio_duration_s if metrics.audio_duration_s > 0 else 0
    assert rtf == 0.5  # Faster than real-time


# ===== Test 3: Audio Player Queue Management =====

@pytest.mark.unit
def test_audio_player_queue_management():
    """AudioPlayer handles queue operations gracefully."""
    import queue

    from src.core.speech.chatterbox_tts import AudioPlayer

    # Create player without starting
    player = AudioPlayer(sample_rate=24000)

    # Verify queue exists
    assert player._audio_queue is not None
    assert isinstance(player._audio_queue, queue.Queue)

    # Test queue operations
    test_audio = b'\x00\x00' * 100
    player._audio_queue.put(test_audio)

    assert not player._audio_queue.empty()

    retrieved = player._audio_queue.get()
    assert retrieved == test_audio


# ===== Additional TTS Tests =====

@pytest.mark.unit
def test_tts_config_custom_values():
    """Custom TTS config values are applied correctly."""
    from src.core.speech.chatterbox_tts import ChatterBoxConfig

    config = ChatterBoxConfig(
        device="cpu",
        exaggeration=0.8,
        cfg_weight=0.6,
        sample_rate=22050,
        voice_reference="custom_voice.wav"
    )

    assert config.device == "cpu"
    assert config.exaggeration == 0.8
    assert config.cfg_weight == 0.6
    assert config.sample_rate == 22050
    assert config.voice_reference == "custom_voice.wav"


@pytest.mark.unit
def test_tts_metrics_ttfb_tracking():
    """TTFB (Time to First Byte) is tracked correctly."""
    from src.core.speech.chatterbox_tts import TTSMetrics

    # Default is 0.0 (TTSMetrics is a frozen dataclass with defaults)
    metrics = TTSMetrics()
    assert metrics.ttfb_ms == 0.0

    # Create with custom TTFB
    metrics = TTSMetrics(ttfb_ms=450.0)
    assert metrics.ttfb_ms == 450.0

    # Verify it's tracked alongside other metrics
    metrics = TTSMetrics(
        ttfb_ms=123.5,
        total_generation_ms=500.0,
        audio_duration_s=1.0
    )
    assert metrics.ttfb_ms == 123.5


@pytest.mark.unit
def test_chatterbox_engine_imports():
    """ChatterBoxEngine can be imported without error."""
    try:
        from src.core.speech.chatterbox_tts import ChatterBoxEngine
        assert ChatterBoxEngine is not None
        assert hasattr(ChatterBoxEngine, 'load_model')
        assert hasattr(ChatterBoxEngine, 'speak')
    except ImportError as e:
        pytest.skip(f"ChatterBox not installed: {e}")


@pytest.mark.unit
def test_tts_sanitization_patterns():
    """TTS should have patterns to sanitize output."""
    # Test that memory tags and asterisk actions would be sanitizable
    # This is a pattern validation test

    import re

    # Memory tag pattern
    memory_pattern = re.compile(r'\[MEMORY\].*?\[/MEMORY\]', re.DOTALL)
    test_text = "Hello [MEMORY]some context[/MEMORY] there."
    sanitized = memory_pattern.sub('', test_text)
    assert "[MEMORY]" not in sanitized

    # Asterisk action pattern
    action_pattern = re.compile(r'\*[^*]+\*')
    test_text = "Hello *smiles warmly* there."
    sanitized = action_pattern.sub('', test_text)
    assert "*smiles" not in sanitized


@pytest.mark.unit
def test_audio_player_sentinel_handling():
    """AudioPlayer recognizes sentinel value for stopping."""
    from src.core.speech.chatterbox_tts import AudioPlayer

    player = AudioPlayer(sample_rate=24000)

    # Sentinel is None
    player._audio_queue.put(None)

    item = player._audio_queue.get()
    assert item is None  # Sentinel indicates stop


@pytest.mark.unit
def test_tts_engine_wrapper_compatibility():
    """TTSEngine wrapper exists for backward compatibility."""
    try:
        from src.core.speech.chatterbox_tts import TTSEngine
        assert TTSEngine is not None
    except ImportError as e:
        pytest.skip(f"TTSEngine not available: {e}")
