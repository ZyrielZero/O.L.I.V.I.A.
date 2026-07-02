"""Speech module - TTS (ChatterBox), STT (Whisper), audio processing."""

# TTS
try:
    from .chatterbox_tts import ChatterBoxConfig, ChatterBoxEngine, TTSEngine

    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    TTSEngine = ChatterBoxEngine = ChatterBoxConfig = None

# Audio processing
try:
    from .audio_processing import AudioQualityConfig, OutputAudioProcessor, ReferenceAudioValidator

    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    AudioQualityConfig = ReferenceAudioValidator = OutputAudioProcessor = None

# STT
try:
    from .stt import HybridSTT, STTEngine

    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    STTEngine = HybridSTT = None

__all__ = [
    "TTSEngine",
    "ChatterBoxEngine",
    "ChatterBoxConfig",
    "CHATTERBOX_AVAILABLE",
    "AudioQualityConfig",
    "ReferenceAudioValidator",
    "OutputAudioProcessor",
    "AUDIO_PROCESSING_AVAILABLE",
    "STTEngine",
    "HybridSTT",
    "STT_AVAILABLE",
]
