"""API config via pydantic-settings.

Configuration values are loaded from environment variables with sensible defaults.
Field validators ensure invalid values are caught at startup, not runtime.
"""

from functools import lru_cache
from typing import List, Optional, Set

from pydantic import field_validator
from pydantic_settings import BaseSettings

# Valid values for enum-like config options
_VALID_LOG_LEVELS: Set[str] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_DEVICES: Set[str] = {"cuda", "cpu", "auto"}
_VALID_COMPUTE_TYPES: Set[str] = {"float16", "float32", "int8", "int8_float16"}
_VALID_STT_MODELS: Set[str] = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v2",
    "large-v3",
}


class APIConfig(BaseSettings):
    """Loads from env vars with sensible defaults.

    Uses pydantic-settings for automatic env var parsing.
    Access via get_api_config() for cached singleton instance.

    Validators ensure configuration errors are caught at startup.
    """

    HF_TOKEN: Optional[str] = None

    HOST: str = "127.0.0.1"
    PORT: int = 8000

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["Content-Type", "X-API-Key", "Accept"]

    LOG_LEVEL: str = "INFO"

    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "olivia-finetuned"

    STT_MODEL_SIZE: str = "small.en"
    STT_DEVICE: str = "cuda"
    STT_COMPUTE_TYPE: str = "float16"

    TTS_DEVICE: str = "cuda"
    TTS_VOICE_REFERENCE: str = "assets/voice/reference.wav"
    TTS_CFG_WEIGHT: float = 0.5
    TTS_EXAGGERATION: float = 0.5

    MEMORY_PERSIST_DIR: str = "data/memory_db"

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @field_validator("PORT")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Ensure port is in valid range (1-65535)."""
        if not 1 <= v <= 65535:
            raise ValueError(f"PORT must be between 1 and 65535, got {v}")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid and uppercase."""
        v_upper = v.upper()
        if v_upper not in _VALID_LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {_VALID_LOG_LEVELS}, got {v}")
        return v_upper

    @field_validator("OLLAMA_HOST")
    @classmethod
    def validate_ollama_host(cls, v: str) -> str:
        """Ensure Ollama host is a valid URL, normalize trailing slash."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"OLLAMA_HOST must start with http:// or https://, got {v}")
        return v.rstrip("/")  # Normalize by removing trailing slash

    @field_validator("STT_DEVICE", "TTS_DEVICE")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Ensure device is valid (cuda, cpu, auto)."""
        v_lower = v.lower()
        if v_lower not in _VALID_DEVICES:
            raise ValueError(f"Device must be one of {_VALID_DEVICES}, got {v}")
        return v_lower

    @field_validator("STT_COMPUTE_TYPE")
    @classmethod
    def validate_compute_type(cls, v: str) -> str:
        """Ensure compute type is valid for faster-whisper."""
        v_lower = v.lower()
        if v_lower not in _VALID_COMPUTE_TYPES:
            raise ValueError(f"STT_COMPUTE_TYPE must be one of {_VALID_COMPUTE_TYPES}, got {v}")
        return v_lower

    @field_validator("STT_MODEL_SIZE")
    @classmethod
    def validate_stt_model_size(cls, v: str) -> str:
        """Ensure STT model size is a valid Whisper model."""
        if v not in _VALID_STT_MODELS:
            raise ValueError(f"STT_MODEL_SIZE must be one of {_VALID_STT_MODELS}, got {v}")
        return v

    @field_validator("TTS_CFG_WEIGHT")
    @classmethod
    def validate_cfg_weight(cls, v: float) -> float:
        """Ensure TTS cfg_weight is in valid range (0.0-1.0)."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"TTS_CFG_WEIGHT must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("TTS_EXAGGERATION")
    @classmethod
    def validate_exaggeration(cls, v: float) -> float:
        """Ensure TTS exaggeration is in valid range (0.0-1.0)."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"TTS_EXAGGERATION must be between 0.0 and 1.0, got {v}")
        return v

    class Config:
        """Pydantic settings behavior."""

        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache(maxsize=1)
def get_api_config() -> APIConfig:
    """Get cached API configuration singleton.

    Complexity: O(1) after first call (LRU cached).
    First call is O(e) where e = number of env vars to parse.

    Returns:
        Cached APIConfig instance with values from env vars.

    Raises:
        ValidationError: If any config values are invalid.
    """
    return APIConfig()
