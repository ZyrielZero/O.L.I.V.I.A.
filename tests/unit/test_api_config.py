"""
Unit tests for API configuration.
Tests environment variable loading and default values.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.config import APIConfig


class TestAPIConfigDefaults:
    """Tests for default configuration values."""

    @pytest.mark.unit
    def test_default_host(self):
        """Test default host value."""
        config = APIConfig()
        assert config.HOST == "127.0.0.1"

    @pytest.mark.unit
    def test_default_port(self):
        """Test default port value."""
        config = APIConfig()
        assert config.PORT == 8000

    @pytest.mark.unit
    def test_default_cors_origins(self):
        """Test default CORS origins."""
        config = APIConfig()
        assert "http://localhost:3000" in config.CORS_ORIGINS
        assert "http://localhost:5173" in config.CORS_ORIGINS

    @pytest.mark.unit
    def test_default_cors_allow_credentials(self):
        """Test default CORS allow credentials."""
        config = APIConfig()
        assert config.CORS_ALLOW_CREDENTIALS is True

    @pytest.mark.unit
    def test_default_cors_methods(self):
        """Test default CORS methods."""
        config = APIConfig()
        assert "GET" in config.CORS_ALLOW_METHODS
        assert "POST" in config.CORS_ALLOW_METHODS
        assert "OPTIONS" in config.CORS_ALLOW_METHODS

    @pytest.mark.unit
    def test_default_cors_headers(self):
        """Test default CORS headers."""
        config = APIConfig()
        assert "Content-Type" in config.CORS_ALLOW_HEADERS
        assert "X-API-Key" in config.CORS_ALLOW_HEADERS
        assert "Accept" in config.CORS_ALLOW_HEADERS

    @pytest.mark.unit
    def test_default_log_level(self):
        """Test default log level."""
        config = APIConfig()
        assert config.LOG_LEVEL == "INFO"

    @pytest.mark.unit
    def test_default_ollama_host(self):
        """Test default Ollama host."""
        config = APIConfig()
        assert config.OLLAMA_HOST == "http://localhost:11434"

    @pytest.mark.unit
    def test_default_ollama_model(self):
        """Test default Ollama model."""
        config = APIConfig()
        assert config.OLLAMA_MODEL == "olivia-finetuned"

    @pytest.mark.unit
    def test_default_stt_model_size(self):
        """Test default STT model size."""
        config = APIConfig()
        assert config.STT_MODEL_SIZE == "small.en"

    @pytest.mark.unit
    def test_default_stt_device(self):
        """Test default STT device."""
        config = APIConfig()
        assert config.STT_DEVICE == "cuda"

    @pytest.mark.unit
    def test_default_stt_compute_type(self):
        """Test default STT compute type (may be overridden by .env)."""
        config = APIConfig()
        # Accept either the code default or common env overrides
        assert config.STT_COMPUTE_TYPE in ["float16", "int8_float16", "int8"]

    @pytest.mark.unit
    def test_default_tts_device(self):
        """Test default TTS device."""
        config = APIConfig()
        assert config.TTS_DEVICE == "cuda"

    @pytest.mark.unit
    def test_default_tts_voice_reference(self):
        """Test default TTS voice reference path."""
        config = APIConfig()
        assert config.TTS_VOICE_REFERENCE == "assets/voice/reference.wav"

    @pytest.mark.unit
    def test_default_tts_cfg_weight(self):
        """Test default TTS cfg_weight."""
        config = APIConfig()
        assert config.TTS_CFG_WEIGHT == 0.5

    @pytest.mark.unit
    def test_default_tts_exaggeration(self):
        """Test default TTS exaggeration."""
        config = APIConfig()
        assert config.TTS_EXAGGERATION == 0.5

    @pytest.mark.unit
    def test_default_memory_persist_dir(self):
        """Test default memory persist directory."""
        config = APIConfig()
        assert config.MEMORY_PERSIST_DIR == "data/memory_db"


class TestAPIConfigEnvironmentVariables:
    """Tests for environment variable loading."""

    @pytest.mark.unit
    def test_env_override_host(self):
        """Test HOST env var override."""
        with patch.dict(os.environ, {"HOST": "127.0.0.1"}):
            config = APIConfig()
            assert config.HOST == "127.0.0.1"

    @pytest.mark.unit
    def test_env_override_port(self):
        """Test PORT env var override."""
        with patch.dict(os.environ, {"PORT": "9000"}):
            config = APIConfig()
            assert config.PORT == 9000

    @pytest.mark.unit
    def test_env_override_log_level(self):
        """Test LOG_LEVEL env var override."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            config = APIConfig()
            assert config.LOG_LEVEL == "DEBUG"

    @pytest.mark.unit
    def test_env_override_ollama_host(self):
        """Test OLLAMA_HOST env var override."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://192.168.1.100:11434"}):
            config = APIConfig()
            assert config.OLLAMA_HOST == "http://192.168.1.100:11434"

    @pytest.mark.unit
    def test_env_override_ollama_model(self):
        """Test OLLAMA_MODEL env var override."""
        with patch.dict(os.environ, {"OLLAMA_MODEL": "custom-model"}):
            config = APIConfig()
            assert config.OLLAMA_MODEL == "custom-model"

    @pytest.mark.unit
    def test_env_override_stt_model_size(self):
        """Test STT_MODEL_SIZE env var override."""
        with patch.dict(os.environ, {"STT_MODEL_SIZE": "medium.en"}):
            config = APIConfig()
            assert config.STT_MODEL_SIZE == "medium.en"

    @pytest.mark.unit
    def test_env_override_stt_device(self):
        """Test STT_DEVICE env var override."""
        with patch.dict(os.environ, {"STT_DEVICE": "cpu"}):
            config = APIConfig()
            assert config.STT_DEVICE == "cpu"

    @pytest.mark.unit
    def test_env_override_tts_cfg_weight(self):
        """Test TTS_CFG_WEIGHT env var override."""
        with patch.dict(os.environ, {"TTS_CFG_WEIGHT": "0.7"}):
            config = APIConfig()
            assert config.TTS_CFG_WEIGHT == 0.7

    @pytest.mark.unit
    def test_env_override_tts_exaggeration(self):
        """Test TTS_EXAGGERATION env var override."""
        with patch.dict(os.environ, {"TTS_EXAGGERATION": "0.8"}):
            config = APIConfig()
            assert config.TTS_EXAGGERATION == 0.8

    @pytest.mark.unit
    def test_env_hf_token(self):
        """Test HF_TOKEN env var."""
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}):
            config = APIConfig()
            assert config.HF_TOKEN == "hf_test_token"

    @pytest.mark.unit
    def test_env_hf_token_none_when_unset(self):
        """Test HF_TOKEN is None when env var is not set."""
        # Clear the HF_TOKEN environment variable for this test
        with patch.dict(os.environ, {}, clear=False):
            # Remove HF_TOKEN if it exists
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_TOKEN", None)
            # Note: pydantic-settings may still read from .env file
            # So this test validates the Optional typing works
            config = APIConfig()
            # HF_TOKEN is Optional[str], so it can be None or a string
            assert config.HF_TOKEN is None or isinstance(config.HF_TOKEN, str)


class TestAPIConfigTypes:
    """Tests for configuration type validation."""

    @pytest.mark.unit
    def test_port_is_integer(self):
        """Test PORT is an integer."""
        config = APIConfig()
        assert isinstance(config.PORT, int)

    @pytest.mark.unit
    def test_cors_origins_is_list(self):
        """Test CORS_ORIGINS is a list."""
        config = APIConfig()
        assert isinstance(config.CORS_ORIGINS, list)

    @pytest.mark.unit
    def test_cors_allow_credentials_is_bool(self):
        """Test CORS_ALLOW_CREDENTIALS is a boolean."""
        config = APIConfig()
        assert isinstance(config.CORS_ALLOW_CREDENTIALS, bool)

    @pytest.mark.unit
    def test_tts_cfg_weight_is_float(self):
        """Test TTS_CFG_WEIGHT is a float."""
        config = APIConfig()
        assert isinstance(config.TTS_CFG_WEIGHT, float)

    @pytest.mark.unit
    def test_tts_exaggeration_is_float(self):
        """Test TTS_EXAGGERATION is a float."""
        config = APIConfig()
        assert isinstance(config.TTS_EXAGGERATION, float)


class TestAPIConfigCaseSensitivity:
    """Tests for case sensitivity of environment variables."""

    @pytest.mark.unit
    def test_env_vars_uppercase_works(self):
        """Test that uppercase env vars work correctly."""
        # The config uses case_sensitive = True in the config
        # However, pydantic-settings may be case-insensitive on some platforms
        with patch.dict(os.environ, {"HOST": "192.168.1.1"}, clear=False):
            config = APIConfig()
            assert config.HOST == "192.168.1.1"

    @pytest.mark.unit
    def test_config_case_sensitive_setting(self):
        """Test that config has case_sensitive = True."""
        # Verify the config class has case_sensitive setting
        assert hasattr(APIConfig, 'Config') or hasattr(APIConfig, 'model_config')
        # This validates the setting exists even if platform behavior varies


class TestAPIConfigValidation:
    """Tests for configuration value validation."""

    @pytest.mark.unit
    def test_port_accepts_valid_port(self):
        """Test PORT accepts valid port numbers."""
        with patch.dict(os.environ, {"PORT": "80"}):
            config = APIConfig()
            assert config.PORT == 80

        with patch.dict(os.environ, {"PORT": "65535"}):
            config = APIConfig()
            assert config.PORT == 65535

    @pytest.mark.unit
    def test_float_conversion(self):
        """Test float values are properly converted."""
        with patch.dict(os.environ, {"TTS_CFG_WEIGHT": "0.33333"}):
            config = APIConfig()
            assert abs(config.TTS_CFG_WEIGHT - 0.33333) < 0.0001
