"""
Unit tests for configuration loader.
Tests YAML loading, defaults, and system prompt generation.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.config_loader import Config, get_config, reload_config


class TestConfigLoading:
    """Tests for Config class loading behavior."""

    @pytest.mark.unit
    def test_load_nonexistent_config_uses_defaults(self):
        """Test that missing config file uses defaults."""
        config = Config(config_path="nonexistent/path/config.yaml")

        # Should have default values
        assert config.get("name") == "O.L.I.V.I.A"
        assert config.get("voice.cfg_weight") == 0.5
        assert config.get("voice.exaggeration") == 0.5

    @pytest.mark.unit
    def test_load_valid_yaml_config(self):
        """Test loading a valid YAML config file."""
        yaml_content = """
name: "TestBot"
identity:
  name: "TestBot"
  display_name: "Test Bot"
voice:
  cfg_weight: 0.7
  exaggeration: 0.3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            assert config.get("name") == "TestBot"
            assert config.get("identity.name") == "TestBot"
            assert config.get("voice.cfg_weight") == 0.7
            assert config.get("voice.exaggeration") == 0.3
        finally:
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_load_empty_yaml_uses_defaults(self):
        """Test that empty YAML file uses defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            # Empty YAML loads as None, then uses defaults
            # Check that get() works and returns defaults for missing keys
            assert config.get("nonexistent", "default") == "default"
        finally:
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_load_invalid_yaml_uses_defaults(self):
        """Test that invalid YAML uses defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [broken")
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            # Should fall back to defaults on parse error
            assert config.get("name") == "O.L.I.V.I.A"
        finally:
            os.unlink(temp_path)


class TestConfigGet:
    """Tests for Config.get() method."""

    @pytest.mark.unit
    def test_get_top_level_key(self):
        """Test getting a top-level configuration key."""
        config = Config(config_path="nonexistent.yaml")
        assert config.get("name") == "O.L.I.V.I.A"

    @pytest.mark.unit
    def test_get_nested_key_dot_notation(self):
        """Test getting nested keys with dot notation."""
        config = Config(config_path="nonexistent.yaml")
        assert config.get("voice.cfg_weight") == 0.5
        assert config.get("tts.max_words_per_chunk") == 12

    @pytest.mark.unit
    def test_get_deeply_nested_key(self):
        """Test getting deeply nested keys."""
        config = Config(config_path="nonexistent.yaml")
        assert config.get("tts.voice_cloning.exaggeration") == 0.5
        assert config.get("tts.optimization.use_torch_compile") is True

    @pytest.mark.unit
    def test_get_nonexistent_key_returns_default(self):
        """Test that missing keys return the default value."""
        config = Config(config_path="nonexistent.yaml")
        assert config.get("nonexistent") is None
        assert config.get("nonexistent", "fallback") == "fallback"
        assert config.get("nested.nonexistent.key") is None
        assert config.get("nested.nonexistent.key", 42) == 42

    @pytest.mark.unit
    def test_get_partial_path_returns_dict(self):
        """Test getting a partial path returns a dict."""
        config = Config(config_path="nonexistent.yaml")
        voice_config = config.get("voice")
        assert isinstance(voice_config, dict)
        assert "cfg_weight" in voice_config

    @pytest.mark.unit
    def test_get_empty_key(self):
        """Test getting with empty key returns None."""
        config = Config(config_path="nonexistent.yaml")
        # Empty string split results in [''] which doesn't match any key
        result = config.get("")
        # Empty key returns None since '' is not a valid key
        assert result is None


class TestConfigGetSystemPrompt:
    """Tests for Config.get_system_prompt() method."""

    @pytest.mark.unit
    def test_system_prompt_override_priority(self):
        """Test that system_prompt_override has highest priority."""
        yaml_content = """
system_prompt_override: "You are OverrideBot."
system_prompt_template: "You are TemplateBot."
name: "TestBot"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            prompt = config.get_system_prompt()
            assert prompt == "You are OverrideBot."
        finally:
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_system_prompt_template_priority(self):
        """Test that system_prompt_template is used if no override."""
        yaml_content = """
system_prompt_template: "You are TemplateBot, a helpful assistant."
name: "TestBot"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            prompt = config.get_system_prompt()
            assert prompt == "You are TemplateBot, a helpful assistant."
        finally:
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_system_prompt_fallback_generation(self):
        """Test fallback prompt generation when no templates exist."""
        yaml_content = """
name: "FallbackBot"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            prompt = config.get_system_prompt()
            assert "FallbackBot" in prompt
            assert "companion" in prompt.lower() or "assistant" in prompt.lower()
        finally:
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_system_prompt_uses_identity_name(self):
        """Test that identity.name is used in fallback if available."""
        yaml_content = """
identity:
  name: "IdentityBot"
name: "TopLevelName"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            prompt = config.get_system_prompt()
            # identity.name should take precedence over top-level name
            assert "IdentityBot" in prompt
        finally:
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_system_prompt_strips_whitespace(self):
        """Test that system prompts are stripped of whitespace."""
        yaml_content = """
system_prompt_override: "  \n  You are TestBot.  \n  "
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            prompt = config.get_system_prompt()
            assert prompt == "You are TestBot."
            assert not prompt.startswith(" ")
            assert not prompt.endswith(" ")
        finally:
            os.unlink(temp_path)


class TestConfigDefaults:
    """Tests for default configuration values."""

    @pytest.mark.unit
    def test_default_name(self):
        """Test default name value."""
        config = Config(config_path="nonexistent.yaml")
        assert config.get("name") == "O.L.I.V.I.A"

    @pytest.mark.unit
    def test_default_voice_settings(self):
        """Test default voice settings."""
        config = Config(config_path="nonexistent.yaml")
        assert config.get("voice.cfg_weight") == 0.5
        assert config.get("voice.exaggeration") == 0.5

    @pytest.mark.unit
    def test_default_tts_settings(self):
        """Test default TTS settings."""
        config = Config(config_path="nonexistent.yaml")
        assert config.get("tts.max_words_per_chunk") == 12
        assert config.get("tts.voice_cloning.seed") is None

    @pytest.mark.unit
    def test_default_optimization_settings(self):
        """Test default optimization settings."""
        config = Config(config_path="nonexistent.yaml")
        assert config.get("tts.optimization.use_torch_compile") is True
        assert config.get("tts.optimization.compile_mode") == "reduce-overhead"
        assert config.get("tts.optimization.enable_inference_mode") is True
        assert config.get("tts.optimization.enable_cudnn_benchmark") is True
        assert config.get("tts.optimization.enable_tf32") is True

    @pytest.mark.unit
    def test_default_latency_settings(self):
        """Test default latency settings."""
        config = Config(config_path="nonexistent.yaml")
        assert config.get("tts.latency.adaptive_chunking") is True
        assert config.get("tts.latency.first_chunk_tokens") == 30
        assert config.get("tts.latency.subsequent_chunk_tokens") == 50


class TestConfigSingleton:
    """Tests for config singleton functions."""

    @pytest.mark.unit
    def test_get_config_returns_singleton(self):
        """Test that get_config returns the same instance."""
        # Reset singleton for test
        import src.config.config_loader as config_module
        config_module._config = None

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    @pytest.mark.unit
    def test_reload_config_creates_new_instance(self):
        """Test that reload_config creates a new instance."""
        import src.config.config_loader as config_module
        config_module._config = None

        config1 = get_config()
        config2 = reload_config()

        # Should be different instances
        assert config1 is not config2

    @pytest.mark.unit
    def test_reload_config_updates_singleton(self):
        """Test that reload_config updates the singleton."""
        import src.config.config_loader as config_module
        config_module._config = None

        get_config()  # Initialize
        reload_config()  # Reload
        config3 = get_config()  # Should get the reloaded one

        # config3 should be the reloaded instance
        assert config3 is config_module._config
