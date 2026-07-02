"""Config loader - YAML settings."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from utils.logger import get_logger

    log = get_logger("config")
except ImportError:
    import logging

    log = logging.getLogger("config")


class Config:
    """Loads and provides access to configuration settings.

    Caches defaults at initialization and lazily merges with loaded config
    for O(k) lookups where k = key depth (typically 2-3).
    """

    def __init__(self, config_path: str = "config/character.yaml"):
        self.config_path = Path(config_path)
        self._data: Dict[str, Any] = {}
        # OPT: Cache defaults at init instead of recomputing per get() call
        self._defaults: Dict[str, Any] = self._get_defaults()
        # OPT: Lazy merge cache for O(1) subsequent lookups
        self._merged: Optional[Dict[str, Any]] = None
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file.

        Complexity: O(f) where f = file size for YAML parsing.
        Invalidates merged cache on reload.
        """
        # Invalidate merged cache on reload
        self._merged = None

        if not self.config_path.exists():
            log.warning(f"Config not found: {self.config_path}, using defaults")
            self._data = self._defaults.copy()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
            log.info(f"Config loaded: {self.config_path}")
        except Exception as e:
            log.error(f"Config load error: {e}, using defaults")
            self._data = self._defaults.copy()

    def _get_defaults(self) -> Dict[str, Any]:
        """Return default configuration.

        Complexity: O(1) - returns static dict literal.
        Called once at init and cached in self._defaults.
        """
        return {
            "name": "O.L.I.V.I.A",
            "voice": {"cfg_weight": 0.5, "exaggeration": 0.5},  # Optimized for natural voice
            "tts": {
                "max_words_per_chunk": 12,
                "voice_cloning": {
                    "exaggeration": 0.5,
                    "cfg_weight": 0.5,
                    "seed": None,
                },
                "reference_audio": {
                    "preprocess": True,
                    "min_duration_sec": 4.0,
                },
                "output_quality": {
                    "enable_post_processing": True,
                    "crossfade_ms": 10,
                },
                # Phase 2: PyTorch optimizations
                "optimization": {
                    "use_torch_compile": True,
                    "compile_mode": "reduce-overhead",
                    "enable_inference_mode": True,
                    "enable_cudnn_benchmark": True,
                    "enable_tf32": True,
                    "torch_dtype": "float32",
                },
                # Phase 4: Latency optimizations
                "latency": {
                    "adaptive_chunking": True,
                    "first_chunk_tokens": 30,
                    "subsequent_chunk_tokens": 50,
                },
                # Phase 5: Performance monitoring
                "performance": {
                    "enable_metrics": True,
                    "log_metrics": False,
                    "memory_cleanup_interval": 10,
                },
            },
        }

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dicts, override takes precedence.

        Complexity: O(n) where n = total keys in both dicts.
        """
        result = base.copy()
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = self._deep_merge(result[key], val)
            else:
                result[key] = val
        return result

    def _get_merged(self) -> Dict[str, Any]:
        """Get merged config (defaults + loaded data).

        Complexity: O(n) first call, O(1) subsequent (cached).
        """
        if self._merged is None:
            self._merged = self._deep_merge(self._defaults, self._data)
        return self._merged

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by key (supports dot notation).

        Complexity: O(k) where k = key depth (typically 2-3).
        Uses cached merged dict for efficient repeated lookups.
        """
        # OPT: Use cached merged dict instead of recomputing defaults per call
        merged = self._get_merged()
        keys = key.split(".")
        value = merged

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value if value is not None else default

    def get_system_prompt(self) -> str:
        """Return the system prompt.

        Priority:
        1. system_prompt_override (explicit override)
        2. system_prompt_template (character template from YAML)
        3. Generated fallback prompt

        Complexity: O(k) where k = key depth for each get() call.
        """
        # Priority 1: Explicit override
        override = self.get("system_prompt_override")
        if override:
            return override.strip()

        # Priority 2: Character template from YAML
        template = self.get("system_prompt_template")
        if template:
            return template.strip()

        # Priority 3: Fallback
        name = self.get("identity.name", self.get("name", "O.L.I.V.I.A."))
        sections = [
            f"You are {name}, a personal AI companion.",
            "Keep responses to 1-3 sentences maximum.",
            "Be warm but direct. Be concise.",
        ]
        return "\n".join(sections)


_config: Optional[Config] = None


def get_config(config_path: str = "config/character.yaml") -> Config:
    """Get or create the config singleton.

    Complexity: O(1) after first call (singleton pattern).
    First call is O(f) where f = YAML file size.
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: str = "config/character.yaml") -> Config:
    """Force reload the config.

    Complexity: O(f) where f = YAML file size.
    Invalidates the singleton and all cached merged data.
    """
    global _config
    _config = Config(config_path)
    return _config
