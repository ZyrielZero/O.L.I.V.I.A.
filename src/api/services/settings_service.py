"""Runtime-tunable settings with JSON persistence (Phase 2).

One validated settings object for everything adjustable while the app runs:
voice-pipeline tuning (VAD, barge-in, TTS expressiveness) plus the client
preference toggles the Flet settings dialog owns. Persisted to
data/settings.json so preferences survive restarts.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

log = logging.getLogger("api.settings")

_SETTINGS_PATH = Path("data/settings.json")


class RuntimeSettings(BaseModel):
    """Everything tunable at runtime, with validation ranges."""

    # Voice pipeline tuning
    vad_threshold: float = Field(0.5, ge=0.1, le=0.95)
    silence_end_s: float = Field(0.5, ge=0.2, le=3.0)
    barge_in_confirm_chunks: int = Field(5, ge=1, le=30)

    # TTS expressiveness (applied to the live ChatterBox config)
    tts_exaggeration: float = Field(0.5, ge=0.0, le=1.0)
    tts_cfg_weight: float = Field(0.5, ge=0.2, le=1.0)

    # Client preference toggles (persisted server-side for the Flet dialog)
    voice_enabled: bool = True
    always_listen_enabled: bool = False
    wake_word_enabled: bool = False
    auto_chat_enabled: bool = False


class SettingsService:
    """Load/update/persist RuntimeSettings; thread-safe."""

    def __init__(self, path: Path = _SETTINGS_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._settings = self._load()

    def _load(self) -> RuntimeSettings:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                return RuntimeSettings(**data)
        except Exception as e:
            log.warning(f"Settings load failed, using defaults: {e}")
        return RuntimeSettings()

    def get(self) -> RuntimeSettings:
        with self._lock:
            return self._settings.model_copy()

    def update(self, partial: dict) -> RuntimeSettings:
        """Validate + apply a partial update, persist, return the new settings."""
        with self._lock:
            merged = self._settings.model_dump()
            merged.update({k: v for k, v in partial.items() if v is not None})
            self._settings = RuntimeSettings(**merged)  # raises on invalid values
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(
                    json.dumps(self._settings.model_dump(), indent=2), encoding="utf-8"
                )
            except Exception as e:
                log.warning(f"Settings persist failed: {e}")
            return self._settings.model_copy()


_service: Optional[SettingsService] = None
_service_lock = threading.Lock()


def get_settings_service() -> SettingsService:
    """Process-wide settings singleton."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = SettingsService()
    return _service


def reset_settings_service() -> None:
    """Testing hook."""
    global _service
    _service = None
