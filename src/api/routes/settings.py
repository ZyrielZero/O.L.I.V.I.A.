"""Runtime settings API (Phase 2)."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from src.api.container import get_container
from src.api.services.settings_service import RuntimeSettings, get_settings_service

log = logging.getLogger("api.settings_routes")

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsUpdate(BaseModel):
    """Partial settings update; omitted fields keep their current values."""

    vad_threshold: Optional[float] = None
    silence_end_s: Optional[float] = None
    barge_in_confirm_chunks: Optional[int] = None
    tts_exaggeration: Optional[float] = None
    tts_cfg_weight: Optional[float] = None
    voice_enabled: Optional[bool] = None
    always_listen_enabled: Optional[bool] = None
    wake_word_enabled: Optional[bool] = None
    auto_chat_enabled: Optional[bool] = None


@router.get("", response_model=RuntimeSettings)
async def get_settings():
    """Current runtime settings."""
    return get_settings_service().get()


@router.put("", response_model=RuntimeSettings)
async def update_settings(update: SettingsUpdate):
    """Apply a partial update; values are validated against their ranges."""
    try:
        new = get_settings_service().update(update.model_dump(exclude_none=True))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    # Apply TTS expressiveness to the live engine config
    tts = get_container().tts
    if tts is not None:
        try:
            tts.config.exaggeration = new.tts_exaggeration
            tts.config.cfg_weight = new.tts_cfg_weight
        except Exception as e:
            log.warning(f"Could not apply TTS settings to live engine: {e}")

    return new
