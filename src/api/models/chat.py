"""Chat models."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request."""

    message: str = Field(..., max_length=10_000)
    context: Optional[str] = None
    stream: bool = True
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
