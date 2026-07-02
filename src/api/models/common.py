"""Common API models."""

from typing import Dict, Optional

from pydantic import BaseModel


class ServiceHealth(BaseModel):
    """Single service health."""

    name: str
    status: str  # up/down
    message: Optional[str] = None


class HealthCheck(BaseModel):
    """Health check response."""

    status: str  # healthy/degraded/unhealthy
    services: Dict[str, ServiceHealth]
    uptime_seconds: float
