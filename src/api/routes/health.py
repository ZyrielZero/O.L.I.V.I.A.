"""Health check endpoint."""

import asyncio
from datetime import datetime
from typing import Any, Coroutine, Optional, Tuple, Union

from fastapi import APIRouter

from src.api import dependencies
from src.api.models.common import HealthCheck, ServiceHealth

router = APIRouter(tags=["health"])

# OPT: Default timeout for individual health checks (seconds)
_HEALTH_CHECK_TIMEOUT = 2.0


async def _check_with_timeout(
    coro: Coroutine[Any, Any, bool],
    name: str,
    timeout: float = _HEALTH_CHECK_TIMEOUT,
) -> Union[bool, Tuple[str, str]]:
    """Wrap coroutine with timeout protection.

    Complexity: O(1) - single async operation with timeout wrapper.

    Prevents a single slow/hung service from blocking the entire health endpoint.
    Returns timeout status instead of raising, allowing partial health results.

    Args:
        coro: Async health check coroutine returning bool.
        name: Component name for error reporting.
        timeout: Maximum wait time in seconds.

    Returns:
        Bool result from health check, or ("timeout", name) tuple on timeout.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return ("timeout", name)


async def _check_svc(
    svc: Optional[Any], key: str, name: str, msg_fn: Optional[callable] = None
) -> Tuple[str, ServiceHealth]:
    """Check single service health.

    Complexity: O(1) per service - single async call with timeout wrapper.

    Args:
        svc: Service instance to check (must have health_check() method).
        key: Dictionary key for result mapping.
        name: Human-readable service name.
        msg_fn: Optional callable returning success message.

    Returns:
        Tuple of (key, ServiceHealth) for dict construction.
    """
    if not svc:
        return (key, ServiceHealth(name=name, status="down", message="Not initialized"))

    try:
        # OPT: Wrap health check with timeout to prevent hung services from blocking
        result = await _check_with_timeout(svc.health_check(), name)

        # Handle timeout case
        if isinstance(result, tuple) and result[0] == "timeout":
            return (
                key,
                ServiceHealth(
                    name=name,
                    status="timeout",
                    message=f"Health check timed out after {_HEALTH_CHECK_TIMEOUT}s",
                ),
            )

        ok = result
        msg = msg_fn() if msg_fn and ok else ("OK" if ok else "Check failed")
        return (key, ServiceHealth(name=name, status="up" if ok else "down", message=msg))
    except Exception as e:
        return (key, ServiceHealth(name=name, status="down", message=str(e)))


@router.get("/health/live")
async def liveness():
    """Liveness probe — returns 200 if server is running."""
    return {"status": "alive"}


@router.get("/health", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """Returns status of all services + uptime.

    Complexity: O(s) where s = number of services (typically 4-5).
    All service checks run in parallel with timeout protection.
    """
    llm = dependencies.services.get("llm")
    memory = dependencies.services.get("memory")
    stt = dependencies.services.get("stt")
    tts = dependencies.services.get("tts")
    state = dependencies.services.get("state")

    results = await asyncio.gather(
        _check_svc(llm, "llm", "LLM (Ollama)", lambda: f"Ollama ({llm.model})"),
        _check_svc(memory, "memory", "Memory (ChromaDB)", lambda: "ChromaDB OK"),
        _check_svc(stt, "stt", "STT (Whisper)", lambda: f"Whisper {stt.model_size}"),
        _check_svc(tts, "tts", "TTS (ChatterBox)", lambda: "ChatterBox loaded"),
        return_exceptions=True,
    )

    svc_health = {}
    status = "healthy"

    # OPT: O(n) iteration where n = number of services (typically 4-5)
    for res in results:
        if isinstance(res, Exception):
            continue
        key, health = res
        svc_health[key] = health
        # Treat both "down" and "timeout" as degraded/unhealthy states
        if health.status in ("down", "timeout"):
            status = (
                "unhealthy" if key == "llm" else ("degraded" if status == "healthy" else status)
            )

    # state manager (sync, fast)
    if state:
        n = state.get_active_sessions()
        svc_health["state"] = ServiceHealth(
            name="State Manager", status="up", message=f"{n} sessions"
        )
    else:
        svc_health["state"] = ServiceHealth(
            name="State Manager", status="down", message="Not initialized"
        )
        if status == "healthy":
            status = "degraded"

    # experimental: dreaming engine
    dreaming = dependencies.services.get("dreaming")
    if dreaming:
        idle_active = (
            dreaming.idle_detector and dreaming.idle_detector._running
        )
        svc_health["dreaming"] = ServiceHealth(
            name="Dreaming Engine",
            status="up",
            message="Idle monitoring active" if idle_active else "Standby",
        )

    # experimental: fact extractor
    extractor = dependencies.services.get("fact_extractor")
    if extractor:
        stats = extractor.get_stats()
        svc_health["fact_extractor"] = ServiceHealth(
            name="Fact Extractor",
            status="up" if stats.get("running") else "down",
            message=f"{stats.get('facts_extracted', 0)} facts extracted",
        )

    from src.api.main import get_startup_time
    from src.api.services.metrics import get_metrics

    t0 = get_startup_time()
    uptime = (datetime.now() - t0).total_seconds() if t0 else 0

    return HealthCheck(
        status=status,
        services=svc_health,
        uptime_seconds=uptime,
        latency=get_metrics().averages() or None,
    )
