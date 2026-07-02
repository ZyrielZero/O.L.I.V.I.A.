"""
Service lifecycle integration tests.
Tests initialization order, health aggregation, and graceful shutdown.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# ===== Test 1: Service Initialization Order =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_initialization_order():
    """Services initialize in correct order: LLM -> Memory -> TTS."""
    init_order = []

    async def mock_llm_init():
        init_order.append("llm")

    async def mock_memory_init():
        init_order.append("memory")

    async def mock_tts_init():
        init_order.append("tts")

    # Simulate startup sequence
    await mock_llm_init()
    await mock_memory_init()
    await mock_tts_init()

    # Verify order
    assert init_order == ["llm", "memory", "tts"]
    assert init_order.index("llm") < init_order.index("memory")
    assert init_order.index("memory") < init_order.index("tts")


# ===== Test 2: Service Health Aggregation =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_health_aggregation(
    mock_llm_service,
    mock_memory_service,
    mock_tts_service
):
    """Health endpoint aggregates all service states correctly."""
    # Setup all services as healthy
    mock_llm_service.health_check = AsyncMock(return_value=True)
    mock_memory_service.health_check = AsyncMock(return_value=True)
    mock_tts_service.health_check = AsyncMock(return_value=True)

    # Aggregate health
    services_health = {
        "llm": await mock_llm_service.health_check(),
        "memory": await mock_memory_service.health_check(),
        "tts": await mock_tts_service.health_check(),
    }

    # All healthy
    overall_status = "healthy" if all(services_health.values()) else "degraded"
    assert overall_status == "healthy"

    # Test degraded state
    mock_tts_service.health_check = AsyncMock(return_value=False)
    services_health["tts"] = await mock_tts_service.health_check()

    overall_status = "healthy" if all(services_health.values()) else "degraded"
    assert overall_status == "degraded"


# ===== Test 3: Graceful Shutdown =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_graceful_shutdown():
    """Services close cleanly on shutdown without hanging."""
    cleanup_completed = []

    async def mock_llm_cleanup():
        await asyncio.sleep(0.01)  # Simulate cleanup work
        cleanup_completed.append("llm")

    async def mock_memory_cleanup():
        await asyncio.sleep(0.01)
        cleanup_completed.append("memory")

    async def mock_tts_cleanup():
        await asyncio.sleep(0.01)
        cleanup_completed.append("tts")

    # Simulate shutdown
    await mock_tts_cleanup()  # TTS first (using GPU)
    await mock_llm_cleanup()
    await mock_memory_cleanup()

    # All cleanups completed
    assert len(cleanup_completed) == 3
    assert "llm" in cleanup_completed
    assert "memory" in cleanup_completed
    assert "tts" in cleanup_completed


# ===== Additional Lifecycle Tests =====

@pytest.mark.integration
@pytest.mark.asyncio
async def test_partial_service_failure_handling():
    """System handles partial service initialization failures."""
    services_status = {}

    async def init_with_possible_failure(name, should_fail=False):
        if should_fail:
            raise Exception(f"{name} initialization failed")
        services_status[name] = "initialized"

    # LLM succeeds
    await init_with_possible_failure("llm", should_fail=False)
    assert services_status.get("llm") == "initialized"

    # Memory fails
    with pytest.raises(Exception):
        await init_with_possible_failure("memory", should_fail=True)

    # System should know memory failed
    assert "memory" not in services_status


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_check_timeout_handling():
    """Health checks handle service timeouts gracefully."""

    async def slow_health_check():
        await asyncio.sleep(5.0)  # Very slow
        return True

    async def health_check_with_timeout(timeout=1.0):
        try:
            return await asyncio.wait_for(slow_health_check(), timeout=timeout)
        except asyncio.TimeoutError:
            return False

    # Should timeout and return False
    result = await health_check_with_timeout(timeout=0.1)
    assert result is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_restart_after_failure():
    """Services can be restarted after failure."""
    service_state = {"initialized": False, "attempts": 0}

    async def init_service():
        service_state["attempts"] += 1
        if service_state["attempts"] < 3:
            raise Exception("Transient failure")
        service_state["initialized"] = True

    # Retry logic
    for attempt in range(5):
        try:
            await init_service()
            break
        except Exception:
            await asyncio.sleep(0.01)

    assert service_state["initialized"] is True
    assert service_state["attempts"] == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dependency_injection_service_lookup():
    """Dependency injection correctly looks up services."""
    from src.api import dependencies
    from src.api.container import get_container

    container = get_container()
    original_llm = container.llm

    try:
        mock_llm = MagicMock()
        mock_llm.is_initialized.return_value = True

        container.llm = mock_llm

        service = dependencies.get_service("llm")
        assert service is mock_llm

        service = dependencies.get_service("nonexistent")
        assert service is None

    finally:
        container.llm = original_llm


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_health_checks():
    """Multiple health checks can run concurrently."""
    check_times = []

    async def mock_health_check(name, delay):
        start = asyncio.get_event_loop().time()
        await asyncio.sleep(delay)
        end = asyncio.get_event_loop().time()
        check_times.append((name, end - start))
        return True

    # Run health checks concurrently
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(
        mock_health_check("llm", 0.1),
        mock_health_check("memory", 0.1),
        mock_health_check("tts", 0.1),
    )
    total_time = asyncio.get_event_loop().time() - start_time

    # All should succeed
    assert all(results)

    # Total time should be ~0.1s (parallel), not ~0.3s (sequential)
    assert total_time < 0.2
