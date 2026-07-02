"""Tests for the Phase 0 background-task and timeout fixes.

Covers: memory-fetch timeout in the chat route (0.4), strong references to
fire-and-forget tasks (0.6), and the TTL maintenance loop wiring (0.1).
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

# ===== 0.4: Memory fetch timeout =====


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_fetch_timeout_does_not_stall_chat():
    """A slow memory backend must not delay chat beyond the timeout."""
    from src.api.routes.chat import _MEMORY_FETCH_TIMEOUT, _fetch_memory_context

    class SlowMemory:
        async def get_relevant_context(self, msg, n_results=3):
            await asyncio.sleep(_MEMORY_FETCH_TIMEOUT + 5)
            return "too late"

    start = time.monotonic()
    result = await _fetch_memory_context(
        SlowMemory(), "a sufficiently long question that triggers a memory lookup"
    )
    elapsed = time.monotonic() - start

    assert result == ""
    assert elapsed < _MEMORY_FETCH_TIMEOUT + 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_fetch_returns_context_when_fast():
    """A responsive memory backend returns its context normally."""
    from src.api.routes.chat import _fetch_memory_context

    class FastMemory:
        async def get_relevant_context(self, msg, n_results=3):
            return "relevant context"

    result = await _fetch_memory_context(
        FastMemory(), "a sufficiently long question that triggers a memory lookup"
    )
    assert result == "relevant context"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_fetch_skips_short_messages():
    """Greetings skip the memory lookup entirely."""
    from src.api.routes.chat import _fetch_memory_context

    memory = AsyncMock()
    result = await _fetch_memory_context(memory, "hi")

    assert result == ""
    memory.get_relevant_context.assert_not_awaited()


# ===== 0.6: Fire-and-forget task references =====


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bg_task_reference_held_until_done():
    """_create_bg_task keeps a strong reference for the task's lifetime."""
    from src.api.routes import chat as chat_module

    started = asyncio.Event()
    release = asyncio.Event()

    async def work():
        started.set()
        await release.wait()

    task = chat_module._create_bg_task(work())
    await started.wait()

    # Referenced while running (this is what prevents GC of the task)
    assert task in chat_module._bg_tasks

    release.set()
    await task

    # Discarded once done — no unbounded growth
    assert task not in chat_module._bg_tasks


# ===== 0.1: TTL maintenance loop =====


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_maintenance_loop_prunes_immediately():
    """The maintenance loop prunes on startup, before its first sleep."""
    from src.api import main as api_main

    mem = AsyncMock()
    mem.prune_expired.return_value = {"conversations": 2, "summaries": 1}

    task = asyncio.create_task(api_main._memory_maintenance_loop(mem))
    try:
        for _ in range(100):
            if mem.prune_expired.await_count:
                break
            await asyncio.sleep(0.01)
        assert mem.prune_expired.await_count == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_maintenance_loop_survives_prune_failure():
    """A pruning failure must not kill the daily loop."""
    from src.api import main as api_main

    mem = AsyncMock()
    mem.prune_expired.side_effect = RuntimeError("chroma hiccup")

    task = asyncio.create_task(api_main._memory_maintenance_loop(mem))
    try:
        for _ in range(100):
            if mem.prune_expired.await_count:
                break
            await asyncio.sleep(0.01)
        # The loop reached its sleep instead of crashing
        await asyncio.sleep(0.05)
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
