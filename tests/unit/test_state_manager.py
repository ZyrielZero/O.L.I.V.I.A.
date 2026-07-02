"""
Unit tests for session state manager.
Tests session lifecycle, state transitions, and cleanup.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.services.state_manager import Session, StateManager


class TestSession:
    """Tests for Session dataclass."""

    @pytest.mark.unit
    def test_session_creation(self):
        """Test creating a session with required fields."""
        session = Session(session_id="test-123")
        assert session.session_id == "test-123"
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)
        assert session.state == "idle"
        assert session.conversation_history == []
        assert isinstance(session.audio_buffer, bytearray)
        assert len(session.audio_buffer) == 0

    @pytest.mark.unit
    def test_session_custom_values(self):
        """Test session with custom values."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        session = Session(
            session_id="custom-session",
            created_at=custom_time,
            last_activity=custom_time,
            state="speaking"
        )
        assert session.session_id == "custom-session"
        assert session.created_at == custom_time
        assert session.state == "speaking"

    @pytest.mark.unit
    def test_session_conversation_history(self):
        """Test session conversation history modification."""
        session = Session(session_id="test")
        session.conversation_history.append({"role": "user", "content": "Hello"})
        session.conversation_history.append({"role": "assistant", "content": "Hi there!"})

        assert len(session.conversation_history) == 2
        assert session.conversation_history[0]["role"] == "user"

    @pytest.mark.unit
    def test_session_audio_buffer_modification(self):
        """Test session audio buffer modification."""
        session = Session(session_id="test")
        session.audio_buffer.extend(b'\x00\x01\x02\x03')

        assert len(session.audio_buffer) == 4
        assert session.audio_buffer[0] == 0


class TestStateManagerSessionCreation:
    """Tests for StateManager session creation."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_session_auto_id(self):
        """Test creating a session with auto-generated ID."""
        manager = StateManager()
        session = await manager.create_session()

        assert session is not None
        assert session.session_id is not None
        assert len(session.session_id) > 0
        # UUID format check
        assert len(session.session_id) == 36

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_session_custom_id(self):
        """Test creating a session with custom ID."""
        manager = StateManager()
        session = await manager.create_session(session_id="my-custom-id")

        assert session.session_id == "my-custom-id"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_multiple_sessions(self):
        """Test creating multiple sessions."""
        manager = StateManager()

        session1 = await manager.create_session(session_id="session-1")
        session2 = await manager.create_session(session_id="session-2")
        await manager.create_session(session_id="session-3")

        assert manager.get_active_sessions() == 3
        assert session1.session_id != session2.session_id

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_session_overwrites_existing(self):
        """Test that creating session with same ID overwrites."""
        manager = StateManager()

        session1 = await manager.create_session(session_id="duplicate")
        session1.state = "speaking"

        session2 = await manager.create_session(session_id="duplicate")

        # New session should replace old
        assert session2.state == "idle"
        assert manager.get_active_sessions() == 1


class TestStateManagerSessionRetrieval:
    """Tests for StateManager session retrieval."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_existing_session(self):
        """Test retrieving an existing session."""
        manager = StateManager()
        created = await manager.create_session(session_id="test-session")

        retrieved = await manager.get_session("test-session")

        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self):
        """Test retrieving a non-existent session returns None."""
        manager = StateManager()

        retrieved = await manager.get_session("nonexistent")

        assert retrieved is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_all_session_ids(self):
        """Test getting all session IDs."""
        manager = StateManager()

        await manager.create_session(session_id="session-a")
        await manager.create_session(session_id="session-b")
        await manager.create_session(session_id="session-c")

        ids = manager.get_all_session_ids()

        assert len(ids) == 3
        assert "session-a" in ids
        assert "session-b" in ids
        assert "session-c" in ids

    @pytest.mark.unit
    def test_get_active_sessions_count(self):
        """Test counting active sessions."""
        manager = StateManager()
        assert manager.get_active_sessions() == 0


class TestStateManagerStateUpdates:
    """Tests for StateManager state update operations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_session_state(self):
        """Test updating session state."""
        manager = StateManager()
        await manager.create_session(session_id="test")

        await manager.update_session_state("test", "listening")

        session = await manager.get_session("test")
        assert session.state == "listening"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_state_updates_last_activity(self):
        """Test that updating state updates last_activity."""
        manager = StateManager()
        session = await manager.create_session(session_id="test")
        original_activity = session.last_activity

        # Small delay
        await asyncio.sleep(0.01)

        await manager.update_session_state("test", "processing")

        updated_session = await manager.get_session("test")
        assert updated_session.last_activity >= original_activity

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_nonexistent_session_state(self):
        """Test updating state of non-existent session does nothing."""
        manager = StateManager()

        # Should not raise
        await manager.update_session_state("nonexistent", "speaking")

        # Verify no session was created
        assert manager.get_active_sessions() == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_valid_states(self):
        """Test all valid session states."""
        manager = StateManager()
        await manager.create_session(session_id="test")

        valid_states = ["idle", "listening", "transcribing", "processing", "speaking"]

        for state in valid_states:
            await manager.update_session_state("test", state)
            session = await manager.get_session("test")
            assert session.state == state


class TestStateManagerAudioBuffer:
    """Tests for StateManager audio buffer operations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_append_to_audio_buffer(self):
        """Test appending data to audio buffer."""
        manager = StateManager()
        await manager.create_session(session_id="test")

        await manager.append_to_audio_buffer("test", b'\x00\x01\x02')
        await manager.append_to_audio_buffer("test", b'\x03\x04\x05')

        session = await manager.get_session("test")
        assert bytes(session.audio_buffer) == b'\x00\x01\x02\x03\x04\x05'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_append_updates_last_activity(self):
        """Test that appending to buffer updates last_activity."""
        manager = StateManager()
        session = await manager.create_session(session_id="test")
        original_activity = session.last_activity

        await asyncio.sleep(0.01)
        await manager.append_to_audio_buffer("test", b'\x00')

        updated_session = await manager.get_session("test")
        assert updated_session.last_activity >= original_activity

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_audio_buffer(self):
        """Test clearing audio buffer."""
        manager = StateManager()
        await manager.create_session(session_id="test")

        await manager.append_to_audio_buffer("test", b'\x00\x01\x02\x03')
        await manager.clear_audio_buffer("test")

        session = await manager.get_session("test")
        assert len(session.audio_buffer) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_append_to_nonexistent_session(self):
        """Test appending to non-existent session does nothing."""
        manager = StateManager()

        # Should not raise
        await manager.append_to_audio_buffer("nonexistent", b'\x00')

        assert manager.get_active_sessions() == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_nonexistent_session_buffer(self):
        """Test clearing non-existent session buffer does nothing."""
        manager = StateManager()

        # Should not raise
        await manager.clear_audio_buffer("nonexistent")


class TestStateManagerCleanup:
    """Tests for StateManager cleanup operations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_session(self):
        """Test cleaning up a specific session."""
        manager = StateManager()
        await manager.create_session(session_id="to-delete")
        await manager.create_session(session_id="to-keep")

        await manager.cleanup_session("to-delete")

        assert await manager.get_session("to-delete") is None
        assert await manager.get_session("to-keep") is not None
        assert manager.get_active_sessions() == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_session(self):
        """Test cleaning up non-existent session does nothing."""
        manager = StateManager()
        await manager.create_session(session_id="existing")

        # Should not raise
        await manager.cleanup_session("nonexistent")

        assert manager.get_active_sessions() == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_stale_sessions(self):
        """Test cleaning up stale sessions."""
        manager = StateManager()

        # Create a session and manually set old last_activity
        session = await manager.create_session(session_id="stale")
        session.last_activity = datetime.now() - timedelta(hours=2)

        await manager.create_session(session_id="fresh")

        # Clean up sessions older than 1 hour
        await manager.cleanup_stale_sessions(timeout_seconds=3600)

        assert await manager.get_session("stale") is None
        assert await manager.get_session("fresh") is not None
        assert manager.get_active_sessions() == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_stale_sessions_all_fresh(self):
        """Test cleanup when all sessions are fresh."""
        manager = StateManager()

        await manager.create_session(session_id="fresh-1")
        await manager.create_session(session_id="fresh-2")

        await manager.cleanup_stale_sessions(timeout_seconds=3600)

        assert manager.get_active_sessions() == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_stale_sessions_empty(self):
        """Test cleanup when no sessions exist."""
        manager = StateManager()

        # Should not raise
        await manager.cleanup_stale_sessions(timeout_seconds=3600)

        assert manager.get_active_sessions() == 0


class TestStateManagerConcurrency:
    """Tests for StateManager concurrent operations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self):
        """Test creating multiple sessions concurrently."""
        manager = StateManager()

        async def create_session(i):
            return await manager.create_session(session_id=f"session-{i}")

        # Create 10 sessions concurrently
        tasks = [create_session(i) for i in range(10)]
        sessions = await asyncio.gather(*tasks)

        assert len(sessions) == 10
        assert manager.get_active_sessions() == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_buffer_operations(self):
        """Test concurrent audio buffer operations."""
        manager = StateManager()
        await manager.create_session(session_id="test")

        async def append_chunk(chunk_id):
            await manager.append_to_audio_buffer("test", bytes([chunk_id] * 10))

        # Append 10 chunks concurrently
        tasks = [append_chunk(i) for i in range(10)]
        await asyncio.gather(*tasks)

        session = await manager.get_session("test")
        assert len(session.audio_buffer) == 100  # 10 chunks * 10 bytes

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_state_updates(self):
        """Test concurrent state updates."""
        manager = StateManager()
        await manager.create_session(session_id="test")

        states = ["idle", "listening", "transcribing", "processing", "speaking"]

        async def update_state(state):
            await manager.update_session_state("test", state)

        # Update state concurrently
        tasks = [update_state(s) for s in states]
        await asyncio.gather(*tasks)

        # Session should have some valid state
        session = await manager.get_session("test")
        assert session.state in states
