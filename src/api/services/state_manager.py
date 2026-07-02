"""Session state management."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Session:
    """Client session (for WS connections)."""

    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    state: str = "idle"  # idle/listening/transcribing/processing/speaking
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    audio_buffer: bytearray = field(default_factory=bytearray)


class StateManager:
    """Manages session state for concurrent clients."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    async def create_session(self, session_id: Optional[str] = None) -> Session:
        """Create session with unique ID."""
        session_id = session_id or str(uuid.uuid4())
        sess = Session(session_id=session_id)
        self._sessions[session_id] = sess
        return sess

    async def get_session(self, sid: str) -> Optional[Session]:
        return self._sessions.get(sid)

    async def update_session_state(self, sid: str, state: str):
        if sess := self._sessions.get(sid):
            sess.state = state
            sess.last_activity = datetime.now()

    async def append_to_audio_buffer(self, sid: str, data: bytes):
        if sess := self._sessions.get(sid):
            sess.audio_buffer.extend(data)
            sess.last_activity = datetime.now()

    async def clear_audio_buffer(self, sid: str):
        if sess := self._sessions.get(sid):
            sess.audio_buffer.clear()

    async def cleanup_session(self, sid: str):
        self._sessions.pop(sid, None)

    async def cleanup_stale_sessions(self, timeout_seconds: int = 3600):
        """Remove sessions inactive > timeout."""
        now = datetime.now()
        stale = [
            s
            for s, sess in self._sessions.items()
            if (now - sess.last_activity).total_seconds() > timeout_seconds
        ]
        for s in stale:
            await self.cleanup_session(s)

    def get_active_sessions(self) -> int:
        return len(self._sessions)

    def get_all_session_ids(self) -> List[str]:
        return list(self._sessions.keys())
