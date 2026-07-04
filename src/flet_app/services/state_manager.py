"""App state management for Flet UI."""

from dataclasses import dataclass, field
from typing import Any, Callable, List

import flet as ft


@dataclass
class AppState:
    """App state container."""

    backend_connected: bool = False
    backend_status: str = "initializing"

    voice_enabled: bool = True
    always_listen_enabled: bool = False
    wake_word_enabled: bool = False
    auto_chat_enabled: bool = False

    is_processing: bool = False
    is_recording: bool = False
    is_speaking: bool = False

    orb_state: str = "initializing"
    status: str = "initializing"
    debug_level: str = "Normal"

    messages: List[dict] = field(default_factory=list)


class StateManager:
    """Central state with pub/sub notifications."""

    def __init__(self, page: ft.Page):
        self.page = page
        self.state = AppState()
        self._listeners: List[Callable[[AppState], None]] = []

    def subscribe(self, cb: Callable[[AppState], None]):
        """Register a listener for state changes."""
        self._listeners.append(cb)

    def unsubscribe(self, cb: Callable[[AppState], None]):
        """Remove a previously registered listener."""
        if cb in self._listeners:
            self._listeners.remove(cb)

    def update(self, **kw):
        """Update state attrs and notify."""
        for k, v in kw.items():
            if hasattr(self.state, k):
                setattr(self.state, k, v)
        self._notify()

    def _notify(self):
        for cb in self._listeners:
            try:
                cb(self.state)
            except Exception as e:
                print(f"State listener error: {e}")

    def get(self, key: str) -> Any:
        """Get a state attribute, or None if it doesn't exist."""
        return getattr(self.state, key, None)

    def add_message(self, role: str, content: str):
        """Append a chat message and notify."""
        self.state.messages.append({"role": role, "content": content})
        self._notify()

    def clear_messages(self):
        """Clear all chat messages and notify."""
        self.state.messages.clear()
        self._notify()
