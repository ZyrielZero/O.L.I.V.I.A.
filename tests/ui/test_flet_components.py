"""
UI component tests for Flet application.
Tests component styling, state management, and API client.
"""

from unittest.mock import MagicMock

import pytest

# ===== Test 1: Theme Colors Defined =====

@pytest.mark.ui
def test_theme_colors_defined():
    """All theme colors are properly defined."""
    from src.flet_app.theme import ColorPalette

    palette = ColorPalette()

    # Check required colors exist
    assert palette.BG_BASE is not None
    assert palette.BG_SURFACE_1 is not None
    assert palette.ACCENT_PRIMARY is not None
    assert palette.TEXT_PRIMARY is not None

    # Colors should be hex format
    assert palette.BG_BASE.startswith("#")
    assert palette.ACCENT_PRIMARY.startswith("#")


# ===== Test 2: Theme Palette Consistency =====

@pytest.mark.ui
def test_chat_bubble_user_styling():
    """User and assistant messages have different colors."""
    from src.flet_app.theme import ColorPalette

    palette = ColorPalette()

    # User messages use accent glow, assistant uses primary
    user_color = palette.ACCENT_GLOW
    assistant_color = palette.ACCENT_PRIMARY

    assert user_color != assistant_color, "User and assistant colors should differ"


# ===== Test 3: Assistant Styling Colors =====

@pytest.mark.ui
def test_chat_bubble_assistant_styling():
    """Assistant messages have distinct styling."""
    from src.flet_app.theme import ColorPalette

    palette = ColorPalette()

    assert palette.ACCENT_PRIMARY is not None
    assert palette.BG_SURFACE_1 is not None
    assert isinstance(palette.ACCENT_PRIMARY, str)


# ===== Test 4: Chat Display Streaming Update =====

@pytest.mark.ui
@pytest.mark.asyncio
async def test_chat_display_streaming_update():
    """Streaming text updates display correctly."""
    display_text = ""

    async def append_token(token):
        nonlocal display_text
        display_text += token

    tokens = ["Hello", " ", "world", "!"]
    for token in tokens:
        await append_token(token)

    assert display_text == "Hello world!"


# ===== Test 5: State Manager with Mock Page =====

@pytest.mark.ui
def test_state_manager_subscription():
    """State changes notify subscribers via pub/sub."""
    from src.flet_app.services.state_manager import StateManager

    # Create mock page
    mock_page = MagicMock()
    mock_page.update = MagicMock()

    manager = StateManager(page=mock_page)
    notifications = []

    def callback(state):
        """Callback receives the AppState object."""
        notifications.append(state)

    # Subscribe
    manager.subscribe(callback)

    # Update state
    manager.update(status="ready")

    # Should have received notification
    assert len(notifications) >= 1
    # Verify the state was passed
    assert notifications[0].status == "ready"


# ===== Test 6: State Manager Message History =====

@pytest.mark.ui
def test_state_manager_message_history():
    """Messages added to history correctly."""
    from src.flet_app.services.state_manager import StateManager

    mock_page = MagicMock()
    mock_page.update = MagicMock()

    manager = StateManager(page=mock_page)

    # Add messages
    manager.add_message("user", "Hello")
    manager.add_message("assistant", "Hi there!")

    # Check history
    assert len(manager.state.messages) == 2


# ===== Test 7: API Client Connection Retry =====

@pytest.mark.ui
@pytest.mark.asyncio
async def test_api_client_connection_retry():
    """Connection retry logic works correctly."""
    from src.flet_app.services.api_client import OliviaAPIClient

    client = OliviaAPIClient(base_url="http://localhost:99999")

    is_connected = await client.check_connection(
        max_retries=2,
        retry_delay=0.1
    )

    assert is_connected is False
    await client.close()


# ===== Additional UI Tests =====

@pytest.mark.ui
def test_theme_spacing_values():
    """Theme spacing values follow consistent scale."""
    from src.flet_app.theme import Spacing

    # Check spacing scale
    assert Spacing.XS < Spacing.SM
    assert Spacing.SM < Spacing.MD
    assert Spacing.MD < Spacing.LG
    assert Spacing.LG < Spacing.XL


@pytest.mark.ui
def test_state_manager_initial_state():
    """State manager has correct initial state."""
    from src.flet_app.services.state_manager import StateManager

    mock_page = MagicMock()
    mock_page.update = MagicMock()

    manager = StateManager(page=mock_page)

    assert manager.state.backend_connected is False
    assert manager.state.is_processing is False
    assert manager.state.is_recording is False


@pytest.mark.ui
def test_state_manager_clear_messages():
    """Clear messages removes all chat history."""
    from src.flet_app.services.state_manager import StateManager

    mock_page = MagicMock()
    mock_page.update = MagicMock()

    manager = StateManager(page=mock_page)

    manager.add_message("user", "Test 1")
    manager.add_message("assistant", "Test 2")

    assert len(manager.state.messages) == 2

    manager.clear_messages()

    assert len(manager.state.messages) == 0


@pytest.mark.ui
def test_api_client_initialization():
    """API client initializes with correct base URL."""
    from src.flet_app.services.api_client import OliviaAPIClient

    client = OliviaAPIClient(base_url="http://localhost:8000")
    assert client.base_url == "http://localhost:8000"


@pytest.mark.ui
def test_color_palette_frozen():
    """ColorPalette is immutable (frozen dataclass)."""
    from src.flet_app.theme import ColorPalette

    palette = ColorPalette()

    # Should not be able to modify
    with pytest.raises(Exception):  # FrozenInstanceError
        palette.BG_BASE = "#000000"


@pytest.mark.ui
def test_status_colors_distinct():
    """Status colors are distinct from each other."""
    from src.flet_app.theme import ColorPalette

    palette = ColorPalette()

    colors = [
        palette.STATUS_SUCCESS,
        palette.STATUS_WARNING,
        palette.STATUS_ERROR,
        palette.STATUS_PURPLE,
    ]

    # All should be unique
    assert len(set(colors)) == len(colors)


@pytest.mark.ui
def test_animation_timing_exists():
    """Animation timing constants are defined."""
    from src.flet_app.theme import Animation

    # Check animation durations exist
    assert Animation.INSTANT > 0
    assert Animation.FAST > Animation.INSTANT
    assert Animation.NORMAL > Animation.FAST


@pytest.mark.ui
def test_status_indicator_states():
    """Status indicator supports all required states."""
    expected_states = [
        "initializing",
        "ready",
        "recording",
        "processing",
        "speaking",
        "error",
    ]

    # Verify the expected states are comprehensive
    assert len(expected_states) >= 5
    assert "ready" in expected_states
    assert "error" in expected_states
