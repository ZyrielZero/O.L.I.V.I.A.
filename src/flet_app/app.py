"""Main Flet app."""

import asyncio
import logging

import flet as ft

from src.flet_app.components.animated_button import (
    AnimatedIconButton,
    AnimatedInputContainer,
    AnimatedSendButton,
)
from src.flet_app.components.chat_display import ChatDisplay
from src.flet_app.components.settings_dialog import SettingsDialog
from src.flet_app.components.status_indicator import StatusIndicator
from src.flet_app.services.api_client import OliviaAPIClient
from src.flet_app.services.state_manager import StateManager
from src.flet_app.services.voice_client import VoiceClient
from src.flet_app.theme import Theme

log = logging.getLogger("flet.app")


class OliviaApp:
    """Main Flet app."""

    def __init__(self, page: ft.Page):
        self.page = page

        # Configure page
        self.page.title = "O.L.I.V.I.A."
        self.page.window.width = 1000
        self.page.window.height = 800
        self.page.window.min_width = 800
        self.page.window.min_height = 600

        # Apply theme
        Theme.apply_to_page(self.page)

        # Initialize services
        self.state_manager = StateManager(self.page)
        self.api_client = OliviaAPIClient()

        # Components
        self.status_indicator = None
        self.chat_display = None
        self.input_field = None
        self.send_button = None
        self.mic_button = None
        self.settings_button = None
        self.settings_dialog = None
        self._is_recording = False
        self._voice_client = None
        self._voice_streaming = False  # assistant bubble open during voice reply

        # Build UI
        self.build()

        # Add window close handler for graceful cleanup
        self.page.window.prevent_close = True
        self.page.window.on_event = self._on_window_event

        # Initialize backend connection (reference kept: asyncio holds only
        # weak refs, and losing this task would silently skip backend init)
        self._init_task = asyncio.create_task(self._initialize_backend())

    def build(self):
        """Build main layout."""
        # Header with glassmorphism
        self.status_indicator = StatusIndicator()

        # Settings button for header
        self.settings_button = AnimatedIconButton(
            icon=ft.Icons.SETTINGS_ROUNDED,
            on_click=self._open_settings,
            icon_size=20,
            button_size=36,
            bgcolor="transparent",
            icon_color=Theme.colors.TEXT_TERTIARY,
            hover_bgcolor=Theme.colors.BG_SURFACE_3,
            tooltip="Settings",
        )

        header = ft.Container(
            height=Theme.dimensions.HEADER_HEIGHT,
            bgcolor=Theme.colors.BG_SURFACE_1,  # Solid background
            padding=ft.padding.symmetric(horizontal=Theme.spacing.LG),
            border=ft.border.only(bottom=ft.BorderSide(1, Theme.colors.BORDER_DEFAULT)),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=16,
                color=Theme.glass.SHADOW_COLOR,
                offset=ft.Offset(0, 4),
            ),
            content=ft.Row(
                [
                    self.status_indicator,
                    ft.Container(expand=True),  # Spacer
                    ft.Column(
                        [
                            ft.Text(
                                "Voice Assistant",
                                size=Theme.typography.SIZE_CAPTION,
                                color=Theme.colors.TEXT_TERTIARY,
                            ),
                            ft.Text(
                                "O.L.I.V.I.A.",
                                size=Theme.typography.SIZE_H3,
                                weight=ft.FontWeight.BOLD,
                                color=Theme.colors.TEXT_PRIMARY,
                            ),
                        ],
                        spacing=0,
                        horizontal_alignment=ft.CrossAxisAlignment.END,
                    ),
                    self.settings_button,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        # Chat display
        self.chat_display = ChatDisplay()

        # Input bar with animated components
        self.input_field = ft.TextField(
            hint_text="Type your message...",
            expand=True,
            text_size=Theme.typography.SIZE_BODY_SM,
            on_submit=lambda e: asyncio.create_task(self.on_send_message()),
            border_radius=Theme.radius.MD,
        )

        # Wrap input field in animated container for focus effects
        input_container = AnimatedInputContainer(self.input_field)

        # Animated send button with hover/press effects
        self.send_button = AnimatedSendButton(on_click=self.on_send_message)

        # Microphone button for voice input (left of input field)
        self.mic_button = AnimatedIconButton(
            icon=ft.Icons.MIC_ROUNDED,
            on_click=self._toggle_voice,
            icon_size=24,
            button_size=48,
            bgcolor=Theme.colors.BG_SURFACE_1,
            icon_color=Theme.colors.MIC_INACTIVE,
            hover_bgcolor=Theme.colors.BG_SURFACE_2,
            tooltip="Push to talk",
        )

        input_bar = ft.Container(
            height=Theme.dimensions.INPUT_BAR_HEIGHT,
            bgcolor=Theme.colors.BG_SURFACE_1,  # Solid background
            padding=Theme.spacing.MD,
            border=ft.border.only(top=ft.BorderSide(1, Theme.colors.BORDER_DEFAULT)),
            content=ft.Row(
                [self.mic_button, input_container, self.send_button],
                spacing=Theme.spacing.SM,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        # Main content
        main_content = ft.Container(
            content=self.chat_display,
            expand=True,
            padding=Theme.spacing.MD,
        )

        # Full layout
        self.page.add(
            ft.Column(
                [header, main_content, input_bar],
                expand=True,
                spacing=0,
            )
        )

    async def _initialize_backend(self):
        """Initialize backend connection."""
        self.status_indicator.set_status("initializing")

        # Check connection
        connected = await self.api_client.check_connection()

        if connected:
            # Get health status
            health = await self.api_client.get_health()
            if health and health.get("status") == "healthy":
                self.state_manager.update(
                    backend_connected=True,
                    orb_state="ready",
                    status="ready",
                )
                # Extract metrics from health response
                metrics = self._extract_metrics(health)
                self.status_indicator.set_status("ready", metrics=metrics)
                log.info("✓ Connected to backend")

                # Send startup greeting to warm up LLM + TTS
                await self._send_startup_greeting()
            else:
                self.state_manager.update(backend_connected=False, status="error")
                self.status_indicator.set_status("error")
                log.warning("Backend unhealthy")
        else:
            self.state_manager.update(backend_connected=False, status="error")
            self.status_indicator.set_status("error")
            log.error("❌ Failed to connect to backend")
            self._show_error_dialog(
                "Backend Not Available",
                "Could not connect to O.L.I.V.I.A. backend.\n\nPlease ensure the FastAPI server is running:\n\ncd src\nuvicorn api.main:app --reload",
            )

    async def _send_startup_greeting(self):
        """Send initial greeting to warm up system and greet user."""
        log.info("Sending startup greeting...")
        self.send_button.set_loading(True)
        self.status_indicator.set_status("processing")
        self.chat_display.start_streaming("O.L.I.V.I.A.")

        try:
            async for token in self.api_client.send_message_stream("Hello"):
                self.chat_display.append_token(token)
                await asyncio.sleep(0.01)
            self.chat_display.end_streaming()
            log.info("Startup greeting complete")
        except Exception as e:
            log.warning(f"Startup greeting failed: {e}")
            self.chat_display.end_streaming()

        self.send_button.set_loading(False)
        self.status_indicator.set_status("ready")

    async def on_send_message(self):
        """Handle send message action."""
        message = self.input_field.value.strip()
        if not message:
            return

        # Clear input
        self.input_field.value = ""
        self.page.update()

        # Set button loading state
        self.send_button.set_loading(True)

        # Update state
        self.state_manager.update(is_processing=True, status="processing")
        self.status_indicator.set_status("processing")

        # Add user message to chat
        self.chat_display.append_message("You", message)

        # Start streaming assistant response
        self.chat_display.start_streaming("O.L.I.V.I.A.")

        try:
            async for token in self.api_client.send_message_stream(message):
                self.chat_display.append_token(token)
                await asyncio.sleep(0.01)  # Prevent UI blocking

            self.chat_display.end_streaming()

        except Exception as e:
            log.error(f"Error during message send: {e}")
            self.chat_display.append_token(f"\n\n[Error: {str(e)}]")
            self.chat_display.end_streaming()

        # Reset button loading state
        self.send_button.set_loading(False)

        # Update state
        self.state_manager.update(is_processing=False, status="ready")
        self.status_indicator.set_status("ready")

    def _voice_ws_url(self) -> str:
        """Derive the /ws/voice URL from the API client's base URL."""
        base = getattr(self.api_client, "base_url", "http://127.0.0.1:8000")
        return base.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws/voice"

    async def _toggle_voice(self, e=None):
        """Start/stop a live voice session over /ws/voice (Phase 1.5)."""
        if not self._is_recording:
            try:
                self._voice_client = VoiceClient(self._voice_ws_url(), self._on_voice_event)
                await self._voice_client.start()
            except Exception as ex:
                log.error(f"Voice session failed to start: {ex}")
                self._voice_client = None
                self._show_error_dialog(
                    "Voice Unavailable",
                    f"Could not start the voice session:\n\n{ex}",
                )
                return

            self._is_recording = True
            self.mic_button.icon_control.color = Theme.colors.MIC_ACTIVE
            self.mic_button._icon_color = Theme.colors.MIC_ACTIVE
            self.mic_button.bgcolor = Theme.colors.BG_SURFACE_2
            self.status_indicator.set_status("recording")
            self.state_manager.update(is_recording=True, status="recording")
            log.info("Voice session started")
        else:
            self._is_recording = False
            if self._voice_client:
                await self._voice_client.stop()
                self._voice_client = None
            if self._voice_streaming:
                self.chat_display.end_streaming()
                self._voice_streaming = False
            self.mic_button.icon_control.color = Theme.colors.MIC_INACTIVE
            self.mic_button._icon_color = Theme.colors.MIC_INACTIVE
            self.mic_button.bgcolor = Theme.colors.BG_SURFACE_1
            self.status_indicator.set_status("ready")
            self.state_manager.update(is_recording=False, status="ready")
            log.info("Voice session stopped")

        self.mic_button.update()

    async def _on_voice_event(self, event: dict):
        """Drive UI state from /ws/voice control events."""
        kind = event.get("type")

        if kind == "speech_start":
            self.status_indicator.set_status("recording")
        elif kind == "transcript_final":
            self.chat_display.append_message("You", event.get("text", ""))
            self.chat_display.start_streaming("O.L.I.V.I.A.")
            self._voice_streaming = True
            self.status_indicator.set_status("processing")
        elif kind == "token":
            if self._voice_streaming:
                self.chat_display.append_token(event.get("text", ""))
        elif kind == "audio_start":
            self.status_indicator.set_status("processing")
            self.state_manager.update(status="speaking")
        elif kind in ("done", "barge_in"):
            if self._voice_streaming:
                self.chat_display.end_streaming()
                self._voice_streaming = False
            if self._is_recording:
                self.status_indicator.set_status("recording")
        elif kind == "error":
            log.error(f"Voice error: {event.get('message')}")
            if self._voice_streaming:
                self.chat_display.end_streaming()
                self._voice_streaming = False
            self.status_indicator.set_status("error")
        self.page.update()

    async def _open_settings(self, e=None):
        """Open settings dialog."""
        # Get current settings from state manager
        state = self.state_manager.state

        self.settings_dialog = SettingsDialog(
            on_close=self._on_settings_close,
            voice_enabled=state.voice_enabled,
            always_listen=state.always_listen_enabled,
            wake_word_enabled=state.wake_word_enabled,
            auto_chat_enabled=state.auto_chat_enabled,
        )

        self.page.dialog = self.settings_dialog
        self.settings_dialog.open = True
        self.page.update()

    def _on_settings_close(self, settings: dict):
        """Handle settings dialog close."""
        # Update state manager with new settings
        self.state_manager.update(**settings)
        log.info(f"Settings updated: {settings}")

    def _show_error_dialog(self, title: str, message: str):
        """Show error dialog."""

        def close_dialog(e):
            dialog.open = False
            self.page.update()

        dialog = ft.AlertDialog(
            title=ft.Text(title),
            content=ft.Text(message),
            actions=[
                ft.TextButton("OK", on_click=close_dialog),
            ],
        )
        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    async def _on_window_event(self, e):
        """Handle window events including close."""
        if e.data == "close":
            log.info("Window closing, cleaning up...")
            await self._cleanup()
            self.page.window.close()

    async def _cleanup(self):
        """Clean up resources before exit."""
        if self._voice_client:
            try:
                await self._voice_client.stop()
            except Exception as e:
                log.warning(f"Voice cleanup error: {e}")
            self._voice_client = None
        try:
            await self.api_client.close()
            log.info("API client closed")
        except Exception as e:
            log.warning(f"Cleanup error: {e}")

    def _extract_metrics(self, health: dict) -> str:
        """Extract metrics string from health response.

        Args:
            health: Health check response dict

        Returns:
            Formatted metrics string
        """
        parts = []

        # Extract LLM model name
        services = health.get("services", {})
        llm_info = services.get("llm", {})
        if llm_info.get("status") == "up":
            message = llm_info.get("message", "")
            # Extract model name from "Connected to Ollama (model-name)"
            if "(" in message and ")" in message:
                model = message.split("(")[1].split(")")[0]
                parts.append(model)

        # Add uptime if available
        uptime = health.get("uptime_seconds", 0)
        if uptime > 0:
            if uptime < 60:
                parts.append(f"{int(uptime)}s")
            elif uptime < 3600:
                parts.append(f"{int(uptime // 60)}m")
            else:
                parts.append(f"{int(uptime // 3600)}h")

        return " | ".join(parts) if parts else ""
