"""Settings dialog component for O.L.I.V.I.A."""

from typing import Callable, Optional

import flet as ft

from src.flet_app.theme import Theme

# Setting definitions: (key, title, description, default)
SETTINGS_CONFIG = [
    ("voice_enabled", "Voice Input", "Enable microphone for voice commands", True),
    ("always_listen", "Always Listen", "Continuously listen for voice input", False),
    ("wake_word_enabled", "Wake Word", "Activate with 'Hey Olivia'", False),
    ("auto_chat_enabled", "Auto Chat", "Automatically send after speech ends", False),
]


class SettingsDialog(ft.AlertDialog):
    """Settings dialog with voice and system toggles."""

    def __init__(
        self,
        on_close: Optional[Callable] = None,
        voice_enabled: bool = True,
        always_listen: bool = False,
        wake_word_enabled: bool = False,
        auto_chat_enabled: bool = False,
    ):
        super().__init__()
        self.on_close_handler = on_close
        self._settings = {
            "voice_enabled": voice_enabled,
            "always_listen": always_listen,
            "wake_word_enabled": wake_word_enabled,
            "auto_chat_enabled": auto_chat_enabled,
        }
        self._toggles: dict[str, ft.Switch] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        """Build settings dialog UI."""
        toggle_rows = []
        for key, title, desc, _ in SETTINGS_CONFIG:
            toggle = ft.Switch(
                value=self._settings[key],
                active_color=Theme.colors.ACCENT_PRIMARY,
                on_change=lambda e, k=key: self._on_toggle(k, e.control.value),
            )
            self._toggles[key] = toggle
            toggle_rows.append(self._create_toggle_row(title, desc, toggle))

        self.modal = True
        self.title = ft.Text(
            "Settings",
            size=Theme.typography.SIZE_H3,
            weight=ft.FontWeight.BOLD,
            color=Theme.colors.TEXT_PRIMARY,
        )
        self.content = ft.Container(
            content=ft.Column(toggle_rows, spacing=Theme.spacing.SM, tight=True),
            width=350,
            padding=Theme.spacing.MD,
        )
        self.actions = [
            ft.TextButton(
                "Close",
                on_click=self._on_close,
                style=ft.ButtonStyle(color=Theme.colors.ACCENT_PRIMARY),
            ),
        ]
        self.actions_alignment = ft.MainAxisAlignment.END
        self.bgcolor = Theme.colors.BG_SURFACE_2
        self.shape = ft.RoundedRectangleBorder(radius=Theme.radius.SM)

    def _create_toggle_row(self, title: str, desc: str, toggle: ft.Switch) -> ft.Container:
        """Create a toggle row with title and description."""
        return ft.Container(
            content=ft.Row(
                [
                    ft.Column(
                        [
                            ft.Text(
                                title,
                                size=Theme.typography.SIZE_BODY_SM,
                                weight=ft.FontWeight.W_500,
                                color=Theme.colors.TEXT_PRIMARY,
                            ),
                            ft.Text(
                                desc,
                                size=Theme.typography.SIZE_CAPTION,
                                color=Theme.colors.TEXT_MUTED,
                            ),
                        ],
                        spacing=2,
                        expand=True,
                    ),
                    toggle,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.padding.symmetric(vertical=Theme.spacing.SM),
            border=ft.border.only(bottom=ft.BorderSide(1, Theme.colors.BORDER_DEFAULT)),
        )

    def _on_toggle(self, key: str, value: bool) -> None:
        """Handle any toggle change."""
        self._settings[key] = value

    def _on_close(self, e) -> None:
        """Handle dialog close."""
        self.open = False
        if self.on_close_handler:
            self.on_close_handler(self.get_settings())
        self.update()

    def get_settings(self) -> dict:
        """Get current settings values."""
        return {
            "voice_enabled": self._settings["voice_enabled"],
            "always_listen_enabled": self._settings["always_listen"],
            "wake_word_enabled": self._settings["wake_word_enabled"],
            "auto_chat_enabled": self._settings["auto_chat_enabled"],
        }
