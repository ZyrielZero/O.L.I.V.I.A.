"""Animated buttons and inputs."""

import asyncio
from typing import Callable

import flet as ft

from src.flet_app.theme import Theme


class AnimatedSendButton(ft.Container):
    """Send button with hover/press effects."""

    def __init__(self, on_click: Callable = None):
        super().__init__()
        self.on_click_handler = on_click
        self.is_loading = False
        self.icon_control = None
        self._base_bg = Theme.colors.ACCENT_PRIMARY
        self._build()

    def _build(self):
        self.icon_control = ft.Icon(ft.Icons.SEND_ROUNDED, size=20, color=Theme.colors.TEXT_PRIMARY)

        self.content = self.icon_control
        self.width = 48
        self.height = 48
        self.border_radius = Theme.radius.MD
        self.bgcolor = self._base_bg
        self.alignment = ft.Alignment(0, 0)

        self.scale = 1.0
        self.opacity = 1.0

        self.animate_scale = ft.Animation(
            duration=Theme.animation.FAST, curve=ft.AnimationCurve.EASE_OUT
        )
        self.animate_opacity = ft.Animation(
            duration=Theme.animation.FAST, curve=ft.AnimationCurve.EASE_OUT
        )
        self.animate_bgcolor = ft.Animation(
            duration=Theme.animation.FAST, curve=ft.AnimationCurve.EASE_OUT
        )

        self.on_hover = self._on_hover
        self.on_click = self._on_click

    def _on_hover(self, e):
        if self.is_loading:
            return
        if e.data == "true":
            self.scale = Theme.animation.HOVER_SCALE
            self.bgcolor = Theme.colors.ACCENT_LIGHT
        else:
            self.scale = 1.0
            self.bgcolor = self._base_bg
        self.update()

    def _on_click(self, e):
        if self.is_loading:
            return
        asyncio.create_task(self._press_anim(e))

    async def _press_anim(self, e):
        self.scale = Theme.animation.PRESS_SCALE
        self.update()
        await asyncio.sleep(Theme.animation.INSTANT / 1000)
        self.scale = 1.0
        self.update()

        if self.on_click_handler:
            if asyncio.iscoroutinefunction(self.on_click_handler):
                await self.on_click_handler()
            else:
                self.on_click_handler(e)

    def set_loading(self, loading: bool):
        self.is_loading = loading
        self.opacity = 0.7 if loading else 1.0
        self.icon_control.name = ft.Icons.HOURGLASS_EMPTY if loading else ft.Icons.SEND_ROUNDED
        self.update()


class AnimatedInputContainer(ft.Container):
    """Wrapper for TextField with animated focus effects."""

    def __init__(self, text_field: ft.TextField):
        super().__init__()
        self.text_field = text_field
        self._is_focused = False
        self._build_ui()

    def _build_ui(self):
        """Build animated input container."""
        # Configure text field styling
        self.text_field.border_color = "transparent"
        self.text_field.focused_border_color = "transparent"
        self.text_field.cursor_color = Theme.colors.ACCENT_PRIMARY
        self.text_field.selection_color = Theme.colors.ACCENT_MUTED

        # Store original handlers
        original_on_focus = self.text_field.on_focus
        original_on_blur = self.text_field.on_blur

        def on_focus_handler(e):
            self._on_focus(e)
            if original_on_focus:
                original_on_focus(e)

        def on_blur_handler(e):
            self._on_blur(e)
            if original_on_blur:
                original_on_blur(e)

        self.text_field.on_focus = on_focus_handler
        self.text_field.on_blur = on_blur_handler

        self.content = self.text_field
        self.expand = True
        self.bgcolor = Theme.colors.BG_BASE  # Darker for contrast (Discord-style)
        self.border_radius = Theme.radius.SM  # Smaller radius for modern look
        self.border = ft.border.all(1, Theme.colors.BORDER_DEFAULT)
        self.padding = ft.padding.symmetric(horizontal=Theme.spacing.MD)

        # Animation setup for border and bgcolor transitions
        self.animate_border = ft.Animation(
            duration=Theme.animation.FAST, curve=ft.AnimationCurve.EASE_OUT
        )
        self.animate_bgcolor = ft.Animation(
            duration=Theme.animation.FAST, curve=ft.AnimationCurve.EASE_OUT
        )

    def _on_focus(self, e):
        """Handle focus with animation."""
        self._is_focused = True
        self.border = ft.border.all(2, Theme.colors.ACCENT_PRIMARY)
        self.bgcolor = Theme.colors.BG_BASE  # Keep same on focus
        self.update()

    def _on_blur(self, e):
        """Handle blur with animation."""
        self._is_focused = False
        self.border = ft.border.all(1, Theme.colors.BORDER_DEFAULT)
        self.bgcolor = Theme.colors.BG_BASE
        self.update()


class AnimatedIconButton(ft.Container):
    """Generic animated icon button with hover and press effects."""

    def __init__(
        self,
        icon: str,
        on_click: Callable = None,
        icon_size: int = 20,
        button_size: int = 40,
        bgcolor: str = None,
        icon_color: str = None,
        hover_bgcolor: str = None,
        tooltip: str = None,
    ):
        super().__init__()
        self.on_click_handler = on_click
        self.icon_name = icon
        self.icon_size = icon_size
        self.button_size = button_size
        self._original_bgcolor = bgcolor or Theme.colors.BG_SURFACE_1
        self._hover_bgcolor = hover_bgcolor or Theme.colors.BG_SURFACE_2
        self._icon_color = icon_color or Theme.colors.TEXT_SECONDARY
        self._tooltip = tooltip
        self.icon_control = None
        self._build_ui()

    def _build_ui(self):
        """Build animated icon button UI."""
        self.icon_control = ft.Icon(
            self.icon_name,
            size=self.icon_size,
            color=self._icon_color,
        )

        self.content = self.icon_control
        self.width = self.button_size
        self.height = self.button_size
        self.border_radius = Theme.radius.SM
        self.bgcolor = self._original_bgcolor
        self.alignment = ft.Alignment(0, 0)
        self.tooltip = self._tooltip

        # Animation setup
        self.scale = 1.0

        self.animate_scale = ft.Animation(
            duration=Theme.animation.FAST, curve=ft.AnimationCurve.EASE_OUT
        )
        self.animate_bgcolor = ft.Animation(
            duration=Theme.animation.FAST, curve=ft.AnimationCurve.EASE_OUT
        )

        # Event handlers
        self.on_hover = self._on_hover
        self.on_click = self._on_click

    def _on_hover(self, e):
        """Handle hover events."""
        if e.data == "true":
            self.scale = Theme.animation.HOVER_SCALE
            self.bgcolor = self._hover_bgcolor
            self.icon_control.color = Theme.colors.ACCENT_PRIMARY
        else:
            self.scale = 1.0
            self.bgcolor = self._original_bgcolor
            self.icon_control.color = self._icon_color
        self.update()

    def _on_click(self, e):
        """Handle click with press animation."""
        asyncio.create_task(self._animate_press(e))

    async def _animate_press(self, e):
        """Animate press effect."""
        self.scale = Theme.animation.PRESS_SCALE
        self.update()

        await asyncio.sleep(Theme.animation.INSTANT / 1000)

        self.scale = 1.0
        self.update()

        if self.on_click_handler:
            if asyncio.iscoroutinefunction(self.on_click_handler):
                await self.on_click_handler(e)
            else:
                self.on_click_handler(e)
