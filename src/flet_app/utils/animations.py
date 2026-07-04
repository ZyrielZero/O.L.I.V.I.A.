"""Animation utilities for O.L.I.V.I.A. Flet application."""

import asyncio
from typing import Callable, Optional

import flet as ft

from src.flet_app.theme import Theme


def create_fade_in_animation(duration_ms: int = None) -> ft.Animation:
    """Create a fade-in animation configuration.

    Args:
        duration_ms: Duration in milliseconds, defaults to Theme.animation.MEDIUM

    Returns:
        Flet Animation object for opacity transitions
    """
    return ft.Animation(
        duration=duration_ms or Theme.animation.MEDIUM, curve=ft.AnimationCurve.EASE_OUT
    )


def create_scale_animation(duration_ms: int = None) -> ft.Animation:
    """Create a scale animation configuration.

    Args:
        duration_ms: Duration in milliseconds, defaults to Theme.animation.FAST

    Returns:
        Flet Animation object for scale transitions
    """
    return ft.Animation(
        duration=duration_ms or Theme.animation.FAST, curve=ft.AnimationCurve.EASE_OUT
    )


def create_slide_animation(duration_ms: int = None) -> ft.Animation:
    """Create a slide animation configuration.

    Args:
        duration_ms: Duration in milliseconds, defaults to Theme.animation.MEDIUM

    Returns:
        Flet Animation object for offset transitions
    """
    return ft.Animation(
        duration=duration_ms or Theme.animation.MEDIUM, curve=ft.AnimationCurve.EASE_OUT_CUBIC
    )


class PulseAnimator:
    """Manages a continuous pulsing/breathing animation for a control.

    Uses scale and opacity to create a breathing effect.
    """

    def __init__(
        self,
        control: ft.Control,
        cycle_duration_ms: int = 2000,
        scale_range: tuple = (0.85, 1.0),
        opacity_range: tuple = (0.6, 1.0),
    ):
        """Initialize pulse animator.

        Args:
            control: Flet control to animate
            cycle_duration_ms: Full pulse cycle duration in ms
            scale_range: (min_scale, max_scale) tuple
            opacity_range: (min_opacity, max_opacity) tuple
        """
        self.control = control
        self.cycle_duration = cycle_duration_ms
        self.scale_min, self.scale_max = scale_range
        self.opacity_min, self.opacity_max = opacity_range
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def start(self):
        """Start the pulse animation loop."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._pulse_loop())

    def stop(self):
        """Stop the pulse animation and reset to default state."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        # Reset to default state
        try:
            self.control.scale = 1.0
            self.control.opacity = 1.0
            self.control.update()
        except Exception:
            pass  # Control may not be mounted

    async def _pulse_loop(self):
        """Internal pulse animation loop."""
        half_cycle = self.cycle_duration / 2 / 1000  # Convert to seconds

        while self._running:
            try:
                # Expand phase
                self.control.scale = self.scale_max
                self.control.opacity = self.opacity_max
                self.control.update()
                await asyncio.sleep(half_cycle)

                if not self._running:
                    break

                # Contract phase
                self.control.scale = self.scale_min
                self.control.opacity = self.opacity_min
                self.control.update()
                await asyncio.sleep(half_cycle)

            except asyncio.CancelledError:
                break
            except Exception:
                break  # Control may have been destroyed


class HoverAnimator:
    """Applies hover animation effects to a container."""

    @staticmethod
    def apply(
        container: ft.Container,
        hover_scale: float = None,
        hover_bgcolor: str = None,
        duration_ms: int = None,
    ):
        """Apply hover animation effects to a container.

        Args:
            container: Container to animate
            hover_scale: Scale on hover (default: Theme.animation.HOVER_SCALE)
            hover_bgcolor: Background color on hover
            duration_ms: Animation duration (default: Theme.animation.FAST)
        """
        hover_scale = hover_scale or Theme.animation.HOVER_SCALE
        duration_ms = duration_ms or Theme.animation.FAST
        original_bgcolor = container.bgcolor

        # Set up animations
        container.animate_scale = ft.Animation(
            duration=duration_ms, curve=ft.AnimationCurve.EASE_OUT
        )
        if hover_bgcolor:
            container.animate_bgcolor = ft.Animation(
                duration=duration_ms, curve=ft.AnimationCurve.EASE_OUT
            )

        # Store original handler if any
        original_on_hover = container.on_hover

        def on_hover_handler(e):
            if e.data == "true":
                container.scale = hover_scale
                if hover_bgcolor:
                    container.bgcolor = hover_bgcolor
            else:
                container.scale = 1.0
                if hover_bgcolor:
                    container.bgcolor = original_bgcolor
            container.update()

            if original_on_hover:
                original_on_hover(e)

        container.on_hover = on_hover_handler


class PressAnimator:
    """Applies press animation effects to a container."""

    @staticmethod
    def apply(
        container: ft.Container,
        press_scale: float = None,
        duration_ms: int = None,
        on_click_callback: Callable = None,
    ):
        """Apply press animation effect to a container.

        Args:
            container: Container to animate
            press_scale: Scale when pressed (default: Theme.animation.PRESS_SCALE)
            duration_ms: Animation duration (default: Theme.animation.INSTANT)
            on_click_callback: Callback to execute after animation
        """
        press_scale = press_scale or Theme.animation.PRESS_SCALE
        duration_ms = duration_ms or Theme.animation.INSTANT

        # Set up animation
        container.animate_scale = ft.Animation(
            duration=duration_ms, curve=ft.AnimationCurve.EASE_OUT
        )

        async def animated_click(e):
            # Press down
            container.scale = press_scale
            container.update()

            await asyncio.sleep(duration_ms / 1000)

            # Release
            container.scale = 1.0
            container.update()

            # Trigger callback
            if on_click_callback:
                if asyncio.iscoroutinefunction(on_click_callback):
                    await on_click_callback(e)
                else:
                    on_click_callback(e)

        container.on_click = lambda e: asyncio.create_task(animated_click(e))
