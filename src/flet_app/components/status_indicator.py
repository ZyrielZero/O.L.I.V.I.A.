"""Status indicator with pulsing dot."""

from typing import Optional

import flet as ft

from src.flet_app.theme import Theme
from src.flet_app.utils.animations import PulseAnimator


class StatusIndicator(ft.Container):
    """Status indicator with animated dot."""

    def __init__(self):
        super().__init__()
        self.status = "initializing"
        self.dot = None
        self.dot_glow = None
        self.label = None
        self.metrics_label = None
        self._pulse_anim: Optional[PulseAnimator] = None
        self._build()

    def _build(self):
        # Simple dot (Discord-style, no glow ring)
        self.dot = ft.Container(
            width=8,
            height=8,
            border_radius=4,
            bgcolor=self._get_color(),
            scale=1.0,
            opacity=1.0,
        )
        # Set animation properties after initialization
        self.dot.animate = ft.Animation(
            duration=Theme.animation.FAST, curve=ft.AnimationCurve.EASE_OUT
        )
        self.dot.animate_scale = ft.Animation(
            duration=Theme.animation.PULSE_CYCLE // 2, curve=ft.AnimationCurve.EASE_IN_OUT
        )
        self.dot.animate_opacity = ft.Animation(
            duration=Theme.animation.PULSE_CYCLE // 2, curve=ft.AnimationCurve.EASE_IN_OUT
        )

        # Keep dot_glow reference pointing to dot for animation compatibility
        self.dot_glow = self.dot

        self.label = ft.Text(
            self._get_text(),
            size=Theme.typography.SIZE_SMALL,
            weight=ft.FontWeight.BOLD,
            color=Theme.colors.TEXT_SECONDARY,
        )
        self.label.animate_opacity = ft.Animation(
            duration=Theme.animation.FAST, curve=ft.AnimationCurve.EASE_OUT
        )

        # Metrics label (shown when ready with system info)
        self.metrics_label = ft.Text(
            "",
            size=Theme.typography.SIZE_CAPTION,
            color=Theme.colors.TEXT_MUTED,
            visible=False,
        )
        self.metrics_label.animate_opacity = ft.Animation(
            duration=Theme.animation.FAST, curve=ft.AnimationCurve.EASE_OUT
        )

        self.content = ft.Row(
            [self.dot, self.label, self.metrics_label],
            spacing=Theme.spacing.SM,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # Container styling
        self.bgcolor = Theme.colors.BG_SURFACE_1
        self.border_radius = Theme.radius.FULL
        self.border = ft.border.all(1, Theme.colors.BORDER_SUBTLE)
        self.padding = ft.padding.symmetric(horizontal=Theme.spacing.MD, vertical=Theme.spacing.SM)

        # Animate container properties
        self.animate = ft.Animation(
            duration=Theme.animation.NORMAL, curve=ft.AnimationCurve.EASE_OUT
        )

        # Start initial animation
        self._start_status_animation()

    def set_status(self, status: str, metrics: Optional[str] = None):
        """Update status with smooth animation.

        Args:
            status: New status string
            metrics: Optional metrics string (e.g., "7.8 GB VRAM")
        """
        self.status = status

        # Stop any existing pulse animation
        if self._pulse_anim:
            self._pulse_anim.stop()
            self._pulse_anim = None

        # Update colors
        new_color = self._get_color()
        if self.dot:
            self.dot.bgcolor = new_color
        if self.label:
            self.label.value = self._get_text()

        # Update metrics label
        if self.metrics_label:
            if status == "ready" and metrics:
                self.metrics_label.value = f"| {metrics}"
                self.metrics_label.visible = True
            else:
                self.metrics_label.visible = False

        # Start status-specific animations
        self._start_status_animation()

        self.update()

    def _start_status_animation(self):
        """Start animation appropriate for current status."""
        # Define pulse parameters per status
        # Active states get pulsing, static states don't
        pulse_configs = {
            "initializing": {"scale_range": (0.8, 1.0), "opacity_range": (0.5, 1.0), "cycle": 1500},
            "processing": {"scale_range": (0.85, 1.1), "opacity_range": (0.6, 1.0), "cycle": 1000},
            "recording": {"scale_range": (0.9, 1.15), "opacity_range": (0.7, 1.0), "cycle": 800},
            "transcribing": {
                "scale_range": (0.85, 1.05),
                "opacity_range": (0.6, 1.0),
                "cycle": 1200,
            },
            "speaking": {"scale_range": (0.9, 1.1), "opacity_range": (0.7, 1.0), "cycle": 600},
        }

        config = pulse_configs.get(self.status)

        if config and self.dot:
            # Update animation durations for this status
            self.dot.animate_scale = ft.Animation(
                duration=config["cycle"] // 2, curve=ft.AnimationCurve.EASE_IN_OUT
            )
            self.dot.animate_opacity = ft.Animation(
                duration=config["cycle"] // 2, curve=ft.AnimationCurve.EASE_IN_OUT
            )

            self._pulse_anim = PulseAnimator(
                control=self.dot,
                cycle_duration_ms=config["cycle"],
                scale_range=config["scale_range"],
                opacity_range=config["opacity_range"],
            )
            self._pulse_anim.start()
        elif self.status in ("ready", "error"):
            # Static states - no pulsing, reset to full visibility
            if self.dot:
                self.dot.scale = 1.0
                self.dot.opacity = 1.0
                try:
                    self.dot.update()
                except Exception:
                    pass

    def _get_color(self) -> str:
        """Get color for current status."""
        status_colors = {
            "initializing": Theme.colors.TEXT_MUTED,
            "ready": Theme.colors.STATUS_SUCCESS,
            "recording": Theme.colors.STATUS_ERROR,
            "transcribing": Theme.colors.STATUS_INFO,
            "processing": Theme.colors.STATUS_WARNING,
            "speaking": Theme.colors.STATUS_PURPLE,
            "error": Theme.colors.STATUS_ERROR,
        }
        return status_colors.get(self.status, Theme.colors.TEXT_MUTED)

    def _get_text(self) -> str:
        """Get text for current status."""
        status_texts = {
            "initializing": "Initializing...",
            "ready": "Ready",
            "recording": "Recording",
            "transcribing": "Transcribing",
            "processing": "Processing",
            "speaking": "Speaking",
            "error": "Error",
        }
        return status_texts.get(self.status, "Unknown")
