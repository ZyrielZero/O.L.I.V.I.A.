"""Chat display with Discord-style messages."""

import asyncio
from datetime import datetime
from typing import Optional

import flet as ft

from src.flet_app.theme import Theme

AVATAR_PATH = "olivia_avatar.png"

# Optimization: Bounded message history to prevent unbounded memory growth.
# O(1) trim operation when appending new messages.
MAX_MESSAGES = 100


class ChatBubble(ft.Row):
    """Message bubble with fade-in animation."""

    def __init__(
        self,
        sender: str,
        message: str,
        is_user: bool = False,
        animate_entrance: bool = True,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__()
        self.sender = sender
        self.message = message
        self.is_user = is_user
        self.animate_entrance = animate_entrance
        self.timestamp = timestamp or datetime.now()
        # Optimization: Cache formatted timestamp string once at construction.
        # Avoids repeated strftime() calls if timestamp is accessed multiple times.
        self._formatted_time = self.timestamp.strftime("%H:%M")
        self.message_text = None
        self.bubble_container = None
        self.avatar = None
        self.expand = True
        self._build()

    def _build(self):
        # All messages left-aligned (Discord-style)
        self.alignment = ft.MainAxisAlignment.START

        # Styling based on sender
        if self.is_user:
            sender_color = Theme.colors.ACCENT_GLOW
            text_color = Theme.colors.TEXT_PRIMARY
            bg_color = Theme.glass.GLASS_BG_USER
        else:
            sender_color = Theme.colors.ACCENT_PRIMARY
            text_color = Theme.colors.TEXT_SECONDARY
            bg_color = Theme.colors.BG_SURFACE_3  # Subtle gray background for contrast

        self.message_text = ft.Text(
            self.message,
            size=Theme.typography.SIZE_BODY_SM,
            color=text_color,
            selectable=True,
        )

        # Sender name with timestamp inline (Discord-style)
        header_row = ft.Row(
            [
                ft.Text(
                    self.sender,
                    size=Theme.typography.SIZE_BODY_SM,
                    weight=ft.FontWeight.BOLD,
                    color=sender_color,
                ),
                ft.Text(
                    self._formatted_time,  # Use cached timestamp
                    size=Theme.typography.SIZE_CAPTION,
                    color=Theme.colors.TEXT_MUTED,
                ),
            ],
            spacing=Theme.spacing.SM,
        )

        # Main message container (flattened, Discord-style)
        self.bubble_container = ft.Container(
            content=ft.Column(
                [
                    header_row,
                    self.message_text,
                ],
                spacing=2,
                tight=True,
            ),
            expand=True,
            bgcolor=bg_color,
            border_radius=Theme.radius.SM,  # Smaller radius for flatter look
            padding=ft.padding.symmetric(horizontal=Theme.spacing.MD, vertical=Theme.spacing.SM),
            # No shadow for flat Discord-style
            # Animation properties for entrance
            opacity=Theme.animation.OPACITY_FADE_START
            if self.animate_entrance
            else Theme.animation.OPACITY_FULL,
            offset=ft.Offset(-0.02, 0) if self.animate_entrance else ft.Offset(0, 0),
        )
        # Set animation properties after initialization
        self.bubble_container.animate_opacity = ft.Animation(
            duration=Theme.animation.MEDIUM, curve=ft.AnimationCurve.EASE_OUT
        )
        self.bubble_container.animate_offset = ft.Animation(
            duration=Theme.animation.MEDIUM, curve=ft.AnimationCurve.EASE_OUT_CUBIC
        )

        # Avatar for both user and assistant (Discord-style)
        avatar_src = AVATAR_PATH if not self.is_user else None
        avatar_bg = Theme.colors.ACCENT_MUTED if self.is_user else Theme.colors.BG_SURFACE_3

        self.avatar = ft.Container(
            content=ft.Image(
                src=avatar_src,
                width=40,
                height=40,
                fit=ft.BoxFit.COVER,
                border_radius=ft.border_radius.all(20),
            )
            if avatar_src
            else ft.Text(
                self.sender[0].upper(),
                size=Theme.typography.SIZE_BODY,
                weight=ft.FontWeight.BOLD,
                color=Theme.colors.TEXT_PRIMARY,
            ),
            width=40,
            height=40,
            border_radius=ft.border_radius.all(20),
            bgcolor=avatar_bg,
            alignment=ft.Alignment(0, 0),
            margin=ft.margin.only(right=Theme.spacing.MD, top=2),
        )

        self.controls = [self.avatar, self.bubble_container]

    def animate_in(self):
        """Trigger entrance animation."""
        if self.bubble_container and self.animate_entrance:
            self.bubble_container.opacity = Theme.animation.OPACITY_FULL
            self.bubble_container.offset = ft.Offset(0, 0)
            self.bubble_container.update()

    def update_text(self, new_text: str):
        """Update message text (for streaming)."""
        self.message = new_text
        if self.message_text:
            self.message_text.value = new_text
            self.update()


class StreamingCursor(ft.Container):
    """Animated blinking cursor for streaming text effect."""

    def __init__(self):
        super().__init__()
        self._visible = True
        self._blink_task: Optional[asyncio.Task] = None
        self._build_ui()

    def _build_ui(self):
        """Build cursor UI."""
        self.width = 2
        self.height = 16
        self.bgcolor = Theme.colors.ACCENT_PRIMARY
        self.border_radius = 1
        self.opacity = 1.0
        self.animate_opacity = ft.Animation(
            duration=Theme.animation.FAST, curve=ft.AnimationCurve.EASE_IN_OUT
        )

    def start_blink(self):
        """Start cursor blink animation."""
        if not self._blink_task:
            self._blink_task = asyncio.create_task(self._blink_loop())

    def stop_blink(self):
        """Stop cursor blink animation."""
        if self._blink_task:
            self._blink_task.cancel()
            self._blink_task = None
        self.opacity = 0.0
        try:
            self.update()
        except Exception:
            pass

    async def _blink_loop(self):
        """Blink animation loop."""
        while True:
            try:
                self.opacity = 0.0 if self._visible else 1.0
                self._visible = not self._visible
                self.update()
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception:
                break


class ChatDisplay(ft.Container):
    """Chat display with message history, streaming support, and animations."""

    def __init__(self):
        super().__init__()
        self.message_column = None
        self.streaming_bubble = None
        self.streaming_text = ""
        self._build_ui()

    def _build_ui(self):
        """Build chat display UI."""
        self.message_column = ft.Column(
            spacing=2,  # Tighter spacing like Discord
            scroll=ft.ScrollMode.AUTO,
            auto_scroll=True,
        )

        self.content = self.message_column
        self.bgcolor = Theme.colors.BG_SURFACE_2  # Discord main content color
        self.border_radius = 0  # No border radius for content area
        self.expand = True
        self.padding = Theme.spacing.MD

    def _trim_old_messages(self) -> None:
        """Trim old messages when exceeding MAX_MESSAGES.

        Optimization: Sliding window prevents unbounded memory growth.
        Complexity: O(k) where k = number of messages to remove (typically 1).
        """
        controls = self.message_column.controls
        if len(controls) > MAX_MESSAGES:
            # Remove oldest messages to maintain bounded history
            excess = len(controls) - MAX_MESSAGES
            del controls[:excess]

    def append_message(self, sender: str, message: str) -> None:
        """Add complete message bubble with fade-in animation.

        Args:
            sender: Message sender ("You" or "O.L.I.V.I.A.")
            message: Message content
        """
        is_user = sender.lower() == "you"
        bubble = ChatBubble(sender=sender, message=message, is_user=is_user, animate_entrance=True)
        self.message_column.controls.append(bubble)

        # Optimization: Trim old messages to prevent unbounded growth
        self._trim_old_messages()

        self.update()

        # Trigger entrance animation after a brief delay for UI update
        asyncio.create_task(self._animate_bubble_in(bubble))

    async def _animate_bubble_in(self, bubble: ChatBubble):
        """Animate bubble entrance after slight delay."""
        await asyncio.sleep(0.02)  # Allow UI to update
        bubble.animate_in()

    def start_streaming(self, sender: str) -> None:
        """Initialize streaming message with animation.

        Args:
            sender: Message sender
        """
        self.streaming_text = ""

        # Create streaming bubble
        self.streaming_bubble = ChatBubble(
            sender=sender, message="", is_user=False, animate_entrance=True
        )

        self.message_column.controls.append(self.streaming_bubble)

        # Optimization: Trim old messages before adding streaming bubble
        self._trim_old_messages()

        self.update()

        # Animate in
        asyncio.create_task(self._start_streaming_animation())

    async def _start_streaming_animation(self):
        """Start streaming animations."""
        await asyncio.sleep(0.02)
        if self.streaming_bubble:
            self.streaming_bubble.animate_in()

    def append_token(self, token: str):
        """Append token to streaming message.

        Args:
            token: Token to append
        """
        if self.streaming_bubble:
            self.streaming_text += token
            self.streaming_bubble.update_text(self.streaming_text)

    def end_streaming(self):
        """Finalize streaming message."""
        self.streaming_bubble = None
        self.streaming_text = ""

    def clear_messages(self):
        """Clear all messages."""
        self.message_column.controls.clear()
        self.update()
