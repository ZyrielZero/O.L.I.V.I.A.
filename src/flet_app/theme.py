"""Flet theme - dark mode with purple accents."""

from dataclasses import dataclass

import flet as ft


@dataclass(frozen=True)
class ColorPalette:
    """Dark theme colors (Discord-inspired)."""

    # Background Layers (Discord-style with better contrast)
    BG_BASE: str = "#1E1F22"  # Darkest - Discord dark mode base
    BG_SURFACE_1: str = "#2B2D31"  # Secondary panels
    BG_SURFACE_2: str = "#313338"  # Main content area
    BG_SURFACE_3: str = "#383A40"  # Elevated elements
    BG_SURFACE_4: str = "#404249"  # Hover states, active items

    # Purple Accent Palette (keeping brand identity)
    ACCENT_PRIMARY: str = "#8B5CF6"
    ACCENT_LIGHT: str = "#A78BFA"
    ACCENT_DARK: str = "#7C3AED"
    ACCENT_GLOW: str = "#C4B5FD"
    ACCENT_MUTED: str = "#6D28D9"

    # Text Hierarchy (Discord-style)
    TEXT_PRIMARY: str = "#F2F3F5"  # Primary text
    TEXT_SECONDARY: str = "#B5BAC1"  # Secondary text
    TEXT_TERTIARY: str = "#949BA4"  # Tertiary/placeholder
    TEXT_MUTED: str = "#6D6F78"  # Muted text
    TEXT_ACCENT: str = "#C4B5FD"  # Accent text

    # Semantic Status Colors (Muted for modern look)
    STATUS_SUCCESS: str = "#23A55A"  # Muted green
    STATUS_WARNING: str = "#3B82F6"  # Soft blue (replacing yellow)
    STATUS_ERROR: str = "#DA373C"  # Muted red
    STATUS_INFO: str = "#3B82F6"  # Soft blue
    STATUS_PURPLE: str = "#8B5CF6"  # Purple for speaking

    # Border Colors (More visible for depth)
    BORDER_DEFAULT: str = "#3F4147"  # Default border
    BORDER_SUBTLE: str = "#2E3035"  # Subtle separator
    BORDER_MEDIUM: str = "#4E5058"  # Medium emphasis
    BORDER_FOCUS: str = "#8B5CF6"  # Focus ring

    # Special States
    MIC_INACTIVE: str = "#6D6F78"  # Muted gray
    MIC_ACTIVE: str = "#DA373C"  # Muted red
    MIC_LISTENING: str = "#23A55A"  # Muted green


# =============================================================================
# TYPOGRAPHY
# =============================================================================


@dataclass(frozen=True)
class Typography:
    """Typography scale for Flet application."""

    # Font Family
    FONT_PRIMARY: str = "Segoe UI Variable"

    # Font Sizes (px)
    SIZE_DISPLAY: int = 48
    SIZE_H1: int = 32
    SIZE_H2: int = 24
    SIZE_H3: int = 20
    SIZE_BODY: int = 16
    SIZE_BODY_SM: int = 14
    SIZE_SMALL: int = 13
    SIZE_CAPTION: int = 12
    SIZE_MICRO: int = 10


# =============================================================================
# SPACING SYSTEM (8px Grid)
# =============================================================================


@dataclass(frozen=True)
class Spacing:
    """8px grid-based spacing system."""

    NONE: int = 0
    XS: int = 4
    SM: int = 8
    MD: int = 16
    LG: int = 24
    XL: int = 32
    XXL: int = 48


# =============================================================================
# BORDER RADIUS
# =============================================================================


@dataclass(frozen=True)
class Radius:
    """Border radius values."""

    NONE: int = 0
    SM: int = 8
    MD: int = 12
    LG: int = 16
    XL: int = 24
    FULL: int = 999


# =============================================================================
# ANIMATION TIMING
# =============================================================================


@dataclass(frozen=True)
class Animation:
    """Animation timing and curve constants for fluid UI."""

    # Timing (ms) - micro-interaction best practices
    INSTANT: int = 100  # Immediate feedback
    FAST: int = 150  # Button press, hover
    NORMAL: int = 250  # Standard transitions
    MEDIUM: int = 350  # Message appearance
    SLOW: int = 500  # Complex transitions
    PULSE_CYCLE: int = 2000  # Breathing animation full cycle

    # Scale factors for subtle professional animations
    HOVER_SCALE: float = 1.03  # Subtle hover enlargement
    PRESS_SCALE: float = 0.97  # Press feedback
    PULSE_MIN: float = 0.85  # Breathing minimum
    PULSE_MAX: float = 1.0  # Breathing maximum

    # Opacity values
    OPACITY_FULL: float = 1.0
    OPACITY_HOVER: float = 0.9
    OPACITY_DISABLED: float = 0.5
    OPACITY_FADE_START: float = 0.0


# =============================================================================
# DIMENSIONS
# =============================================================================


@dataclass(frozen=True)
class Dimensions:
    """Component dimensions."""

    HEADER_HEIGHT: int = 80
    SIDEBAR_WIDTH: int = 280
    INPUT_BAR_HEIGHT: int = 80
    ORB_SIZE: int = 120


# =============================================================================
# GLASSMORPHISM STYLING
# =============================================================================


@dataclass(frozen=True)
class GlassmorphismStyle:
    """Discord-style solid backgrounds (replacing glassmorphism)."""

    # Solid backgrounds for modern Discord-style look
    GLASS_BG_USER: str = "#3C3E46"  # User message background (subtle elevation)
    GLASS_BG_ASSISTANT: str = "#2B2D31"  # Assistant message (matches surface)
    GLASS_BG_SURFACE: str = "#2B2D31"  # General surface
    GLASS_BG_HEADER: str = "#2B2D31"  # Header (solid)

    # Borders (visible for depth separation)
    BORDER_GLASS: str = "#3F4147"  # Visible border
    BORDER_GLASS_ACCENT: str = "#8B5CF6"  # Accent border
    BORDER_GLASS_SUBTLE: str = "#2E3035"  # Subtle separator

    # Shadow for depth (more pronounced)
    SHADOW_COLOR: str = "#00000066"  # Black 40%
    SHADOW_BLUR: int = 24  # Larger blur for depth
    SHADOW_OFFSET: int = 8  # More offset


# =============================================================================
# THEME INSTANCE
# =============================================================================


class Theme:
    """Central theme instance for the application."""

    colors = ColorPalette()
    typography = Typography()
    spacing = Spacing()
    radius = Radius()
    animation = Animation()
    dimensions = Dimensions()
    glass = GlassmorphismStyle()

    @staticmethod
    def get_flet_theme() -> ft.Theme:
        """Generate Flet theme configuration.

        Returns:
            Flet Theme object
        """
        return ft.Theme(
            color_scheme_seed=Theme.colors.ACCENT_PRIMARY,
            use_material3=True,
            font_family=Theme.typography.FONT_PRIMARY,
        )

    @staticmethod
    def apply_to_page(page: ft.Page):
        """Apply theme to Flet page.

        Args:
            page: Flet Page object
        """
        page.theme_mode = ft.ThemeMode.DARK
        page.theme = Theme.get_flet_theme()
        page.bgcolor = Theme.colors.BG_BASE
        page.padding = 0
