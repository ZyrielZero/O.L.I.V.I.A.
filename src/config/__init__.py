"""Configuration module.

Handles character configuration, settings, and theming.
"""

from .config_loader import Config, get_config, reload_config

# Theme exports
try:
    from .theme import (
        ModernDarkTheme,
        StatusStyles,
        Theme,
        apply_theme,
        get_status_style,
    )

    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False
    ModernDarkTheme = None
    Theme = None
    StatusStyles = None
    apply_theme = None
    get_status_style = None

__all__ = [
    # Config
    "Config",
    "get_config",
    "reload_config",
    # Theme
    "ModernDarkTheme",
    "Theme",
    "StatusStyles",
    "apply_theme",
    "get_status_style",
    "THEME_AVAILABLE",
]
