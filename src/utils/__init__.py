"""Utilities - logging, tray, timing."""

from .logger import ConversationLogger, DebugLevel, get_logger, log_timing, timed

try:
    from .system_tray import PYSTRAY_AVAILABLE, OliviaTray, SystemTrayIcon

    TRAY_AVAILABLE = PYSTRAY_AVAILABLE
except ImportError:
    TRAY_AVAILABLE = False
    SystemTrayIcon = OliviaTray = None

__all__ = [
    "get_logger",
    "DebugLevel",
    "ConversationLogger",
    "timed",
    "log_timing",
    "SystemTrayIcon",
    "OliviaTray",
    "TRAY_AVAILABLE",
]
