"""Logging system - debug levels, colored output, timing decorators."""

import atexit
import logging
import sys
import time
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


class DebugLevel(Enum):
    """Debug verbosity."""

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


class Colors:
    """ANSI color codes for console output."""

    RESET = "\033[0m"
    DIM = "\033[2m"
    DEBUG = "\033[36m"
    INFO = "\033[32m"
    WARNING = "\033[33m"
    ERROR = "\033[31m"
    LLM = "\033[94m"
    STT = "\033[95m"
    TTS = "\033[96m"
    MEMORY = "\033[93m"
    GUI = "\033[92m"
    PERF = "\033[91m"
    WEB = "\033[94m"
    WAKE = "\033[95m"
    DREAM = "\033[96m"
    FACTS = "\033[93m"
    EXT = "\033[92m"
    TRAY = "\033[91m"


class DebugConfig:
    """Runtime-configurable debug settings with GUI integration."""

    _level: DebugLevel = DebugLevel.NORMAL
    _enabled_components: Set[str] = set()
    _disabled_components: Set[str] = set()
    _callbacks: List[Callable[[DebugLevel], None]] = []
    _initialized: bool = False

    @classmethod
    def initialize(cls):
        """Initialize with default settings."""
        if cls._initialized:
            return
        cls._level = DebugLevel.NORMAL
        cls._enabled_components = set()
        cls._disabled_components = set()
        cls._callbacks = []
        cls._initialized = True

    @classmethod
    def set_level(cls, level: DebugLevel):
        """Set the global debug level and notify callbacks."""
        cls._level = level
        cls._update_logger_levels()
        for callback in cls._callbacks:
            try:
                callback(level)
            except Exception:
                pass

    @classmethod
    def get_level(cls) -> DebugLevel:
        """Get the current debug level."""
        return cls._level

    @classmethod
    def set_level_by_name(cls, name: str):
        """Set level by string name (for GUI integration)."""
        level_map = {
            "quiet": DebugLevel.QUIET,
            "normal": DebugLevel.NORMAL,
            "verbose": DebugLevel.VERBOSE,
            "debug": DebugLevel.DEBUG,
        }
        if name.lower() in level_map:
            cls.set_level(level_map[name.lower()])

    @classmethod
    def get_level_name(cls) -> str:
        """Get current level as string."""
        return cls._level.name.capitalize()

    @classmethod
    def enable_component(cls, name: str):
        """Enable logging for a specific component."""
        cls._enabled_components.add(name.lower())
        cls._disabled_components.discard(name.lower())

    @classmethod
    def disable_component(cls, name: str):
        """Disable logging for a specific component."""
        cls._disabled_components.add(name.lower())
        cls._enabled_components.discard(name.lower())

    @classmethod
    def is_component_enabled(cls, name: str) -> bool:
        """Check if a component should log."""
        name_lower = name.lower()
        if name_lower in cls._disabled_components:
            return False
        if cls._enabled_components and name_lower not in cls._enabled_components:
            return False
        return True

    @classmethod
    def add_change_callback(cls, callback: Callable[[DebugLevel], None]):
        """Register a callback for level changes (GUI updates)."""
        if callback not in cls._callbacks:
            cls._callbacks.append(callback)

    @classmethod
    def remove_change_callback(cls, callback: Callable[[DebugLevel], None]):
        """Remove a registered callback."""
        if callback in cls._callbacks:
            cls._callbacks.remove(callback)

    @classmethod
    def _update_logger_levels(cls):
        """Update all logger levels based on current debug level."""
        level_map = {
            DebugLevel.QUIET: logging.ERROR,
            DebugLevel.NORMAL: logging.INFO,
            DebugLevel.VERBOSE: logging.DEBUG,
            DebugLevel.DEBUG: logging.DEBUG,
        }
        root = logging.getLogger("olivia")
        root.setLevel(level_map.get(cls._level, logging.INFO))

        for handler in root.handlers:
            handler.setLevel(level_map.get(cls._level, logging.INFO))

    @classmethod
    def should_log_performance(cls) -> bool:
        """Check if performance logging is enabled."""
        return cls._level == DebugLevel.DEBUG


class ColoredFormatter(logging.Formatter):
    """Colored console formatter with component awareness."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
    }

    COMPONENT_COLORS = {
        "llm": Colors.LLM,
        "stt": Colors.STT,
        "tts": Colors.TTS,
        "memory": Colors.MEMORY,
        "gui": Colors.GUI,
        "perf": Colors.PERF,
        "web": Colors.WEB,
        "wake": Colors.WAKE,
        "dream": Colors.DREAM,
        "facts": Colors.FACTS,
        "ext": Colors.EXT,
        "tray": Colors.TRAY,
    }

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Optimization: Cache component names extracted from logger names.
        # Avoids repeated string.split(".") on every log message.
        # Complexity: O(1) lookup vs O(n) split per message.
        self._component_cache: Dict[str, str] = {}

    def _get_component(self, name: str) -> str:
        """Get component name with caching."""
        if name not in self._component_cache:
            parts = name.split(".")
            self._component_cache[name] = parts[1] if len(parts) > 1 else "main"
        return self._component_cache[name]

    def format(self, record: logging.LogRecord) -> str:
        """Format a record with level and component colors, or empty string if the component is disabled."""
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        component = self._get_component(record.name)

        if not DebugConfig.is_component_enabled(component):
            return ""

        component_color = self.COMPONENT_COLORS.get(component, Colors.RESET)
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        return f"{Colors.DIM}{timestamp}{Colors.RESET} {level_color}{record.levelname:<8}{Colors.RESET} {component_color}[{component.upper():<6}]{Colors.RESET} {record.getMessage()}"


class FileFormatter(logging.Formatter):
    """File formatter with structured output."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Optimization: Cache component names to avoid repeated string.split().
        self._component_cache: Dict[str, str] = {}

    def _get_component(self, name: str) -> str:
        """Get component name with caching."""
        if name not in self._component_cache:
            parts = name.split(".")
            self._component_cache[name] = parts[1] if len(parts) > 1 else "main"
        return self._component_cache[name]

    def format(self, record: logging.LogRecord) -> str:
        """Format a record as a pipe-delimited line for file output."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        component = self._get_component(record.name)
        return (
            f"{timestamp} | {record.levelname:<8} | {component.upper():<6} | {record.getMessage()}"
        )


class ComponentFilter(logging.Filter):
    """Filter that checks DebugConfig for component enablement."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Optimization: Cache component names to avoid repeated string.split().
        self._component_cache: Dict[str, str] = {}

    def _get_component(self, name: str) -> str:
        """Get component name with caching."""
        if name not in self._component_cache:
            parts = name.split(".")
            self._component_cache[name] = parts[1] if len(parts) > 1 else "main"
        return self._component_cache[name]

    def filter(self, record: logging.LogRecord) -> bool:
        """Pass the record only if its component is enabled."""
        component = self._get_component(record.name)
        return DebugConfig.is_component_enabled(component)


class OliviaLogger:
    """Singleton logger manager for O.L.I.V.I.A."""

    _instance: Optional["OliviaLogger"] = None
    _initialized: bool = False
    MAX_LOG_FILES = 10

    def __new__(cls):
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if OliviaLogger._initialized:
            return

        DebugConfig.initialize()

        self.log_dir = Path("data/logs/system")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._cleanup_old_logs()

        self.root_logger = logging.getLogger("olivia")
        self.root_logger.setLevel(logging.DEBUG)
        self.root_logger.handlers.clear()

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG)
        console.setFormatter(ColoredFormatter())
        console.addFilter(ComponentFilter())
        self.root_logger.addHandler(console)

        log_filename = datetime.now().strftime("olivia_%Y-%m-%d_%H%M%S.log")
        file_handler = logging.FileHandler(self.log_dir / log_filename, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FileFormatter())
        self.root_logger.addHandler(file_handler)

        self.root_logger.propagate = False
        OliviaLogger._initialized = True

        self.root_logger.info(f"Logging initialized: {self.log_dir / log_filename}")

    def _cleanup_old_logs(self) -> None:
        """Keep only the latest MAX_LOG_FILES log files."""
        log_files = sorted(
            self.log_dir.glob("olivia_*.log"), key=lambda f: f.stat().st_mtime, reverse=True
        )
        for old_file in log_files[self.MAX_LOG_FILES :]:
            try:
                old_file.unlink()
            except Exception:
                pass

    def get_logger(self, component: str) -> logging.Logger:
        """Get a logger for the specified component."""
        return logging.getLogger(f"olivia.{component}")


class ConversationLogger:
    """Logs conversation transcripts to separate files with buffered writes."""

    MAX_CONVERSATION_FILES = 50  # Keep only the latest 50 conversation logs
    # Optimization: Buffer writes to reduce I/O overhead.
    # Flush every BUFFER_FLUSH_COUNT entries instead of per-write.
    BUFFER_FLUSH_COUNT = 10

    def __init__(self):
        self.log_dir = Path("data/logs/conversations")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Cleanup old conversation logs
        self._cleanup_old_conversations()

        filename = datetime.now().strftime("chat_%Y-%m-%d_%H%M%S.txt")
        self.file_path = self.log_dir / filename

        # Optimization: Buffered writes to reduce disk I/O.
        # Complexity: Amortized O(1) per write, O(n) flush every BUFFER_FLUSH_COUNT writes.
        self._buffer: List[str] = []
        self._write_count = 0

        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("O.L.I.V.I.A. Conversation Transcript\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

        # Register cleanup on exit to flush remaining buffer
        atexit.register(self._flush_buffer)

    def _cleanup_old_conversations(self) -> None:
        """Keep only the latest MAX_CONVERSATION_FILES conversation logs."""
        try:
            log_files = sorted(
                self.log_dir.glob("chat_*.txt"), key=lambda f: f.stat().st_mtime, reverse=True
            )
            for old_file in log_files[self.MAX_CONVERSATION_FILES :]:
                try:
                    old_file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete old conversation log {old_file}: {e}")
        except Exception as e:
            logging.warning(f"Failed to cleanup conversation logs: {e}")

    def _flush_buffer(self) -> None:
        """Flush buffered entries to disk."""
        if not self._buffer:
            return
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write("".join(self._buffer))
            self._buffer.clear()
        except Exception:
            pass

    def log_interaction(self, sender: str, message: str) -> None:
        """Append a message to the transcript with buffered writes.

        Optimization: Buffers writes and flushes every BUFFER_FLUSH_COUNT entries.
        Reduces disk I/O from O(n) file opens to O(n/BUFFER_FLUSH_COUNT).
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._buffer.append(f"[{timestamp}] {sender}: {message}\n\n")
        self._write_count += 1

        if self._write_count >= self.BUFFER_FLUSH_COUNT:
            self._flush_buffer()
            self._write_count = 0

    def flush(self) -> None:
        """Manually flush the buffer (call when conversation ends)."""
        self._flush_buffer()
        self._write_count = 0


_manager: Optional[OliviaLogger] = None


def get_logger(component: str = "main") -> logging.Logger:
    """Get logger for component: 'llm', 'stt', 'tts', 'memory', 'gui', 'perf', 'web', 'wake', etc."""
    global _manager
    if _manager is None:
        _manager = OliviaLogger()
    return _manager.get_logger(component)


def set_debug_level(level: DebugLevel):
    """Set the global debug level."""
    DebugConfig.set_level(level)


def get_debug_level() -> DebugLevel:
    """Get the current debug level."""
    return DebugConfig.get_level()


def timed(name: Optional[str] = None):
    """Decorator to time function execution (respects debug level)."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not DebugConfig.should_log_performance():
                return func(*args, **kwargs)

            log = get_logger("perf")
            func_name = name or func.__name__
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                log.info(f"{func_name}: {time.perf_counter() - start:.3f}s")
                return result
            except Exception as e:
                log.error(f"{func_name} failed: {e}")
                raise

        return wrapper

    return decorator


class TimingContext:
    """Context manager for timing code blocks (respects debug level)."""

    def __init__(self, name: str):
        self.name = name
        self.log = get_logger("perf")
        self.start: float = 0
        self._should_log = DebugConfig.should_log_performance()

    def __enter__(self) -> "TimingContext":
        if self._should_log:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if not self._should_log:
            return False
        elapsed = time.perf_counter() - self.start
        if exc_type:
            self.log.error(f"{self.name} failed after {elapsed:.3f}s")
        else:
            self.log.info(f"{self.name}: {elapsed:.3f}s")
        return False


def log_timing(name: str) -> TimingContext:
    """Create a timing context manager."""
    return TimingContext(name)
