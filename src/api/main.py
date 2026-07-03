"""O.L.I.V.I.A. FastAPI Backend - main entry point."""

import asyncio
import concurrent.futures
import logging
import logging.handlers
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import APIConfig
from src.api.container import get_container
from src.api.middleware import ErrorHandlingMiddleware, LoggingMiddleware
from src.api.routes import chat, health, voice

_LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _setup_logging():
    """Configure logging with file rotation + console output."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(_LOG_FMT)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file handler (10MB, 3 backups)
    try:
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "olivia.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except Exception:
        logging.warning("File logging unavailable, using console only")


_setup_logging()
log = logging.getLogger("api")

cfg = APIConfig()
startup_time: Optional[datetime] = None
_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

# OPT: Background tasks for lazy-loading heavy models (STT/TTS)
# Reduces startup time from ~30s to ~5s by deferring model loading
_lazy_load_tasks: list[asyncio.Task] = []

# Daily TTL maintenance task (referenced so it can't be GC'd, cancelled on shutdown)
_memory_maintenance_task: Optional[asyncio.Task] = None

_MAINTENANCE_INTERVAL_S = 24 * 3600


async def _memory_maintenance_loop(mem_svc) -> None:
    """Run TTL pruning immediately on startup, then once a day.

    Without this loop, prune_expired() is never called and conversations/
    summaries accumulate forever.
    """
    while True:
        try:
            pruned = await mem_svc.prune_expired()
            if pruned.get("conversations") or pruned.get("summaries"):
                log.info(f"Memory TTL pruning removed: {pruned}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"Memory TTL pruning failed: {e}")
        await asyncio.sleep(_MAINTENANCE_INTERVAL_S)


async def _lazy_load_stt() -> None:
    """Background task to load STT model.

    OPT: Deferred loading - STT is not needed until user speaks.
    Reduces startup blocking time by ~8-10 seconds.
    """
    try:
        from src.api.services.stt_service import STTService

        stt = STTService(
            model_size=cfg.STT_MODEL_SIZE,
            device=cfg.STT_DEVICE,
            compute_type=cfg.STT_COMPUTE_TYPE,
        )
        await stt.initialize()
        get_container().stt = stt
        log.info("STT ready (lazy-loaded)")
    except Exception as e:
        log.error(f"STT lazy-load failed: {e}")


async def _lazy_load_tts() -> None:
    """Background task to load TTS model.

    OPT: Deferred loading - TTS is not needed until first response.
    Reduces startup blocking time by ~12-15 seconds.
    """
    try:
        from src.api.services.tts_service import TTSService

        tts = TTSService(
            voice_reference=cfg.TTS_VOICE_REFERENCE,
            device=cfg.TTS_DEVICE,
            cfg_weight=cfg.TTS_CFG_WEIGHT,
            exaggeration=cfg.TTS_EXAGGERATION,
        )
        await tts.initialize()
        get_container().tts = tts
        log.info("TTS ready (lazy-loaded)")
    except Exception as e:
        log.error(f"TTS lazy-load failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle.

    OPT: Two-phase startup for faster availability:
    1. Critical services (LLM, Memory) - blocking, required for /health
    2. Heavy models (STT, TTS) - lazy-loaded in background

    This reduces startup time from ~30s to ~5s while maintaining full functionality.
    """
    global startup_time, _executor, _lazy_load_tasks, _memory_maintenance_task

    log.info("=" * 60)
    log.info("  O.L.I.V.I.A. API Starting")
    log.info("=" * 60)

    try:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        asyncio.get_event_loop().set_default_executor(_executor)
        log.info("Thread pool ready (10 workers)")

        startup_time = datetime.now()

        # ===== PHASE 1: Critical services (blocking) =====
        # These must be ready before /health returns 200

        # load character config
        from src.config.config_loader import get_config

        char_cfg = get_config()
        log.info(f"Config loaded: {char_cfg.get('name')}")

        # Get typed container
        container = get_container()

        # LLM + Memory - parallel initialization for faster startup
        from src.api.services.llm_service import LLMService
        from src.api.services.memory_service import MemoryService

        llm_svc = LLMService(
            model=cfg.OLLAMA_MODEL,
            system_prompt=char_cfg.get_system_prompt(),
            host=cfg.OLLAMA_HOST,
        )
        mem_svc = MemoryService(persist_directory=cfg.MEMORY_PERSIST_DIR)

        # OPT: Parallel init - saves ~1-2s vs sequential
        await asyncio.gather(
            llm_svc.initialize(),
            mem_svc.initialize(),
        )
        container.llm = llm_svc
        container.memory = mem_svc
        log.info("LLM + Memory ready")

        # TTL maintenance: prune expired conversations/summaries now and daily
        _memory_maintenance_task = asyncio.create_task(_memory_maintenance_loop(mem_svc))
        log.info("Memory TTL maintenance scheduled (startup + daily)")

        # State manager - lightweight, no model loading
        from src.api.services.state_manager import StateManager

        container.state = StateManager()
        log.info("State manager ready")

        # Web search - lightweight, no model loading
        from src.core.tools.web_search import get_web_tools

        get_web_tools()
        log.info("Web search ready")

        # ===== Experimental memory integration =====
        try:
            from src.experimental.memory.dreaming import DreamConfig, create_dreaming_engine
            from src.experimental.memory.fact_extractor import create_fact_extractor

            dream_cfg = DreamConfig(max_conversations_per_dream=10)
            dreaming = create_dreaming_engine(mem_svc._db, dream_cfg)
            dreaming.start_idle_monitoring()
            container.dreaming = dreaming
            log.info("DreamingEngine ready (idle monitoring started)")

            extractor = create_fact_extractor(mem_svc._db)
            extractor.start()
            container.fact_extractor = extractor
            log.info("HybridFactExtractor ready (background worker started)")
        except Exception as e:
            log.warning(f"Experimental memory integration failed: {e}")

        log.info("=" * 60)
        log.info(f"  O.L.I.V.I.A. Ready @ {cfg.HOST}:{cfg.PORT}")
        log.info("  (STT/TTS loading in background...)")
        log.info("=" * 60)

        # ===== PHASE 2: Heavy models (lazy-loaded in background) =====
        # OPT: Fire-and-forget tasks - don't block startup
        _lazy_load_tasks = [
            asyncio.create_task(_lazy_load_stt()),
            asyncio.create_task(_lazy_load_tts()),
        ]

    except Exception as e:
        log.error(f"Startup failed: {e}", exc_info=True)
        raise

    yield

    # shutdown
    log.info("Shutting down...")

    # OPT: Cancel any pending lazy-load tasks first
    for task in _lazy_load_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    if _memory_maintenance_task and not _memory_maintenance_task.done():
        _memory_maintenance_task.cancel()
        try:
            await _memory_maintenance_task
        except asyncio.CancelledError:
            pass

    container = get_container()

    # Stop experimental systems first (they use Ollama/memory)
    if container.fact_extractor:
        try:
            container.fact_extractor.stop()
            log.info("Fact extractor stopped")
        except Exception as e:
            log.warning(f"Fact extractor cleanup failed: {e}")

    if container.dreaming:
        try:
            container.dreaming.stop_idle_monitoring()
            container.dreaming.dream_on_shutdown()
            log.info("Shutdown dreaming complete")
        except Exception as e:
            log.warning(f"Shutdown dreaming failed: {e}")

    # Stop the session TTS queue and release the audio device
    try:
        from src.api.services.audio_output import close_audio_output
        from src.api.services.tts_service import get_active_tts_queue

        q = get_active_tts_queue()
        if q:
            await q.stop()
        close_audio_output()
    except Exception as e:
        log.warning(f"Audio pipeline cleanup failed: {e}")

    # OPT: Parallel cleanup for independent services
    async def _cleanup_tts():
        if container.tts and container.tts.is_initialized():
            await container.tts.stop()
            await container.tts.cleanup()

    async def _cleanup_llm():
        if container.llm:
            await container.llm.cleanup()

    cleanup_tasks = []
    if container.tts and container.tts.is_initialized():
        cleanup_tasks.append(_cleanup_tts())
    if container.llm:
        cleanup_tasks.append(_cleanup_llm())

    if cleanup_tasks:
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.warning(f"Cleanup task {i} failed: {result}")

    # State cleanup last (may reference other services)
    if container.state:
        for sid in container.state.get_all_session_ids():
            await container.state.cleanup_session(sid)

    if _executor:
        _executor.shutdown(wait=True, cancel_futures=True)

    log.info("Shutdown complete")


app = FastAPI(
    title="O.L.I.V.I.A. API",
    version="1.0.0",
    description="Offline Local Intelligent Voice Interactive Assistant",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_credentials=cfg.CORS_ALLOW_CREDENTIALS,
    allow_methods=cfg.CORS_ALLOW_METHODS,
    allow_headers=cfg.CORS_ALLOW_HEADERS,
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(voice.router)


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "O.L.I.V.I.A. API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


def get_startup_time():
    """Return the timestamp recorded when the API process started."""
    return startup_time
