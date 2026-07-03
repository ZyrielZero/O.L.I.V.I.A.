# O.L.I.V.I.A. Comprehensive Technical Documentation

**O.L.I.V.I.A.** (Offline Local Intelligent Voice Interactive Assistant)
**Version:** 1.0.0
**Status:** Active Development (95% Core Complete)
**Target Hardware:** RTX 4080 SUPER 16GB VRAM

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Systems](#3-core-systems)
   - 3.1 [LLM Integration (Brain)](#31-llm-integration-brain)
   - 3.2 [Memory System](#32-memory-system)
   - 3.3 [Speech-to-Text (Ears)](#33-speech-to-text-ears)
   - 3.4 [Text-to-Speech (Mouth)](#34-text-to-speech-mouth)
4. [API Layer](#4-api-layer)
   - 4.1 [FastAPI Application](#41-fastapi-application)
   - 4.2 [Service Container](#42-service-container)
   - 4.3 [API Endpoints](#43-api-endpoints)
   - 4.4 [Configuration](#44-configuration)
   - 4.5 [Constants](#45-constants)
5. [Desktop UI (Flet)](#5-desktop-ui-flet)
   - 5.1 [Application Architecture](#51-application-architecture)
   - 5.2 [Theme System](#52-theme-system)
   - 5.3 [Components](#53-components)
6. [Background Memory Systems](#6-background-memory-systems)
   - 6.1 [Dreaming Engine](#61-dreaming-engine)
   - 6.2 [Fact Extractor](#62-fact-extractor)
7. [Personality System](#7-personality-system)
8. [Performance Optimizations](#8-performance-optimizations)
9. [VRAM Budget](#9-vram-budget)
10. [Directory Structure](#10-directory-structure)
11. [Critical Notes](#11-critical-notes)

---

## 1. Project Overview

O.L.I.V.I.A. is a fully local AI companion featuring:

- **Voice Interaction**: Push-to-talk, continuous listening with VAD, and a full-duplex `/ws/voice` WebSocket pipeline
- **Persistent Memory**: Three-tier ChromaDB storage with semantic search
- **Character-Driven Personality**: Warm but direct, concise, no corporate AI speak
- **Memory Consolidation**: LLM-powered dreaming system for fact extraction (runs in the background)

Fully offline: no web search, telemetry, or external network dependencies beyond the local Ollama server.

### The Body Metaphor

| Part | Function | Technology | Location | VRAM |
|------|----------|------------|----------|------|
| **Brain** | LLM | Ollama (olivia-finetuned 8B Q4) | `src/core/llm/` | ~6GB |
| **Ears** | STT | faster-whisper (small.en) | `src/core/speech/stt.py` | ~1GB |
| **Mouth** | TTS | ChatterBox Turbo | `src/core/speech/chatterbox_tts.py` | ~2GB |
| **Memory** | ChromaDB | 3-tier architecture | `src/core/memory/smart_memory.py` | ~0.5GB |
| **Dreaming** | Consolidation | LLM summarization | `src/core/memory/dreaming.py` | - |
| **Instincts** | Fact Extraction | Regex + LLM hybrid | `src/core/memory/fact_extractor.py` | - |

---

## 2. Architecture Overview

```
+-------------------------------------------+
|           Flet Desktop UI                 |
|  (src/flet_app/)                          |
|  - Modern Material Design                 |
|  - Chat interface with streaming          |
|  - Status indicators                      |
+--------------------+----------------------+
                     | HTTP/SSE
                     v
+--------------------+----------------------+
|         FastAPI Backend                   |
|  (src/api/)                               |
|  - REST + /ws/voice WebSocket endpoints   |
|  - Service wrappers with DI              |
|  - Two-phase startup                      |
+--------------------+----------------------+
                     |
                     v
+-------------------------------------------+
|           O.L.I.V.I.A. Core               |
|  (src/core/)                              |
|  - Ollama LLM client                      |
|  - ChromaDB memory                        |
|  - faster-whisper STT                     |
|  - ChatterBox TTS                         |
|  - Dreaming + fact extraction             |
+-------------------------------------------+
```

### Request Flow

```
User Input -> FastAPI -> Memory Prefetch -> LLM Streaming
          -> Store Conversation -> Session TTS Queue -> Audio Output
```

---

## 3. Core Systems

### 3.1 LLM Integration (Brain)

**Location:** `src/core/llm/ollama_client.py`

#### ConversationManager Class

The central class for LLM interaction using Ollama's chat API.

```python
class ConversationManager:
    def __init__(
        self,
        system_prompt: str = "You are a helpful AI assistant.",
        model: str = "olivia-finetuned",
        host: str = "http://localhost:11434",
    ):
```

**Key Features:**

1. **Async Streaming**: Native async via httpx with connection pooling
2. **ChatML Format**: Compatible with Llama 3.1 and olivia-finetuned models
3. **Context Injection**: Via system message, NOT conversation history
4. **History Management**: Auto-trim with configurable retention

**Generation Parameters:**

```python
GEN_PARAMS = {
    "temperature": 0.3,      # Low for consistency
    "top_p": 0.7,            # Nucleus sampling
    "top_k": 15,             # Limited vocabulary
    "repeat_penalty": 1.3,   # Prevent repetition
    "num_ctx": 8192,         # Context window
    "num_predict": 150,      # Token limit (250 with context)
}
```

**Stop Tokens:**

```python
STOP_TOKENS = [
    # Llama 3.1 format
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
    # ChatML format (olivia-finetuned)
    "<|im_start|>", "<|im_end|>",
    # General
    "\n\n\n",
]
```

**Connection Pooling Optimization:**

```python
self._client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=10.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
)
```

**Streaming Implementation:**

```python
async def chat_stream_async(self, user_input: str, context: Optional[str] = None, ...):
    # List append + join instead of string += (O(n) vs O(n^2))
    resp_chunks: List[str] = []

    async with self._client.stream("POST", "/api/chat", json=payload) as resp:
        async for line in resp.aiter_lines():
            data = json.loads(line)
            if "message" in data:
                tok = data["message"].get("content", "")
                if tok:
                    yield tok
                    resp_chunks.append(tok)

    # Store complete response
    self.history.append({"role": "assistant", "content": "".join(resp_chunks)})
```

---

### 3.2 Memory System

**Location:** `src/core/memory/smart_memory.py`

#### SmartMemoryDB - Three-Tier Architecture

| Tier | Collection | Purpose | Retention |
|------|-----------|---------|-----------|
| 1 | `olivia_facts` | Permanent user facts | Forever |
| 2 | `olivia_conversations` | Recent exchanges | 7 days |
| 3 | `olivia_summaries` | Session summaries | 1 year |

**Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions)
**ChromaDB Batch Limit:** ~5,400 embeddings per add()

**HNSW Index Configuration:**

```python
HNSW_METADATA = {
    "hnsw:search_ef": 100,   # Default 10 is too low for quality recall
    "hnsw:num_threads": 4,   # Parallel search threads
}
```

#### Pre-Compiled Regex Patterns

Module-level compilation for O(1) reuse:

```python
_FACT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?:my name is|i'm called|call me) ([a-zA-Z]+)", re.IGNORECASE), "name"),
    (re.compile(r"(?:i (?:really )?(?:like|love|enjoy|prefer)) (.+?)(?:\.|$|,|!)", re.IGNORECASE), "preference"),
    (re.compile(r"(?:i work (?:at|as|for)) (.+?)(?:\.|$|,|!)", re.IGNORECASE), "personal"),
    (re.compile(r"(?:i live in|i'm from) (.+?)(?:\.|$|,|!)", re.IGNORECASE), "personal"),
    # ... more patterns
]
```

#### Parallel Tier Searches

```python
def search_all(self, query: str, n_results: int = 3) -> str:
    # ThreadPoolExecutor for parallel tier queries
    # O(3 * query_time) sequential -> O(query_time) parallel
    futures = [
        self._executor.submit(search_facts),
        self._executor.submit(search_conversations),
        self._executor.submit(search_summaries),
    ]

    for future in as_completed(futures):
        tier_results = future.result(timeout=5.0)
        results_list.extend(tier_results)
```

#### Batch Duplicate Checking

```python
def batch_check_duplicates(self, facts: List[str], threshold: float = 0.3) -> List[bool]:
    # Single batch query instead of n individual queries
    # O(n) individual queries -> O(1) batch query
    results = self.facts.query(query_texts=facts, n_results=1)

    duplicates = []
    for fact_distances in results["distances"]:
        duplicates.append(fact_distances[0] < threshold)
    return duplicates
```

#### Fact Extraction from Conversations

```python
def add_conversation(self, user_msg: str, ai_msg: str, auto_extract_facts: bool = True):
    # Store conversation
    self.conversations.add(documents=[conversation], ...)

    if auto_extract_facts:
        extracted = self.extract_facts_from_conversation(user_msg, ai_msg)
        # Batch duplicate check instead of per-fact queries
        facts_only = [fact for fact, _ in extracted]
        is_duplicate = self.batch_check_duplicates(facts_only)

        for i, (fact, category) in enumerate(extracted):
            if not is_duplicate[i]:
                self.add_fact(fact, category)
```

---

### 3.3 Speech-to-Text (Ears)

**Location:** `src/core/speech/stt.py`

#### STTEngine (faster-whisper)

```python
class STTEngine:
    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        self.sample_rate = 16000
```

**Critical Configuration:**
- **beam_size=5 ALWAYS** - Must match warmup parameters for CUDA kernel consistency

**Model Warmup:**

```python
def _warmup_model(self) -> None:
    """Pre-compile CUDA kernels to eliminate first-inference latency."""
    warmup_audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
    # beam_size=5 matches transcribe_audio() for proper kernel warmup
    _ = list(self.model.transcribe(warmup_audio, beam_size=5)[0])
```

#### ContinuousSTT (VAD)

Voice Activity Detection with barge-in support:

```python
class ContinuousSTT:
    BASE_THRESHOLD = 0.005       # Normal listening
    TTS_ACTIVE_THRESHOLD = 0.01  # During TTS (allows barge-in while filtering echo)

    def __init__(self, ...):
        self.silence_duration = 1.5    # Speech end detection (seconds)
        self.min_speech_duration = 0.3  # Minimum valid speech (seconds)

        # Bounded deque prevents memory growth
        self._audio_buffer: Deque[np.ndarray] = deque(maxlen=self.MAX_SPEECH_CHUNKS)
```

**Audio Callback Optimization:**

```python
def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any):
    # Cache frequently accessed attributes for hot path
    volume = float(np.abs(indata).mean())
    threshold = get_threshold()  # Varies based on TTS state

    if volume > threshold:
        if not self._is_speaking:
            self._is_speaking = True
            audio_buffer.clear()
            if on_speech_start:
                on_speech_start()
        audio_buffer.append(indata.copy())
        self._silence_frames = 0
```

#### HybridSTT

Unified interface for PTT and continuous modes:

```python
class HybridSTT:
    def __init__(self, stt_engine: STTEngine):
        self.ptt = PushToTalkSTT(stt_engine)
        self.continuous = ContinuousSTT(stt_engine)
        self.mode = "ptt"

    def set_tts_active(self, active: bool):
        """Raises threshold during TTS to prevent self-detection."""
        self.continuous.set_tts_active(active)
```

---

### 3.4 Text-to-Speech (Mouth)

**Location:** `src/core/speech/chatterbox_tts.py`

#### ChatterBoxEngine

Zero-shot voice cloning with streaming support:

```python
@dataclass
class ChatterBoxConfig:
    device: str = "cuda"
    voice_reference: str = "assets/voice/reference.wav"
    sample_rate: int = 24000

    # Voice cloning parameters
    exaggeration: float = 0.5   # Emotion intensity (0.25-2.0)
    cfg_weight: float = 0.5     # Voice adherence (0.2-1.0, lower for fast speakers)

    # Adaptive chunking for lower TTFB
    adaptive_chunking: bool = True
    first_chunk_tokens: int = 30      # Smaller first chunk
    subsequent_chunk_tokens: int = 50  # Larger subsequent chunks

    # PyTorch optimizations
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"
    enable_cudnn_benchmark: bool = True
    enable_tf32: bool = True  # For Ampere+ GPUs
```

**CUDA Optimizations:**

```python
def _enable_cuda_optimizations(self):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

**Streaming Synthesis:**

```python
def _generate_streaming(self, text: str, gen_kwargs: dict, ...):
    chunk_size = self.config.first_chunk_tokens if self.config.adaptive_chunking else self.config.chunk_size

    for chunk_idx, (audio_chunk, _) in enumerate(
        self._model.generate_stream(text, chunk_size=chunk_size, **gen_kwargs)
    ):
        if not ttfb_recorded:
            metrics.ttfb_ms = (time.perf_counter() - generation_start) * 1000
            ttfb_recorded = True

        audio_np = self._to_numpy(audio_chunk)

        if post_processor:
            audio_np = post_processor.process_chunk(audio_np, ...)

        self._player.play_chunk(audio_np)

        # Increase chunk size after first chunk
        if self.config.adaptive_chunking and chunk_idx == 0:
            chunk_size = self.config.subsequent_chunk_tokens
```

**Performance Metrics:**

```python
@dataclass
class TTSMetrics:
    ttfb_ms: float = 0.0              # Time to first byte
    total_generation_ms: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0                  # Real-time factor
    chunks_generated: int = 0
    model_inference_ms: float = 0.0
```

**Memory Management:**

```python
def _cleanup_cuda_memory(self):
    """Called every N generations to prevent VRAM fragmentation."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

---

## 4. API Layer

### 4.1 FastAPI Application

**Location:** `src/api/main.py`

#### Two-Phase Startup

Optimized startup reduces time from ~30s to ~5s:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ===== PHASE 1: Critical services (blocking) =====
    # Must be ready before /health returns 200

    # LLM + Memory - parallel initialization
    await asyncio.gather(
        llm_svc.initialize(),
        mem_svc.initialize(),
    )
    container.llm = llm_svc
    container.memory = mem_svc

    log.info("O.L.I.V.I.A. Ready")
    log.info("(STT/TTS loading in background...)")

    # ===== PHASE 2: Heavy models (lazy-loaded in background) =====
    _lazy_load_tasks = [
        asyncio.create_task(_lazy_load_stt()),
        asyncio.create_task(_lazy_load_tts()),
    ]

    yield  # Application runs

    # ===== SHUTDOWN =====
    # Parallel cleanup for independent services
    cleanup_tasks = []
    if container.tts and container.tts.is_initialized():
        cleanup_tasks.append(_cleanup_tts())
    if container.llm:
        cleanup_tasks.append(_cleanup_llm())

    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
```

#### Middleware Stack

```python
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
```

---

### 4.2 Service Container

**Location:** `src/api/container.py`

Typed dataclass with dict-like backward compatibility:

```python
@dataclass
class ServiceContainer:
    llm: Optional["LLMService"] = None
    memory: Optional["MemoryService"] = None
    stt: Optional["STTService"] = None
    tts: Optional["TTSService"] = None
    state: Optional["StateManager"] = None
    dreaming: Optional["DreamingEngine"] = None
    fact_extractor: Optional["HybridFactExtractor"] = None

    def get(self, name: str):
        """Dict-like access for backward compatibility."""
        return getattr(self, name, None)

    def __setitem__(self, name: str, value):
        """Dict-like assignment."""
        setattr(self, name, value)

    def is_critical_ready(self) -> bool:
        """Check if LLM and Memory are initialized."""
        return (
            self.llm is not None and self.llm.is_initialized() and
            self.memory is not None and self.memory.is_initialized()
        )

# Module-level singleton
_container: Optional[ServiceContainer] = None

def get_container() -> ServiceContainer:
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container
```

---

### 4.3 API Endpoints

#### Health Check (GET /health)

```python
@router.get("/health")
async def health_check():
    services = {}
    for name, svc in [("llm", llm), ("memory", memory), ("tts", tts)]:
        if svc and hasattr(svc, "health_check"):
            try:
                status = await asyncio.wait_for(svc.health_check(), timeout=2.0)
                services[name] = {"status": "up" if status else "degraded"}
            except asyncio.TimeoutError:
                services[name] = {"status": "timeout"}

    return HealthCheck(
        status="healthy" if all_healthy else "degraded",
        services=services,
        uptime_seconds=(datetime.now() - startup_time).total_seconds(),
    )
```

#### Chat Endpoint (POST /api/chat)

**Location:** `src/api/routes/chat.py`

```python
@router.post("/chat")
async def chat(request: ChatRequest, llm: LLMServiceDep, memory: MemoryServiceDep):
    # 1. Prefetch memory context (skipped for short/greeting messages,
    #    hard-capped at 1.5s so a slow Chroma query can't stall startup)
    mem_ctx = await _fetch_memory_context(memory, request.message)

    # 2. Stream (SSE) or return JSON
    if request.stream:
        return StreamingResponse(gen_sse(), media_type="text/event-stream")
    else:
        return ChatResponse(message=full_resp, ...)
```

During SSE streaming, completed sentences are pushed into a **session-scoped
TTS queue** (owned by the service layer, not the request) as they arrive, so
the first sentence plays while the LLM is still generating. Because the queue
outlives the request, a client closing the SSE stream does not stop playback
(Phase 1.1). After the stream finishes, the conversation is stored and queued
for background LLM fact extraction via a fire-and-forget task.

There is no web search or search-intent detection — the assistant is offline.

`DELETE /api/history` clears the LLM conversation history.

**SSE Format:**

```
data: {"token": "Hello", "done": false}
data: {"token": " there", "done": false}
data: {"token": "", "done": true}
```

**Pre-built SSE Templates:**

```python
# Avoids json.dumps() overhead per token (~1.5us -> ~0.3us)
_JSON_TOKEN_TEMPLATE = '{{"token": {}, "done": false}}'
_JSON_DONE = '{"token": "", "done": true}'
_JSON_ERROR_TEMPLATE = '{{"error": {}, "done": true}}'
```

#### Voice Endpoint (WebSocket /ws/voice)

**Location:** `src/api/routes/voice.py`

Full-duplex voice pipeline over a single WebSocket (Phase 1.3/1.4). The client
streams raw 16 kHz mono PCM up and receives JSON control events (`speech_start`,
`transcript_final`, `token`, `audio_start`/`audio_end`, `done`, `barge_in`)
interleaved with binary TTS audio frames. Audio playback happens client-side, so
the backend stays containerizable.

#### Other Routers

- `src/api/routes/memory.py` — memory inspection/management endpoints (`/api/memory/*`)
- `src/api/routes/settings.py` — runtime settings endpoints (`/api/settings/*`)
- `src/api/routes/health.py` — `/health` liveness/readiness with rolling voice-pipeline latency metrics

---

### 4.4 Configuration

**Location:** `src/api/config.py`

Pydantic-settings with field validators:

```python
class APIConfig(BaseSettings):
    HF_TOKEN: Optional[str] = None
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000", "http://localhost:5173",
        "http://127.0.0.1:3000", "http://127.0.0.1:5173",
    ]

    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "olivia-finetuned"

    STT_MODEL_SIZE: str = "small.en"
    STT_DEVICE: str = "cuda"
    STT_COMPUTE_TYPE: str = "float16"

    TTS_DEVICE: str = "cuda"
    TTS_VOICE_REFERENCE: str = "assets/voice/reference.wav"
    TTS_CFG_WEIGHT: float = 0.5
    TTS_EXAGGERATION: float = 0.5

    MEMORY_PERSIST_DIR: str = "data/memory_db"

    @field_validator("PORT")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"PORT must be between 1 and 65535, got {v}")
        return v

    @field_validator("OLLAMA_HOST")
    @classmethod
    def validate_ollama_host(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"OLLAMA_HOST must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("STT_MODEL_SIZE")
    @classmethod
    def validate_stt_model_size(cls, v: str) -> str:
        valid = {"tiny", "tiny.en", "base", "base.en", "small", "small.en",
                 "medium", "medium.en", "large", "large-v2", "large-v3"}
        if v not in valid:
            raise ValueError(f"STT_MODEL_SIZE must be one of {valid}")
        return v

@lru_cache(maxsize=1)
def get_api_config() -> APIConfig:
    """Cached singleton configuration."""
    return APIConfig()
```

---

### 4.5 Constants

**Location:** `src/api/constants.py`

Centralized configuration values:

```python
@dataclass(frozen=True)
class Timeouts:
    HEALTH_CHECK: float = 2.0
    MEMORY_OPERATION: float = 10.0
    MEMORY_INIT: float = 30.0
    STT_TRANSCRIBE: float = 30.0
    TTS_SYNTH_BASE: float = 20.0

# Greeting patterns for O(1) lookup (short messages skip memory prefetch)
GREETING_PATTERNS: FrozenSet[str] = frozenset([
    "hi", "hello", "hey", "thanks", "thank you"
])

GREETING_REGEX: Pattern = re.compile(
    r"^(hi|hello|hey|thanks|thank you)\s*[!?.]?$",
    re.IGNORECASE
)

# Pre-built SSE templates (~5x faster than json.dumps per token)
SSE_TOKEN_TEMPLATE: str = '{{"token": {}, "done": false}}'
SSE_DONE: str = '{"token": "", "done": true}'
SSE_ERROR_TEMPLATE: str = '{{"error": {}, "done": true}}'
```

---

## 5. Desktop UI (Flet)

### 5.1 Application Architecture

**Location:** `src/flet_app/app.py`

```python
class OliviaApp:
    def __init__(self, page: ft.Page):
        # Window configuration
        self.page.title = "O.L.I.V.I.A."
        self.page.window.width = 1000
        self.page.window.height = 800
        self.page.window.min_width = 800
        self.page.window.min_height = 600

        # Apply theme
        Theme.apply_to_page(self.page)

        # Initialize services
        self.state_manager = StateManager(self.page)
        self.api_client = OliviaAPIClient()

        # Build UI
        self.build()

        # Initialize backend connection
        asyncio.create_task(self._initialize_backend())
```

**Component Hierarchy:**

```
Page
+-- Column
    +-- Header (Container)
    |   +-- Row
    |       +-- StatusIndicator
    |       +-- Title Column
    |       +-- SettingsButton
    +-- ChatDisplay (Container, expand=True)
    +-- InputBar (Container)
        +-- Row
            +-- MicButton
            +-- InputField (AnimatedInputContainer)
            +-- SendButton (AnimatedSendButton)
```

**Message Streaming:**

```python
async def on_send_message(self):
    message = self.input_field.value.strip()

    self.send_button.set_loading(True)
    self.status_indicator.set_status("processing")

    self.chat_display.append_message("You", message)
    self.chat_display.start_streaming("O.L.I.V.I.A.")

    try:
        async for token in self.api_client.send_message_stream(message):
            self.chat_display.append_token(token)
            await asyncio.sleep(0.01)  # Prevent UI blocking

        self.chat_display.end_streaming()
    except Exception as e:
        self.chat_display.append_token(f"\n\n[Error: {str(e)}]")

    self.send_button.set_loading(False)
    self.status_indicator.set_status("ready")
```

---

### 5.2 Theme System

**Location:** `src/flet_app/theme.py`

#### Color Palette (Discord-Inspired)

```python
@dataclass(frozen=True)
class ColorPalette:
    # Background Layers
    BG_BASE: str = "#1E1F22"      # Darkest
    BG_SURFACE_1: str = "#2B2D31"  # Secondary panels
    BG_SURFACE_2: str = "#313338"  # Main content
    BG_SURFACE_3: str = "#383A40"  # Elevated elements
    BG_SURFACE_4: str = "#404249"  # Hover states

    # Purple Accent Palette
    ACCENT_PRIMARY: str = "#8B5CF6"
    ACCENT_LIGHT: str = "#A78BFA"
    ACCENT_DARK: str = "#7C3AED"

    # Text Hierarchy
    TEXT_PRIMARY: str = "#F2F3F5"
    TEXT_SECONDARY: str = "#B5BAC1"
    TEXT_TERTIARY: str = "#949BA4"

    # Status Colors
    STATUS_SUCCESS: str = "#23A55A"   # Green
    STATUS_WARNING: str = "#3B82F6"   # Blue
    STATUS_ERROR: str = "#DA373C"     # Red
    STATUS_PURPLE: str = "#8B5CF6"    # Speaking
```

#### Animation Timings

```python
@dataclass(frozen=True)
class Animation:
    INSTANT: int = 100      # Immediate feedback
    FAST: int = 150         # Button press, hover
    NORMAL: int = 250       # Standard transitions
    MEDIUM: int = 350       # Message appearance
    SLOW: int = 500         # Complex transitions
    PULSE_CYCLE: int = 2000 # Breathing animation

    HOVER_SCALE: float = 1.03   # Subtle hover enlargement
    PRESS_SCALE: float = 0.97   # Press feedback
```

---

### 5.3 Components

#### StatusIndicator

| State | Color | Pulse Cycle |
|-------|-------|-------------|
| initializing | Muted | 1500ms |
| ready | Green | None |
| processing | Blue | 1000ms |
| recording | Red | 800ms |
| speaking | Purple | 600ms |
| error | Red | None |

#### ChatDisplay

- Message bubbles with user/assistant styling
- Streaming token append
- Auto-scroll to bottom
- Bounded to 100 messages

#### AnimatedButton Components

- AnimatedIconButton: Mic, settings buttons
- AnimatedSendButton: Send with loading state
- AnimatedInputContainer: Focus highlight effects

---

## 6. Background Memory Systems

Both systems below are production and wired into the FastAPI lifespan
(`src/api/main.py`): the dreaming engine starts idle monitoring at startup and
runs a final pass on shutdown, and the hybrid fact extractor's background worker
starts at startup. They are tracked on the service container as
`container.dreaming` and `container.fact_extractor`.

The only module still under `src/experimental/` is
`src/experimental/speech/wake_word_enhanced.py`, which is not yet wired in.

### 6.1 Dreaming Engine

**Location:** `src/core/memory/dreaming.py`

Memory consolidation during idle/shutdown:

```python
@dataclass
class DreamConfig:
    dream_on_shutdown: bool = True
    dream_on_idle: bool = True
    idle_threshold_minutes: int = 5
    age_threshold_hours: int = 24
    max_conversations_per_dream: int = 50

    summary_model: str = "olivia-finetuned"
    summary_max_tokens: int = 200

    keep_raw_conversations_days: int = 7
    save_dream_reports: bool = True
```

**Process Flow:**

1. **Trigger**: Idle detection (5 min) or shutdown
2. **Fetch**: Get un-dreamed conversations from ChromaDB (server-side filtering)
3. **Group**: Group conversations by date
4. **Summarize**: LLM-powered summarization per group
5. **Extract**: LLM fact extraction with hallucination filtering
6. **Store**: Save summaries and facts to respective tiers
7. **Mark**: Mark conversations as dreamed

**Hallucination Detection:**

```python
# O(1) frozenset lookup + O(1) regex match
_HALLUCINATED_EXACT: frozenset = frozenset({
    "sister's birthday", "march 15", "example fact", "placeholder",
})

_HALLUCINATED_PATTERN: re.Pattern = re.compile(
    r"sister's birthday|march 15|example fact|placeholder|john doe",
    re.IGNORECASE
)

def _is_hallucinated_fact(self, fact: str) -> bool:
    fact_lower = fact.lower()
    for pattern in _HALLUCINATED_EXACT:
        if pattern in fact_lower:
            return True
    return bool(_HALLUCINATED_PATTERN.search(fact_lower))
```

**ChromaDB Server-Side Filtering:**

```python
def _get_conversations(self, age_threshold_hours: int):
    # Build where clause for server-side filtering
    # O(n) Python-side filtering -> O(log n) database filtering
    where_conditions = [{"dreamed": {"$ne": True}}]

    if age_threshold_hours > 0:
        threshold_time = datetime.now() - timedelta(hours=age_threshold_hours)
        where_conditions.append({"timestamp": {"$lt": threshold_time.isoformat()}})

    where_clause = {"$and": where_conditions}

    return self.memory.conversations.get(
        limit=self.config.max_conversations_per_dream,
        where=where_clause,
    )
```

---

### 6.2 Fact Extractor

**Location:** `src/core/memory/fact_extractor.py`

Hybrid extraction using regex + LLM:

```python
class HybridFactExtractor:
    def extract(self, user_msg: str, ai_msg: str):
        # 1. Quick regex extraction (immediate)
        quick_facts = self._regex_extract(user_msg)

        if quick_facts:
            # Batch duplicate check
            facts_only = [fact for fact, _ in quick_facts]
            is_duplicate = self.memory.batch_check_duplicates(facts_only)

            for i, (fact, category) in enumerate(quick_facts):
                if not is_duplicate[i]:
                    self.memory.add_fact(fact, category)

        # 2. Queue for LLM extraction (background)
        self.llm_extractor.queue_extraction(user_msg, ai_msg)
```

**Pre-Compiled Quick Patterns:**

```python
_QUICK_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"(?:my name is|i'm called|call me) ([a-zA-Z]+)", re.IGNORECASE),
     "personal", "User's name is {0}"),
    (re.compile(r"(?:i am|i'm) (\d+)(?: years old)?", re.IGNORECASE),
     "personal", "User is {0} years old"),
    (re.compile(r"(?:i live in|i'm from) ([^.,!]+)", re.IGNORECASE),
     "personal", "User lives in {0}"),
    (re.compile(r"(?:i work (?:at|as|for)) ([^.,!]+)", re.IGNORECASE),
     "work", "User works {0}"),
    # ... 10 total patterns
]
```

**LLM Extraction Worker:**

```python
class LLMFactExtractor:
    def _worker_loop(self):
        while self._running:
            item = self._queue.get(timeout=1.0)
            user_msg, ai_msg, timestamp = item

            facts = self._extract_facts(user_msg, ai_msg)

            # Batch duplicate check
            valid_facts = [f for f in facts if f.confidence >= self.config.min_confidence]
            fact_texts = [f.fact for f in valid_facts]
            is_duplicate = self._batch_check_duplicates(fact_texts)

            for i, fact in enumerate(valid_facts):
                if not is_duplicate[i]:
                    self._store_fact(fact)

            time.sleep(self.config.batch_delay_seconds)
```

**Integration:** Wired into the FastAPI lifespan. The `/api/chat` handler queues
each completed exchange for background LLM fact extraction via
`container.fact_extractor.llm_extractor.queue_extraction(...)`.

---

## 7. Personality System

**Location:** `config/character.yaml`

### Core Traits

```yaml
personality:
  core_traits:
    warmth: 0.75          # Kind and caring, but not saccharine
    directness: 0.80      # Frank and honest, doesn't sugarcoat
    confidence: 0.70      # Self-assured without arrogance
    playfulness: 0.60     # Has fun without being silly
    sensitivity: 0.65     # Emotionally aware
    protectiveness: 0.75  # Genuinely cares about user wellbeing
```

### Speaking Style

```yaml
speaking_style:
  tone: "warm but direct"
  formality: "casual"
  verbosity: "concise"

  characteristics:
    uses_contractions: true
    uses_filler_words: "occasionally"  # "um", "hmm", "well"
    asks_followup_questions: true
    uses_emojis: false           # CRITICAL: Never use emojis
    uses_kaomoji: false          # CRITICAL: Never use kaomoji
    uses_asterisk_actions: false # CRITICAL: No *actions*
```

### Forbidden Phrases

```yaml
forbidden:
  phrases:
    - "Certainly!"
    - "Absolutely!"
    - "I'd be delighted to"
    - "I'd be happy to"
    - "Great question!"
    - "I hope that helps"
    - "As an AI"
    - "I'm sorry, but"
    - "Is there anything else"
```

### System Prompt

```yaml
system_prompt_template: |
  You are Olivia, a personal AI companion. Keep responses conversational and natural.

  Voice: casual, warm, direct. Speak like a close friend chatting.
  Format: plain text only. No code, no lists, no markdown.
  Length: Be concise but complete your thoughts. 2-4 sentences is typical.

  Forbidden: Certainly, Great question, I'd be happy to, As an AI, I apologize, language model.
```

### TTS Voice Configuration

```yaml
tts:
  voice_cloning:
    exaggeration: 0.5       # 0.25-2.0: emotion intensity
    cfg_weight: 0.5         # 0.2-1.0: adherence to reference voice

  latency:
    adaptive_chunking: true
    first_chunk_tokens: 30
    subsequent_chunk_tokens: 50
```

---

## 8. Performance Optimizations

### Summary Table

| System | Optimization | Complexity Change |
|--------|-------------|-------------------|
| Memory | Pre-compiled regex | O(n) compile -> O(1) reuse |
| Memory | Parallel tier searches | O(3t) -> O(t) |
| Memory | Batch duplicate check | O(n) queries -> O(1) batch |
| LLM | Connection pooling | Reduced TCP overhead |
| LLM | List append + join | O(n^2) -> O(n) |
| TTS | Adaptive chunking | Lower TTFB |
| TTS | torch.compile() | Faster inference |
| API | Parallel startup | O(t1+t2) -> O(max(t1,t2)) |
| API | Parallel shutdown | O(t1+t2) -> O(max(t1,t2)) |
| API | Pre-built SSE templates | ~5x faster per token |
| Dreaming | ChromaDB where clause | Server-side filtering |
| Facts | Batch LLM extraction | Background queue |

### Key Patterns

1. **Pre-compiled Regex**: All regex patterns compiled at module level
2. **FrozenSet Lookups**: O(1) membership testing for greetings, domains, keywords
3. **Parallel I/O**: asyncio.gather() for independent operations
4. **Connection Pooling**: httpx limits for Ollama client
5. **Lazy Loading**: STT/TTS loaded in background after startup
6. **Caching**: LRU cache for repeated operations (greeting detection)
7. **Template Strings**: Pre-built JSON templates avoid serialization overhead

---

## 9. VRAM Budget

**Target: RTX 4080 SUPER 16GB**

| Component | VRAM Usage |
|-----------|------------|
| olivia-finetuned 8B Q4 | ~6GB |
| Whisper small.en | ~1GB |
| ChatterBox Turbo | ~2GB |
| ChromaDB/Embeddings | ~0.5GB |
| **Total Active** | **~9.5GB** |
| **Available** | ~6.5GB (for future: vision, emotion) |

---

## 10. Directory Structure

```
project-olivia/
+-- src/
|   +-- api/                      # FastAPI Backend
|   |   +-- routes/               # chat.py, voice.py, memory.py, settings.py, health.py
|   |   +-- services/             # Service wrappers (llm, memory, stt, tts,
|   |   |                         #   tts_queue, audio_output, metrics, ...)
|   |   +-- models/               # Pydantic schemas
|   |   +-- utils/                # sentence_buffer, tts_sanitizer, audio_utils, ...
|   |   +-- main.py               # FastAPI app entry + lifespan
|   |   +-- config.py             # Configuration
|   |   +-- container.py          # DI container
|   |   +-- constants.py          # Centralized values
|   |   +-- protocols.py          # Service interfaces
|   |   +-- dependencies.py       # FastAPI dependencies
|   |   +-- middleware.py         # Logging + error handling
|   |
|   +-- flet_app/                 # Desktop UI
|   |   +-- components/           # UI components
|   |   +-- services/             # api_client, voice_client, state
|   |   +-- main.py               # Flet entry point
|   |   +-- app.py                # Main app class
|   |   +-- theme.py              # Design system
|   |
|   +-- core/                     # Core Systems
|   |   +-- llm/                  # Ollama integration
|   |   |   +-- ollama_client.py  # ConversationManager
|   |   +-- speech/               # STT/TTS
|   |   |   +-- stt.py            # faster-whisper
|   |   |   +-- chatterbox_tts.py # ChatterBox Turbo
|   |   |   +-- audio_processing.py
|   |   +-- memory/               # ChromaDB + background memory systems
|   |       +-- smart_memory.py   # Three-tier system
|   |       +-- dreaming.py       # DreamingEngine
|   |       +-- fact_extractor.py # HybridFactExtractor
|   |
|   +-- config/                   # Character config loader
|   +-- utils/                    # Shared utilities (logger)
|   +-- experimental/
|       +-- speech/
|           +-- wake_word_enhanced.py  # Not yet wired in
|
+-- config/
|   +-- character.yaml            # Personality configuration
|
+-- tools/                        # Data-gen, fine-tuning, benchmarking
|   +-- bench.py, bench_compare.py
|   +-- model_engineering/        # finetune, DPO, GGUF convert, merge/ kit
+-- benchmarks/results/           # Committed benchmark results
+-- tests/                        # Test suite
+-- docs/                         # Documentation (see MASTER_PLAN.md for roadmap)
+-- data/                         # Runtime data (memory_db, logs)
+-- assets/voice/                 # Voice reference files
+-- run_olivia.py                 # Unified launcher
+-- CLAUDE.md                     # Project instructions
```

---

## 11. Critical Notes

### Must-Follow Rules

1. **beam_size=5 ALWAYS** for STT consistency (must match warmup)
2. **Memory injection via system message** - NOT conversation history
3. **TTS sanitizer removes**: [MEMORY], ###, *actions*, :emoji:
4. **Sentence buffer**: min 6 words, max 30 words
5. **FORBIDDEN personality elements**: emojis, kaomoji, asterisk actions

### Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Time to first audio | <500ms | ~500ms |
| Total response time | <3s | ~2.5s |
| Tokens per second | >40 | ~45 |
| Startup (to ready) | <10s | ~5s |
| Personality consistency | >90% | ~85% |

### Status & Roadmap

Done: DreamingEngine + HybridFactExtractor are wired into the FastAPI lifespan,
and the memory API endpoints (`/api/memory`) are live. Forward direction now
lives in `docs/MASTER_PLAN.md`.

---

*Documentation generated from codebase analysis on 2026-01-28; audited against current code 2026-07-03.*
