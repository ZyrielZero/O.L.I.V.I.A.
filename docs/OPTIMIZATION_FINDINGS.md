# O.L.I.V.I.A. Production Optimization Findings

> Multi-agent analysis (5 specialists) of the O.L.I.V.I.A. codebase for production readiness.
> Date: 2026-02-10 | Agents: backend-optimizer, memory-optimizer, voice-optimizer, infra-hardener, devils-advocate
>
> **Implementation Status**: 32/33 APPROVED items IMPLEMENTED. 1 DEFERRED (ChatService consolidation — chat.py already clean after dead code removal). 26 files changed, 664 insertions, 665 deletions. All tests passing.

## Executive Summary

The codebase is already well-optimized for a local AI assistant. Previous optimization passes addressed the highest-impact items (CUDA sync removal, Silero VAD, LLM parameter tuning, streaming TTS pipeline). The remaining opportunities fall into four categories:

1. **Bug fixes & correctness** -- ID collisions, HNSW metadata loss, TTL not enforced, missing TTS sanitizer, unbounded history
2. **Experimental integration** -- DreamingEngine + HybridFactExtractor (Sprint Goal #1)
3. **Security hardening** -- error info disclosure, HOST binding, input validation, CORS
4. **Performance tweaks** -- fire-and-forget storage, ASGI middleware, CUDA cleanup tuning

**Top 10 highest-impact changes:**

| # | Change | Category | Est. Impact | Effort |
|---|--------|----------|-------------|--------|
| 1 | Integrate DreamingEngine + HybridFactExtractor | Memory | Sprint Goal #1 completion | ~50 LOC |
| 2 | Auto-trim conversation history | Backend | Prevents OOM + personality loss | 1 LOC |
| 3 | Wire barge-in (STT -> TTS stop) | Voice | Critical UX -- stops TTS when user speaks | ~10 LOC |
| 4 | Implement TTS sanitizer | Voice | Prevents speaking [MEMORY], ###, *actions* | ~10 LOC |
| 5 | Fix ID generation (race condition + collision) | Memory | Prevents data corruption | ~40 LOC |
| 6 | Enforce TTL retention (7d convos, 1y summaries) | Memory | Prevents unbounded growth | ~20 LOC |
| 7 | Fix error info disclosure | Infra | Prevents stack trace leak to clients | 5 LOC |
| 8 | Bind default HOST to 127.0.0.1 | Infra | Eliminates LAN exposure | 2 LOC |
| 9 | Fire-and-forget memory storage | Backend | 50-80ms latency reduction | 2 LOC |
| 10 | Fix clear_all() HNSW metadata bug | Memory | Prevents 10x search quality degradation | 3 LOC |

**VRAM Budget (Corrected):**

| Component | Documented | Actual |
|-----------|-----------|--------|
| LLM (olivia-merged Q4) | ~6GB | ~6GB |
| Whisper small.en | ~1GB | ~1GB |
| ChatterBox + torch.compile | ~2GB | ~3-4GB |
| ChromaDB/Embeddings | ~0.5GB | ~0.5GB |
| CUDA overhead/fragmentation | -- | ~0.5-1GB |
| **Total** | **~9.5GB** | **~11-12.5GB** |
| **Headroom** | **~6.5GB** | **~3.5-5GB** |

Note: torch.compile `reduce-overhead` adds 1-2GB vs documented estimate. CLAUDE.md VRAM table needs updating.

---

## Backend Optimizations

### IMPLEMENTED: Auto-trim conversation history [CRITICAL]
- **Description**: `ConversationManager.history` is unbounded (`src/core/llm/ollama_client.py:60`). `trim_history(keep=20)` exists at line 83 but **nothing calls it**. After 40+ messages, the system prompt gets pushed out of `num_ctx=4096`, causing personality loss.
- **Impact**: Prevents OOM, payload bloat, and silent personality degradation.
- **Complexity**: 1 LOC -- add `self.trim_history()` after line 158 in `chat_stream_async()`.
- **Files**: `src/core/llm/ollama_client.py`
- **Consensus**: APPROVED unanimously.

### IMPLEMENTED: Fire-and-forget memory storage
- **Description**: In streaming path (`chat.py:206-209`), `memory.add_conversation()` is awaited before done signal. Memory storage involves ChromaDB writes + fact extraction.
- **Impact**: 50-80ms latency reduction (UI gets done signal immediately).
- **Complexity**: 2 LOC -- wrap in `asyncio.create_task()`.
- **Files**: `src/api/routes/chat.py`
- **Dependency**: ID generation fix (Memory F5) MUST be done first -- concurrent writes with count-based IDs will collide.
- **Consensus**: APPROVED by backend/memory/voice specialists. Devils-advocate concerned about silent data loss -- mitigated by existing error logging in memory service. Memory failures are already non-fatal.

### IMPLEMENTED: Convert BaseHTTPMiddleware to pure ASGI
- **Description**: Both middlewares (`middleware.py:13-42`) use `BaseHTTPMiddleware`, adding per-SSE-event overhead. With 100-300 tokens per response, cumulative overhead is 10-150ms.
- **Impact**: 20-30% latency reduction on SSE streams. Also fixes error info disclosure (generic error instead of `str(e)`).
- **Complexity**: ~40 LOC rewrite.
- **Files**: `src/api/middleware.py`, `src/api/main.py`
- **Consensus**: APPROVED by backend/voice/infra specialists. Voice-optimizer showed per-event overhead cascades into TTFB. Devils-advocate rejected ("1 req/min") but the concern is events-per-stream, not requests-per-minute. 4/5 majority.

### IMPLEMENTED: Remove dead search code from chat route
- **Description**: `chat.py:118` hardcodes `search_q, search_mode = None, ""` but lines 124-140 contain unreachable search logic.
- **Impact**: ~20 LOC removed.
- **Complexity**: 10 min.
- **Files**: `src/api/routes/chat.py`

### DEFERRED: Consolidate chat.py to use ChatService
- **Description**: `ChatService` duplicates route logic (greeting/search patterns, SSE generation). Route should delegate.
- **Impact**: ~100 LOC reduction, single sanitization point for TTS, enables unit testing.
- **Complexity**: 30-60 min. Must be 1:1 behavior-preserving.
- **Files**: `src/api/routes/chat.py`, `src/api/services/chat_service.py`
- **Reason**: After Phase 1+2 dead code removal, chat.py is already clean (207 lines). ChatService had stale search patterns removed. Consolidation deferred to avoid unnecessary churn.

### IMPLEMENTED: Reduce ThreadPoolExecutor 20 -> 10
- **Description**: `main.py:92` creates 20 workers. Peak concurrent usage is ~5-6 threads.
- **Impact**: ~10MB memory savings.
- **Complexity**: 1 LOC.
- **Files**: `src/api/main.py`
- **Consensus**: Team converged on 10 (safety margin for planned WebSocket feature).

### IMPLEMENTED: Fix asyncio event loop caching in MemoryService
- **Description**: Caches deprecated `asyncio.get_event_loop()`. Use `asyncio.get_running_loop()` inline.
- **Complexity**: 3 LOC.
- **Files**: `src/api/services/memory_service.py`

### IMPLEMENTED: Fix middleware timing to use perf_counter
- **Description**: `middleware.py:17,22` uses `time.time()` (~1ms resolution). `perf_counter()` has ~100ns.
- **Complexity**: 2 LOC.
- **Files**: `src/api/middleware.py`

### IMPLEMENTED: Reuse httpx client for health checks (sync only)
- **Description**: `check_ollama_connection_async()` creates new `httpx.AsyncClient` per call instead of reusing persistent `self._client`.
- **Complexity**: 5-10 LOC.
- **Files**: `src/core/llm/ollama_client.py`

### IMPLEMENTED: Add timeout on get_relevant_context
- **Description**: Memory context retrieval has no timeout. ChromaDB hang = chat hang.
- **Complexity**: 1 LOC.
- **Files**: `src/api/routes/chat.py`

### DEFERRED: Web search re-enablement
- **Description**: Fully built but disabled due to sync `requests.get()` blocking. Fix: `asyncio.to_thread()` + SSRF mitigations (block private IPs, sanitize scraped content). Opt-in via `/search` prefix.
- **Status**: Deferred -- requires security review. Not a quick fix.
- **Files**: `src/core/tools/web_search.py`, `src/api/routes/chat.py`

---

## Memory System

### IMPLEMENTED: Integrate DreamingEngine [SPRINT GOAL #1]
- **Description**: Wire `src/experimental/memory/dreaming.py` into app lifecycle. Fully implemented, not connected.
- **Impact**: Enables memory consolidation -- conversations summarized, facts LLM-extracted, stale data marked. Core missing feature.
- **Complexity**: ~30-50 LOC.
- **Files**: `src/api/main.py`, `src/api/routes/health.py`
- **Integration**: `create_dreaming_engine(mem_svc._db)` -> `start_idle_monitoring()` on startup -> `dream_on_shutdown()` on shutdown.
- **Constraint**: Shutdown dream can block 60-120s. Add `max_shutdown_conversations=10`.

### IMPLEMENTED: Integrate HybridFactExtractor [SPRINT GOAL #1]
- **Description**: Wire `src/experimental/memory/fact_extractor.py` into chat flow. Replaces 7-pattern regex with hybrid regex (10 patterns, immediate) + LLM extraction (background).
- **Impact**: Dramatically richer fact extraction -- LLM catches implicit facts regex never will.
- **Complexity**: ~20-30 LOC.
- **Files**: `src/api/main.py`, `src/api/services/memory_service.py`
- **CRITICAL CONSTRAINT**: Background LLM calls `ollama.chat()` in a thread, competing with user's chat (Ollama serializes inference). MUST gate extraction on idle state -- 1-3s latency penalty otherwise.

### IMPLEMENTED: Enforce TTL/Retention [BUG]
- **Description**: CLAUDE.md documents "conversations (7 days), summaries (1 year)" but ZERO code enforces this. Collections grow unbounded.
- **Impact**: Prevents unbounded growth, stale context surfacing, query latency degradation.
- **Complexity**: ~20 LOC -- `prune_expired()` in SmartMemoryDB, wire into dream cycle. Only prune dreamed conversations. 30-day hard TTL safety net.
- **Files**: `src/core/memory/smart_memory.py`

### IMPLEMENTED: Fix clear_all() missing HNSW metadata [BUG]
- **Description**: `smart_memory.py:440-458` recreates collections without `metadata=self.HNSW_METADATA`. `search_ef` drops 100->10, `num_threads` 4->1.
- **Impact**: 10x search quality degradation after any clear. Silent.
- **Complexity**: 3 LOC.
- **Files**: `src/core/memory/smart_memory.py`

### IMPLEMENTED: Fix ID generation [BUG]
- **Description**: `smart_memory.py:98,231,303` uses `count()+1`. Race condition on concurrent adds; deletion causes collision.
- **Impact**: Data corruption. Required before fire-and-forget can be implemented.
- **Complexity**: ~40 LOC -- timestamp-based IDs, rewrite `get_recent_conversations` to use `where` filter + `limit`.
- **Files**: `src/core/memory/smart_memory.py`

### IMPLEMENTED: Add memory backup before dreaming
- **Description**: No backup mechanism. ChromaDB corruption during dream = permanent data loss.
- **Complexity**: ~15 LOC -- `shutil.copytree` before dream, keep last 3.
- **Files**: `src/core/memory/smart_memory.py`

### IMPLEMENTED: Add explicit close() to SmartMemoryDB
- **Description**: `ThreadPoolExecutor(max_workers=3)` only cleaned in unreliable `__del__`.
- **Complexity**: ~5 LOC.
- **Files**: `src/core/memory/smart_memory.py`

### NOTED: ChromaDB `$ne` on missing metadata
- **Description**: `dreaming.py` filters `{"dreamed": {"$ne": True}}` but conversations stored without "dreamed" metadata. Behavior varies by version.
- **Action**: Test during integration. Fallback exists.

### NOTED: Embedding 256 token limit
- **Description**: `all-MiniLM-L6-v2` silently truncates beyond ~200 words. Long messages produce poor embeddings.
- **Action**: Document limit. Consider chunking in future sprint.

---

## Voice Pipeline

### IMPLEMENTED: Wire barge-in [HIGH UX]
- **Description**: `on_speech_start` callback (`stt.py:258,386-397`) and `TTS.stop()` (`chatterbox_tts.py:559-567`) exist but are **never connected**. TTS plays to completion even if user speaks. No cancellation.
- **Impact**: Critical UX -- estimated ~111ms from speech detection to TTS stop.
- **Complexity**: ~10 LOC -- wire callback + add global TTS queue reference for cross-request cancellation.
- **Files**: `src/api/services/tts_service.py`, `src/api/services/stt_service.py`
- **Additional**: `SentenceTTSQueue` is per-request with no global registry. Store active queue on container.

### IMPLEMENTED: Implement TTS text sanitizer [FUNCTIONAL GAP]
- **Description**: CLAUDE.md states sanitizer must strip `[MEMORY]`, `###`, `*actions*`, `:emoji:` but **no implementation exists**. Memory tags and markdown spoken verbatim.
- **Impact**: Correctness -- prevents speaking formatting/control text.
- **Complexity**: ~10 LOC regex utility + 2 integration points (or 1 if ChatService consolidated).
- **Files**: `src/api/utils/` (new utility), `src/api/routes/chat.py`

### IMPLEMENTED: CUDA cleanup interval tuning
- **Description**: `chatterbox_tts.py:549-550` runs `empty_cache()` + `synchronize()` every 10 generations. `synchronize()` forces GPU stall.
- **Impact**: Eliminates ~90% of cleanup stalls (50-100ms per event).
- **Proposal**: Increase to 50. Add VRAM threshold (cleanup at >80%). Remove `synchronize()`.
- **Files**: `src/core/speech/chatterbox_tts.py`

### IMPLEMENTED: Monitor torch.compile VRAM growth [HIGH]
- **Description**: `compile_mode="reduce-overhead"` (`chatterbox_tts.py:69`) uses CUDA graphs. PyTorch #128424, #159669 confirm VRAM growth with dynamic shapes. `empty_cache()` does NOT free CUDA graph memory.
- **Impact**: Prevents OOM in long sessions. Potentially recovers 1-2GB.
- **Proposal**: Monitor over 100+ generations. If confirmed: switch to `compile_mode="default"` (~10-20ms slower, stable VRAM).
- **Files**: `src/core/speech/chatterbox_tts.py`, `src/api/config.py`

### IMPLEMENTED: Pin Silero VAD version
- **Description**: `stt.py:42-44` loads via `torch.hub.load()` with `trust_repo=True`, no version pin. Supply chain risk.
- **Impact**: Version stability, security. VAD v5 is 3x faster, same API.
- **Complexity**: Pin commit hash or bundle locally.
- **Files**: `src/core/speech/stt.py`

### IMPLEMENTED: Non-streaming TTS fire-and-forget
- **Description**: Non-streaming path (`chat.py:256-262`) blocks response for entire playback.
- **Complexity**: 1 LOC -- `asyncio.create_task(tts.speak(full_resp))`.
- **Files**: `src/api/routes/chat.py`

### IMPLEMENTED: Fix synthesize_stream threading
- **Description**: Uses `threading.Thread` instead of `_gpu_exec`, bypassing CUDA contention protection.
- **Complexity**: ~10 LOC.
- **Files**: `src/api/services/tts_service.py`

### IMPLEMENTED: Optimize sentence buffer set creation
- **Description**: Creates `set(self._buf)` per token. Check only new token for terminators.
- **Complexity**: ~5 LOC.
- **Files**: `src/api/utils/sentence_buffer.py`

### DEFERRED: Reusable TTS engine (measure first)
- **Description**: `_make_temp_engine()` recreates wrappers per synthesis. May be <1ms overhead.
- **Files**: `src/api/services/tts_service.py`

---

## Infrastructure

### IMPLEMENTED: Fix error information disclosure [SECURITY]
- **Description**: `middleware.py:39-40` returns `str(e)` and `type(e).__name__` to clients. Leaks paths, library names, stack traces.
- **Impact**: Eliminates info leakage (OWASP A01).
- **Complexity**: 5 LOC -- return generic `{"error": "Internal server error"}`. Merged with ASGI rewrite.
- **Files**: `src/api/middleware.py`

### IMPLEMENTED: Bind default HOST to 127.0.0.1
- **Description**: `config.py:45` defaults to `"0.0.0.0"`. `run_olivia.py:99` hardcodes it.
- **Complexity**: 2 LOC.
- **Files**: `src/api/config.py`, `run_olivia.py`

### IMPLEMENTED: Add request size limits
- **Description**: `ChatRequest.message` has no `max_length`. Multi-MB message triggers embedding + LLM + storage.
- **Complexity**: 1 LOC -- `Field(..., max_length=10_000)`.
- **Files**: `src/api/models/chat.py`

### IMPLEMENTED: CORS methods/headers lockdown
- **Description**: `config.py:55-56` uses wildcards for methods/headers. Origins are properly scoped.
- **Complexity**: 2 LOC -- `["GET", "POST", "OPTIONS"]` and `["Content-Type", "X-API-Key", "Accept"]`.
- **Files**: `src/api/config.py`

### IMPLEMENTED: Rename MemoryError to MemoryServiceError
- **Description**: Shadows Python builtin `MemoryError`. Could mask OOM on GPU-intensive app.
- **Complexity**: ~15 LOC.
- **Files**: `src/api/utils/exceptions.py`, `src/api/services/memory_service.py`

### IMPLEMENTED: Graceful shutdown signal handling (already existed in run_olivia.py)
- **Description**: No SIGINT/SIGTERM handler. CUDA cleanup needed on Ctrl+C.
- **Complexity**: ~10-15 LOC.
- **Files**: `src/api/main.py` or `run_olivia.py`

### IMPLEMENTED: GitHub Actions CI enhancement
- **Description**: Current CI runs fast tests + ruff. Add coverage reporting, `pip-audit` for vulnerability scanning.
- **Complexity**: ~30 LOC YAML.
- **Files**: `.github/workflows/ci.yml`

### IMPLEMENTED: Dependency pinning
- **Description**: Most deps use `>=` (`chromadb>=0.4.22` but `1.4.1` installed). Non-reproducible builds.
- **Complexity**: 1 command -- `pip freeze > requirements.lock`.
- **Files**: `requirements.lock` (new)

### IMPLEMENTED: Log rotation
- **Description**: `main.py:18-19` uses `basicConfig` to stdout only. No rotation, no file output.
- **Complexity**: ~15 LOC -- `RotatingFileHandler` (10MB, 3 backups).
- **Files**: `src/api/main.py`

### IMPLEMENTED: Health check liveness/readiness split
- **Description**: Single `/health` checks all services. Can't distinguish "crashed" from "still loading models."
- **Complexity**: ~25 LOC -- `/health/live` (always 200), `/health/ready` (all services up).
- **Files**: `src/api/routes/health.py`

### DEFERRED: API authentication
- **Status**: Implement when LAN access (HOST=0.0.0.0) is needed. Design ready (ASGI middleware, X-API-Key).

### DEFERRED: Docker/Docker Compose
- **Status**: Implement when deploying to other machines. Design ready (NVIDIA CUDA base, `compile_mode="default"`, VRAM health monitoring). Near-zero VRAM overhead confirmed.

### DEFERRED: Prometheus metrics
- **Status**: YAGNI for single-user. `tts_service.get_status()` already reports key metrics.

---

## Rejected Proposals

| Proposal | Category | Reason |
|----------|----------|--------|
| Connection pooling for Ollama | Backend | Already implemented (`httpx.Limits`) |
| Circuit breaker for Ollama | Backend | Local service, binary failure modes. Existing timeouts suffice. |
| Response caching for /health | Backend | Negligible benefit at local usage |
| LLM response caching | Backend | Conversations are unique |
| Message queuing (Redis/RabbitMQ) | Backend | YAGNI -- asyncio.Queue sufficient |
| Switch vector DB | Memory | Working, migration cost massive |
| Upgrade embedding model | Memory | Requires full re-embedding. nomic-embed-text-v1.5 as future path. |
| Semantic cache for memory queries | Memory | Queries rarely repeat. Cache invalidation complexity. |
| Fact cache LRU | Memory | Parallel queries already 15-30ms. Marginal gain. |
| Switch TTS engine | Voice | ChatterBox tuned + working. Migration cost enormous. |
| Switch from faster-whisper | Voice | CTranslate2-optimized. beam_size=5 mandatory. |
| Change STT model (tiny.en) | Voice | 0.85GB savings not worth +2-3% WER for voice assistant. |
| Audio buffer pooling | Voice | GC pressure negligible at ~1 chunk/second. |
| WebSocket voice streaming | Voice | Feature request, not optimization. Separate sprint. |
| Full CI/CD (K8s, staging) | Infra | Over-engineering for personal project |
| JWT/OAuth | Infra | Overkill for single-user local app |
| Rate limiting | Infra | Single user, single machine |
| Structured JSON logging | Infra | No log aggregation system |
| Health check dashboard | Infra | `/health` JSON response sufficient |

---

## Implementation Roadmap

### Phase 1: Quick Wins (high impact, low effort) -- ~1 hour

| # | Change | LOC | Time | Dependency |
|---|--------|-----|------|------------|
| 1 | Bind HOST to 127.0.0.1 | 2 | 2 min | -- |
| 2 | Add request size limits (max_length=10000) | 1 | 1 min | -- |
| 3 | Fix error info disclosure (generic error) | 5 | 5 min | -- |
| 4 | CORS methods/headers lockdown | 2 | 2 min | -- |
| 5 | Rename MemoryError -> MemoryServiceError | 15 | 15 min | -- |
| 6 | Auto-trim conversation history | 1 | 1 min | -- |
| 7 | Fix clear_all() HNSW metadata bug | 3 | 5 min | -- |
| 8 | Fix middleware timing (perf_counter) | 2 | 2 min | -- |
| 9 | Remove dead search code | -20 | 10 min | -- |
| 10 | Fix asyncio event loop caching | 3 | 5 min | -- |
| 11 | Reduce thread pool 20 -> 10 | 1 | 1 min | -- |
| 12 | Dependency pinning (pip freeze) | 0 | 5 min | -- |

### Phase 2: Correctness & Safety -- ~2 hours

| # | Change | LOC | Time | Dependency |
|---|--------|-----|------|------------|
| 1 | Fix ID generation (timestamp + UUID) | 40 | 30 min | -- |
| 2 | Fire-and-forget memory storage | 2 | 5 min | Requires #1 |
| 3 | Implement TTS text sanitizer | 10 | 15 min | -- |
| 4 | Wire barge-in (STT -> TTS stop) | 10 | 30 min | -- |
| 5 | Add memory backup mechanism | 15 | 15 min | -- |
| 6 | Add SmartMemoryDB.close() | 5 | 5 min | -- |
| 7 | Enforce TTL retention | 20 | 20 min | -- |
| 8 | Pin Silero VAD version | 3 | 5 min | -- |
| 9 | Log rotation (RotatingFileHandler) | 15 | 15 min | -- |

### Phase 3: Experimental Integration -- ~3-4 hours

| # | Change | LOC | Time | Dependency |
|---|--------|-----|------|------------|
| 1 | Integrate DreamingEngine | 30-50 | 1-2 hrs | Phase 2 #5 (backup), #6 (close), #7 (TTL) |
| 2 | Integrate HybridFactExtractor (with idle gating) | 20-30 | 1 hr | Phase 2 #1 (ID fix) |
| 3 | Consolidate chat.py -> ChatService | 50 | 30-60 min | Phase 2 #3 (sanitizer) |
| 4 | ASGI middleware rewrite | 40 | 30 min | -- |

### Phase 4: Polish & Monitoring -- ~2 hours

| # | Change | LOC | Time |
|---|--------|-----|------|
| 1 | CUDA cleanup interval tuning (10 -> 50, pressure-based) | 15 | 30 min |
| 2 | torch.compile VRAM monitoring / mode switch | 10 | 30 min |
| 3 | Fix synthesize_stream threading | 10 | 30 min |
| 4 | Health check liveness/readiness split | 25 | 15 min |
| 5 | Graceful shutdown signal handling | 15 | 15 min |
| 6 | GitHub Actions CI enhancement | 30 | 30 min |
| 7 | Sentence buffer optimization | 5 | 10 min |
| 8 | Reuse httpx client for health checks | 5 | 10 min |
| 9 | Non-streaming TTS fire-and-forget | 1 | 2 min |

### Phase 5: Future Considerations (separate sprints)

| # | Change | Notes |
|---|--------|-------|
| 1 | Memory API endpoints (`/api/memory`) | Sprint Goal #3 |
| 2 | Web search re-enablement (async + SSRF) | Needs security review |
| 3 | WebSocket voice endpoint | Feature, not optimization |
| 4 | Embedding model upgrade (nomic-embed-text-v1.5) | Requires full re-embedding |
| 5 | API authentication | When LAN access needed |
| 6 | Docker/Docker Compose | When deploying elsewhere |
| 7 | LLM runner evaluation | OPTIMIZATION_ROADMAP.md Day 8 |

---

## Cross-Domain Constraints

Implementation ordering requirements from multi-specialist debate:

1. **ID fix -> Fire-and-forget**: Count-based IDs must be fixed before enabling concurrent writes via `asyncio.create_task()`.
2. **HybridFactExtractor -> Idle gating**: Background LLM extraction MUST gate on idle. Ollama serializes inference -- 1-3s latency otherwise.
3. **DreamingEngine -> Shutdown limit**: Add `max_shutdown_conversations=10` to prevent 60-120s blocking.
4. **Memory backup -> Before dreaming**: Backup must exist before DreamingEngine writes to the database.
5. **torch.compile in Docker**: Use `compile_mode="default"` (not `"reduce-overhead"`) to avoid CUDA graph VRAM leak (PyTorch #128424, #159669).

---

## What's Already Optimized (Do Not Re-Propose)

- Lazy-loading STT/TTS (2-phase startup)
- Pre-compiled regex patterns at module level
- SSE JSON templates (avoid json.dumps per token)
- Parallel memory tier searches (ThreadPoolExecutor)
- Batch ChromaDB duplicate checking
- TTS queue with overlapped synthesis/playback (SentenceTTSQueue)
- httpx AsyncClient with connection pooling
- CUDA optimizations (torch.compile, cudnn_benchmark, tf32, inference_mode)
- Silero VAD integration
- Adaptive TTS chunking (smaller first chunk for lower TTFB)
- num_ctx reduced to 4096, num_predict to 100
- CUDA sync barrier removed
- Float32 audio path (no int16 round-trip)
- Single GPU thread for TTS (CUDA contention prevention)
- Event-based queue waiting (not polling)
