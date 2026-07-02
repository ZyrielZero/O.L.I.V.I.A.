# O.L.I.V.I.A. Full Optimization Roadmap

## Context

Despite running on an RTX 4080 SUPER 16GB, O.L.I.V.I.A. has higher latency than expected. Three parallel audits revealed critical bugs, pipeline bottlenecks, dead code, and SOTA upgrade opportunities. This roadmap covers everything — organized into daily chunks that can be referenced session by session.

Current estimated voice-to-voice latency: 2500-4000ms
Target after Day 1-2: ~1200ms
Target after all days: <500ms

---

## Day 1: Critical Bug Fixes & Quick Wins

### 1.1 Fix Broken Memory Service Attributes

- `src/api/services/memory_service.py:121` — change `db.facts_collection` -> `db.facts`
- `src/api/services/memory_service.py:129` — change `db.conversations_collection` -> `db.conversations`
- Verify: start API, send chat, confirm no AttributeError in logs and memory context is returned

### 1.2 Remove CUDA Sync Barrier

- `src/api/routes/chat.py:87-103` — remove `_sync_cuda()` function entirely
- `src/api/routes/chat.py:~221` — remove the `await _sync_cuda()` call
- `src/api/services/chat_service.py:197-212` — remove `_sync_cuda()` function
- `src/api/services/chat_service.py:~293` — remove the `await _sync_cuda()` call
- Remove unused torch imports if no other usage remains
- Verify: send multiple chat messages, confirm no 50-500ms gap before TTS

### 1.3 Reduce Context Window

- `src/core/llm/ollama_client.py:30` — change `"num_ctx": 8192` -> `"num_ctx": 4096`
- Verify: chat still works, responses are contextually appropriate

### 1.4 Reduce num_predict

- `src/core/llm/ollama_client.py:31` — change `"num_predict": 150` -> `"num_predict": 100`
- Verify: Olivia responses still complete properly (she targets <50 words anyway)

### 1.5 Dead File Cleanup

- Delete `src/config/theme.py` (dead CustomTkinter theme, never imported)
- Delete `nul` in project root (0-byte Windows artifact)
- Delete `voice_reference.processed.wav` in root (auto-generated, can be regenerated)
- Move `src/core/speech/wake_word_enhanced.py` -> `src/experimental/speech/wake_word_enhanced.py`
- Create `src/experimental/speech/__init__.py` if needed

### 1.6 Dependency Cleanup

- Remove `customtkinter` from requirements.txt (legacy only)
- Remove `pillow` from requirements.txt (not imported anywhere)
- Remove `pystray` from requirements.txt (never integrated)
- Remove `openwakeword` from requirements.txt (wake word not integrated)
- Verify: `pip install -r requirements.txt` still works, `python run_olivia.py` starts fine

### 1.7 Ollama Environment Tuning (no code changes)

- Document/create a startup script or .env addition:
  ```
  OLLAMA_KV_CACHE_TYPE=q8_0
  OLLAMA_NUM_PARALLEL=1
  OLLAMA_KEEP_ALIVE=24h
  OLLAMA_FLASH_ATTENTION=1
  ```
- Restart Ollama with these vars set
- Verify: `nvidia-smi` shows lower VRAM usage for LLM

### Day 1 Verification

- Run `pytest tests/ -m "not slow" -v --tb=short` — all pass
- Run `ruff check --fix && ruff format` — clean
- Run `python run_olivia.py --api-only` — starts without errors
- Send test chat message — memory context appears, response is fast

---

## Day 2: Silero VAD Integration

### 2.1 Add Silero VAD Dependency

- Add silero-vad (or use `torch.hub` load) to requirements
- Verify Silero VAD model loads (~1.8MB, runs on CPU)

### 2.2 Replace Amplitude VAD in ContinuousSTT

- `src/core/speech/stt.py` — load Silero VAD model in `__init__`
- Replace `np.abs(indata).mean()` check (line ~317) with Silero VAD inference
- Change `silence_duration` default from 1.5 -> 0.4 (line 191)
- Keep `vad_threshold` parameter but repurpose for Silero confidence threshold (~0.5)
- Keep amplitude check as a fast pre-filter (skip Silero if volume is clearly silence)
- Ensure Silero runs on CPU to avoid GPU contention with LLM/TTS

### 2.3 Test VAD Integration

- Test with quiet room — no false triggers
- Test with speech — reliable detection, fast end-of-speech recognition
- Test with background noise — no phantom speech detection
- Measure time from user stops speaking to transcription start (target: <500ms)
- Run `/test-voice` command to validate

### Day 2 Verification

- Voice interaction works end-to-end
- Perceived latency noticeably reduced (1.1s savings from VAD alone)
- Run `pytest tests/ -m "not slow" -v --tb=short` — all pass

---

## Day 3: LLM->TTS Streaming Pipeline

### 3.1 Feed LLM Tokens into TTS During Streaming

- `src/api/services/chat_service.py` — modify `speak_response()` to accept a token stream instead of a complete string
- Implement sentence buffering: accumulate tokens until sentence boundary (`.`, `!`, `?`, `\n`)
- Feed each complete sentence to `SentenceTTSQueue` immediately
- The existing `tts_queue.py` already has the right architecture — wire it up
- First sentence plays while LLM is still generating the rest

### 3.2 Update SSE Chat Route

- `src/api/routes/chat.py` — during SSE streaming loop, simultaneously:
  - Yield SSE token events to the UI
  - Accumulate sentences and feed to TTS queue
- Ensure TTS doesn't block SSE streaming (both run concurrently)

### 3.3 Handle Edge Cases

- Short responses (<1 sentence): flush buffer after LLM completes
- TTS sanitizer runs on each sentence before queuing
- Barge-in: if user starts speaking, cancel pending TTS queue
- Error recovery: if TTS fails on one sentence, continue with next

### Day 3 Verification

- Send a long question — first sentence of response plays before LLM finishes
- UI still shows streaming tokens simultaneously
- Short responses still work correctly
- Run `pytest tests/ -m "not slow" -v --tb=short` — all pass

---

## Day 4: Audio Pipeline & ChatterBox Optimization

### 4.1 Persistent Audio OutputStream

- `src/api/services/tts_service.py` — replace per-chunk `sd.play()`/`sd.wait()` with persistent `sd.OutputStream`
- Implement ring buffer that TTS chunks feed into
- Eliminate inter-chunk gaps (clicks, stutters)
- Handle stream lifecycle (start on first chunk, close after silence timeout)

### 4.2 Optimize Audio Format Conversions

- `src/api/services/tts_service.py` — eliminate unnecessary float32->int16->float32 round-trips
- Keep internal audio as float32, only convert at the final output stage
- Reduce memory allocations in the audio callback path

### 4.3 ChatterBox Tuning

- Verify `torch.compile(mode="reduce-overhead")` warmup actually triggers compilation
- Reduce `crossfade_ms` from 10 to 5 (less post-processing overhead)
- Verify streaming chunk sizes are optimal (`first_chunk_tokens=30`, `subsequent=50`)
- Ensure `memory_cleanup_interval=10` is working (since we removed manual CUDA sync)

### Day 4 Verification

- Audio playback is smooth, no gaps between chunks
- TTS latency improved (measure TTFB)
- No audio artifacts or clicks
- Run full test suite

---

## Day 5: Embedding Model Upgrade

### 5.1 Replace all-MiniLM-L6-v2

- `src/core/memory/smart_memory.py:55` — change model to `bge-small-en-v1.5`
  - Same 384 dimensions, similar speed, ~25% better retrieval quality
- Update any other references to the embedding model name
- Note: this requires re-embedding ALL existing vectors (incompatible vector spaces)

### 5.2 Re-embed Existing Data

- Write a migration script that:
  - Reads all documents from each collection (facts, conversations, summaries)
  - Deletes and recreates collections with new embedding function
  - Re-inserts all documents (respect ~5,400 batch limit)
- Back up `data/memory_db/` before running
- Run migration and verify collection counts match

### 5.3 Test Memory Quality

- Query known facts — results should be more relevant
- Test conversation context retrieval
- Verify no data loss after migration

### Day 5 Verification

- All memory collections have same document counts as before
- Memory queries return better-ranked results
- Run `pytest tests/ -m "memory" -v --tb=short`

---

## Day 6: Architecture Hardening

### 6.1 Fix time.time() in Middleware

- `src/api/middleware.py:17,22` — change `time.time()` -> `time.perf_counter()`
- Minor but correct for a performance-critical system

### 6.2 Optimize Memory Service (Fix Dead Code Path)

- `src/api/services/memory_service.py` — the optimized `query_memory()` method is never called
- Wire it up properly or remove it and ensure `get_relevant_context()` uses parallel queries
- Verify memory retrieval uses `asyncio.gather()` for parallel collection queries

### 6.3 Add Health Metrics

- Add latency tracking to the voice pipeline (STT time, LLM TTFT, TTS TTFB, total)
- Log these metrics on each interaction for ongoing monitoring
- Add to `/health` endpoint: average latencies

### 6.4 Connection Pool Tuning

- `src/core/llm/ollama_client.py:65-68` — adjust httpx limits if benchmarks show contention
- May not need changes for single-user, but measure first

### Day 6 Verification

- Latency metrics appear in logs
- `/health` endpoint shows timing data
- No performance regressions

---

## Day 7: WebSocket Voice Endpoint

### 7.1 Implement /ws/voice Endpoint

- `src/api/routes/` — create `voice.py` with WebSocket endpoint
- Bidirectional: client streams audio chunks, server streams audio responses
- Integrate with Silero VAD (server-side) for end-of-speech detection
- Integrate with the full pipeline: VAD -> STT -> LLM (streaming) -> TTS (streaming) -> audio back

### 7.2 Barge-In Support

- Detect user speech while TTS is playing
- Cancel current TTS playback and LLM generation
- Start new STT capture immediately
- This is critical for natural conversation flow

### 7.3 Update Flet UI

- `src/flet_app/` — add WebSocket client for voice mode
- Voice button: hold-to-talk or continuous listening mode
- Visual feedback during speech detection, processing, and response

### Day 7 Verification

- WebSocket voice endpoint accepts audio and returns audio
- Barge-in works: speaking interrupts current response
- Flet UI can use voice mode via WebSocket

---

## Day 8: LLM Runner Evaluation

### 8.1 Benchmark Ollama vs llama.cpp Server

- Install llama.cpp server (`llama-server`)
- Run same GGUF model with identical parameters
- Measure: time-to-first-token, tokens/sec, total response time
- Compare with Ollama numbers

### 8.2 Test Speculative Decoding (llama.cpp only)

- Download a small draft model (Qwen3-0.6B GGUF)
- Run `llama-server --model-draft` with draft model
- Measure token generation speedup (expected: 2-2.5x)
- Verify output quality is identical to non-speculative

### 8.3 Decision: Switch or Stay

- If llama.cpp is significantly faster: update OllamaClient to use llama.cpp's OpenAI-compatible API
  - Minimal code changes — same `/v1/chat/completions` format
  - Remove Ollama-specific code paths
- If difference is marginal: stay with Ollama for convenience

### Day 8 Verification

- Benchmark results documented
- If switched: all chat functionality works through new backend
- Run full test suite

---

## Day 9: Advanced Voice Pipeline

### 9.1 Speculative STT Trigger

- Start LLM call with partial transcript while user is still speaking
- When final transcript arrives, compare with partial:
  - If same: continue LLM generation (saves 200-400ms)
  - If different: restart with correct transcript
- Configurable: can be toggled off if it causes issues

### 9.2 Persistent TTS Process (GPU Isolation)

- Consider running ChatterBox in a separate process with its own CUDA context
- Prevents VRAM fragmentation between STT and TTS
- Could use a lightweight gRPC or Unix socket interface
- Evaluate if the process isolation overhead is worth the GPU scheduling benefit

### 9.3 Wake Word Integration

- Move `wake_word_enhanced.py` from experimental back to core (if ready)
- Integrate with the WebSocket voice pipeline
- Support always-listening mode with wake word trigger

### Day 9 Verification

- Speculative STT measurably reduces perceived latency
- Wake word triggers reliably
- System is stable under continuous use

---

## Day 10: Memory System Upgrades

### 10.1 Integrate Experimental Memory Systems

- Wire DreamingEngine (`src/experimental/memory/dreaming.py`) into app lifecycle
  - Trigger during idle periods (using IdleDetector)
  - Consolidate conversation memories into summaries
- Wire HybridFactExtractor (`src/experimental/memory/fact_extractor.py`)
  - Extract facts from conversations in real-time
  - Feed into facts collection automatically

### 10.2 Memory API Endpoints

- `GET /api/memory` — query memory entries (facts, conversations, summaries)
- `POST /api/memory` — manually store a memory
- `DELETE /api/memory/{id}` — remove a memory
- `GET /api/memory/stats` — collection sizes and health

### 10.3 Evaluate ChromaDB -> LanceDB Migration

- Benchmark ChromaDB vs LanceDB for O.L.I.V.I.A.'s workload
- LanceDB advantages: embedded (no batch limits), disk-based (lower RAM), lighter deps
- If beneficial: plan migration (separate sprint)

### Day 10 Verification

- Dreaming runs during idle periods, creates summaries
- Fact extraction works on new conversations
- Memory API endpoints work in Swagger UI
- Run `pytest tests/ -m "memory" -v --tb=short`

---

## Running Checklist Summary

| Day | Focus | Est. Time | Impact |
|-----|-------|-----------|--------|
| 1 | Bug fixes, cleanup, Ollama tuning | 2-3 hrs | -1000ms (bugs + num_ctx) |
| 2 | Silero VAD | 2-3 hrs | -1100ms (silence_duration) |
| 3 | LLM->TTS streaming | 3-4 hrs | -500 to -2000ms (pipeline overlap) |
| 4 | Audio pipeline & ChatterBox | 2-3 hrs | -50 to -200ms (smoother audio) |
| 5 | Embedding model upgrade | 2-3 hrs | Better memory quality |
| 6 | Architecture hardening | 2-3 hrs | Metrics + correctness |
| 7 | WebSocket voice endpoint | 4-5 hrs | Real-time bidirectional voice |
| 8 | LLM runner evaluation | 3-4 hrs | Potential -200 to -500ms |
| 9 | Advanced voice pipeline | 4-5 hrs | -200 to -400ms + wake word |
| 10 | Memory system integration | 4-5 hrs | Sprint goal #1 completion |

Cumulative latency reduction estimate: 2500-4000ms -> <500ms voice-to-voice

---

## Key Files Reference

| File | Role |
|------|------|
| `src/api/services/memory_service.py` | Memory query service |
| `src/api/routes/chat.py` | SSE chat endpoint |
| `src/api/services/chat_service.py` | Chat business logic |
| `src/core/llm/ollama_client.py` | Ollama client |
| `src/core/speech/stt.py` | STT + VAD |
| `src/core/speech/chatterbox_tts.py` | ChatterBox TTS engine |
| `src/api/services/tts_service.py` | TTS service wrapper |
| `src/api/services/tts_queue.py` | Sentence-based TTS queue |
| `src/core/memory/smart_memory.py` | ChromaDB memory system |
| `src/api/middleware.py` | Logging middleware |
| `src/experimental/speech/wake_word_enhanced.py` | Wake word detection |
