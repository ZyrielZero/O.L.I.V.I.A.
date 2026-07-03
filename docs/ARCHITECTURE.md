# O.L.I.V.I.A. Architecture

> **Getting Started?** See the [main README](../README.md) for setup instructions and features.

## Overview

O.L.I.V.I.A. is built with a clean separation between backend and frontend:
- **Backend**: FastAPI server exposing REST/WebSocket APIs
- **Frontend**: Flet desktop UI (with support for other UIs)

This design lets you use O.L.I.V.I.A. through different interfaces — desktop app, web UI, direct API calls, whatever you need. Audio capture and playback live on the client side of `/ws/voice`, so the backend can run on a different machine than the microphone.

## System Architecture

```
┌─────────────────────────────────────────┐
│         Flet Desktop UI                 │
│  (src/flet_app/)                        │
│  - Chat interface with streaming        │
│  - Mic capture -> /ws/voice             │
│  - Status indicators                    │
└──────────────┬──────────────────────────┘
               │ HTTP + WebSocket
               │
┌──────────────┴──────────────────────────┐
│       FastAPI Backend                   │
│  (src/api/)                             │
│  - REST + WebSocket endpoints           │
│  - Service container (DI)               │
│  - LLM, Memory, STT, TTS services       │
│  - Background memory (dreaming +        │
│    fact extraction) in the lifespan     │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│         O.L.I.V.I.A. Core               │
│  (src/core/)                            │
│  - Ollama client (LLM)                  │
│  - ChromaDB memory + dreaming           │
│  - faster-whisper STT, ChatterBox TTS   │
└─────────────────────────────────────────┘
```

## Running O.L.I.V.I.A.

### Launch Everything (Recommended)

```bash
python run_olivia.py
```

Starts both the FastAPI backend and Flet UI in one command.
Flags: `--api-only`, `--ui-only`, `--no-reload`.

Access interactive API docs at http://localhost:8000/docs

There is also a console client: `python run_console.py`.

## Directory Structure

```
project-olivia-ai/
├── src/
│   ├── api/                   # FastAPI Backend
│   │   ├── models/            # Pydantic schemas
│   │   ├── routes/            # chat, health, memory, settings, voice
│   │   ├── services/          # Service wrappers + audio output/queue
│   │   ├── utils/             # Utilities
│   │   ├── main.py            # FastAPI app + lifespan
│   │   ├── container.py       # Service container
│   │   ├── config.py          # Configuration
│   │   └── dependencies.py    # Dependency injection
│   │
│   ├── flet_app/              # Flet Desktop UI
│   │   ├── components/        # UI components
│   │   ├── services/          # API client, voice client, state
│   │   ├── utils/             # UI utilities
│   │   ├── main.py            # Flet entry point
│   │   ├── app.py             # Main app class
│   │   └── theme.py           # Theme system
│   │
│   ├── core/                  # Core Systems
│   │   ├── llm/               # Ollama integration
│   │   ├── speech/            # Speech I/O (STT/TTS)
│   │   └── memory/            # ChromaDB storage, dreaming, fact extraction
│   │
│   ├── experimental/          # Unwired experiments (enhanced wake word)
│   ├── config/                # Configuration
│   └── utils/                 # Shared utilities
│
├── tools/                     # bench.py, bench_compare.py, model_engineering/
├── benchmarks/results/        # Committed benchmark history
├── tests/                     # Pytest suite (unit, integration, smoke, benchmarks)
└── run_olivia.py              # Unified launcher
```

## API Endpoints

- **GET /health** — full health check for all services; includes rolling voice-pipeline latency metrics
- **GET /health/live** — liveness probe
- **POST /api/chat** — text chat; non-streaming or Server-Sent Events stream
- **DELETE /api/history** — clear conversation history
- **GET/POST /api/memory**, **DELETE /api/memory/{entry_id}**, **GET /api/memory/stats** — memory management
- **GET/PUT /api/settings** — runtime settings
- **WebSocket /ws/voice** — full-duplex voice pipeline (client mic audio in, STT → LLM → TTS audio out, barge-in)

## Current Features

### Backend (FastAPI)

- Service container with dependency injection and lifespan management
- Streaming chat (SSE) and full-duplex voice over WebSocket
- Persistent audio output stream with ring buffer, session-scoped TTS queue
- Background memory systems: dreaming engine (idle consolidation) and hybrid fact extractor
- Memory TTL pruning scheduled at startup + daily
- Health checks with rolling latency metrics

### Desktop UI (Flet)

- Chat interface with streaming token display
- Mic capture wired to the voice WebSocket
- Status indicator and connection checking
- State management and error handling

### Known Limitations / Planned

- Barge-in is threshold-based — headphones recommended; acoustic echo cancellation (AEC) is planned
- Forward direction lives in [MASTER_PLAN.md](MASTER_PLAN.md)

## Development

### Testing

```bash
pytest tests/ -m "not slow and not gpu"   # fast suite (what CI runs)
ruff check src/ tests/                    # lint (Google-style docstrings enforced)
python tools/bench.py                     # voice-pipeline benchmarks (GPU machine)
```

### Configuration

Copy `.env.example` to `.env` and adjust. Key settings:

```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=olivia-finetuned
STT_MODEL_SIZE=small.en
MEMORY_PERSIST_DIR=data/memory_db
HOST=127.0.0.1
PORT=8000
```

The persona config is private: copy `config/character.template.yaml` to
`config/character.yaml` (gitignored) and fill in your own character.

## Why This Architecture?

**Separation of Concerns**: Backend logic lives separately from UI code. Makes everything easier to test and maintain.

**Multiple Frontends**: The same backend can power desktop, web, mobile — whatever you want to build.

**Client-side audio**: `/ws/voice` carries the audio, so the backend can live on a home server GPU while clients connect over the LAN.

**Better Testing**: You can test the API independently without dealing with UI.

**Modern Stack**: FastAPI's async architecture with streaming support. Flet's Flutter-based Material Design.

## Troubleshooting

**Backend won't start**
- Is Ollama running? Check with `ollama list`
- Is port 8000 available?
- Check logs for service initialization errors

**UI can't connect**
- Make sure backend is running first
- Check http://localhost:8000/health in browser
- Verify firewall isn't blocking port 8000

**Streaming not working**
- Check browser console for errors
- Verify FastAPI streaming endpoint works: `curl -N http://localhost:8000/api/chat -d '{"message":"test","stream":true}'`

## Technical Details

For deep technical documentation, see [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).
