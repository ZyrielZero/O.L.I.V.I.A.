# O.L.I.V.I.A. Architecture

> **Getting Started?** See the [main README](../README.md) for setup instructions and features.

## Overview

O.L.I.V.I.A. is built with a clean separation between backend and frontend:
- **Backend**: FastAPI server exposing REST/WebSocket APIs
- **Frontend**: Flet desktop UI (with support for other UIs)

This design lets you use O.L.I.V.I.A. through different interfaces — desktop app, web UI, direct API calls, whatever you need.

## System Architecture

```
┌─────────────────────────────────────────┐
│         Flet Desktop UI                 │
│  (src/flet_app/)                        │
│  - Modern Material Design               │
│  - Chat interface with streaming        │
│  - Status indicators                    │
└──────────────┬──────────────────────────┘
               │ HTTP/WebSocket
               │
┌──────────────┴──────────────────────────┐
│       FastAPI Backend                   │
│  (src/api/)                             │
│  - REST API endpoints                   │
│  - Service wrappers                     │
│  - LLM, Memory, STT, TTS                │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│         O.L.I.V.I.A. Core               │
│  (src/core/)                            │
│  - Ollama, ChromaDB, Whisper, etc.     │
└─────────────────────────────────────────┘
```

## Running O.L.I.V.I.A.

### Launch Everything (Recommended)

```bash
python run_olivia.py
```

Starts both the FastAPI backend and Flet UI in one command.

### Launch Components Separately

If you want more control:

**Terminal 1 - Backend:**
```bash
cd src
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Desktop UI:**
```bash
python src/flet_app/main.py
```

### API Only (for development/testing)

```bash
python run_olivia.py --api-only
```

Access interactive API docs at http://localhost:8000/docs

## Directory Structure

```
project-olivia/
├── src/
│   ├── api/                    # FastAPI Backend
│   │   ├── models/            # Pydantic schemas
│   │   ├── routes/            # API endpoints
│   │   ├── services/          # Service wrappers
│   │   ├── utils/             # Utilities
│   │   ├── main.py            # FastAPI app
│   │   ├── config.py          # Configuration
│   │   └── dependencies.py    # Dependency injection
│   │
│   ├── flet_app/              # Flet Desktop UI
│   │   ├── components/        # UI components
│   │   ├── services/          # API client, state
│   │   ├── layouts/           # Layout components
│   │   ├── main.py            # Flet entry point
│   │   ├── app.py             # Main app class
│   │   └── theme.py           # Theme system
│   │
│   ├── core/                  # Core Systems
│   │   ├── llm/              # Language model integration
│   │   ├── speech/           # Speech I/O (STT/TTS)
│   │   ├── memory/           # Memory system
│   │   └── tools/            # Tools (web search, etc.)
│   │
│   ├── config/               # Configuration
│   ├── utils/                # Shared utilities
│   └── legacy/               # Legacy CustomTkinter GUI
│
└── run_olivia.py             # Unified launcher
```

## API Endpoints

### Available Now

- **GET /health** - Health check for all services
- **POST /api/chat** - Text chat with streaming support
  - Non-streaming: Returns complete response
  - Streaming: Server-Sent Events (SSE) stream

### Planned

- **GET /api/memory** - Query memory entries
- **POST /api/memory** - Store conversation
- **GET /api/settings** - Get configuration
- **PUT /api/settings** - Update configuration
- **WebSocket /ws/voice** - Real-time voice streaming

## Current Features

### Backend (FastAPI)

- [x] Service wrappers (LLM, Memory, STT, TTS)
- [x] REST API with streaming support
- [x] Health check endpoint
- [x] CORS middleware
- [x] Logging middleware
- [x] Dependency injection
- [x] Lifespan management
- [x] Web search integration

### Desktop UI (Flet)

- [x] Modern dark theme with purple accents
- [x] Chat interface with message bubbles
- [x] Streaming token display
- [x] Status indicator
- [x] API client (REST)
- [x] State management
- [x] Error handling
- [x] Connection status checking

### In Progress

- [ ] Memory API endpoints
- [ ] Settings API endpoints
- [ ] WebSocket voice streaming
- [ ] AI Orb with animations
- [ ] Sidebar with feature toggles
- [ ] Voice input button
- [ ] Wake word integration
- [ ] Auto-chat mode

## Development

### Testing the API

```bash
# Start API server
python run_olivia.py --api-only

# Visit http://localhost:8000/docs to test endpoints interactively
```

### Testing the UI

```bash
# Make sure backend is running first
python run_olivia.py --ui-only
```

### Configuration

Set these in your `.env` file:
```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=olivia-merged
STT_MODEL_SIZE=small.en
MEMORY_PERSIST_DIR=data/chroma_db
API_HOST=0.0.0.0
API_PORT=8000
```

## Why This Architecture?

**Separation of Concerns**: Backend logic lives separately from UI code. Makes everything easier to test and maintain.

**Multiple Frontends**: The same backend can power desktop, web, mobile — whatever you want to build.

**Better Testing**: You can test the API independently without dealing with UI.

**Flexible Deployment**: Desktop app, web app, or direct API access. Your choice.

**Modern Stack**: FastAPI's async architecture with streaming support. Flet's Flutter-based Material Design.

## Legacy GUI

The original CustomTkinter GUI (`src/legacy/gui_app.py`) is still available:

```bash
python src/legacy/gui_app.py
```

It's preserved for reference and backwards compatibility, but new features are built for the FastAPI + Flet architecture.

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

For deep technical documentation, see:
- [TTS Optimization Guide](technical/TTS_OPTIMIZATION.md) - ChatterBox Turbo optimization details
