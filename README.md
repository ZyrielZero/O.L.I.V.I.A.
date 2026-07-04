# O.L.I.V.I.A.

**O**ffline **L**ocal **I**ntelligent **V**oice **I**nteractive **A**ssistant

[![CI](https://github.com/ZyrielZero/O.L.I.V.I.A./actions/workflows/ci.yml/badge.svg)](https://github.com/ZyrielZero/O.L.I.V.I.A./actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

A voice AI companion that runs entirely on your own hardware. No cloud APIs, no
telemetry, no subscription — your conversations never leave your machine.

## Features

- **Full-duplex voice** — speak and listen over a single `/ws/voice` WebSocket:
  client mic audio in, streaming STT → LLM → TTS audio out. Audio lives on the
  client side, so the backend can run on a home-server GPU.
- **Barge-in** — interrupt her mid-sentence and she stops talking.
  *Currently threshold-based: headphones recommended; acoustic echo
  cancellation is planned.*
- **Persistent memory** — ChromaDB-backed long-term memory with background
  consolidation ("dreaming"), automatic fact extraction from conversations,
  and TTL pruning. Manageable over a REST API.
- **Streaming text chat** — Server-Sent Events when you'd rather type.
- **Runtime settings API** — adjust behavior without restarts.
- **Desktop UI** — Flet (Flutter-based) chat interface with mic capture.
- **Latency accountability** — rolling voice-pipeline metrics in `/health`,
  a benchmark harness (`tools/bench.py`), and committed benchmark history in
  `benchmarks/results/`.

## Stack

| Layer | Tech |
|---|---|
| LLM | [Ollama](https://ollama.com/) (local models, streaming) |
| STT | faster-whisper + Silero VAD |
| TTS | ChatterBox (streaming, voice-cloned) |
| Memory | ChromaDB + sentence-transformers |
| Backend | FastAPI (REST + WebSocket) |
| UI | Flet desktop app |

```
Flet UI / any client ──HTTP+WS──▶ FastAPI backend ──▶ Ollama · Whisper · ChatterBox · ChromaDB
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full picture and
[docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) for where this is going.

## Requirements

- Windows 11 + NVIDIA GPU (developed on an RTX 4080 SUPER; CUDA 12.8 wheels —
  the pinned stack also covers RTX 50xx/Blackwell)
- Python 3.10+ (developed on 3.11)
- [Ollama](https://ollama.com/) installed and running

## Quickstart

```bash
git clone https://github.com/ZyrielZero/O.L.I.V.I.A..git
cd O.L.I.V.I.A.
python -m venv .venv && .venv\Scripts\activate

# Two-step install (chatterbox-tts hard-pins torch and must skip deps)
pip install -r requirements.txt
pip install chatterbox-tts==0.1.6 --no-deps

# Configuration
copy .env.example .env                                        # then edit
copy config\character.template.yaml config\character.yaml     # your persona

# Pull a model and point OLLAMA_MODEL at it in .env
ollama pull llama3.1:8b

# Run (backend + desktop UI)
python run_olivia.py
```

API docs live at http://localhost:8000/docs once it's running. There's also a
console client (`python run_console.py`) and `--api-only` / `--ui-only` flags.

**Private assets:** your persona (`config/character.yaml`) and TTS voice
reference (`assets/voice/reference.wav`) are gitignored by design — the repo
ships templates only. See [docs/PRIVATE_ASSETS.md](docs/PRIVATE_ASSETS.md).

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -m "not slow and not gpu"   # fast suite (what CI runs)
ruff check src/ tests/                    # lint; Google-style docstrings enforced
python tools/bench.py                     # voice-pipeline benchmarks (GPU machine)
```

CI runs tests + coverage (ratchet only goes up), lint, CPU micro-benchmarks,
and a pip-audit security gate. Benchmark results are committed to
`benchmarks/results/` — performance claims come with receipts.

Model fine-tuning and merge experiments live in
[`tools/model_engineering/`](tools/model_engineering/README.md).

## License

[MIT](LICENSE)
