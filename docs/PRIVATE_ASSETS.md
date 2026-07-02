# Private Assets

This repo ships the O.L.I.V.I.A. engine; the persona, voice, and training data are
**private, gitignored, per-machine assets**. A fresh clone will not run until you
create them. This page is the reconstitution contract.

| Path | What it is | How to create it |
|------|-----------|------------------|
| `config/character.yaml` | Persona configuration (identity, speaking style, TTS params) | Copy `config/character.template.yaml` and customize |
| `assets/voice/reference.wav` | Voice-clone reference clip for TTS (never publish — it lets anyone clone that voice) | Record ~5-15s of clean speech; set its transcript as `tts.reference_text` in `character.yaml` |
| `models/ollama/Modelfile.*` | Ollama model definitions embedding the persona system prompt | Copy `config/Modelfile.template`, then `ollama create <name> -f <file>` |
| `models/checkpoints/` | Fine-tuned model weights | Train with `tools/model_engineering/`, or use a stock Ollama base model |
| `data/training/*.jsonl` | Personal fine-tuning conversation data | Generate with `tools/generate_training_data.py` / your own data |
| `.env` | Environment config (tokens, device settings) | Copy `.env.example` and fill in values |
| `data/memory_db/` | ChromaDB long-term memory | Created automatically on first run |

## Provenance

This repository's history was re-initialized on 2026-07-02 (master plan Phase -1) so that
no private asset ever appears in a public commit. The prior history (through commit
`e090062`) is archived locally at `.git-archive/` and in the private GitHub repo
`ZyrielZero/O.L.I.V.I.A.`. Do not push this repository to that remote.
