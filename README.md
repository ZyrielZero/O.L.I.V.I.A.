# O.L.I.V.I.A. Claude Code Setup

Ready-to-use Claude Code configuration for the O.L.I.V.I.A. project.

**Repository:** https://github.com/ZyrielZero/project-olivia

## What's Included

```
olivia-claude-setup/
├── CLAUDE.md                    # Main project context
├── .mcp.json                    # MCP server configuration
├── .claude/
│   ├── settings.json            # Permissions & hooks
│   ├── agents/
│   │   ├── memory-architect.md  # Memory + Dreaming + Instincts
│   │   ├── voice-engineer.md    # STT + TTS
│   │   ├── llm-specialist.md    # Ollama + prompts
│   │   ├── flet-developer.md    # Desktop UI
│   │   ├── fastapi-developer.md # Backend API
│   │   ├── personality-tester.md # Personality compliance
│   │   └── humanizer.md         # Code humanization
│   ├── commands/
│   │   ├── dream.md             # Trigger dreaming
│   │   ├── test-voice.md        # Voice pipeline test
│   │   └── personality-check.md # Personality validation
│   └── skills/
│       ├── memory-systems/      # ChromaDB patterns
│       ├── dreaming-system/     # Memory consolidation
│       ├── fact-extraction/     # LLM fact extraction
│       ├── character-personality/ # Character guide
│       └── voice-processing/    # STT/TTS config
└── README.md                    # This file
```

## Quick Install

### 1. Extract to Project Root

```bash
# From your project-olivia directory
unzip olivia-claude-setup.zip
cp -r olivia-claude-setup/* .
```

Or manually copy:
- `CLAUDE.md` → project root
- `.mcp.json` → project root
- `.claude/` → project root

### 2. Start Claude Code

```bash
cd project-olivia
claude
```

### 3. Verify Setup

```
> What project am I in?
# Should describe O.L.I.V.I.A. with body metaphor architecture

> Use @memory-architect to explain the dreaming system
# Should describe DreamingEngine, IdleDetector, integration status

> /personality-check
# Should run personality compliance tests
```

## Available Agents

| Agent | Use For |
|-------|---------|
| `@memory-architect` | ChromaDB, dreaming, fact extraction |
| `@voice-engineer` | STT, TTS, streaming pipeline |
| `@llm-specialist` | Ollama, prompts, fine-tuning |
| `@flet-developer` | Flet UI, async patterns |
| `@fastapi-developer` | REST API, WebSocket |
| `@personality-tester` | Personality compliance testing |
| `@humanizer` | Transform AI code to human-like |

## Available Commands

| Command | Description |
|---------|-------------|
| `/dream` | Trigger dreaming manually |
| `/test-voice` | Test voice pipeline latency |
| `/personality-check` | Run personality compliance tests |

## Key Technical Notes

### Memory System
- **Three collections**: facts (permanent), conversations (7 days), summaries (1 year)
- **ChromaDB batch limit**: ~5,400 per add()
- **Memory injection**: Via system message, NOT history
- **Experimental features** in `src/experimental/memory/` need integration

### Voice Pipeline
- **beam_size=5** always for faster-whisper
- **TTS sanitizer** removes [MEMORY], ###, *actions*
- **Sentence buffer**: min 6 words, max 30 words

### Personality
- Warm but direct, concise
- No emojis, no *asterisks*
- Target: <50 words, max 1 question

## Customization

### Personal Settings (gitignored)

Create `CLAUDE.local.md` for personal preferences:
```markdown
# Personal Preferences
- Prefer verbose explanations
- Always run tests before committing
```

Create `.claude/settings.local.json` for personal permissions.

### Adding MCP Servers

Edit `.mcp.json` to add servers. Keep under 10 total.

### Adding Agents

Create new `.md` files in `.claude/agents/` following existing patterns.

### Adding Skills

Create new directories in `.claude/skills/` with `SKILL.md` files.

## Current Sprint Focus

1. **Integrate experimental memory** → DreamingEngine + HybridFactExtractor
2. **Complete QLoRA fine-tuning** → olivia-finetuned model
3. **Memory API endpoints** → `/api/memory`

## Need Help?

- Check `OLIVIA_MASTER_DOCUMENTATION.md` in project knowledge
- Use `@memory-architect` for memory questions
- Use `@voice-engineer` for audio questions
- Use `@llm-specialist` for model questions
