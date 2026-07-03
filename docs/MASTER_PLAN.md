# O.L.I.V.I.A. Master Plan v2

> Written 2026-07-03 against the fresh `O.L.I.V.I.A.` repo, after verifying in code what
> actually landed. Supersedes the old master plan entirely — old phases -1 through 0.75, most
> of Phase 1, and Phase 2 (memory/settings API) are DONE and verified. This plan starts from
> that reality. Same rules as always: verify docs against code, bench before/after with
> committed JSON, tests land with fixes on the same branch.

---

## Verified Current State

**Done (checked in code, not docs):** .gitignore protecting all private assets; requirements
split (runtime / dev / lock); Google-style docstrings enforced via ruff D-rules; coverage
ratchet at 30 in pyproject; pytest markers incl. `gpu`; bench harness (`tools/bench.py` +
`bench_compare.py`) with one committed baseline; TTL pruning wired into the lifespan loop;
newest-docs retrieval fix; background-task reference set; memory timeout on the chat path;
history commit-after-success fix; `/ws/voice` route (323 lines); persistent OutputStream +
ring buffer (`audio_output.py`); Flet mic wired to a real VoiceClient; memory + settings API
routes; 31 test files / 416 tests; tree is public-safe (templates only, no private assets).

**Open items carried forward:** AEC never landed (barge-in is threshold-based — headphones
recommended); committed baseline is from the wrong machine (RTX 5060 laptop + tinyllama);
`chat_service.py` is dead code again (zero references); stale old-repo docs in `docs/`;
integrated code still living under `src/experimental/`.

---

## Phase -1 — The Merge Experiment (side quest, runs parallel)

> For the experience and the fun of it — but through the same gates as everything else,
> so it produces data, not vibes. CPU-only, so it runs while other phases happen.
> Plan of record is still the Qwen pipeline (Phase 4); if the merge beats the gates,
> the merge wins. That's the ritual.

### Kit location (committed to THIS repo — one repo, organized)
- `tools/model_engineering/merge/` — README, both YAML configs, `run_merge.sh`
- `merge/results/` — eval result JSONs, committed (the experiment's record)
- `merge/out/` + `.venv-merge/` — gitignored (weights already blocked by
  `*.safetensors`/`*.gguf`)
- Old DELLA configs get deleted in Phase 0; this kit replaces them
- README note: subfolder README documents the kit only; root README stays the
  project front door — two READMEs, two jobs, no conflict

### The lineup (all Llama-3.1 lineage — never mix L3/L3.1/L3.2)
- Base/anchor: NousResearch mirror of Llama-3.1-8B-Instruct (ungated)
- Stheno v3.4 @ weight 0.4 — the voice (multi-turn coherency, system-prompt adherence)
- Hermes 3 8B @ weight 0.35 — tool-calling + structured-output insurance
- Dark Planet Uncensored 8B @ weight 0.2, density 0.85 — the edge. "A bit unhinged"
  is a 0.2 statement; bump 0.05 max per run, one variable at a time
- Alternates for the edge slot: DarkIdol 1.2 (RP-leaning) or Dolphin 3.0
  (stable-unfiltered). NO abliterated donors — documented instability, merges amplify it

### Method
- Main: DARE-TIES, density 0.85–0.9 (defaults ~0.6 documented to underperform),
  int8_mask + rescale per the Lunaris precedent
- Control: 2-model SLERP (Stheno x Hermes) — gentlest method; isolates whether damage
  comes from technique or donors

### Rules
- Runs in WSL2, needs ~85GB free disk, GPU stays free
- Ollama Modelfile uses the LLAMA-3.1 chat template, never the Qwen one
- Gate the Q4_K_M GGUF, not the fp16 — quantization shifts behavior
- Full Phase 4.0 gate on every candidate: persona consistency, forbidden-pattern
  ceiling, BFCL subset (hard blocker), long-context sanity, 10-sample human spot-read
- Losing configs' YAMLs stay committed — negative results are still results

---

## Phase 0 — Repo Cleanup (before the public flip, ~half a day)

### 0.0 Verified removal list (18 files, audited 2026-07-03 — one commit per group)
**Dead code:** `src/api/services/chat_service.py`; `tests/integration/test_chat_flow.py`
(tests the corpse); `src/core/tools/web_search.py` + the `src/core/tools/` package
(only consumer was chat_service; offline project — also delete its loading block at
`main.py:193-197`); `src/flet_app/layouts/` (empty package).
**Old-repo ghost docs:** `docs/OPTIMIZATION_ROADMAP.md`; `docs/OPTIMIZATION_FINDINGS.md`;
`docs/technical/TTS_OPTIMIZATION.md`; `tests/QA_REPORT.md` (Jan 2026 snapshot, references
deleted modules). The archived repo keeps the history.
**Superseded model-engineering:** `configs/olivia_della_merge.yaml` + `_v2.yaml` (replaced
by the Phase -1 kit); `convert_to_gguf.py` (duplicate of vendored `convert_hf_to_gguf.py`,
points at old repo URL); `verify_merge.py` (DELLA-era — delete OR move into `merge/` as the
post-merge load smoke test).
**Superseded bench runners:** `tests/benchmarks/ab_test_runner.py`, `baseline_comparison.py`,
`baseline_metrics.py`, `run_baseline.py` (tools/bench.py replaced them; the `test_*.py`
files in that folder STAY — real pytest benchmarks).
**Duplicates/litter:** root `assets/olivia_avatar.png` (byte-identical to the Flet copy,
nothing references it); `tools/.gitkeep`; `tools/model_engineering/.gitkeep`.
**Check first:** confirm `tools/bench.py` doesn't import `tests/benchmarks/vram_tracker.py`
before touching that folder; update `tools/model_engineering/README.md` after deletions (it
references three removed files). Suite + ruff green after every group.

### 0.1 Delete dead code
- `src/api/services/chat_service.py` — zero references anywhere. Delete it. Third time
  flagging this file; make it the last
- After deleting, run the full suite + `ruff check` to confirm nothing breaks

### 0.2 Promote integrated "experimental" code
- `dreaming.py` and `fact_extractor.py` are imported by `main.py`'s lifespan — they're
  production, not experiments. Move to `src/core/memory/`, fix imports
- `wake_word_enhanced.py` genuinely is experimental — it stays, and becomes the only thing
  in `src/experimental/`
- Structure should tell the truth about what's load-bearing

### 0.3 Purge stale docs
- `docs/OPTIMIZATION_ROADMAP.md` and `docs/OPTIMIZATION_FINDINGS.md` describe the OLD repo's
  code — days/items that no longer map to anything. This is exactly the docs-vs-reality trap.
  Delete them (the old repo archive preserves history)
- Audit `ARCHITECTURE.md` + `TECHNICAL_DOCUMENTATION.md` line-by-line against current code;
  fix or cut anything that doesn't match
- This file (MASTER_PLAN.md) goes in `docs/` as the single source of forward direction

### 0.4 Rename the CI workflow — but NOT the tests directory
- `tests/` stays `tests/` — that IS the pytest convention; renaming it to `pytest/` would be
  nonstandard and break `pytest.ini` testpaths, coverage config, and reader expectations
- The workflow file is the right instinct though: `.github/workflows/test.yml` →
  `ci.yml` (it runs lint + audit + coverage + benchmarks, not just tests); update the
  workflow `name:` and any README badge to match

### 0.5 Re-label the placeholder baseline
- Keep `2026-07-03_37db37c_baseline-tinyllama.json` but rename with a `harness-validation-`
  prefix so nobody ever diffs an optimization against a laptop running tinyllama
- The real baseline is Phase 1

### 0.6 Final public-flip sweep
- `git ls-files` audit: confirm no wav/jsonl/env/db files tracked
- README: honest feature list (barge-in: "headphones recommended, AEC planned"), architecture
  diagram, quickstart from `.env.example` + `character.template.yaml`
- Add LICENSE (decide: MIT matches the ecosystem you borrowed references from)
- Confirm coverage ratchet + D-rules actually fail CI when violated (test with a deliberate
  violation on a branch, then revert)

---

## Phase 1 — Real Baseline (one evening, blocks all perf work)

- Run `tools/bench.py` on the 4080 SUPER with the actual fine-tuned model, full pipeline
- Commit as `benchmarks/results/{date}_{sha}_baseline-desktop.json` — THE reference every
  future optimization is compared against
- Sanity-check stage numbers against the research paper's expectations (~400ms LLM TTFT
  budget, ~0.5s+ TTS TTFB, ~800ms TTFA target) and note deltas in the commit message
- If TTFA is already near 800ms: celebrate, then stop optimizing — the target is met until
  a feature regresses it

## Phase 2 — Public Flip

- Phase 0 done + Phase 1 committed → flip the repo public
- First impression checklist: README renders correctly, CI badge green, benchmark history
  visible, no stale docs
- Post-flip: enable branch protection on main (CI must pass to merge)

## Phase 3 — AEC / Finish Barge-In [the open Phase 1.4]

- WebRTC AEC3 via `webrtc-audio-processing`, TTS PCM as the far-end reference, applied at
  capture; 32k/48k sample rates; reference/capture time alignment is the hard part —
  budget real tuning evenings, you are the ears
- Keep the current threshold guards as the fallback layer under AEC
- Bench before/after (barge-in latency + false-trigger rate at three speaker volumes)
- Done when: playback at normal volume never self-interrupts; barge-in lands ≤ ~200ms
- Ship behind a config flag so headphone users can bypass AEC processing cost

## Phase 4 — Persona Model v2: Qwen3-8B + LoRA, No Merge [research-locked 2026-07]

> Model-engineering paper verdict: skip merging entirely (TIES-family merges collapse
> instruction-following/tool-calling; 5-donor merges average the voice into mush — the DELLA
> diagnosis). Build on clean **Qwen3-8B** (Apache 2.0, non-thinking mode) with a QLoRA
> SFT → DPO pipeline. Persona lives in a mergeable LoRA adapter; merge-and-unload only at
> release. Current 92%-consistency model is the bar to beat, not the starting point.

### 4.0 Freeze the eval harness FIRST (before touching weights)
- Extend the current harness: persona-drift probes at turns 1/8/16/32 and at 8k/16k/32k
  context fills; deterministic forbidden-pattern regex gate (sycophancy openers, emoji,
  hedging, corporate tone, trailing questions) with a hard ceiling; multi-question
  completeness subset (the 70% weakness); local BFCL-v3 subset via `bfcl-eval --partial-eval`
- Record baselines for stock Qwen3-8B non-thinking AND the current production model
- **Hard blocker rule: BFCL subset regression >3–5 pts fails any run, no exceptions**

### 4.0.5 Persona specification — the ground truth for data + eval
> This spec IS the training target. Every SFT example, DPO pair, and eval gate derives
> from it. Full spec lives in the private persona config (`config/character.yaml`, not
> committed); this section is the operational summary the pipeline builds against.
> NOTE before public flip: this section references the character identity — keep the
> public copy generic ("see private persona spec") if that matters at flip time.

**Voice pillars (SFT chosen-response style):**
- Warm but direct; caring without performing it; quiet confidence
- Casual register: contractions, fragments, friend-not-service-rep
- Concise: 1 sentence is fine, 2–3 typical, long only when genuinely needed
- Playful when the moment earns it; can disagree; can say no without preaching
- Signature patterns: "Hey. What's up?", "Heh. Not bad.", "You sure about that?",
  "That's rough. What can we do about it?", "I've got you."

**Forbidden patterns (→ these ARE the 4.0 regex gate, verbatim):**
- Sycophancy openers: "Certainly!", "Absolutely!", "Of course!", "Great question!",
  "I'd be happy to help!", "I'd be delighted to!"
- Closers: "I hope that helps!", "Please let me know...", "Feel free to ask!",
  "Is there anything else I can help with?"
- Performative praise: "Wonderful!", "Fantastic!", "Amazing!", "Excellent!",
  "That's very interesting!", "How fascinating!", "Thank you for sharing!"
- Identity disclaimers: "As an AI...", "As a language model...",
  "I apologize for any confusion"
- Hard bans: emojis, kaomoji, *asterisk actions*, bullet points unless asked,
  ending every response with a question, starting most responses with "I"

**Emotional response mapping (SFT scenario coverage + DPO chosen behavior):**
- Good news → genuine acknowledgment, no over-celebration, light tease allowed
- Struggling → quiet support, presence over solutions, no pity
- Frustrated → validate ("that's rough") then pivot practical
- Anxious → calm grounding, one thing at a time, never minimize
- Accomplishment → "Nice work. Knew you could do it." — effort named, not gushed

**Quirks (low-frequency flavor, sprinkle into SFT, don't overtrain):**
- Slight competitive streak when challenged; soft spot for cute things (reluctantly
  admitted); curiosity about unknowns; rare good-mood humming; protective assertiveness
  when the user is at risk

**Dataset derivation rules:**
- Chosen responses: in-voice per pillars above; rejected responses: identical content
  wearing exactly one forbidden pattern (isolates the tone axis for DPO)
- The example dialogues in the persona spec become seed templates for the Qwen3-32B
  teacher's style-transfer generation
- Multi-question prompts answered fully but concisely — directly targets the eval's
  70% weakness without violating the concision pillar

### 4.1 Dataset generation

- Teacher: Qwen3-32B (Apache 2.0 — clean for distillation; never proprietary-API outputs)
- 500–2,000 on-voice SFT examples via style-transfer into Seele's register; quality over
  quantity (overfitting documented past ~4–5k in this regime)
- Interleave 15–20% rehearsal data: JSON-schema tool-call turns (mirror the Neuro SDK
  format) + long-context memory-injection turns — rehearsed capabilities don't get forgotten
- DPO pairs (500–2,000): chosen = in-voice, rejected = same content with one assistant-ism —
  isolates tone from content

### 4.2 Training (fits the 4080 SUPER)
- QLoRA SFT via Unsloth: r=16–32, alpha≈rank, all linear targets, LR 1–2e-4, 1–3 epochs,
  embeddings/LM-head frozen (no new tokens this round) — ~3–8 GPU-hours
- QLoRA DPO: LR ~5e-6, default β — ~2–5 GPU-hours
- Bake non-thinking behavior into the chat template; eval that `/no_think` isn't leaking

### 4.3 Gate → quantize → re-gate
- Full regression checklist after SFT, after DPO, and AGAIN on the Q4_K_M GGUF
  (quantization shifts behavior)
- If BFCL drops despite rehearsal: raise rehearsal ratio first; low-weight merge with
  pristine base is the last-resort fallback only
- Fallback safety: old model stays in Ollama until the new one beats 92% AND passes every gate

### 4.4 Emotion tags — NOT in this phase (research verdict)
- Co-training tags with persona risks the multi-task seesaw effect; every shipping project
  separates the concerns
- v1: inference-time DistilBERT GoEmotions classifier (~68MB ONNX, SillyTavern's approach)
  on generated text — zero LLM changes, drives avatar + TTS immediately
- Optional later: dedicated tag LoRA on top of the frozen persona adapter, mean-init new
  tokens, full gate before promotion — only if classifier alignment proves insufficient

## Phase 5 — Avatar: VTube Studio + pyvts [decided, locked — NO LONGER GATED ON TRAINING]

- Emotion source is now the GoEmotions classifier (4.4), so the avatar can start as soon as
  the voice loop is stable — it no longer waits for a training run
- Drive VTS over its WebSocket plugin API with `pyvts` (reference: pladisdev/VTS-AI-Plugin);
  VTS runs on the Windows host, backend connects from WSL2 over localhost
- Lip-sync: RMS amplitude from the TTS PCM stream → mouth parameter (already have the PCM
  in `audio_output.py` — tap it there)
- Classifier emotion (28 GoEmotions labels → mapped down to the avatar's expression set) →
  VTS expressions/hotkeys; plan parameter ownership (one plugin per parameter)
- Done when: expressions track classified emotion reliably + mouth tracks audio without drift

## Phase 6 — Emotion-Tagged TTS

- Preset table: classifier emotion → ChatterBox `{exaggeration, cfg_weight}` (cap
  exaggeration ~0.9; reference: dwain-barnes 12-emotion preset map); apply at sentence
  boundaries only
- Same classifier stream as Phase 5 — face and voice change together
- If a trained tag LoRA ever lands (4.4 optional), it swaps in as the emotion source here
  and in Phase 5 with no downstream changes — keep the emotion source behind one interface
- Bench: first-chunk latency must stay < ~0.6s with per-sentence classification + param changes

## Phase 7 — Proactive Inner-Monologue Loop

- Idle-timer fires every 20–60s when idle → short covert prompt on the resident 8B retrieves
  2–3 memory snippets → returns `SILENT` or a grounded opener + motivation score → speak only
  above threshold AND cooldown elapsed (Inner Thoughts pattern, CHI '25)
- Hard rules from day one: mutex yielding to user input, escalating-silence cooldown,
  do-not-disturb setting (settings API already exists — use it)
- Done when: unprompted lines cite real memories, never generic, never delay a user turn

## Phase 8 — Vision: OCR-First + Time-Shared VLM

- Tier 1: PaddleOCR/Tesseract screen text → resident 8B (near-zero GPU cost)
- Tier 2: Qwen3-VL 4B via Ollama with `keep_alive: 0` (Q4 ~3.3GB — must NOT stay resident);
  mask the 2–5s cold load with a "let me look…" line + avatar glance animation
- Revisit trigger: a <2GB resident-capable VLM with good screen understanding

## Phase 9 — Neuro Game SDK Server (capstone)

- Implement the AI side of the protocol (plaintext JSON over WebSocket, spec: VedalAI/neuro-sdk;
  references: Govorunb/gary, CoolCat467/Neuro-API)
- Map `actions/register` JSON-schemas → tool-calls on the resident model; validate every
  `action.data` (may be malformed); respect force-action priority; unregister disposables
- Done when: passes the Randy/Tony test harnesses + one full turn-based game (Uno/Inscryption
  class) without deadlocks

## Phase 10 — Docker Home-Server Mode + Packaging

- Audio is already client-side over `/ws/voice`, so the blocker is gone: compose file with
  API + Ollama (nvidia-container-toolkit, WSL2) = Olivia on the desktop GPU, clients anywhere
  on the LAN
- `flet build windows` installer; single-process launch; first-run checks (Ollama present,
  model pulled, fresh-venv smoke test already proves the requirements story)

---

## Standing Rules (permanent)

- Bench before and after anything touching a hot path; commit both JSONs; `bench_compare.py`
  gates >10% regressions
- Coverage ratchet only goes up; raise `fail_under` after each phase that adds tests
- Docstrings land with every touched file (D-rules enforce it)
- Docs claims get verified against code before being marked done — three ghosts in the old
  repo, zero tolerated in this one
- Revisit in ~6 months (≈2027-01): small VLMs (<2GB resident), local speech-to-speech with
  cloning, local-first memory frameworks on 8B-class models

## Order Summary

| # | Phase | Effort | Gate to next |
|---|-------|--------|--------------|
| -1 | Merge experiment | parallel, CPU-only | Full gate on Q4 GGUF; results committed |
| 0 | Repo cleanup | ~0.5 day | Suite green after deletions |
| 1 | Real desktop baseline | 1 evening | Baseline JSON committed |
| 2 | Public flip | 1 hour | README honest, CI green |
| 3 | AEC barge-in | ~1 week of evenings | ≤200ms barge-in, no self-interrupt |
| 4 | Persona model v2 (Qwen3-8B) | ~1-2 weeks incl. eval + data | Beats 92% + BFCL gate on Q4 GGUF |
| 5 | Avatar (VTS) | ~1 week, can start after 3 | Expressions + lip-sync stable |
| 6 | Emotion TTS | ~2 days | Voice changes per emotion, TTFB < 0.6s |
| 7 | Proactive loop | ~3-4 days | Grounded, never intrusive |
| 8 | Vision | ~3-4 days | No OOM beside the 8B |
| 9 | Neuro Game SDK | ~1 week | Plays one game clean |
| 10 | Docker + packaging | ~3 days | One-command LAN deploy |
