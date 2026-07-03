# The Merge Experiment

A side quest, run through the same gates as everything else so it produces data, not
vibes. Plan of record is still the Qwen3-8B + LoRA pipeline (Master Plan Phase 4); if a
merge beats the gates, the merge wins. That's the ritual.

This README documents the kit only — the project front door is the root `README.md`,
and the forward direction lives in `docs/MASTER_PLAN.md`.

## Why this lineup (and why DELLA died)

The DELLA v1/v2 attempts mixed ChatML donors (Hermes-as-ChatML era, Dolphin) with
Llama-3-Instruct donors over a *base* (non-instruct) foundation — template leakage and
tokenizer collision followed. This kit fixes the frame:

- **Everything is Llama-3.1 lineage.** Never mix L3 / L3.1 / L3.2 — close-but-wrong
  lineages merge into subtle brain damage.
- **The base is L3.1-8B-Instruct** (ungated NousResearch mirror), so donor deltas are
  computed against the format the donors were actually trained on.
- **`tokenizer_source: base`** everywhere, always.

## The lineup

| Model | Role | Weight | Density |
|---|---|---|---|
| `NousResearch/Meta-Llama-3.1-8B-Instruct` | Base / anchor | — | — |
| `Sao10K/Llama-3.1-8B-Stheno-v3.4` | The voice: multi-turn coherency, system-prompt adherence | 0.4 | 0.9 |
| `NousResearch/Hermes-3-Llama-3.1-8B` | Tool-calling + structured-output insurance | 0.35 | 0.9 |
| `DavidAU/L3.1-Dark-Planet-8B` | The edge: reduced positivity bias, non-slop prose | 0.2 | 0.85 |

Rules for the edge slot:

- "A bit unhinged" is a 0.2 statement. Bump 0.05 max per run, one variable at a time.
- Verify the Dark Planet repo id resolves on Hugging Face before the first download —
  DavidAU ships many variants and it must be the **L3.1** one.
- Alternates: DarkIdol 1.2 (RP-leaning), Dolphin 3.0 (stable-unfiltered).
- **NO abliterated donors.** Documented instability; merges amplify it.

## The two arms

- **`olivia_dare_ties.yaml`** — main arm. DARE-TIES at density 0.85–0.9 (mergekit's
  ~0.6 default is documented to underperform), `int8_mask` + `rescale` per the Lunaris
  precedent.
- **`olivia_slerp_control.yaml`** — control arm. 2-model SLERP (Stheno × Hermes), the
  gentlest method there is. Isolates whether damage comes from technique or donors; if
  the control beats the main arm, the extra donor isn't paying its way.

### SLERP `slices:` fallback

If the installed mergekit rejects the `models:` shorthand for slerp, use the classic
form (L3.1-8B has 32 layers), keeping every other key identical:

```yaml
slices:
  - sources:
      - model: Sao10K/Llama-3.1-8B-Stheno-v3.4
        layer_range: [0, 32]
      - model: NousResearch/Hermes-3-Llama-3.1-8B
        layer_range: [0, 32]
```

## Running a merge

Runs in **WSL2**, needs **~85GB free disk**, and the **GPU stays free** — the merge is
CPU-only (no `--cuda`), so it can run while other work happens.

```bash
cd tools/model_engineering/merge
./run_merge.sh olivia_dare_ties.yaml
# → out/olivia_dare_ties/  (gitignored)
python verify_merge.py out/olivia_dare_ties   # tokenizer + load smoke test
```

The script creates a kit-local venv (`.venv-merge/`, gitignored) and installs mergekit
into it. mergekit is deliberately **not** in `pyproject.toml` — its dependency tree
would fight the app's pins, and the kit never runs inside the app venv.

## Gating a candidate

Gate the **Q4_K_M GGUF, not the fp16** — quantization shifts behavior. The Ollama
Modelfile uses the **Llama-3.1 chat template, never the Qwen one**.

Every candidate goes through the full Master Plan Phase 4.0 gate:

1. Persona consistency (current production model's 92% is the bar)
2. Forbidden-pattern regex ceiling (sycophancy openers, emoji, hedging, corporate tone)
3. **BFCL subset — hard blocker**: regression >3–5 pts fails the run, no exceptions
4. Long-context sanity
5. 10-sample human spot-read

Eval result JSONs get committed to `results/` — that is the experiment's record.
Losing configs' YAMLs stay committed too: negative results are still results.
