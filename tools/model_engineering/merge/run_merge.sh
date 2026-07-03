#!/usr/bin/env bash
# O.L.I.V.I.A. merge runner — run inside WSL2, from this directory.
#
# CPU-only merge: the GPU must stay free for Ollama/eval workloads, so this
# never passes --cuda. mergekit merges fine on CPU with lazy unpickling.
#
# The kit uses its own venv (.venv-merge/, gitignored) — mergekit is
# deliberately NOT in pyproject.toml; its dependency tree would fight the
# app's pins.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv-merge"
REQUIRED_GB=85 # base + 3 donors (~16GB each fp16) + bf16 output + slack

usage() {
    echo "Usage: $0 <config.yaml>"
    echo "  e.g. $0 olivia_dare_ties.yaml"
    exit 1
}
[[ $# -eq 1 ]] || usage
CONFIG="$1"
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }

# --- Disk space check (filesystem hosting the kit) --------------------------
AVAIL_GB=$(df -BG --output=avail "$SCRIPT_DIR" | tail -1 | tr -dc '0-9')
if (( AVAIL_GB < REQUIRED_GB )); then
    echo "ERROR: ${AVAIL_GB}GB free, need ~${REQUIRED_GB}GB (source models + output)." >&2
    echo "Tip: the HF cache lives at \${HF_HOME:-~/.cache/huggingface} — if that is" >&2
    echo "a different mount, check its free space too." >&2
    exit 1
fi

# --- Kit-local venv ----------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating merge venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet mergekit

# --- Merge (CPU — leave the GPU alone) ---------------------------------------
NAME="$(basename "$CONFIG" .yaml)"
OUT_DIR="$SCRIPT_DIR/out/$NAME"
mkdir -p "$OUT_DIR"
echo "Merging $CONFIG -> $OUT_DIR (CPU merge; GPU stays free)"
mergekit-yaml "$CONFIG" "$OUT_DIR" --out-shard-size 5B --lazy-unpickle --verbose

echo
echo "Done. Next steps:"
echo "  1. python verify_merge.py \"$OUT_DIR\"   # tokenizer + load smoke test"
echo "  2. Quantize to Q4_K_M and gate THAT — the fp16 is not what ships."
