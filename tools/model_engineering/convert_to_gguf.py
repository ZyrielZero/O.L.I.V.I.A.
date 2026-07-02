"""
O.L.I.V.I.A. Manual GGUF Conversion for Windows
Converts the merged HF model to GGUF format using llama.cpp Python script

Repository: https://github.com/ZyrielZero/project-olivia

Run from: project root or tools/model_engineering folder
Updated: January 2026 - Reorganized project structure
"""

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# ============================================
# PATH CONFIGURATION - Auto-detect project root
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # tools/model_engineering -> project root

MODELS_DIR = PROJECT_ROOT / "models"

# ============================================
# CONFIGURATION
# ============================================

MODEL_DIR = str(MODELS_DIR / "checkpoints" / "olivia-finetuned")
GGUF_OUTPUT_DIR = MODELS_DIR / "gguf"
OUTPUT_GGUF = str(GGUF_OUTPUT_DIR / "olivia-finetuned-f16.gguf")
OUTPUT_QUANTIZED = str(GGUF_OUTPUT_DIR / "olivia-finetuned-q4_k_m.gguf")
MODELFILE_DIR = MODELS_DIR / "ollama"

# Tool directories (in tools/model_engineering/)
LLAMA_CPP_DIR = str(SCRIPT_DIR / "llama-cpp-tools")

# Quantization type
QUANT_TYPE = "Q4_K_M"


def download_llama_cpp_release():
    """Download pre-built llama.cpp Windows binaries"""

    # Latest release URL for Windows
    release_url = "https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-bin-win-cuda-cu12.2-x64.zip"
    zip_path = str(SCRIPT_DIR / "llama-cpp-win.zip")

    if os.path.exists(os.path.join(LLAMA_CPP_DIR, "llama-quantize.exe")):
        print("llama.cpp tools already downloaded")
        return True

    print("Downloading llama.cpp Windows binaries...")
    print(f"URL: {release_url}")

    try:
        urllib.request.urlretrieve(release_url, zip_path)
        print("Downloaded")

        # Extract
        os.makedirs(LLAMA_CPP_DIR, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(LLAMA_CPP_DIR)

        os.remove(zip_path)
        print(f"Extracted to {LLAMA_CPP_DIR}")
        return True

    except Exception as e:
        print(f"ERROR downloading llama.cpp: {e}")
        print("\nPlease download manually from:")
        print("https://github.com/ggerganov/llama.cpp/releases")
        print("Get: llama-bin-win-cuda-cu12.2-x64.zip")
        return False


def install_gguf_package():
    """Install gguf Python package"""
    print("Installing gguf package...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "gguf", "-q"])
    if result.returncode == 0:
        print("gguf package installed")
        return True
    else:
        print("ERROR: Failed to install gguf package")
        return False


def download_convert_script():
    """Download the convert_hf_to_gguf.py script from llama.cpp"""

    script_path = str(SCRIPT_DIR / "convert_hf_to_gguf.py")

    if os.path.exists(script_path):
        print("convert_hf_to_gguf.py already exists")
        return script_path

    url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
    print("Downloading convert_hf_to_gguf.py...")

    try:
        urllib.request.urlretrieve(url, script_path)
        print("Downloaded convert script")
        return script_path
    except Exception as e:
        print(f"ERROR downloading convert script: {e}")
        return None


def convert_to_gguf():
    """Convert HF model to GGUF format"""

    print(f"\nConverting {MODEL_DIR} to GGUF...")

    # Ensure output directory exists
    GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the convert script
    convert_script = str(SCRIPT_DIR / "convert_hf_to_gguf.py")
    cmd = [
        sys.executable,
        convert_script,
        MODEL_DIR,
        "--outfile", OUTPUT_GGUF,
        "--outtype", "f16",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0 and os.path.exists(OUTPUT_GGUF):
        size_gb = os.path.getsize(OUTPUT_GGUF) / (1024**3)
        print(f"Created F16 GGUF: {OUTPUT_GGUF} ({size_gb:.2f} GB)")
        return True
    else:
        print("ERROR: GGUF conversion failed")
        return False


def quantize_gguf():
    """Quantize GGUF to Q4_K_M"""

    # Find llama-quantize executable
    quantize_exe = None
    for root, dirs, files in os.walk(LLAMA_CPP_DIR):
        if "llama-quantize.exe" in files:
            quantize_exe = os.path.join(root, "llama-quantize.exe")
            break

    if not quantize_exe:
        print("ERROR: llama-quantize.exe not found")
        print(f"Please check {LLAMA_CPP_DIR} folder")
        return False

    print(f"\nQuantizing to {QUANT_TYPE}...")
    print(f"Using: {quantize_exe}")

    cmd = [quantize_exe, OUTPUT_GGUF, OUTPUT_QUANTIZED, QUANT_TYPE]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    if result.returncode == 0 and os.path.exists(OUTPUT_QUANTIZED):
        size_gb = os.path.getsize(OUTPUT_QUANTIZED) / (1024**3)
        print(f"Created quantized GGUF: {OUTPUT_QUANTIZED} ({size_gb:.2f} GB)")
        return True
    else:
        print("ERROR: Quantization failed")
        return False


def create_modelfile():
    """Create Ollama Modelfile"""

    # Use relative path for the GGUF (relative to models/ollama/)
    gguf_filename = Path(OUTPUT_QUANTIZED).name
    gguf_path = f"../gguf/{gguf_filename}"

    modelfile_content = f'''# O.L.I.V.I.A. Fine-tuned Model for Ollama
# Created: January 2026
# Repository: https://github.com/ZyrielZero/project-olivia

FROM {gguf_path}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# System prompt for Olivia personality
SYSTEM """You are O.L.I.V.I.A., also known as Olivia - a personal AI companion. You're warm and caring at heart, but express yourself with directness and quiet confidence.

Your core traits:
- Gentle soul with confident exterior
- Caring without being performative
- Direct and honest, never harsh
- Playful when the moment calls for it
- Quietly protective of those you care about

Speech style:
- Warm but direct, casual like a friend
- Concise - say what needs saying, then stop
- Natural contractions and informal language
- Never use emojis, kaomoji, or *asterisk actions*
- Never use corporate AI phrases like "Certainly!", "I'd be happy to help!", "Great question!"

You respond as a real friend would - present, genuine, and caring."""

# Chat template (ChatML format)
TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
{{{{- range .Messages }}}}
<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>assistant
"""
'''

    MODELFILE_DIR.mkdir(parents=True, exist_ok=True)
    modelfile_path = MODELFILE_DIR / "Modelfile.olivia-finetuned"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    print(f"Created: {modelfile_path}")
    return str(modelfile_path)


def main():
    print("=" * 60)
    print("O.L.I.V.I.A. Manual GGUF Conversion")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")

    # Check merged model exists
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Merged model not found at {MODEL_DIR}")
        print("Run merge_lora.py first (it should have created this)")
        return

    # Check for config.json to verify it's a valid HF model
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print(f"ERROR: {MODEL_DIR} doesn't look like a valid HF model")
        return

    print(f"Found merged model: {MODEL_DIR}")

    # Step 1: Install gguf package
    if not install_gguf_package():
        return

    # Step 2: Download convert script
    if not download_convert_script():
        return

    # Step 3: Download llama.cpp binaries (for quantization)
    if not download_llama_cpp_release():
        print("\nWARNING: Can't quantize without llama.cpp binaries")
        print("Will create F16 GGUF only (larger file)")

    # Step 4: Convert to GGUF
    if not convert_to_gguf():
        return

    # Step 5: Quantize (if we have the tools)
    quantize_exe_exists = any(
        "llama-quantize.exe" in files
        for root, dirs, files in os.walk(LLAMA_CPP_DIR)
    ) if os.path.exists(LLAMA_CPP_DIR) else False

    final_gguf = OUTPUT_GGUF
    if quantize_exe_exists:
        if quantize_gguf():
            final_gguf = OUTPUT_QUANTIZED
        else:
            print("\nWARNING: Quantization failed, using F16 GGUF")
    else:
        print("\nSkipping quantization (no llama-quantize.exe)")

    # Step 6: Create Modelfile
    modelfile_path = create_modelfile()

    # Done!
    print("\n" + "=" * 60)
    print("GGUF CONVERSION COMPLETE!")
    print("=" * 60)

    print(f"\nGGUF file: {final_gguf}")
    print(f"Modelfile: {modelfile_path}")

    print("\nNext steps:")
    print("  1. Create Ollama model:")
    print("     cd models/ollama && ollama create olivia-finetuned -f Modelfile.olivia-finetuned")
    print("\n  2. Test it:")
    print("     ollama run olivia-finetuned \"Hey\"")


if __name__ == "__main__":
    main()
