#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║   Installing Evaluation Metrics         ║"
echo "╚══════════════════════════════════════════╝"

source .venv/bin/activate

echo ""
echo "This installs heavy packages for evaluation."
echo "Only run this when you're ready to evaluate audio."
echo ""

# Stage 1: SpeechBrain (speaker similarity)
echo "▶ [1/7] SpeechBrain (speaker similarity)..."
pip install --no-cache-dir speechbrain -q
echo "  ✓ Installed"

# Stage 2: Resemblyzer (speaker similarity)
echo "▶ [2/7] Resemblyzer..."
pip install --no-cache-dir resemblyzer -q
echo "  ✓ Installed"

# Stage 3: Parselmouth (prosody)
echo "▶ [3/7] Praat Parselmouth (prosody)..."
pip install --no-cache-dir praat-parselmouth -q
echo "  ✓ Installed"

# Stage 4: NISQA (naturalness)
echo "▶ [4/7] NISQA (MOS prediction)..."
pip install --no-cache-dir nisqa -q 2>/dev/null || echo "  ⚠️  NISQA skipped (optional)"
echo "  ✓ Installed"

# Stage 5: PESQ (signal quality)
echo "▶ [5/7] PESQ (signal quality)..."
pip install --no-cache-dir pesq -q
echo "  ✓ Installed"

# Stage 6: TorchMetrics
echo "▶ [6/7] TorchMetrics..."
pip install --no-cache-dir torchmetrics -q
echo "  ✓ Installed"

# Stage 7: Local TTS models (optional, heavy)
echo "▶ [7/7] Local TTS models (Kokoro, Piper, XTTS)..."
echo "  This is OPTIONAL and large (~2GB). Install? (y/N)"
read -r install_local

if [[ "$install_local" =~ ^[Yy]$ ]]; then
    pip install --no-cache-dir kokoro kokoro-onnx onnxruntime -q
    pip install --no-cache-dir piper-tts -q
    pip install --no-cache-dir TTS -q
    echo "  ✓ Local models installed"
else
    echo "  ⊘ Skipped local models"
fi

echo ""
echo "════════════════════════════════════════════"
echo "  ✅ Evaluation metrics installed!"
echo ""
echo "  Ready to run:"
echo "     python evaluate.py"
echo "════════════════════════════════════════════"
