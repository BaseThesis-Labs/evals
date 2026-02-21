#!/bin/bash
# QUICK INSTALL - Bypasses slow dependency resolution
# Run this instead: bash install-quick.sh

set -e

echo "🚀 QUICK INSTALL - Starting..."

# Activate venv
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Upgrade pip quietly
pip install --upgrade pip -q 2>/dev/null

echo "✓ Installing PyTorch CPU (2-3 min)..."
pip install --no-deps torch torchaudio -q
pip install typing-extensions sympy networkx jinja2 fsspec filelock -q

echo "✓ Installing audio libraries..."
pip install --no-deps soundfile librosa pydub -q
pip install numba decorator resampy audioread pooch joblib scikit-learn threadpoolctl lazy-loader -q

echo "✓ Installing core utilities..."
pip install numpy scipy pandas -q
pip install pyyaml click -q

echo "✓ Installing APIs (fastest part)..."
pip install deepgram-sdk cartesia assemblyai -q

echo "✓ Installing progress bars..."
pip install tqdm rich -q

echo "✓ Installing visualization..."
pip install matplotlib seaborn plotly kaleido -q

echo ""
echo "════════════════════════════════════════════"
echo "  ✅ QUICK INSTALL DONE! (~5 min total)"
echo ""
echo "  Ready to use:"
echo "  • Deepgram TTS"
echo "  • Cartesia TTS"
echo "  • AssemblyAI ASR"
echo "  • Audio generation"
echo ""
echo "  Set API keys:"
echo "  export DEEPGRAM_API_KEY='your_key'"
echo "  export CARTESIA_API_KEY='your_key'"
echo "  export ASSEMBLYAI_API_KEY='your_key'"
echo ""
echo "  Test it:"
echo "  source .venv/bin/activate"
echo "  python -c 'import torch; print(\"✓ Works!\")"
echo "════════════════════════════════════════════"
