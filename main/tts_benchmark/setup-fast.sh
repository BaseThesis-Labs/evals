#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║   TTS Benchmark - FAST Setup            ║"
echo "╚══════════════════════════════════════════╝"

# Create venv
if [ ! -d ".venv" ]; then
    echo "▶ Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Upgrade pip with caching
echo "▶ Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

echo ""
echo "Installing in stages for faster feedback..."
echo ""

# Stage 1: Core (fast)
echo "▶ [1/5] Core dependencies (numpy, scipy)..."
pip install --no-cache-dir numpy scipy -q
echo "  ✓ Core installed"

# Stage 2: PyTorch CPU (largest, install first)
echo "▶ [2/5] PyTorch CPU-only (this may take 2-3 min)..."
pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q
echo "  ✓ PyTorch installed"

# Stage 3: Audio libraries (medium)
echo "▶ [3/5] Audio libraries..."
pip install --no-cache-dir soundfile librosa pydub -q
echo "  ✓ Audio libraries installed"

# Stage 4: Essential TTS & ASR (only what we need)
echo "▶ [4/5] TTS models & ASR..."
pip install --no-cache-dir \
    deepgram-sdk \
    cartesia \
    assemblyai \
    jiwer -q
echo "  ✓ APIs installed"

# Stage 5: Data & visualization (fast)
echo "▶ [5/5] Data processing & visualization..."
pip install --no-cache-dir \
    pandas pyyaml tqdm rich click \
    matplotlib seaborn plotly kaleido -q
echo "  ✓ Utilities installed"

echo ""
echo "════════════════════════════════════════════"
echo "  ✅ FAST SETUP COMPLETE!"
echo ""
echo "  ⚠️  OPTIONAL: Install evaluation metrics"
echo "  (Run when you need them, not required for generation)"
echo ""
echo "  To install evaluation metrics later:"
echo "  source .venv/bin/activate"
echo "  bash install-evaluation.sh"
echo ""
echo "  🚀 Ready to use:"
echo "     - Deepgram TTS"
echo "     - Cartesia TTS"
echo "     - AssemblyAI ASR"
echo ""
echo "  Next: export API keys and run:"
echo "     source .venv/bin/activate"
echo "     python generate.py  # Generate audio first"
echo "════════════════════════════════════════════"
