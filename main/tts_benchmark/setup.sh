#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║   TTS Benchmark - Environment Setup     ║"
echo "╚══════════════════════════════════════════╝"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "▶ Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✓ Virtual environment exists"
fi

source .venv/bin/activate

echo "▶ Upgrading pip..."
pip install --upgrade pip -q

# Core dependencies
echo "▶ Installing PyTorch (CPU-only)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q

echo "▶ Installing audio libraries..."
pip install soundfile librosa pydub numpy scipy -q

echo "▶ Installing data utilities..."
pip install pandas pyyaml tqdm rich click -q

# TTS Models
echo "▶ Installing TTS models..."
pip install kokoro kokoro-onnx onnxruntime -q  # Kokoro
pip install piper-tts -q                        # Piper
pip install TTS -q                              # Coqui XTTS v2
pip install deepgram-sdk -q                     # Deepgram Aura
pip install cartesia -q                         # Cartesia Sonic

# Evaluation Metrics
echo "▶ Installing evaluation metrics..."
pip install assemblyai -q        # ASR for intelligibility
pip install jiwer -q             # WER/CER
pip install speechbrain -q       # Speaker similarity
pip install resemblyzer -q       # Speaker similarity
pip install praat-parselmouth -q # Prosody
pip install nisqa -q             # MOS prediction
pip install pesq -q              # Signal quality

echo "▶ Installing visualization..."
pip install matplotlib seaborn plotly kaleido -q

echo ""
echo "════════════════════════════════════════════"
echo "  ✅ Setup Complete!"
echo ""
echo "  Set API keys:"
echo "  export DEEPGRAM_API_KEY='your_key'"
echo "  export CARTESIA_API_KEY='your_key'"
echo "  export ASSEMBLYAI_API_KEY='your_key'"
echo ""
echo "  Run: bash run_all.sh"
echo "════════════════════════════════════════════"
