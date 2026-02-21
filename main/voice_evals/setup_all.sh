#!/bin/bash
#
# Complete Setup Script for Voice Evaluation Framework
# Installs all dependencies and downloads all models
#

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Voice Evaluation Framework - Complete Setup                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"
echo ""

# Core dependencies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Installing Core Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install -q --upgrade pip
pip install -q \
    librosa \
    numpy \
    scipy \
    soundfile \
    torch \
    torchaudio \
    scikit-learn \
    pandas

echo "✅ Core dependencies installed"
echo ""

# Transcription
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Installing Transcription (Whisper)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install -q \
    transformers \
    accelerate \
    openai-whisper

echo "✅ Whisper installed"
echo ""

# Semantic metrics
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Installing Semantic Metrics (BERT, LaBSE)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install -q \
    sentence-transformers \
    tiktoken

echo "✅ Semantic metrics installed"
echo ""

# Emotion detection
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Installing Emotion Detection (FunASR)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install -q \
    funasr \
    modelscope

echo "✅ Emotion detection installed"
echo ""

# Speaker diarization
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. Installing Speaker Diarization (Pyannote)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install -q pyannote.audio

echo "✅ Pyannote.audio installed"
echo ""

# Additional tools
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. Installing Additional Tools"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install -q \
    matplotlib \
    seaborn \
    jupyter

echo "✅ Additional tools installed"
echo ""

# Download models
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. Downloading Pre-trained Models"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Download UTMOS for speech quality
echo "Downloading UTMOS (Speech Quality/MOS)..."
python -c "
import torch
import warnings
warnings.filterwarnings('ignore')
try:
    model = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)
    print('✓ UTMOS model downloaded')
except Exception as e:
    print(f'⚠️  UTMOS download failed: {e}')
"
echo ""

# Pre-download Whisper model
echo "Downloading Whisper base model..."
python -c "
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import warnings
warnings.filterwarnings('ignore')
try:
    model_id = 'openai/whisper-base'
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    print('✓ Whisper base model downloaded')
except Exception as e:
    print(f'⚠️  Whisper download failed: {e}')
"
echo ""

# Pre-download BERT for semantic metrics
echo "Downloading BERT for semantic metrics..."
python -c "
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')
try:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    print('✓ BERT model downloaded')
except Exception as e:
    print(f'⚠️  BERT download failed: {e}')
"
echo ""

# Pre-download LaBSE for SAER
echo "Downloading LaBSE for SAER..."
python -c "
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')
try:
    model = SentenceTransformer('sentence-transformers/LaBSE')
    print('✓ LaBSE model downloaded')
except Exception as e:
    print(f'⚠️  LaBSE download failed: {e}')
"
echo ""

# Pre-download emotion model
echo "Downloading emotion2vec model..."
python -c "
from funasr import AutoModel
import warnings
warnings.filterwarnings('ignore')
try:
    model = AutoModel(model='iic/emotion2vec_plus_seed', device='cpu', hub='huggingface')
    print('✓ Emotion2vec model downloaded')
except Exception as e:
    print(f'⚠️  Emotion model download failed: {e}')
"
echo ""

echo "✅ All models downloaded"
echo ""

# Verify installation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. Verifying Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -c "
import sys

checks = {
    'Core': ['librosa', 'numpy', 'torch', 'torchaudio'],
    'Transcription': ['transformers', 'whisper'],
    'Semantic': ['sentence_transformers', 'tiktoken'],
    'Emotion': ['funasr', 'modelscope'],
    'Diarization': ['pyannote.audio'],
    'Utils': ['sklearn', 'pandas', 'matplotlib']
}

all_ok = True
for category, packages in checks.items():
    print(f'\n{category}:')
    for package in packages:
        try:
            __import__(package.replace('-', '_').replace('.', '_'))
            print(f'  ✓ {package}')
        except ImportError:
            print(f'  ✗ {package} - MISSING')
            all_ok = False

if all_ok:
    print('\n✅ All packages verified!')
else:
    print('\n⚠️  Some packages missing - check errors above')
    sys.exit(1)
"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ SETUP COMPLETE!                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "You can now run:"
echo ""
echo "  # Full evaluation with all features:"
echo "  python run.py audio.wav transcript.txt --num-speakers 2"
echo ""
echo "  # Real-time ASR evaluation:"
echo "  python test_realtime_asr.py"
echo ""
echo "  # Test specific features:"
echo "  python -c 'from metrics.realtime_asr import RealtimeASREvaluator; print(\"✓ Ready\")'"
echo ""
