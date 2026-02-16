#!/usr/bin/env python3
"""
Complete Setup Script for Voice Evaluation Framework (Python version)
Installs all dependencies and downloads all models
"""

import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n→ {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  ✓ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {description} failed: {e}")
        if e.stderr:
            print(f"     {e.stderr}")
        return False


def install_package(package, description=None):
    """Install a pip package"""
    desc = description or f"Installing {package}"
    return run_command(f"pip install -q {package}", desc)


def download_model(code, description):
    """Download a model using Python code"""
    print(f"\n→ {description}...")
    try:
        exec(code)
        print(f"  ✓ {description} complete")
        return True
    except Exception as e:
        print(f"  ✗ {description} failed: {e}")
        return False


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     Voice Evaluation Framework - Complete Setup                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    # Check Python version
    print(f"Python version: {sys.version}")
    print("")

    # 1. Core dependencies
    print("━"*70)
    print("1. Installing Core Dependencies")
    print("━"*70)

    core_packages = [
        "librosa numpy scipy soundfile torch torchaudio scikit-learn pandas"
    ]

    for pkg in core_packages:
        install_package(pkg, "Core dependencies")

    # 2. Transcription
    print("\n━"*70)
    print("2. Installing Transcription (Whisper)")
    print("━"*70)

    install_package("transformers accelerate openai-whisper", "Whisper & Transformers")

    # 3. Semantic metrics
    print("\n━"*70)
    print("3. Installing Semantic Metrics")
    print("━"*70)

    install_package("sentence-transformers tiktoken", "Semantic metrics")

    # 4. Emotion detection
    print("\n━"*70)
    print("4. Installing Emotion Detection")
    print("━"*70)

    install_package("funasr modelscope", "FunASR & ModelScope")

    # 5. Speaker diarization
    print("\n━"*70)
    print("5. Installing Speaker Diarization")
    print("━"*70)

    install_package("pyannote.audio", "Pyannote.audio")

    # 6. Additional tools
    print("\n━"*70)
    print("6. Installing Additional Tools")
    print("━"*70)

    install_package("matplotlib seaborn jupyter", "Visualization & Jupyter")

    # 7. Download models
    print("\n━"*70)
    print("7. Downloading Pre-trained Models")
    print("━"*70)

    # UTMOS
    download_model("""
import torch
model = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)
""", "UTMOS (Speech Quality/MOS)")

    # Whisper
    download_model("""
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
model_id = 'openai/whisper-base'
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
""", "Whisper base model")

    # BERT
    download_model("""
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
""", "BERT for semantic metrics")

    # LaBSE
    download_model("""
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/LaBSE')
""", "LaBSE for SAER")

    # Emotion2vec
    download_model("""
from funasr import AutoModel
model = AutoModel(model='iic/emotion2vec_plus_seed', device='cpu', hub='huggingface')
""", "Emotion2vec model")

    # 8. Verify installation
    print("\n━"*70)
    print("8. Verifying Installation")
    print("━"*70)

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
                __import__(package.replace('-', '_').replace('.', '_').split('.')[0])
                print(f'  ✓ {package}')
            except ImportError:
                print(f'  ✗ {package} - MISSING')
                all_ok = False

    print("\n" + "╔" + "═"*68 + "╗")
    if all_ok:
        print("║" + " "*20 + "✅ SETUP COMPLETE!" + " "*26 + "║")
    else:
        print("║" + " "*15 + "⚠️  SETUP INCOMPLETE - Check errors" + " "*16 + "║")
    print("╚" + "═"*68 + "╝\n")

    if all_ok:
        print("You can now run:")
        print("")
        print("  # Full evaluation with all features:")
        print("  python run.py audio.wav transcript.txt --num-speakers 2")
        print("")
        print("  # Real-time ASR evaluation:")
        print("  python test_realtime_asr.py")
        print("")
        print("  # Test individual features:")
        print("  python -c 'from metrics.realtime_asr import RealtimeASREvaluator'")
        print("")


if __name__ == "__main__":
    main()
