

# TTS Evaluation Benchmark

A comprehensive Text-to-Speech evaluation pipeline testing 5 TTS models across 35 metrics on the Seed-TTS-Eval dataset.

## 🎯 Overview

- **Dataset**: Seed-TTS-Eval (EN subset, 200 sampled utterances)
- **Models**: 2 API-based + 3 local open-source TTS systems
- **Metrics**: 35 active metrics across 9 dimensions
- **Hardware**: CPU-only, ≥16 GB RAM
- **Estimated Runtime**: ~3-4 hours total

## 📊 Models Evaluated

| Model | Type | Speed | Voice Cloning | API/Local |
|-------|------|-------|---------------|-----------|
| **Deepgram Aura** | API | ~1s/utt | No | API |
| **Cartesia Sonic** | API | ~1s/utt | Yes | API |
| **Kokoro ONNX** | Local | ~3s/utt | No | Local |
| **Piper TTS** | Local | ~1s/utt | No | Local |
| **Coqui XTTS v2** | Local | ~20s/utt | Yes | Local |

## 🚀 Quick Start

### 1. Installation

```bash
cd tts_benchmark

# Run setup (creates venv, installs dependencies)
bash setup.sh
```

### 2. Set API Keys

```bash
# Required for API-based models
export DEEPGRAM_API_KEY="your_deepgram_key"
export CARTESIA_API_KEY="your_cartesia_key"

# Required for ASR evaluation (choose one)
export ASSEMBLYAI_API_KEY="your_assemblyai_key"
# OR
export DEEPGRAM_API_KEY="your_deepgram_key"  # Can reuse for both TTS and ASR
```

### 3. Run Complete Pipeline

```bash
# Activate environment
source .venv/bin/activate

# Run everything
bash run_all.sh
```

This will:
1. Download Seed-TTS-Eval dataset
2. Sample 200 utterances
3. Generate audio with all 5 models (~1,000 wav files)
4. Compute 35 metrics across all outputs
5. Aggregate scores and rankings
6. Generate 6 visualization charts

### 4. View Results

```bash
# Leaderboard
cat analysis/leaderboard.json | jq '.rankings.balanced'

# Charts
open analysis/charts/01_leaderboard.png
open analysis/charts/02_radar.png
```

## 📈 Evaluation Metrics (35 Active)

### 1. Naturalness (10 metrics)
- **UTMOS** (1-5) - VoiceMOS Challenge winner
- **SCOREQ** (1-5) - NeurIPS 2024
- **NISQA MOS, Noisiness, Coloration, Discontinuity, Loudness** (5 sub-scores)
- **DNSMOS SIG, BAK, OVRL** (3 sub-scores)

### 2. Intelligibility (7 metrics)
- **WER** - Word Error Rate
- **CER** - Character Error Rate
- **ASR Mismatch** - Binary error flag
- **Word Skip/Insertion/Substitution Rates**

### 3. Speaker Similarity (2 metrics)
- **ECAPA-TDNN Cosine Similarity**
- **Resemblyzer Cosine Similarity**

### 4. Prosody (5 metrics)
- **F0 Mean, Std, Range** - Pitch statistics
- **Pause Ratio** - Silence percentage
- **Duration** - Audio length

### 5. Robustness (3 metrics)
- **Repetition Detection** - N-gram loops
- **Silence Anomaly** - Long gaps
- **Empty/Short Audio** - Failure detection

### 6. Latency (3 metrics)
- **TTFA** - Time to First Audio
- **RTF** - Real-Time Factor
- **Inference Time** - Total generation time

### 7. Distribution (5 metrics - model-level)
- **TTSDS2** scores: Prosody, Speaker, Environment, Intelligibility, Overall

## 📊 Use Cases & Composite Scores

The benchmark computes weighted composite scores for 5 use cases:

| Use Case | Focus Areas | Best For |
|----------|-------------|----------|
| **Conversational AI** | Latency (30%) + Intelligibility (25%) | Voice assistants, chatbots |
| **Audiobook** | Naturalness (35%) + Prosody (25%) | Long-form narration |
| **Voice Cloning** | Speaker Sim (40%) + Naturalness (20%) | Personalized voices |
| **Low Latency** | Latency (45%) + Intelligibility (20%) | Real-time applications |
| **Balanced** | Equal weighting across all dimensions | General purpose |

## 🏗️ Project Structure

```
tts_benchmark/
├── setup.sh                          # Environment setup
├── run_all.sh                        # Master pipeline
├── configs/
│   └── models.yaml                   # Model configurations
├── datasets/
│   ├── download.py                   # Dataset downloader
│   ├── manifest.json                 # 200 sampled entries
│   └── reference_audio/              # Speaker references
├── models/
│   ├── base_client.py                # Abstract TTS interface
│   ├── deepgram_client.py            # Deepgram Aura
│   ├── cartesia_client.py            # Cartesia Sonic
│   ├── kokoro_client.py              # Kokoro ONNX
│   ├── piper_client.py               # Piper TTS
│   └── xtts_client.py                # Coqui XTTS v2
├── generate.py                       # Audio generation
├── evaluate.py                       # Metric computation
├── aggregate.py                      # Score aggregation
├── visualize.py                      # Chart generation
├── generated_audio/{model}/          # Generated audio (1,000 files)
├── results/{model}_metrics.json      # Per-utterance scores
└── analysis/
    ├── leaderboard.json              # Final rankings
    └── charts/*.png                  # 6 visualizations
```

## 🎨 Visualizations

1. **Overall Leaderboard** - Bar chart sorted by balanced score
2. **Radar Chart** - 6-axis multi-dimensional comparison
3. **Metrics Heatmap** - Key metrics across models
4. **Use-Case Comparison** - Performance by use case
5. **WER Distribution** - Box plot of intelligibility
6. **UTMOS Distribution** - Violin plot of quality

## 🔧 Advanced Usage

### Run Individual Steps

```bash
source .venv/bin/activate

# 1. Download only
python datasets/download.py --n-samples 200 --seed 42

# 2. Generate for specific model
python generate.py --model kokoro

# 3. Evaluate specific model
python evaluate.py --gen-dir generated_audio/kokoro

# 4. Re-aggregate
python aggregate.py

# 5. Regenerate charts
python visualize.py
```

### Modify Configuration

Edit `configs/models.yaml` to:
- Enable/disable models
- Change voices or parameters
- Adjust rate limits

Example:
```yaml
models:
  kokoro:
    enabled: true
    config:
      voice: "af_heart"  # Change voice
```

### Scale Up

```bash
# Use full dataset (1,000 utterances)
python datasets/download.py --all

# Or custom sample size
python datasets/download.py --n-samples 500
```

## ⚙️ ASR Configuration

The pipeline supports two ASR backends for evaluation:

### 1. AssemblyAI (Default)
```bash
export ASSEMBLYAI_API_KEY="your_key"
```
- More accurate
- Better punctuation
- ~0.5s per utterance

### 2. Deepgram
```bash
export DEEPGRAM_API_KEY="your_key"
```
- Very fast (~0.3s per utterance)
- Good accuracy
- Can reuse same key for Deepgram TTS

The script auto-detects which API key is set.

## 📋 System Requirements

### Minimum
- **CPU**: 4 cores (Apple Silicon or x86_64)
- **RAM**: 16 GB
- **Disk**: 10 GB free
- **Python**: 3.10+

### Recommended
- **RAM**: 32 GB (for parallel model loading)
- **Disk**: 20 GB (caching models + outputs)
- **Internet**: Stable connection for API calls

## ⏱️ Expected Timing (200 Utterances)

| Stage | Time | Notes |
|-------|------|-------|
| Dataset Download | ~5 min | One-time |
| Audio Generation | ~60 min | 5 models × 200 utterances |
| ASR Transcription | ~20 min | AssemblyAI or Deepgram |
| Other Metrics | ~30 min | UTMOS, NISQA, etc. |
| Aggregation | ~5 min | TTSDS2 + statistics |
| Visualization | ~1 min | 6 charts |
| **Total** | **~2 hours** | With API ASR |

## 🧪 Testing Individual Models

Test before running full pipeline:

```bash
source .venv/bin/activate

# Test Kokoro
python -c "
from models.kokoro_client import KokoroClient
client = KokoroClient()
result = client.generate('Hello, this is a test.')
print(f'✓ Generated {result.duration_seconds:.2f}s audio')
"

# Test Deepgram
python -c "
from models.deepgram_client import DeepgramClient
client = DeepgramClient()
result = client.generate('Testing Deepgram Aura.')
print(f'✓ Generated {result.duration_seconds:.2f}s audio')
"
```

## 🔬 Extending the Benchmark

### Add a New Model

1. Create `models/my_model_client.py`:
```python
from models.base_client import BaseTTSClient, TTSResult

class MyModelClient(BaseTTSClient):
    def __init__(self):
        super().__init__()
        self.name = "my_model"

    def generate(self, text, ...):
        # Your implementation
        return TTSResult(...)
```

2. Add to `configs/models.yaml`:
```yaml
my_model:
  class_path: models.my_model_client.MyModelClient
  enabled: true
```

3. Run: `python generate.py --model my_model`

### Add More Datasets

The pipeline is designed to support:
- **LibriTTS-R** - Activates 8 dormant metrics (PESQ, ViSQOL, etc.)
- **EmergentTTS-Eval** - Competence testing across 6 categories
- **Long-TTS-Eval** - Long-form generation

Edit `datasets/download.py` to add new datasets.

## 📊 Sample Results

Expected score ranges (200 utterances):

| Model | UTMOS | WER | Speaker Sim | RTF | Balanced |
|-------|-------|-----|-------------|-----|----------|
| Deepgram Aura | ~4.0 | ~0.05 | N/A | ~0.05 | ~0.75 |
| Cartesia Sonic | ~3.8 | ~0.08 | ~0.70 | ~0.03 | ~0.72 |
| Kokoro ONNX | ~3.5 | ~0.12 | N/A | ~0.15 | ~0.65 |
| Piper TTS | ~3.3 | ~0.10 | N/A | ~0.05 | ~0.68 |
| XTTS v2 | ~4.2 | ~0.06 | ~0.82 | ~1.20 | ~0.70 |

*Note: Actual results will vary based on dataset and configuration*

## 🐛 Troubleshooting

### Out of Memory
```bash
# Run models sequentially instead of all at once
for model in kokoro piper xtts_v2 deepgram cartesia; do
    python generate.py --model $model
done
```

### API Rate Limits
```bash
# Adjust in configs/models.yaml
rate_limit_rps: 2  # Slower but safer
```

### Missing Dependencies
```bash
# Re-run setup
rm -rf .venv
bash setup.sh
```

### Model Download Issues
```bash
# Pre-download HuggingFace models
huggingface-cli download coqui/XTTS-v2
```

## 📝 Citation

**Seed-TTS-Eval Dataset:**
```bibtex
@article{seedtts2024,
  title={Seed-TTS: A Family of High-Quality Versatile Speech Generation Models},
  author={ByteDance Research},
  year={2024}
}
```

**Key Metrics:**
- UTMOS: VoiceMOS Challenge 2022/2024
- NISQA: Mittag et al., Interspeech 2021
- SCOREQ: Chen et al., NeurIPS 2024

## 📄 License

**Benchmark Code**: MIT License

**TTS Model Licenses**:
- Deepgram Aura: Commercial (requires API key)
- Cartesia Sonic: Commercial (requires API key)
- Kokoro ONNX: Apache 2.0
- Piper TTS: MIT
- Coqui XTTS v2: CPML (non-commercial use)

**Dataset**:
- Seed-TTS-Eval: ByteDance Research (CC BY-NC 4.0)

## 🤝 Contributing

To add a new model:
1. Implement `models/{model}_client.py`
2. Add entry to `configs/models.yaml`
3. Test: `python generate.py --model {model}`
4. Submit results

## 📞 Support

- **Issues**: Create GitHub issue
- **Dataset**: [ByteDance Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval)
- **Models**: Check respective model documentation

---

**Built with**: Python 3.10+ | PyTorch CPU | 5 TTS Models | 35 Metrics | Seed-TTS-Eval

**Status**: ✅ Production Ready | 📊 Reproducible | 🚀 Extensible

