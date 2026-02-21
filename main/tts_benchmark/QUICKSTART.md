# TTS Benchmark - Quick Start Guide

## 🎯 What You Have

A complete TTS evaluation pipeline with:
- **5 TTS models**: Deepgram Aura, Cartesia Sonic, Kokoro ONNX, Piper, Coqui XTTS v2
- **35 evaluation metrics** across naturalness, intelligibility, speaker similarity, prosody, robustness, latency
- **Automated pipeline** from dataset download to visualizations
- **200 utterances** from Seed-TTS-Eval dataset

## ⚡ 3-Step Setup

### Step 1: Install Dependencies (5 minutes)

```bash
cd tts_benchmark
bash setup.sh
```

This creates a virtual environment and installs all dependencies.

### Step 2: Set API Keys

```bash
# Required for API models
export DEEPGRAM_API_KEY="your_deepgram_api_key"
export CARTESIA_API_KEY="your_cartesia_api_key"

# Required for ASR evaluation (choose one)
export ASSEMBLYAI_API_KEY="your_assemblyai_key"
# OR use Deepgram for both TTS and ASR
```

**Get API Keys:**
- Deepgram: https://console.deepgram.com/
- Cartesia: https://cartesia.ai/
- AssemblyAI: https://www.assemblyai.com/

### Step 3: Run Everything (2-3 hours)

```bash
source .venv/bin/activate
bash run_all.sh
```

## 📊 What You'll Get

After completion, you'll have:

```
tts_benchmark/
├── generated_audio/          # 1,000 wav files (5 models × 200 utterances)
├── results/                  # Detailed metrics per model
└── analysis/
    ├── leaderboard.json      # Final rankings
    └── charts/               # 6 PNG visualizations
```

## 🎨 View Results

### Leaderboard
```bash
cat analysis/leaderboard.json | python -m json.tool | head -30
```

### Charts
```bash
open analysis/charts/01_leaderboard.png    # Overall ranking
open analysis/charts/02_radar.png          # Multi-dimensional comparison
open analysis/charts/03_heatmap.png        # Detailed metrics
open analysis/charts/04_use_cases.png      # Use-case performance
open analysis/charts/05_wer_distribution.png  # Intelligibility
open analysis/charts/06_utmos_distribution.png # Quality
```

## 🔧 Test Individual Models (Before Full Run)

### Test Local Models (No API Key Needed)

```bash
source .venv/bin/activate

# Test Kokoro
python -c "
from models.kokoro_client import KokoroClient
client = KokoroClient()
result = client.generate('Hello world, this is a test.')
print(f'✓ Kokoro: {result.duration_seconds:.2f}s audio, {result.inference_time_ms:.0f}ms')
"

# Test Piper
python -c "
from models.piper_client import PiperClient
client = PiperClient()
result = client.generate('Testing Piper TTS.')
print(f'✓ Piper: {result.duration_seconds:.2f}s audio, {result.inference_time_ms:.0f}ms')
"
```

### Test API Models (Requires Keys)

```bash
# Test Deepgram
python -c "
from models.deepgram_client import DeepgramClient
client = DeepgramClient()
result = client.generate('Testing Deepgram Aura TTS.')
print(f'✓ Deepgram: {result.duration_seconds:.2f}s audio, {result.inference_time_ms:.0f}ms')
"

# Test Cartesia
python -c "
from models.cartesia_client import CartesiaClient
client = CartesiaClient()
result = client.generate('Testing Cartesia Sonic.')
print(f'✓ Cartesia: {result.duration_seconds:.2f}s audio, {result.inference_time_ms:.0f}ms')
"
```

## 🎯 Run Partially (Step-by-Step)

If you want to run stages separately:

```bash
source .venv/bin/activate

# 1. Download dataset (5 min)
python datasets/download.py --n-samples 200 --seed 42

# 2. Generate audio for one model (test)
python generate.py --model kokoro

# 3. Generate for all models (60 min)
python generate.py

# 4. Evaluate (2 hours)
python evaluate.py --asr assemblyai

# 5. Aggregate scores (5 min)
python aggregate.py

# 6. Visualize (1 min)
python visualize.py
```

## 🚨 Common Issues

### Issue: "DEEPGRAM_API_KEY not set"
**Solution:**
```bash
export DEEPGRAM_API_KEY="your_key_here"
# Or skip Deepgram by disabling it in configs/models.yaml
```

### Issue: Out of memory
**Solution:** Run models one at a time:
```bash
for model in kokoro piper xtts_v2 deepgram cartesia; do
    python generate.py --model $model
done
```

### Issue: Slow evaluation
**Solution:** Use Deepgram for faster ASR:
```bash
python evaluate.py --asr deepgram
```

### Issue: Model download fails
**Solution:** Pre-download models:
```bash
# Kokoro
python -c "from kokoro_onnx import Kokoro; Kokoro('kokoro-v0_19.onnx', 'voices.json')"

# XTTS v2
huggingface-cli download coqui/XTTS-v2
```

## 📈 Expected Timeline (200 Utterances)

| Stage | Duration | CPU Usage |
|-------|----------|-----------|
| Setup | 5 min | Low |
| Dataset Download | 5 min | Low |
| Audio Generation | 60 min | High |
| Evaluation (ASR) | 20 min | Medium |
| Evaluation (Other) | 30 min | Medium |
| Aggregation | 5 min | Low |
| Visualization | 1 min | Low |
| **TOTAL** | **~2 hours** | - |

## 🎯 Minimal Test Run (Quick Validation)

To test the pipeline works before committing 2 hours:

```bash
# Use only 10 utterances and 2 models
python datasets/download.py --n-samples 10 --seed 42

# Edit configs/models.yaml - disable all except kokoro and piper
# Then run:
python generate.py
python evaluate.py --asr assemblyai
python aggregate.py
python visualize.py

# Should complete in ~15 minutes
```

## 📊 Understanding Results

### Key Metrics to Check

1. **UTMOS** (1-5): Overall quality
   - \>4.0 = Excellent
   - 3.5-4.0 = Good
   - <3.5 = Needs improvement

2. **WER** (0-1): Intelligibility
   - <0.05 = Excellent
   - 0.05-0.15 = Good
   - \>0.15 = Poor

3. **Speaker Similarity** (0-1, for cloning models):
   - \>0.85 = Excellent
   - 0.75-0.85 = Good
   - <0.75 = Needs improvement

4. **RTF** (Real-Time Factor):
   - <0.5 = Very fast
   - 0.5-1.0 = Fast
   - \>1.0 = Slow

### Composite Scores

Check `analysis/leaderboard.json` for:
- **Balanced**: Overall best model
- **Conversational AI**: Best for voice assistants
- **Audiobook**: Best for narration
- **Voice Cloning**: Best for personalized voices
- **Low Latency**: Best for real-time apps

## 🔄 Re-run After Changes

If you modify configs/models.yaml:

```bash
# Skip existing files
bash run_all.sh

# Or force regenerate
rm -rf generated_audio results analysis
bash run_all.sh
```

## 📞 Need Help?

- **Documentation**: Read `README.md`
- **Model Details**: Check `models/*_client.py`
- **Configuration**: Edit `configs/models.yaml`
- **Metrics**: See `evaluate.py` for all 35 metrics

---

**Ready?** Run `bash run_all.sh` and grab a coffee! ☕

