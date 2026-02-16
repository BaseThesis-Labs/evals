# Voice Evaluation Framework with Speaker Diarization

Comprehensive voice AI evaluation system with speaker separation, transcription accuracy, semantic analysis, and prosody metrics.

## ✨ Features

### 🎙️ **Speaker Diarization** (NEW!)
- **Automatic speaker separation** using Pyannote.audio 3.1
- **Per-speaker metrics**: speaking time, turn-taking analysis
- **Visual timeline** of speaker interactions
- **RTTM format export** for compatibility

### 📊 **Comprehensive Metrics**

#### String-Level Accuracy
- **WER** (Word Error Rate)
- **CER** (Character Error Rate)
- **MER** (Match Error Rate)
- **WIP/WIL** (Word Information Preserved/Lost)
- **Normalized WER** (with filler removal)

#### Semantic Accuracy
- **SeMaScore** - BERT-based semantic similarity with MER penalty
- **SAER** - Semantic-Aware Error Rate for multilingual/code-switching
- **ASD** - Aligned Semantic Distance for fine-grained meaning comparison

#### Performance Metrics
- **RTFx** - Real-time factor (processing speed)
- **Response Latency** - AI reaction time
- **SNR** - Signal-to-noise ratio
- **Talk Ratio** - Speaking time balance

#### Quality Analysis
- **Emotion Detection** - Dominant emotion and distribution
- **Prosody Analysis** - Pitch, pace, intonation
- **Speech Quality** - UTMOS-based quality score

## 📁 Project Structure

```
voice_evals/
├── __init__.py              # Package initialization
├── run.py                   # Main CLI entry point
├── requirements.txt         # Dependencies
├── README.md               # This file
│
├── core/                   # Core components
│   ├── __init__.py
│   ├── diarization.py     # Speaker diarization (Pyannote.audio)
│   └── pipeline.py        # Main evaluation pipeline
│
├── metrics/               # Metric implementations
│   ├── __init__.py
│   └── enhanced_metrics.py  # Re-exports from pipeline2
│
├── utils/                 # Utilities
│   └── __init__.py
│
├── reports/               # Report generation
│   ├── __init__.py
│   └── formatter.py       # Report formatting
│
└── models/               # Model cache (auto-created)
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (required for audio processing)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### HuggingFace Token Setup (for diarization)

1. Create account: https://huggingface.co
2. Accept model terms:
   - https://huggingface.co/pyannote/speaker-diarization
   - https://huggingface.co/pyannote/segmentation
3. Generate token: https://huggingface.co/settings/tokens
4. Set environment variable:
   ```bash
   export HF_TOKEN="your-token-here"
   ```

### Basic Usage

```bash
# Simple evaluation with diarization
python run.py audio.wav

# With ground truth transcript
python run.py audio.wav transcript.txt

# With explicit HF token
python run.py audio.wav --hf-token hf_xxxxxxxxxxxxx

# Fixed number of speakers
python run.py audio.wav --num-speakers 2

# Without diarization (faster)
python run.py audio.wav --no-diarization
```

### Python API

```python
from voice_evals import VoiceEvaluationPipeline

# Create pipeline
pipeline = VoiceEvaluationPipeline(
    audio_path="audio.wav",
    transcript_path="transcript.txt",  # Optional
    hf_token="hf_xxxxx",               # Optional (uses env var)
    enable_diarization=True,
    num_speakers=2
)

# Run evaluation
metrics = pipeline.evaluate()

# Generate report
report = pipeline.generate_report(metrics)
print(report)

# Save results
pipeline.save_results(metrics)
```

## 📊 Output

The pipeline generates:

1. **Console Report** - Formatted metrics display
2. **JSON Results** - `<filename>_diarized_eval.json`
3. **RTTM File** - `<filename>_diarization.rttm` (speaker timestamps)

### Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║           ENHANCED VOICE EVALUATION REPORT                        ║
╚══════════════════════════════════════════════════════════════════╝

📊 AUDIO QUALITY
──────────────────────────────────────────────────────────────────
  Signal to Noise Ratio:    28.5 dB
  Speech Quality Score:     4.2

⚡ TRANSCRIPTION PERFORMANCE
──────────────────────────────────────────────────────────────────
  RTFx (Speed Factor):      2.8x ✓ faster than real-time

📝 TRANSCRIPTION ACCURACY
──────────────────────────────────────────────────────────────────
  Word Error Rate (WER):         0.0850 (8.50%)
  Word Accuracy:                 0.9150 (91.50%)

🧠 SEMANTIC ACCURACY
──────────────────────────────────────────────────────────────────
  SeMaScore:                     0.9245 ↑ higher = better
  SAER:                          0.0785 ↓ lower = better
  ASD (Aligned Distance):        0.0425 ↓ lower = better

📊 SPEAKER DIARIZATION ANALYSIS
──────────────────────────────────────────────────────────────────
Detected 2 speakers

SPEAKER_00:
  Speaking time:    45.2s (51.0%)
  Number of turns:  12
  Avg turn length:  3.8s

SPEAKER_01:
  Speaking time:    43.4s (49.0%)
  Number of turns:  11
  Avg turn length:  3.9s

SPEAKER TIMELINE
──────────────────────────────────────────────────────────────────
SPEAKER_00: ███    ███    ███    ███    ███    ███
SPEAKER_01:    ███    ███    ███    ███    ███    ███
```

## 🔧 Advanced Usage

### Standalone Diarization

```python
from voice_evals.core.diarization import SpeakerDiarizer

# Initialize diarizer
diarizer = SpeakerDiarizer(hf_token="hf_xxxxx")

# Perform diarization
result = diarizer.diarize("audio.wav", num_speakers=2)

# Visualize
diarizer.visualize_diarization(result)

# Save RTTM
diarizer.save_diarization_rttm(result, "output.rttm")
```

### Custom Evaluation Pipeline

```python
from voice_evals import VoiceEvaluationPipeline

pipeline = VoiceEvaluationPipeline(
    audio_path="audio.wav",
    transcript_path="transcript.txt",
    hf_token="hf_xxxxx",
    enable_diarization=True,
    num_speakers=None,  # Auto-detect
    saer_lambda=0.5,    # SAER form/semantic balance
    saer_lang="en",     # Language code
    per_speaker_evaluation=False  # Set True for per-speaker detailed metrics
)

metrics = pipeline.evaluate()

# Access specific results
print(f"WER: {metrics.overall_metrics['wer_score']:.4f}")
print(f"Speakers: {metrics.num_speakers}")

for speaker in metrics.speaker_metrics:
    print(f"{speaker.speaker_id}: {speaker.speaking_time:.2f}s")
```

## 📚 Metrics Guide

### String-Level Metrics

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| WER | [0, ∞) | Lower | (Substitutions + Deletions + Insertions) / N_ref |
| CER | [0, ∞) | Lower | Character-level WER |
| MER | [0, 1] | Lower | Match Error Rate (symmetric for D/I) |
| WIP | [0, 1] | Higher | Word Information Preserved |
| WIL | [0, 1] | Lower | Word Information Lost (1 - WIP) |

### Semantic Metrics

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| SeMaScore | [0, 1] | Higher | BERT cosine + MER penalty + importance weighting |
| SAER | [0, ∞) | Lower | λ×F_form + (1-λ)×ε_sem (multilingual) |
| ASD | [0, 1] | Lower | Aligned token-level semantic distance |

### Performance Benchmarks

| Metric | Excellent | Good | Needs Improvement |
|--------|-----------|------|-------------------|
| WER | < 5% | 5-15% | > 15% |
| Response Latency | < 300ms | 300-800ms | > 800ms |
| RTFx | > 2.0x | 1.0-2.0x | < 1.0x |
| SNR | > 30dB | 20-30dB | < 20dB |
| Talk Ratio | 0.8-1.5 | 0.5-2.0 | < 0.5 or > 2.0 |

## 🔍 Troubleshooting

### "HuggingFace token required"
- Set `HF_TOKEN` environment variable
- Or pass `--hf-token` argument
- Get token from: https://huggingface.co/settings/tokens

### "pyannote.audio not available"
```bash
pip install pyannote.audio
```

### "Model not found" errors
- Accept model terms at https://huggingface.co/pyannote/speaker-diarization
- Ensure your HF token has read access

### Low SNR values
- Check microphone quality
- Reduce background noise
- Ensure proper audio levels

### Slow processing
- Use `--no-diarization` to skip speaker separation
- Use smaller models (modify pipeline2.py)
- Process on GPU if available

## 📄 File Formats

### Supported Audio Formats
- WAV (recommended)
- MP3
- FLAC
- OGG
- Any format supported by librosa

### Transcript Format
- Plain text file (.txt)
- UTF-8 encoding
- One continuous paragraph or sentences

### RTTM Format (Output)
Standard format for speaker diarization:
```
SPEAKER <file> <channel> <start> <duration> <na> <na> <speaker> <conf> <na>
```

## 🤝 Contributing

This is an evaluation framework built on top of:
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Whisper](https://github.com/openai/whisper) - Transcription
- [pipeline2.py](../pipeline2.py) - Enhanced metrics

## 📝 License

MIT License - See individual dependencies for their licenses

## 🔗 References

- **SeMaScore**: Sasindran et al., Interspeech 2024
- **SAER**: SwitchLingua, 2025
- **Pyannote.audio**: Bredin et al., 2020
- **Whisper**: Radford et al., 2022

## 💡 Tips

1. **For best diarization**: Use stereo audio with clear speaker separation
2. **For best transcription**: Use clean audio, minimal background noise
3. **For semantic metrics**: Provide ground truth transcripts
4. **For speed**: Disable diarization and use smaller models
5. **For accuracy**: Enable all metrics and use largest models

---

**Need help?** Open an issue or check the [documentation](README.md)
