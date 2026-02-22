# STT Benchmark

End-to-end evaluation pipeline for Speech-to-Text models. Measures 70+ metrics across 7 dimensions — intelligibility, semantic accuracy, formatting, hallucination, entity recognition, latency, and safety — with statistical significance testing and use-case weighted leaderboards.

---

## Models evaluated

| Model | Type | Cost/min |
|---|---|---|
| faster-whisper-large-v3-turbo | Local (CPU) | free |
| faster-whisper-large-v3 | Local (CPU) | free |
| groq-whisper-large-v3-turbo | API | $0.00067 |
| groq-whisper-large-v3 | API | $0.00111 |
| assemblyai-universal-2 | API | $0.0025 |
| deepgram-nova-3 | API | $0.0043 |
| elevenlabs-scribe-v2 | API | $0.005 |
| openai-gpt4o-transcribe | API (disabled) | $0.006 |
| openai-gpt4o-mini-transcribe | API (disabled) | $0.003 |
| google-chirp-2 | API (disabled) | $0.016 |

Enable/disable models in `configs/models.yaml`.

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Copy and fill in API keys
cp .env.example .env   # then edit .env

# 4. Run the full pipeline (LibriSpeech test-clean, all enabled models)
bash run_all.sh

# Quick smoke-test with 50 samples
bash run_all.sh --quick
```

---

## Environment variables (`.env`)

```
DEEPGRAM_API_KEY=...
ASSEMBLYAI_API_KEY=...
ELEVENLABS_API_KEY=...
GROQ_API_KEY=...
OPENAI_API_KEY=...          # only needed for openai-gpt4o-* models
GOOGLE_APPLICATION_CREDENTIALS=...   # only needed for google-chirp-2
```

Never commit `.env` — it is listed in `.gitignore`.

---

## Supported datasets

| Dataset | Domain | Size | Flag |
|---|---|---|---|
| LibriSpeech test-clean | Read speech | 2,620 utt | `--dataset librispeech --subset test-clean` |
| LibriSpeech test-other | Read speech (harder) | 2,939 utt | `--dataset librispeech --subset test-other` |
| TED-LIUM 3 | Spontaneous lectures | 1,155 utt | `--dataset tedlium` |
| VoxPopuli EN | Parliament recordings | 1,842 utt | `--dataset voxpopuli` |
| Common Voice EN | Crowdsourced | varies | `--dataset commonvoice` |
| Earnings22 | Financial calls | varies | `--dataset earnings22` |
| Kincaid46 | Diverse speakers | 46 utt | `--dataset kincaid --kincaid-dir /path` |

Common Voice requires a HuggingFace account and dataset terms acceptance:
```bash
huggingface-cli login
```

---

## Pipeline steps

```
generate.py   →  evaluate.py   →  aggregate.py   →  visualize.py
(download)       (transcribe       (score,            (charts)
                  + metrics)        rank)
```

All steps are orchestrated by `run_all.sh`. You can also run them individually:

```bash
# Download dataset
python generate.py --source librispeech --subsets test-clean --max-samples 200

# Transcribe + compute metrics
python evaluate.py \
  --dataset datasets/librispeech/test-clean_manifest.jsonl \
  --output-dir results/librispeech/test-clean

# Aggregate into leaderboard
python aggregate.py \
  --metrics-dir results/librispeech/test-clean/metrics \
  --output-dir  analysis/librispeech/test-clean \
  --case-study  balanced

# Generate charts
python visualize.py \
  --leaderboard analysis/librispeech/test-clean/leaderboard.json \
  --metrics-dir results/librispeech/test-clean/metrics \
  --output      analysis/librispeech/test-clean/charts
```

---

## Metrics

### Dimension breakdown

| Dimension | Key metrics |
|---|---|
| Intelligibility | WER, CER, SER |
| Semantic | SemDist, ASD, SeMaScore |
| Formatting | FWER, Punctuation F1, Capitalization Acc |
| Hallucination | Hallucination Rate, HER, Shallow (SF/PF/RL/LC) |
| Entity | Entity F1 (spaCy NER), KRR |
| Latency | RTFx, NIC |
| Safety | Error Severity (avg/max), Impact Score |

### Use-case composite weights (`configs/case_studies.yaml`)

| Use case | Intelligibility | Semantic | Latency | Formatting | Hallucination | Entity |
|---|---|---|---|---|---|---|
| conversational_ai | 30% | 25% | 30% | — | 15% | — |
| audiobook | 35% | 25% | — | 25% | — | 10% |
| voice_cloning_qa | 30% | 25% | — | 10% | — | 20% |
| low_latency | 30% | 10% | 45% | — | 15% | — |
| balanced | 25% | 20% | 20% | 15% | 10% | 10% |

---

## Outputs

```
results/{dataset}/{subset}/
  transcriptions/{model}.jsonl   # raw hypotheses + references
  metrics/{model}.json           # per-sample metrics

analysis/{dataset}/{subset}/
  leaderboard.json               # ranked models with composite scores
  total_metrics.csv              # all metrics in one CSV
  charts/                        # PNG visualizations

comparison_charts/{dataset}/     # cross-subset comparisons (--subset both)
```

---

## Robustness testing (SPUD)

```bash
python spud.py --model groq-whisper-large-v3-turbo \
               --dataset datasets/librispeech/test-clean_manifest.jsonl \
               --output  results/spud/
```

SPUD applies AWGN noise at 6 SNR levels (∞, 40, 30, 20, 10, 5 dB) and reports the area under the robustness curve.

---

## Configuration

| File | Purpose |
|---|---|
| `configs/models.yaml` | Enable/disable models, set API keys, cost |
| `configs/evaluation.yaml` | Metric settings, normalizer config, stat test params |
| `configs/case_studies.yaml` | Use-case weight profiles and metric bounds |
| `configs/datasets.yaml` | Dataset definitions and manifest paths |

---

## Requirements

- Python 3.10+
- 16 GB RAM (32 GB recommended for large-v3 + all metrics)
- Internet access for API models and HuggingFace downloads
- `en_core_web_sm` spaCy model (auto-downloaded by `run_all.sh`)
