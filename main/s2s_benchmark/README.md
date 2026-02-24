# S2S Benchmark

End-to-end evaluation pipeline for Speech-to-Speech models. Measures 60+ metrics across 9 dimensions — content preservation, audio quality, speaker similarity, prosody, emotion, latency, response quality, interaction, and multi-turn agent capabilities — with [0,1] normalization, use-case weighted composites, and multi-turn dialogue evaluation.

---

## Models evaluated

### Cascaded (Whisper ASR + LLM + TTS)

| Model | LLM | TTS | Type |
|---|---|---|---|
| cascaded_cartesia | — (echo) | Cartesia | Echo |
| cascaded_deepgram | — (echo) | Deepgram | Echo |
| cascaded_elevenlabs | — (echo) | ElevenLabs | Echo |
| cascaded_groq_cartesia | Llama 3.3-70b (Groq) | Cartesia | Generative |
| cascaded_groq_deepgram | Llama 3.3-70b (Groq) | Deepgram | Generative |
| cascaded_claude_cartesia | Claude Haiku 4.5 | Cartesia | Generative |
| cascaded_claude_deepgram | Claude Haiku 4.5 | Deepgram | Generative |
| cascaded_claude_elevenlabs | Claude Haiku 4.5 | ElevenLabs | Generative |
| cascaded_together_deepgram | Llama 3.3-70b (Together) | Deepgram | Generative |
| cascaded_llama3_deepgram | Llama 3.2 (Ollama) | Deepgram | Generative |

### Native S2S (end-to-end)

| Model | Type |
|---|---|
| gpt4o_realtime | Generative |
| gemini_live (Gemini 2.0 Flash Live) | Generative |
| ultravox (v0.7) | Generative |
| qwen25_omni_3b / 7b | Generative |
| moshi | Generative |
| elevenlabs_s2s | Echo |

Enable/disable models in `config/eval_config.yaml`.

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Set API keys in environment
export OPENAI_API_KEY=...
export GROQ_API_KEY=...
export CARTESIA_API_KEY=...
export DEEPGRAM_API_KEY=...
export ELEVENLABS_API_KEY=...
export ANTHROPIC_API_KEY=...       # only for cascaded_claude_* models
export GOOGLE_API_KEY=...          # only for gemini_live
export TOGETHER_API_KEY=...        # only for cascaded_together_*

# 4. Prepare datasets
python datasets/prepare_s2s_testsets.py

# 5. Run full pipeline (inference + evaluation)
python cli.py run-all --mode both

# Or run evaluation only (if inference outputs already exist)
python cli.py eval --manifest datasets/manifests/s2s_manifest.json
```

---

## Evaluation modes

| Mode | Description | Command |
|---|---|---|
| Single-turn echo | Model repeats/resynthesizes input speech | `python cli.py eval --mode single` |
| Single-turn generative | Model generates a spoken response | `python cli.py eval --mode single` |
| Multi-turn agent | Multi-turn dialogue with context probes | `python cli.py eval --mode multiturn` |
| Both | Run single-turn and multi-turn | `python cli.py eval --mode both` |

---

## Pipeline steps

```
datasets/prepare_*.py  →  inference/run_s2s_inference.py  →  pipeline.py  →  report.py
(download + manifest)     (run models)                       (metrics +      (markdown
                                                              scoring)        report)
```

All steps are orchestrated by `run_all.py` or `cli.py run-all`. You can also run them individually:

```bash
# Prepare datasets
python datasets/prepare_s2s_testsets.py
python datasets/prepare_voicebench.py

# Run inference for a specific model
python inference/run_s2s_inference.py --model cascaded_groq_deepgram

# Evaluate metrics
python cli.py eval --manifest datasets/manifests/s2s_manifest.json

# Run multi-turn scenarios
python cli.py eval --mode multiturn

# Generate report
python report.py --results results --output results/report.md

# Export to CSV/Excel
python export_csv.py
python scripts/export_excel.py
```

---

## Metrics

### Dimension breakdown

| Dimension | Key metrics |
|---|---|
| Content | WER, CER, BERTScore F1, SemDist, ROUGE-L |
| ASR Quality | Insertion/Deletion/Substitution rate, SER, HER |
| Audio Quality | UTMOS, DNSMOS, NISQA, PESQ, MCD |
| Speaker | SECS, WavLM-sim, ECAPA-sim |
| Prosody | F0-RMSE, Pitch-corr, Energy-corr, Duration-ratio, Speaking-rate |
| Emotion | Emotion-match, Emotion-sim (emotion2vec), ESIM |
| Latency | TTFB, E2E latency, RTF |
| Response Quality | LLM-as-Judge (coherence, relevance, helpfulness, safety, naturalness), Instruction-following |
| Interaction | TOR-up, TOR-down |

### Multi-turn agent metrics

| Metric | Method |
|---|---|
| Task Completion | LLM Judge against success criteria |
| Context Retention | Context probe pass rate |
| Voice Consistency | WavLM cosine similarity across turns |
| Dialogue Coherence | LLM Judge per-turn coherence |
| Error Recovery | Recovery rate after failures |
| Quality Degradation | UTMOS/DNSMOS slope over turns |
| Session Verdict | LLM Judge pass/fail |
| Avg Turn Latency | Mean E2E latency across turns |

### Use-case composite weights (`config/weights/`)

| Use case | Content | Quality | Speaker | Prosody | Emotion | Latency | Response | Interaction |
|---|---|---|---|---|---|---|---|---|
| balanced | 14% | 14% | 14% | 13% | 10% | 10% | 10% | 5% |
| conversational | 10% | 15% | 10% | 10% | 15% | 20% | 15% | 5% |
| audiobook | 15% | 20% | 15% | 20% | 10% | 5% | 10% | 5% |
| voice_cloning | 10% | 15% | 25% | 20% | 10% | 5% | 10% | 5% |
| expressive | 10% | 15% | 10% | 15% | 25% | 10% | 10% | 5% |
| agent | 5% | 10% | 10% | 5% | 5% | 15% | 15% | 5% |

---

## Scoring pipeline

```
Per-utterance raw metrics (60+ values)
        ↓  normalize to [0,1] (higher/lower/target strategies)
Per-utterance normalized scores
        ↓  weighted average per dimension
Per-utterance dimension scores (9 dimensions)
        ↓  aggregate across utterances per model
Per-model dimension scores
        ↓  weighted sum by use case
Per-model composite score (0–1)
        ↓  rank
Leaderboard
```

Normalization config: `config/normalization.yaml`

---

## Multi-turn scenarios

12 YAML scenario files in `datasets/multiturn/scenarios/`:

| Scenario | Category |
|---|---|
| customer_service | Customer support |
| restaurant_ordering | Task completion |
| appointment_booking | Scheduling |
| tech_support | Technical assistance |
| information_retrieval | Knowledge lookup |
| instruction_following | Directive compliance |
| emotional_support | Empathetic dialogue |
| negotiation | Multi-party negotiation |
| context_retention | Long-context memory |
| adversarial | Robustness testing |

Each scenario defines: system prompt, user turns, context probes, success criteria, and voice checks.

```bash
# List available scenarios
python cli.py scenarios

# Run multi-turn evaluation
python cli.py eval --mode multiturn
```

---

## Datasets

| Dataset | Domain | Use |
|---|---|---|
| LJSpeech | Read speech, single speaker | Echo baseline |
| RAVDESS | Emotional speech, multi-speaker | Emotion evaluation |
| TESS | Emotional speech, female | Emotion evaluation |
| SAVEE | Emotional speech, male | Emotion evaluation |
| CMU Arctic | Multi-speaker, controlled | Speaker similarity |
| VoiceBench | Diverse voices | General evaluation |
| FullDuplex | Overlapping speech | Interaction metrics |

Prepare datasets:
```bash
python datasets/prepare_s2s_testsets.py
python datasets/prepare_voicebench.py
python datasets/prepare_fullduplex.py
python datasets/download_additional.py   # Kaggle datasets
```

---

## Outputs

```
results/{dataset}/
  {model}_utterances.jsonl        # per-utterance metrics
  {model}_metrics.json            # aggregated model metrics
  leaderboard_echo.json           # echo model rankings
  leaderboard_generative.json     # generative model rankings
  leaderboard_combined.json       # all models ranked

results/multiturn/
  {model}_multiturn.json          # per-session agent metrics
  leaderboard_agent.json          # agent model rankings

results/report.md                 # full markdown report
```

---

## Configuration

| File | Purpose |
|---|---|
| `config/eval_config.yaml` | Model definitions, metric toggles, multi-turn settings |
| `config/normalization.yaml` | Per-metric [0,1] scaling rules (floor, ceiling, direction) |
| `config/weights/balanced.yaml` | Dimension and metric weights for balanced use case |
| `config/weights/conversational.yaml` | Weights optimized for conversational AI |
| `config/weights/agent.yaml` | Weights optimized for multi-turn agents |

---

## Project structure

```
s2s_benchmark/
├── cli.py                        # CLI entry point
├── pipeline.py                   # Master evaluator (metrics + aggregate)
├── run_all.py                    # Full automation (inference + eval)
├── run_integrated.py             # Integrated pipeline runner
├── report.py                     # Markdown report generation
├── export_csv.py                 # CSV export
├── config/
│   ├── eval_config.yaml          # Model and metric configuration
│   ├── normalization.yaml        # Metric scaling rules
│   └── weights/                  # Use-case weight profiles
├── inference/
│   ├── adapters/                 # Model adapters (cascaded + native S2S)
│   ├── multiturn_runner.py       # Multi-turn session orchestrator
│   ├── run_s2s_inference.py      # Inference driver
│   └── streaming_harness.py      # Streaming latency measurement
├── metrics/
│   ├── content.py                # WER, BERTScore, ROUGE, SemDist
│   ├── quality.py                # UTMOS, DNSMOS, NISQA, PESQ, MCD
│   ├── speaker.py                # SECS, WavLM-sim, ECAPA-sim
│   ├── prosody.py                # F0, pitch, energy, duration
│   ├── emotion.py                # emotion2vec, emotion_match
│   ├── latency.py                # TTFB, E2E, RTF
│   ├── judge.py                  # LLM-as-Judge (GPT-4o)
│   ├── interaction.py            # Turn-taking metrics
│   └── multiturn/                # Session-level agent metrics
├── scoring/
│   ├── aggregate.py              # Utterance → dimension → composite
│   ├── normalize.py              # Metric [0,1] normalization
│   └── significance.py           # Statistical significance testing
├── datasets/
│   ├── manifests/                # JSON test set manifests
│   ├── multiturn/scenarios/      # YAML agent scenarios
│   ├── prepare_s2s_testsets.py   # Dataset preparation
│   └── download_additional.py    # Kaggle dataset fetcher
├── scripts/                      # Utility scripts
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- 16 GB RAM (32 GB recommended for full metric suite)
- GPU optional (required for Qwen, Ultravox, Moshi local models)
- Internet access for API models and HuggingFace model downloads
