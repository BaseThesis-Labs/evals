# PROJECT CONTEXT: Interspeech 2026 TTS Benchmark Paper

> **This document gives you (Claude Code) complete context for our project. Read this ENTIRELY before doing any work. This is your single source of truth.**

## MISSION

We are a 3-person team submitting a paper to **Interspeech 2026** by **25 February 2026 AoE** (today is Feb 22, 2026 — we have 3 days). The paper is about our TTS evaluation benchmark framework. You have access to the full codebase via this repo.

**You have Scientific Writer plugin installed.** Use its skills (`@research-lookup`, `@citation-management`, `@peer-review`, `@scholar-evaluation`, `@venue-templates`, `@scientific-schematics`) as tools — but the actual paper content must be written by us/you with deep understanding of the code and results, not generated generically.

---

## THE CODEBASE — COMPLETE MAP

### A. `tts_benchmark/` — The Main Evaluation Pipeline (6 Stages)

#### Stage 1: `datasets/download.py` — Data Preparation
- Downloads/prepares **3 datasets**:
  - **Seed-TTS-Eval**: 200 utterances, the standard TTS benchmark dataset
  - **Harvard Sentences**: Phonetically balanced sentences
  - **Challenge Set** (NOVEL — our key contribution): 7 adversarial categories:
    - `numbers` — numeric strings, dates, phone numbers, currency
    - `proper_nouns` — brand names, place names, foreign-origin names
    - `tongue_twisters` — phonetically difficult sequences
    - `long_form` — paragraphs (200+ words) testing coherence over long utterances
    - `questions` — interrogative prosody (rising intonation patterns)
    - `emotional` — text requiring emotional expression (excitement, sadness, anger)
    - `code_switching` — English text with embedded French/German loanwords and phrases
- Also supports: LJSpeech, VCTK, CMU Arctic as additional sources
- Each challenge category has ~50 utterances with ground-truth transcripts

#### Stage 2: `generate.py` — Speech Generation
- Dynamic TTS client instantiation from YAML config (`configs/models.yaml`)
- Exponential backoff retry for API 429 rate limits
- All audio resampled to **24kHz mono float32** for fair comparison
- Saves per-utterance WAV files organized by model/dataset

#### Stage 3: `evaluate.py` — Metric Computation
- Runs **28 active metrics across 6 dimensions** on every generated utterance
  (plus 13 additional diagnostic metrics stored in raw output):

**Naturalness (5 active metrics in dimension scoring):**
- UTMOS (UTokyo-SaruLab MOS predictor)
- DNSMOS overall (+ signal/background sub-scores computed but not in dimension)
- Output SNR (Silero VAD-based, reference-free)
- PESQ (ITU-T P.862, requires reference audio)
- STOI (Short-Time Objective Intelligibility, requires reference audio)
- ~~SCOREQ~~ — **STUBBED**: `compute_scoreq()` returns `None` (line 241)
- ~~NISQA (5 sub-scores)~~ — **STUBBED**: `compute_nisqa()` returns all `None` (line 246-253)

**Intelligibility (5 active metrics):**
- WER (Word Error Rate via **Whisper base** model — line 60)
- CER (Character Error Rate)
- ASR transcript mismatch (binary)
- Word skip rate (deletion rate)
- Semantic distance (sentence-transformers all-MiniLM-L6-v2 cosine distance)
- (Also computed but not in dimension: MER, insertion rate, substitution rate)

**Speaker Similarity (1 active metric):**
- Resemblyzer GE2E cosine similarity (lines 476-524)
- ~~ECAPA-TDNN cosine similarity~~ — **DISABLED**: hardcoded `None` (line 472, "torchaudio compatibility issues")
- (Only computed when reference audio is available)

**Prosody (13 active metrics):**
- F0 range, jitter, shimmer, HNR (via **Parselmouth/Praat** — line 538)
- Pause ratio, pause count, pause mean duration (via Silero VAD — lines 641-692)
- Speaking rate, syllable rate (words/syllables per second)
- Duration ratio (actual / expected from text length)
- Energy mean, energy std (via Parselmouth intensity — line 571)
- Dynamic range in dB (frame-level RMS percentile range)

**Robustness (3 active metrics):**
- Repetition detection (n-gram repeated 3+ times)
- Silence anomaly detection (from VAD pause data, line 838-839)
- Insertion rate (from intelligibility, reused here)

**Latency (1 active metric in dimension scoring):**
- RTF (Real-Time Factor) — from generation metadata
- (Also computed but not in dimension: TTFA, raw inference time)

#### Stage 4: `aggregate.py` — Score Aggregation (CORE NOVELTY)
- **Normalizes** all 28 dimension metrics to [0, 1] scale (min-max per metric, direction-aware)
- **Averages** into 6 dimension scores: Naturalness, Intelligibility, Speaker Similarity, Prosody, Robustness, Latency
- **Computes 5 weighted use-case composite scores** (THIS IS THE KEY CONTRIBUTION):
  - **Audiobook**: Naturalness (0.35), Prosody (0.25), Intelligibility (0.15), Robustness (0.15), Speaker Similarity (0.10)
  - **Conversational AI**: Latency (0.30), Intelligibility (0.25), Naturalness (0.20), Robustness (0.15), Prosody (0.10)
  - **Voice Clone**: Speaker Similarity (0.40), Naturalness (0.20), Intelligibility (0.15), Latency (0.10), Robustness (0.10), Prosody (0.05)
  - **Low-Latency**: Latency (0.45), Intelligibility (0.20), Naturalness (0.15), Robustness (0.15), Prosody (0.05)
  - **Balanced**: Naturalness (0.20), Intelligibility (0.20), Latency (0.20), Speaker Similarity (0.15), Robustness (0.15), Prosody (0.10)
- Computes **Wilcoxon signed-rank p-values** for all pairwise model comparisons (statistical significance)
- Outputs per-model dimension scores, composite scores, and p-value matrix as JSON/CSV

#### Stage 5: `visualize.py` — Chart Generation
- 6 visualization types:
  - Leaderboard bar chart (composite scores)
  - Radar/spider chart (dimension profiles per model)
  - Heatmap (models × metrics with color intensity)
  - Use-case grouped bar chart (5 composites × 5 models)
  - WER distribution box plots per model
  - UTMOS violin plots per model
- Also generates **dataset difficulty analysis** (which challenge categories are hardest for which models)

#### Stage 6: `architecture_diagram.py` — Pipeline Visualization
- Renders the full pipeline as a figure using matplotlib/graphviz

### 5 TTS Model Clients

All inherit from `BaseTTSClient` → return `TTSResult` (audio array + sample_rate + metadata):

| Model | File | Type | Cloning | Speed | License |
|-------|------|------|---------|-------|---------|
| **Deepgram Aura** | `deepgram_client.py` | Cloud API | No | Fast | Commercial |
| **Cartesia Sonic** | `cartesia_client.py` | Cloud API | Yes | Fast | Commercial |
| **Kokoro** | `kokoro_client.py` | Local ONNX | No | Medium | Apache 2.0 |
| **Piper** | `piper_client.py` | Local | No | Fastest | MIT |
| **Coqui XTTS v2** | `xtts_client.py` | Local | Yes | Slowest (~20s/utt) | CPML |

Config: `configs/models.yaml` — each model has name, client class, API key reference, default voice, capabilities flags.

### B. `voice_evals/` — Conversational Voice AI Evaluation (Secondary Framework)

This is a separate but related framework for evaluating voice AI in conversational settings:

- **`core/pipeline.py`**: Full pipeline with Pyannote.audio 3.1 speaker diarization, per-speaker analysis, timeline visualization, RTTM export
- **`core/diarization.py`**: Diarization wrapper
- **`pipeline2.py`** (`EnhancedVoiceEvaluator`): WER/CER/MER/WIP/WIL + **semantic metrics**:
  - **SeMaScore**: BERT cosine similarity + MER penalty + importance weighting
  - **SAER**: LaBSE multilingual embeddings, λ-weighted form/semantic blend
  - **ASD**: Aligned token-level semantic distance
- **`test2.py` / `testev.py`**: VAD chunking, streaming WER, noise perturbation robustness, emotion detection (emotion2vec), UTMOS, prosody analysis
- **`metrics/task_completion.py`**: PARADISE framework slot-filling evaluator for goal-oriented dialogue

> **For the paper**: voice_evals provides supplementary depth (semantic metrics, diarization) but the primary contribution is `tts_benchmark/`. Consider mentioning SeMaScore/SAER as future work or briefly in methodology if space allows.

### Key Files to Reference

```
main/tts_benchmark/
├── datasets/download.py      # Data preparation + Challenge Set definition
├── generate.py               # TTS generation orchestrator
├── evaluate.py               # 28-metric evaluation engine (+ 13 diagnostic)
├── aggregate.py              # Normalization + composites + Wilcoxon
├── visualize.py              # 6 chart types
├── architecture_diagram.py   # Pipeline figure
├── models/
│   ├── base_client.py        # BaseTTSClient abstract class
│   ├── deepgram_client.py
│   ├── cartesia_client.py
│   ├── kokoro_client.py
│   ├── piper_client.py
│   └── xtts_client.py
├── configs/models.yaml       # Model configuration
├── README.md
├── QUICKSTART.md
├── run_all.sh                # Full pipeline execution script
└── setup.sh                  # Environment setup

voice_evals/
├── core/pipeline.py          # Diarization pipeline
├── core/diarization.py
├── metrics/enhanced_metrics.py
├── metrics/task_completion.py
├── run.py
└── README.md

pipeline2.py                  # EnhancedVoiceEvaluator (SeMaScore, SAER, ASD)
```

---

## INTERSPEECH 2026 — COMPLETE REQUIREMENTS

### Logistics
- **Conference**: 27 Sep – 1 Oct 2026, ICC Sydney, Australia
- **Theme**: "Speaking Together" — speech across languages, cultures, modalities
- **Submission Deadline**: **25 February 2026 AoE** (3 days from now)
- **Paper Update Deadline**: 4 March 2026 AoE (1 week polish window after initial submission)
- **Rebuttal Period**: 24 Apr – 1 May 2026
- **Notification**: 5 June 2026
- **Camera-Ready**: 19 June 2026

### Paper Format
- **Regular Paper**: 4 pages content + 2 pages references/acknowledgments (~50% acceptance)
- **Long Paper** (NEW 2026): 8 pages content + 2 pages references (<30% acceptance target)
- **Our choice: Regular 4-page paper** (achievable in 3 days, still competitive at ~50% acceptance)
- Template: **Interspeech 2026 Paper Kit** from Overleaf (ISCA archive LaTeX style)
- Two-column format, specific ISCA margins, embedded fonts, searchable PDF, sanitized metadata

### Topic Areas (Target These)
- **Resources and Evaluation** ← PRIMARY (historically underserved, our paper fits perfectly)
- **Speech Synthesis** ← SECONDARY
- **Generative AI for Speech and Language Processing** ← NEW for 2026, committee actively seeking papers here
- **Conversational AI Systems** ← if we emphasize the conversational-AI use-case composite

### Explicit Review Criteria
1. **Novelty and originality** — weighted composites + adversarial challenge set
2. **Technical correctness** — 28 validated metrics across 6 dimensions, Wilcoxon p-values, proper normalization
3. **Clarity of presentation** — figures, tables, concise methodology
4. **Key strengths** — what sets this apart from existing benchmarks
5. **Quality of references** — MUST be primarily peer-reviewed (arXiv "kept to a minimum")

### Scientific Checklist (Reviewers Check These)
- [ ] Clear claims with explicit novelty statement
- [ ] Limitations acknowledged
- [ ] Model/architecture details (TTS model specs, metric implementations)
- [ ] Dataset details: languages, duration, splits, preprocessing, exclusion criteria
- [ ] Evaluation metrics explained with justification
- [ ] **Statistical significance** reported (Wilcoxon p-values — we have this!)
- [ ] Code availability or reproducibility details
- [ ] Computing infrastructure, runtimes, parameter counts

### Mandatory Requirements
- **Double-blind**: Paper must NOT reveal author identity → rejected without review if it does
- **Generative AI Disclosure**: REQUIRED section between Acknowledgments and References acknowledging use of AI tools (we MUST disclose use of Scientific Writer / Claude)
- **No anonymity period for arXiv** (changed from previous years) — submission must be anonymous but preprints are OK
- **References**: primarily peer-reviewed, minimize arXiv-only citations
- **PDF**: embedded fonts, searchable text, no author-identifying metadata
- **Optional supplementary**: 100 MB ZIP, must be anonymized

### IMPLICIT Requirements (Inferred From Accepted Papers & Review Patterns)

**What reviewers actually look for but don't explicitly state:**

1. **"Resources and Evaluation" is underserved** — most submissions cluster around ASR, synthesis, speaker ID. A rigorous benchmark paper stands out IF it advances methodology, not just runs existing metrics.

2. **Papers proposing frameworks > papers reporting numbers.** Accepted TTS evaluation papers at Interspeech 2024-2025 include:
   - "The State of TTS" (2025) — human fooling rates as novel evaluation angle
   - "Contextual Interactive Evaluation of TTS" (2024) — TTS in dialogue context
   - "Benchmarking Responsiveness" (2025) — latency as first-class dimension
   - TTSDS benchmark (Minixhofer et al., 2024) — correlated objective metrics with human evals
   - Papers that just run MOS tests on few models → REJECTED

3. **Reviewers aggressively check for statistical significance.** Many TTS papers get rejected for p-hacking or no significance testing. Our Wilcoxon p-values put us ahead.

4. **Reproducibility is quasi-mandatory.** Plan anonymous GitHub repo + Zenodo archive. Mention "code and data will be released upon acceptance" in the paper.

5. **Analysis beyond tables is the differentiator.** Raw metric tables are commodity. Our use-case composites, dataset difficulty analysis, and per-category breakdowns are what make this publishable.

6. **Theme alignment matters for borderline papers.** "Speaking Together" rewards multilingualism, cross-cultural work. Our code-switching challenge set (French/German in English) directly aligns. EMPHASIZE THIS.

7. **The rebuttal period is 1 week.** Prepare for these reviewer questions:
   - "Why these 5 models and not [X]?" → Justify: 2 commercial API + 3 open-source spanning different architectures
   - "No human evaluation?" → Acknowledge as limitation, cite TTSDS correlation work
   - "How do composites correlate with human preferences?" → Future work, but weights are principled
   - "Challenge set is too small" → Qualitative insight matters more than scale for adversarial testing

8. **Pre-print strategy**: Put on arXiv immediately after submission. Timestamps your work, builds visibility.

---

## PAPER STRATEGY

### Title (Working)
**"TTS-Bench: A Multi-Dimensional Evaluation Framework with Use-Case-Driven Composite Scoring for Text-to-Speech Systems"**

### Core Claims (In Order of Novelty)
1. **Use-case-driven composite scoring**: 5 weighted profiles (Audiobook, Conversational-AI, Voice Clone, Low-Latency, Balanced) that aggregate 28 metrics into actionable scores — practitioners can pick the composite matching their deployment scenario. NO EXISTING BENCHMARK DOES THIS.
2. **7-category adversarial Challenge Set**: stress-tests TTS beyond standard read speech — code-switching, emotional, tongue twisters, long-form, questions, numbers, proper nouns. Most benchmarks only use clean read speech.
3. **28 metrics across 6 dimensions** with principled normalization and aggregation, vs. the typical MOS + WER.
4. **Statistical rigor**: Wilcoxon signed-rank tests for all pairwise comparisons.
5. **Commercial vs. open-source comparison**: Deepgram/Cartesia APIs alongside Kokoro/Piper/XTTS — rarely done in academic benchmarks due to API cost/access.

### Paper Structure (4 Pages + 2 Refs)

```
Page 1:  Abstract (150 words) + Introduction (motivation, gap, contributions)
Page 1-2: Related Work (0.5 page — existing TTS benchmarks and their limitations)
Page 2-3: Methodology
         - Framework Architecture (pipeline diagram, 6 stages)
         - Evaluation Metrics (table: 28 metrics grouped by 6 dimensions)
         - Challenge Set (7 categories with examples, motivation for each)
         - Composite Scoring (formulas, use-case weight tables)
Page 3-4: Experiments
         - Setup (5 models, hardware, config)
         - Results (leaderboard table, radar chart, 1-2 key findings)
         - Analysis (which models excel where, challenge set difficulty, p-values)
Page 4:  Conclusion + Limitations + Future Work
         Generative AI Use Disclosure
         Acknowledgments
Pages 5-6: References (15-20 peer-reviewed citations)
```

### What Makes This Paper Publishable
- The composites are a **genuine methodological contribution** — they change HOW TTS is evaluated, not just WHAT is measured
- The challenge set reveals model weaknesses invisible to standard benchmarks
- Statistical significance via Wilcoxon (ahead of most TTS papers)
- Direct alignment with "Speaking Together" theme (code-switching) and "Generative AI for Speech" topic area
- Open-source: framework + challenge set will be released

---

## HOW TO USE SCIENTIFIC WRITER SKILLS

You have these skills available. Use them as tools, not as the primary author:

### `@research-lookup`
**Use for**: Finding peer-reviewed references on TTS evaluation, speech quality metrics (UTMOS, NISQA, DNSMOS), existing benchmarks (Seed-TTS-Eval, TTSDS, VoiceMOS Challenge), adversarial evaluation.
**Requires**: OPENROUTER_API_KEY in `.env` (powers Perplexity Sonar Pro search)
**CRITICAL**: Verify EVERY citation returned. It hallucinates references. Cross-check on Google Scholar.

### `@citation-management`
**Use for**: Generating BibTeX entries from verified paper lists. Output to `references.bib`.
**CRITICAL**: Verify DOIs, page numbers, venue names after generation.

### `@peer-review`
**Use for**: Running ScholarEval 8-dimension scoring on draft. Scores: Problem Formulation, Literature Review, Methodology, Data Collection, Analysis, Results, Writing Quality, Citations. Scale 1-5. Target 4.0+ overall.

### `@scholar-evaluation`
**Use for**: Comprehensive evaluation of submission readiness. Check double-blind compliance, figure references, citation completeness, statistical claims, page limits.

### `@venue-templates`
**Use for**: LaTeX formatting guidance. NOTE: It does NOT have Interspeech template. Use official Interspeech 2026 Paper Kit from Overleaf (ISCA style). Use this skill for general best practices only.

### `@scientific-schematics`
**Use for**: Generating pipeline architecture diagrams, metric taxonomy figures. Uses Nano Banana Pro.

### General Paper Writing
Just describe what you want: "Write the Introduction section focusing on [X]" — the `scientific-writing` skill activates automatically.

---

## CONSTRAINTS AND RULES

1. **DOUBLE-BLIND**: Never include author names, affiliations, or identifying information in any generated content. No "our previous work [X]" with identifiable citations.

2. **AI DISCLOSURE**: The paper MUST include a "Generative AI Use Disclosure" section stating that Claude / Scientific Writer was used for literature search assistance and draft review. AI tools were NOT used to produce a significant part of the manuscript.

3. **REFERENCES**: Prioritize peer-reviewed venues (Interspeech, ICASSP, IEEE/ACM TASLP, ACL, EMNLP). Minimize arXiv-only papers. Each reference must be real and verifiable.

4. **PAGE LIMIT**: 4 pages content MAXIMUM. Every sentence must earn its place. No filler, no redundancy. References/acknowledgments on pages 5-6 only.

5. **LATEX**: Use ISCA Interspeech template. Two-column. 9pt font. Specific margins per template.

6. **STATISTICAL CLAIMS**: Every performance comparison must cite the Wilcoxon p-value. No unsupported superlatives.

7. **REPRODUCIBILITY**: State that code and challenge set will be released. Mention computing hardware, Python version, key library versions.

8. **FIGURES**: Must be referenced in text. Must be readable at print size. Prefer vector (PDF/SVG) over raster for charts.

---

## 3-DAY EXECUTION PLAN

### Day 1 (Feb 22 — TODAY): Foundation
- [ ] Run full pipeline: `run_all.sh` → get actual results CSV + figures
- [ ] `@research-lookup` → find 20 key references, verify each
- [ ] `@citation-management` → generate references.bib
- [ ] Set up Interspeech LaTeX template in repo
- [ ] Write skeleton: all section headers, figure/table placeholders

### Day 2 (Feb 23): Writing Sprint
- [ ] Write Methodology section (framework, metrics table, challenge set, composites)
- [ ] Write Experiments section (setup, results table, analysis)
- [ ] Write Introduction (gap → contribution → "this paper presents")
- [ ] Write Related Work (existing benchmarks → limitations → our advance)
- [ ] Insert figures: architecture diagram, leaderboard, radar chart

### Day 3 (Feb 24): Polish & Submit
- [ ] Write Abstract (150 words, last thing written)
- [ ] Write Conclusion + Limitations + Future Work
- [ ] Add Generative AI Disclosure
- [ ] `@peer-review` → ScholarEval scoring → fix weaknesses
- [ ] `@scholar-evaluation` → submission readiness check
- [ ] Double-blind compliance check (no author info anywhere)
- [ ] Compile PDF, verify formatting, sanitize metadata
- [ ] **SUBMIT before AoE Feb 25**

### Post-Submission (Feb 25 – Mar 4):
- Update deadline is March 4 AoE — use this week to polish
- Prepare anonymous GitHub + Zenodo for reproducibility
- Prepare arXiv preprint

---

## QUICK COMMAND REFERENCE

```bash
# Run the full benchmark pipeline
cd main/tts_benchmark && bash run_all.sh

# Or run stages individually
python datasets/download.py          # Stage 1
python generate.py                    # Stage 2
python evaluate.py                    # Stage 3
python aggregate.py                   # Stage 4
python visualize.py                   # Stage 5
python architecture_diagram.py        # Stage 6
```

---

## IMPORTANT CONTEXT FOR WRITING QUALITY

When writing any section of this paper, remember:
- **Interspeech reviewers are speech/audio experts** — don't over-explain WER or MOS but DO explain novel elements (composites, challenge set design rationale)
- **Be concise** — 4 pages means every paragraph must advance the argument
- **Lead with the "so what"** — not "we computed 28 metrics" but "existing benchmarks reduce TTS quality to a single MOS score, obscuring critical deployment-specific tradeoffs"
- **Use active voice** — "We propose" not "A framework is proposed"
- **Quantify everything** — "7 categories, 350 total utterances" not "several categories"
- **One key finding per paragraph** in Results
- **Figures should be self-contained** — caption must explain what's shown without reading the text
