# References v3 Summary — VoiceBench (Interspeech 2026)

**Total: 33 entries | Peer-reviewed: 20 | arXiv-only: 3 | Software/misc: 10**

## Changes from v2 → v3

### Publication Status Verified (2 entries)
- `srinivasavaradhan2025statetts` → **Confirmed** published in Interspeech 2025 proceedings (ISCA archive), not arXiv-only
- `minixhofer2025ttsds2` → **Confirmed** published in SSW 2025 proceedings (ISCA archive), booktitle updated to "13th ISCA Speech Synthesis Workshop"

### Replaced (1 entry)
- `ravanelli2021speechbrain` → **Replaced** with `ravanelli2024speechbrain` — JMLR Volume 25, No. 249, 2024 (was arXiv-only, now peer-reviewed)

### Added (5 new entries)
- `taal2011stoi` — STOI intelligibility metric (IEEE TASLP 2011, 2000+ citations) — **used in evaluate.py**
- `jemine2019resemblyzer` — Resemblyzer speaker embeddings (@misc) — **used in evaluate.py lines 476-524**
- `kokoro2025` — Kokoro TTS 82M model (@misc, HuggingFace) — evaluated system
- `piper2023` — Piper TTS (@misc, GitHub) — evaluated system
- `deepgram2024aura` — Deepgram Aura TTS API (@misc) — evaluated system
- `cartesia2024sonic` — Cartesia Sonic TTS (@misc) — evaluated system

### Title Confirmed (1 entry)
- `deja2022calibrated` → Title confirmed as "Automatic Evaluation of Speaker Similarity" (was already correct in v2)

### Code Verification Annotations Added
Every entry now has `USED IN CODE:` annotation indicating whether/how it's used in evaluate.py/aggregate.py.

---

## Code-Verified Metric Usage (from evaluate.py + aggregate.py)

| Metric | Code Status | Reference | Lines |
|--------|------------|-----------|-------|
| UTMOS | **Active** | saeki2022utmos | ~800 |
| DNSMOS (sig/bak/ovrl) | **Active** | reddy2021dnsmos | 259-323 |
| PESQ | **Active** | itu2001pesq | 327-352 |
| STOI | **Active** | taal2011stoi | ~805 |
| Whisper ASR (WER/CER/MER) | **Active** | radford2023whisper | 60 |
| Parselmouth (F0/jitter/shimmer/HNR) | **Active** | jadoul2018parselmouth | 538-602 |
| Resemblyzer (GE2E cosine sim) | **Active** | jemine2019resemblyzer | 476-524 |
| Silero VAD (SNR/pauses) | **Active** | *(no paper — used as tool)* | 381-437, 641-692 |
| sentence-transformers (semantic dist) | **Active** | *(no paper — used as tool)* | 168-201 |
| ECAPA-TDNN | **Disabled** | desplanques2020ecapa | 472 (always None) |
| NISQA | **Stubbed** | mittag2021nisqa | 244-253 (returns None) |
| SCOREQ | **Stubbed** | *(removed from refs)* | 238-241 (returns None) |

---

## Category 1: TTS Benchmarks & Evaluation Frameworks (9 papers)

| Key | Title | Venue | Year | Cite For | Status |
|-----|-------|-------|------|----------|--------|
| minixhofer2024ttsds | TTSDS Distribution Score | SLT 2024 | 2024 | Related Work (primary competitor) | VERIFIED |
| minixhofer2025ttsds2 | TTSDS2 Human-Quality Eval | SSW 2025 | 2025 | Related Work (human correlation) | VERIFIED |
| srinivasavaradhan2025statetts | State of TTS / Human Fooling Rate | Interspeech 2025 | 2025 | Intro + Related Work (MOS insufficient) | VERIFIED |
| huang2022voicemos | VoiceMOS Challenge 2022 | Interspeech 2022 | 2022 | Related Work (community benchmark) | VERIFIED |
| huang2024voicemos | VoiceMOS Challenge 2024 | Interspeech 2024 | 2024 | Related Work (multi-task eval trend) | VERIFIED |
| kirkland2023mos | Stuck in the MOS Pit | SSW 2023 | 2023 | Introduction (MOS unreliable) | VERIFIED |
| yang2025responsible | Responsible TTS Evaluation | arXiv 2025 | 2025 | Related Work (3-level framework) | arXiv-only |
| wang2024contextual | Contextual TTS Evaluation | Interspeech 2024 | 2024 | Related Work (beyond sentence-level) | VERIFIED |
| manocha2021noresqa | NORESQA Non-Matching Refs | NeurIPS 2021 | 2021 | Related Work (beyond-MOS) | VERIFIED |

## Category 2: Speech Quality & Similarity Metrics (9 papers)

| Key | Title | Venue | Year | Cite For | Code Status |
|-----|-------|-------|------|----------|-------------|
| saeki2022utmos | UTMOS MOS Predictor | Interspeech 2022 | 2022 | Methods (naturalness) | **Active** |
| mittag2021nisqa | NISQA Multidimensional | Interspeech 2021 | 2021 | Methods (planned metric) | Stubbed |
| reddy2021dnsmos | DNSMOS Non-Intrusive | ICASSP 2021 | 2021 | Methods (naturalness) | **Active** |
| desplanques2020ecapa | ECAPA-TDNN Speaker Verif | Interspeech 2020 | 2020 | Methods (speaker sim) | Disabled |
| itu2001pesq | PESQ ITU Standard | ITU-T P.862 | 2001 | Methods (signal quality) | **Active** |
| taal2011stoi | STOI Intelligibility | IEEE TASLP 2011 | 2011 | Methods (intelligibility) | **Active** |
| jadoul2018parselmouth | Parselmouth Praat | J. Phonetics 2018 | 2018 | Methods (prosody) | **Active** |
| deja2022calibrated | Speaker Similarity | Interspeech 2022 | 2022 | Methods (embedding valid.) | VERIFIED |
| lei2022msemotts | MsEmoTTS Emotional | TASLP 2022 | 2022 | Related Work (emotional) | VERIFIED |

## Category 3: Adversarial / Robustness Evaluation (4 papers)

| Key | Title | Venue | Year | Cite For | Status |
|-----|-------|-------|------|----------|--------|
| manku2025emergentttseval | EmergentTTS-Eval | arXiv 2025 | 2025 | Related Work (challenge set) | arXiv-only |
| feng2024robustness | TTS Robustness to Noise | Interspeech 2024 | 2024 | Related Work (robustness) | VERIFIED |
| wu2024mole | MoLE Multilingual TTS | Interspeech 2024 | 2024 | Related Work (code-switching) | VERIFIED |
| zhou2020codeswitching | Code-Switching TTS | ICASSP 2020 | 2020 | Related Work (code-switching) | VERIFIED |

## Category 4: TTS Models & Systems (3 papers)

| Key | Title | Venue | Year | Cite For | Status |
|-----|-------|-------|------|----------|--------|
| casanova2024xtts | XTTS Multilingual | Interspeech 2024 | 2024 | Exp. Setup (evaluated) | VERIFIED |
| anastassiou2024seedtts | Seed-TTS Generation | arXiv 2024 | 2024 | Exp. Setup (dataset) | arXiv-only |
| radford2023whisper | Whisper ASR | ICML 2023 | 2023 | Methods (WER/CER) | VERIFIED |

## Category 5: Toolkits & Software (5 entries)

| Key | Title | Type | Year | Cite For | Code Status |
|-----|-------|------|------|----------|-------------|
| ravanelli2024speechbrain | SpeechBrain 1.0 | JMLR 2024 | 2024 | Methods (ECAPA-TDNN) | Disabled |
| jemine2019resemblyzer | Resemblyzer GE2E | @misc (GitHub) | 2019 | Methods (speaker sim) | **Active** |
| kokoro2025 | Kokoro 82M TTS | @misc (HuggingFace) | 2025 | Exp. Setup (evaluated) | **Active** |
| piper2023 | Piper TTS | @misc (GitHub) | 2023 | Exp. Setup (evaluated) | **Active** |
| deepgram2024aura | Deepgram Aura | @misc (API docs) | 2024 | Exp. Setup (evaluated) | **Active** |
| cartesia2024sonic | Cartesia Sonic | @misc (API docs) | 2024 | Exp. Setup (evaluated) | **Active** |
| elevenlabs2024 | ElevenLabs TTS API | @misc (API docs) | 2024 | Exp. Setup (evaluated) | **Active** |
| hume2024 | Hume AI EVI | @misc (API docs) | 2024 | Exp. Setup (evaluated) | **Active** |
| lmnt2024 | LMNT TTS API | @misc (API docs) | 2024 | Exp. Setup (evaluated) | **Active** |

---

## Summary Statistics

| Category | Count | Peer-reviewed | arXiv | @misc |
|----------|-------|--------------|-------|-------|
| TTS Benchmarks & Eval | 9 | 8 | 1 | 0 |
| Quality & Similarity Metrics | 9 | 9 | 0 | 0 |
| Adversarial / Robustness | 4 | 3 | 1 | 0 |
| TTS Models & Systems | 3 | 2 | 1 | 0 |
| Toolkits & Software | 9 | 1 | 0 | 8 |
| **Total** | **33** | **23** | **3** | **8** |

Note: @misc entries are intentional software citations with no paper equivalent. These are standard practice for citing open-source tools and commercial APIs.

### Added in v3.1 (model count correction)
- `elevenlabs2024` — ElevenLabs TTS API (@misc) — evaluated system
- `hume2024` — Hume AI EVI (@misc) — evaluated system
- `lmnt2024` — LMNT TTS API (@misc) — evaluated system

## arXiv-only papers (3/30)
1. `yang2025responsible` — position paper, no conference version
2. `manku2025emergentttseval` — recent, may appear at venue later
3. `anastassiou2024seedtts` — ByteDance industry paper, widely cited

Previous arXiv-only entries now resolved:
- ~~`ravanelli2021speechbrain`~~ → Now `ravanelli2024speechbrain` (JMLR V25)
- ~~`srinivasavaradhan2025statetts`~~ → Confirmed Interspeech 2025
