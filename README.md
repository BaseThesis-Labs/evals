Enhanced Voice Evaluation Pipeline
A comprehensive speech-to-text evaluation framework that computes string-level accuracy metrics (WER, CER, MER, WIP, WIL), semantic accuracy metrics (SeMaScore, SAER, ASD), real-time processing speed (RTFx), prosody analysis, emotion detection, and more.

Features

String-level accuracy — WER, normalized WER, CER, MER, WIP, WIL
Semantic accuracy — SeMaScore (BERT + MER penalty), SAER (LaBSE), ASD (aligned cosine distance)
Transcription — Whisper-based ASR with processing time tracking
RTFx — Real-time factor; measures how fast transcription runs vs. audio duration
Prosody analysis — Pitch std dev, monotone score, pace, intonation, overall prosody
Emotion detection — Via emotion2vec, with per-emotion confidence scores
Speech quality — UTMOS MOS prediction (optional)
Timing metrics — AI/user speaking time, talk ratio, WPM, average latency
Behavioral flags — Interruption detection, early termination detection
Stereo channel separation — AI and user channels analyzed independently


Installation
Core dependencies
bashpip install librosa numpy torch torchaudio
pip install transformers          # Whisper transcription + BERT (SeMaScore, ASD)
pip install tiktoken              # Token counting
pip install sentence-transformers # LaBSE for SAER
pip install funasr                # Emotion2vec emotion detection
Optional: UTMOS speech quality scoring
bash# Download the UTMOS model and pass its directory via --utmos_model_dir

Quick Start
bash# Basic usage (no ground truth)
python enhanced_voice_eval.py audio.wav

# With ground truth transcript
***python enhanced_voice_eval.py audio.wav transcript.txt***

# Full options
python enhanced_voice_eval.py audio.wav transcript.txt <utmos_dir> <saer_lambda> <saer_lang>
Examples
bash# English monolingual
python enhanced_voice_eval.py call.wav ground_truth.txt '' 0.5 en

# Chinese (uses CER for form-error in SAER)
python enhanced_voice_eval.py call.wav ground_truth.txt '' 0.5 zh

# Code-switching / mixed language
python enhanced_voice_eval.py call.wav ground_truth.txt '' 0.5 mixed
Results are printed to stdout and saved as <audio_stem>_enhanced_eval.json.
