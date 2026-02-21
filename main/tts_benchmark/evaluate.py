#!/usr/bin/env python3
"""Evaluate TTS outputs across all metrics."""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import tempfile
import warnings
warnings.filterwarnings('ignore')


# Global model cache
MODELS = {}


def load_manifest(manifest_path: Path) -> List[Dict]:
    """Load utterance manifest."""
    with open(manifest_path) as f:
        return json.load(f)


def load_gen_metadata(model_dir: Path) -> Dict:
    """Load generation metadata."""
    meta_path = model_dir / "gen_meta.jsonl"
    if not meta_path.exists():
        return {}

    metadata = {}
    with open(meta_path) as f:
        for line in f:
            entry = json.loads(line)
            metadata[entry['id']] = entry

    return metadata


def get_asr_transcript(audio_path: str, provider: str = "whisper") -> str:
    """Get ASR transcript using Whisper, AssemblyAI or Deepgram."""
    if provider == "whisper":
        return get_whisper_transcript(audio_path)
    elif provider == "assemblyai":
        return get_assemblyai_transcript(audio_path)
    elif provider == "deepgram":
        return get_deepgram_transcript(audio_path)
    else:
        raise ValueError(f"Unknown ASR provider: {provider}")


def get_whisper_transcript(audio_path: str) -> str:
    """Transcribe audio using Whisper (local, free)."""
    import whisper

    if 'whisper' not in MODELS:
        print("  Loading Whisper model (one-time, ~1 min)...")
        model = whisper.load_model("base")  # base, small, medium, large
        MODELS['whisper'] = model

    model = MODELS['whisper']
    result = model.transcribe(audio_path, language='en')

    return result['text'].strip().lower()


def get_assemblyai_transcript(audio_path: str) -> str:
    """Transcribe audio using AssemblyAI."""
    import assemblyai as aai

    if 'assemblyai_client' not in MODELS:
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not set")
        aai.settings.api_key = api_key
        MODELS['assemblyai_client'] = True  # Just mark as initialized

    # Create new transcriber each time (API)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path)

    # Check if transcription succeeded
    if transcript and transcript.text:
        return transcript.text.strip().lower()
    else:
        return ""  # Return empty string if failed


def get_deepgram_transcript(audio_path: str) -> str:
    """Transcribe audio using Deepgram."""
    from deepgram import DeepgramClient, PrerecordedOptions

    if 'deepgram_client' not in MODELS:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")
        MODELS['deepgram_client'] = DeepgramClient(api_key)

    client = MODELS['deepgram_client']

    with open(audio_path, "rb") as f:
        source = {"buffer": f.read()}

    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
    )

    response = client.listen.prerecorded.v("1").transcribe_file(source, options)
    text = response.results.channels[0].alternatives[0].transcript

    return text.strip().lower()


def compute_intelligibility_metrics(reference: str, hypothesis: str) -> Dict:
    """Compute WER, CER, MER, normalized WER, and alignment-based metrics."""
    from jiwer import wer, cer, mer, process_words

    # Normalize
    reference = reference.strip().lower()
    hypothesis = hypothesis.strip().lower()

    metrics = {}

    # WER, CER, and MER
    try:
        metrics['wer'] = wer(reference, hypothesis)
        metrics['cer'] = cer(reference, hypothesis)
        metrics['mer'] = mer(reference, hypothesis)  # Match Error Rate

        # Normalized WER (0-100 scale)
        metrics['wer_normalized'] = metrics['wer'] * 100.0
        metrics['mer_normalized'] = metrics['mer'] * 100.0
    except:
        metrics['wer'] = 1.0
        metrics['cer'] = 1.0
        metrics['mer'] = 1.0
        metrics['wer_normalized'] = 100.0
        metrics['mer_normalized'] = 100.0

    # ASR mismatch
    metrics['asr_mismatch'] = 1 if metrics['wer'] > 0 else 0
    metrics['asr_mismatch_rate'] = metrics['wer']

    # Word-level alignment
    try:
        alignment = process_words(reference, hypothesis)
        ref_words = len(reference.split())

        if ref_words > 0:
            metrics['word_skip_rate'] = alignment.deletions / ref_words
            metrics['insertion_rate'] = alignment.insertions / ref_words
            metrics['substitution_rate'] = alignment.substitutions / ref_words
        else:
            metrics['word_skip_rate'] = 0.0
            metrics['insertion_rate'] = 0.0
            metrics['substitution_rate'] = 0.0
    except:
        metrics['word_skip_rate'] = None
        metrics['insertion_rate'] = None
        metrics['substitution_rate'] = None

    return metrics


def compute_semantic_distance(reference: str, hypothesis: str) -> Optional[float]:
    """Compute semantic distance between reference text and ASR hypothesis.

    Uses sentence-transformers cosine distance. Range [0, 1]:
      0.0 = identical meaning, 1.0 = completely unrelated.
    Lower is better for intelligibility.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        if 'sentence_transformer' not in MODELS:
            # all-MiniLM-L6-v2 is small (80 MB) and fast
            MODELS['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')

        model = MODELS['sentence_transformer']

        ref = reference.strip().lower()
        hyp = hypothesis.strip().lower()

        if not ref or not hyp:
            return None

        embeddings = model.encode([ref, hyp], convert_to_tensor=True)
        cos_sim = float(
            torch.nn.functional.cosine_similarity(
                embeddings[0].unsqueeze(0),
                embeddings[1].unsqueeze(0)
            ).item()
        )
        # Convert similarity → distance (0 = identical, 1 = opposite)
        return max(0.0, 1.0 - cos_sim)
    except Exception as e:
        return None


def compute_utmos(audio: np.ndarray, sr: int) -> float:
    """Compute UTMOS score."""
    try:
        import torch

        if 'utmos' not in MODELS:
            print("  Loading UTMOS model (one-time)...")
            model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            model = model.cpu().eval()
            MODELS['utmos'] = model

        model = MODELS['utmos']

        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Ensure 1D array
        if audio.ndim > 1:
            audio = audio.flatten()

        # Convert to tensor - UTMOS expects 1D tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            score = model(audio_tensor, sr)

        return float(score.item())
    except Exception as e:
        print(f"    UTMOS error: {e}")
        return None


def compute_scoreq(audio: np.ndarray, sr: int) -> float:
    """Compute SCOREQ score."""
    # SCOREQ installation is unreliable - skip for now
    return None


def compute_nisqa(audio_path: str) -> Dict:
    """Compute NISQA scores."""
    # NISQA is complex - skip for now
    return {
        'nisqa_mos': None,
        'nisqa_noisiness': None,
        'nisqa_coloration': None,
        'nisqa_discontinuity': None,
        'nisqa_loudness': None
    }


def compute_dnsmos(audio: np.ndarray, sr: int) -> Dict:
    """Compute DNSMOS scores using ONNX combined model."""
    try:
        import onnxruntime as ort
        from pathlib import Path

        # Load ONNX model (single combined model predicts all 3 scores)
        if 'dnsmos' not in MODELS:
            model_dir = Path.home() / '.cache' / 'dnsmos'
            model_path = model_dir / 'sig_bak_ovr.onnx'

            if not model_path.exists():
                return {'dnsmos_sig': None, 'dnsmos_bak': None, 'dnsmos_ovrl': None}

            # Load the combined model
            MODELS['dnsmos'] = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )

        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.flatten()

        # DNSMOS requires exactly 144160 samples (9.01 seconds at 16kHz)
        target_length = 144160
        if len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            # Truncate
            audio = audio[:target_length]

        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        audio = audio.astype(np.float32).reshape(1, -1)

        # Run inference - combined model outputs all 3 scores at once
        model = MODELS['dnsmos']
        outputs = model.run(None, {'input_1': audio})

        # The model outputs [sig, bak, ovrl] as separate tensors or a single tensor
        # Try to extract the scores based on output structure
        if len(outputs) == 3:
            sig_score, bak_score, ovr_score = outputs[0][0], outputs[1][0], outputs[2][0]
        elif len(outputs) == 1 and len(outputs[0][0]) == 3:
            sig_score, bak_score, ovr_score = outputs[0][0]
        else:
            # Fallback: assume first output contains all scores
            scores = outputs[0][0] if len(outputs[0].shape) > 1 else outputs[0]
            sig_score, bak_score, ovr_score = scores[0], scores[1], scores[2]

        return {
            'dnsmos_sig': float(sig_score),
            'dnsmos_bak': float(bak_score),
            'dnsmos_ovrl': float(ovr_score)
        }
    except Exception as e:
        print(f"    DNSMOS error: {e}")
        return {
            'dnsmos_sig': None,
            'dnsmos_bak': None,
            'dnsmos_ovrl': None
        }


def compute_pesq(synth_audio: np.ndarray, synth_sr: int,
                 ref_audio: np.ndarray, ref_sr: int) -> Optional[float]:
    """Compute PESQ score (ITU-T P.862).

    Requires reference audio. Range: -0.5 to 4.5 (higher = better).
    Only valid when reference audio is available.
    """
    try:
        from pesq import pesq

        target_sr = 16000  # wideband mode

        if synth_sr != target_sr:
            synth_audio = librosa.resample(synth_audio, orig_sr=synth_sr, target_sr=target_sr)
        if ref_sr != target_sr:
            ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=target_sr)

        # Align lengths (PESQ requires same length)
        min_len = min(len(synth_audio), len(ref_audio))
        if min_len < target_sr:  # Need at least 1 second
            return None

        score = pesq(target_sr, ref_audio[:min_len], synth_audio[:min_len], 'wb')
        return float(score)
    except Exception:
        return None


def compute_stoi(synth_audio: np.ndarray, synth_sr: int,
                 ref_audio: np.ndarray, ref_sr: int) -> Optional[float]:
    """Compute STOI (Short-Time Objective Intelligibility).

    Requires reference audio. Range: 0 to 1 (higher = better).
    """
    try:
        from pystoi import stoi

        target_sr = 16000

        if synth_sr != target_sr:
            synth_audio = librosa.resample(synth_audio, orig_sr=synth_sr, target_sr=target_sr)
        if ref_sr != target_sr:
            ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=target_sr)

        min_len = min(len(synth_audio), len(ref_audio))
        if min_len < target_sr:
            return None

        score = stoi(ref_audio[:min_len], synth_audio[:min_len], target_sr, extended=False)
        return float(score)
    except Exception:
        return None


def compute_output_snr(audio: np.ndarray, sr: int) -> Optional[float]:
    """Estimate SNR of generated audio using Silero VAD.

    Compares speech frame power vs non-speech (noise) frame power.
    Reference-free. Higher = cleaner audio. Range: ~0-40 dB.
    """
    try:
        import torch

        if 'silero_vad' not in MODELS:
            vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad',
                force_reload=False, trust_repo=True
            )
            MODELS['silero_vad'] = (vad_model, utils)

        vad_model, (get_speech_timestamps, *_) = MODELS['silero_vad']

        audio_mono = audio.flatten() if audio.ndim > 1 else audio.copy()
        audio_16k = (librosa.resample(audio_mono, orig_sr=sr, target_sr=16000)
                     if sr != 16000 else audio_mono)

        # Normalise amplitude
        peak = np.max(np.abs(audio_16k))
        if peak == 0:
            return None
        audio_16k = audio_16k / peak

        timestamps = get_speech_timestamps(
            torch.from_numpy(audio_16k).float(), vad_model,
            sampling_rate=16000, threshold=0.5,
            min_speech_duration_ms=250, min_silence_duration_ms=100
        )

        if not timestamps:
            return None

        # Collect speech and non-speech frames
        total_samples = len(audio_16k)
        speech_mask = np.zeros(total_samples, dtype=bool)
        for ts in timestamps:
            speech_mask[ts['start']:ts['end']] = True

        speech_frames = audio_16k[speech_mask]
        noise_frames = audio_16k[~speech_mask]

        if len(speech_frames) == 0 or len(noise_frames) == 0:
            return None

        signal_power = np.mean(speech_frames ** 2)
        noise_power = np.mean(noise_frames ** 2)

        if noise_power < 1e-12:  # Near-silent background — clamp to high SNR
            return 40.0

        snr_db = 10.0 * np.log10(signal_power / noise_power)
        return float(np.clip(snr_db, 0.0, 60.0))
    except Exception:
        return None


def compute_dynamic_range(audio: np.ndarray, sr: int) -> Optional[float]:
    """Compute dynamic range of the generated audio (dB).

    Defined as: 95th-percentile frame energy – 5th-percentile frame energy
    (among voiced frames only, using 20 ms frames).
    Higher = more expressive variation in loudness.
    """
    try:
        frame_len = int(0.02 * sr)  # 20 ms
        hop_len = frame_len // 2
        frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)

        # RMS energy per frame in dB
        rms = np.sqrt(np.mean(frames ** 2, axis=0))
        rms = rms[rms > 1e-6]  # Remove near-silent frames
        if len(rms) < 10:
            return None

        db_vals = 20.0 * np.log10(rms)
        dynamic_range = float(np.percentile(db_vals, 95) - np.percentile(db_vals, 5))
        return max(0.0, dynamic_range)
    except Exception:
        return None


def compute_speaker_similarity(audio1: np.ndarray, sr1: int,
                               audio2: np.ndarray, sr2: int) -> Dict:
    """Compute speaker similarity using Resemblyzer."""
    metrics = {
        'resemblyzer_cosine_sim': None,
        'ecapa_cosine_sim': None  # Disabled due to torchaudio compatibility issues
    }

    # Resemblyzer similarity
    try:
        # ── Compatibility shims for Resemblyzer on modern NumPy / librosa ──
        # 1) np.bool was removed in NumPy 1.24
        if not hasattr(np, 'bool'):
            np.bool = bool

        # 2) librosa >= 0.10 made melspectrogram() fully keyword-only.
        #    Resemblyzer calls it as melspectrogram(S, sr, n_mels=...) — patch
        #    to remap the first two positional args to kwargs.
        import librosa.feature as _lf
        if not getattr(_lf.melspectrogram, '_compat_patched', False):
            _orig_mel = _lf.melspectrogram
            def _compat_mel(*args, **kwargs):
                if len(args) >= 1 and 'y' not in kwargs and 'S' not in kwargs:
                    first = args[0]
                    kwargs['S' if (hasattr(first, 'ndim') and first.ndim == 2) else 'y'] = first
                if len(args) >= 2 and 'sr' not in kwargs:
                    kwargs['sr'] = args[1]
                return _orig_mel(**kwargs)
            _compat_mel._compat_patched = True
            _lf.melspectrogram = _compat_mel

        from resemblyzer import VoiceEncoder, preprocess_wav

        if 'resemblyzer' not in MODELS:
            MODELS['resemblyzer'] = VoiceEncoder()

        encoder = MODELS['resemblyzer']

        # Resample to 16 kHz with current librosa API before handing off to
        # preprocess_wav — old resemblyzer calls librosa.resample() with
        # positional args which breaks on librosa >= 0.10.
        RESEMBLYZER_SR = 16000
        audio1_16k = librosa.resample(audio1, orig_sr=sr1, target_sr=RESEMBLYZER_SR) if sr1 != RESEMBLYZER_SR else audio1
        audio2_16k = librosa.resample(audio2, orig_sr=sr2, target_sr=RESEMBLYZER_SR) if sr2 != RESEMBLYZER_SR else audio2

        wav1 = preprocess_wav(audio1_16k)
        wav2 = preprocess_wav(audio2_16k)

        # Get embeddings
        embed1 = encoder.embed_utterance(wav1)
        embed2 = encoder.embed_utterance(wav2)

        # Cosine similarity
        similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
        metrics['resemblyzer_cosine_sim'] = float(similarity)
    except Exception as e:
        print(f"    Resemblyzer error: {e}")
        metrics['resemblyzer_cosine_sim'] = None

    return metrics


def compute_prosody_metrics(audio: np.ndarray, sr: int, transcript: str = "") -> Dict:
    """Compute prosody metrics covering pitch, rhythm/timing, energy, and voice quality.

    Pillars:
      Pitch   — f0_mean, f0_std, f0_range, jitter, shimmer, hnr
      Timing  — pause_ratio, pause_count, pause_mean_duration, speaking_rate, syllable_rate
      Energy  — energy_mean, energy_std, dynamic_range_db
      Duration — duration, duration_ratio
    """
    import parselmouth
    import torch

    metrics = {}

    # Build Parselmouth Sound once; reused by all Praat-based metrics
    sound = None
    try:
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
    except Exception:
        pass

    # ── Pitch (F0) ────────────────────────────────────────────────────────
    try:
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        f0_values = [
            pitch.get_value_in_frame(i)
            for i in range(pitch.get_number_of_frames())
            if pitch.get_value_in_frame(i) > 0
        ]
        if f0_values:
            metrics['f0_mean'] = float(np.mean(f0_values))
            metrics['f0_std'] = float(np.std(f0_values))
            metrics['f0_range'] = float(
                np.percentile(f0_values, 95) - np.percentile(f0_values, 5)
            )
        else:
            metrics['f0_mean'] = metrics['f0_std'] = metrics['f0_range'] = None
    except Exception:
        metrics['f0_mean'] = metrics['f0_std'] = metrics['f0_range'] = None

    # ── Energy (Intensity) ────────────────────────────────────────────────
    try:
        intensity = sound.to_intensity(minimum_pitch=75, time_step=0.01)
        vals = intensity.values[0]                  # shape: (n_frames,)
        voiced_vals = vals[vals > 30]               # exclude near-silence frames (< 30 dB)
        if len(voiced_vals) > 0:
            metrics['energy_mean'] = float(np.mean(voiced_vals))
            metrics['energy_std'] = float(np.std(voiced_vals))
        else:
            metrics['energy_mean'] = metrics['energy_std'] = None
    except Exception:
        metrics['energy_mean'] = metrics['energy_std'] = None

    # ── Jitter & Shimmer (cycle-to-cycle perturbation) ────────────────────
    try:
        pp = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter = parselmouth.praat.call(
            pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer = parselmouth.praat.call(
            [sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        metrics['jitter'] = float(jitter) if (jitter is not None and not np.isnan(jitter)) else None
        metrics['shimmer'] = float(shimmer) if (shimmer is not None and not np.isnan(shimmer)) else None
    except Exception:
        metrics['jitter'] = metrics['shimmer'] = None

    # ── HNR (Harmonics-to-Noise Ratio) ────────────────────────────────────
    try:
        harmonicity = parselmouth.praat.call(
            sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
        )
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        metrics['hnr'] = float(hnr) if (hnr is not None and not np.isnan(hnr)) else None
    except Exception:
        metrics['hnr'] = None

    # ── Duration ──────────────────────────────────────────────────────────
    duration = len(audio) / sr
    metrics['duration'] = duration

    # ── Speaking Rate (words / second) ────────────────────────────────────
    word_count = len(transcript.split()) if transcript.strip() else 0
    try:
        metrics['speaking_rate'] = float(word_count / duration) if duration > 0 else None
    except Exception:
        metrics['speaking_rate'] = None

    # ── Syllable Rate (approx: words × 1.5 syllables/word per second) ─────
    try:
        metrics['syllable_rate'] = float(word_count * 1.5 / duration) if duration > 0 else None
    except Exception:
        metrics['syllable_rate'] = None

    # ── Duration Ratio (actual / expected from text length) ───────────────
    # Expected duration at natural speaking rate: 3 words per second
    try:
        expected_duration = word_count / 3.0 if word_count > 0 else None
        if expected_duration and expected_duration > 0:
            metrics['duration_ratio'] = float(duration / expected_duration)
        else:
            metrics['duration_ratio'] = None
    except Exception:
        metrics['duration_ratio'] = None

    # ── Dynamic Range ─────────────────────────────────────────────────────
    try:
        metrics['dynamic_range_db'] = compute_dynamic_range(audio, sr)
    except Exception:
        metrics['dynamic_range_db'] = None

    # ── Pause Ratio, Pause Count & Mean Pause Duration (Silero VAD) ───────
    try:
        if 'silero_vad' not in MODELS:
            vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad',
                force_reload=False, trust_repo=True
            )
            MODELS['silero_vad'] = (vad_model, utils)

        vad_model, (get_speech_timestamps, *_) = MODELS['silero_vad']

        audio_mono = audio.flatten() if audio.ndim > 1 else audio
        audio_16k = (
            librosa.resample(audio_mono, orig_sr=sr, target_sr=16000)
            if sr != 16000 else audio_mono.copy()
        )
        if np.max(np.abs(audio_16k)) > 0:
            audio_16k = audio_16k / np.max(np.abs(audio_16k))

        timestamps = get_speech_timestamps(
            torch.from_numpy(audio_16k).float(), vad_model,
            sampling_rate=16000, threshold=0.5,
            min_speech_duration_ms=250, min_silence_duration_ms=100
        )

        total_dur = len(audio_16k) / 16000

        if timestamps:
            speech_dur = sum((ts['end'] - ts['start']) / 16000 for ts in timestamps)
            metrics['pause_ratio'] = max(0.0, 1.0 - speech_dur / total_dur)

            # Compute inter-speech gaps (pauses > 200 ms)
            pause_durations = []
            for i in range(1, len(timestamps)):
                gap_start = timestamps[i - 1]['end'] / 16000
                gap_end = timestamps[i]['start'] / 16000
                gap_dur = gap_end - gap_start
                if gap_dur >= 0.2:  # 200 ms threshold
                    pause_durations.append(gap_dur)

            metrics['pause_count'] = len(pause_durations)
            metrics['pause_mean_duration'] = (float(np.mean(pause_durations))
                                              if pause_durations else 0.0)
        else:
            metrics['pause_ratio'] = 1.0  # All silence
            metrics['pause_count'] = 0
            metrics['pause_mean_duration'] = 0.0

    except Exception:
        metrics['pause_ratio'] = None
        metrics['pause_count'] = None
        metrics['pause_mean_duration'] = None

    return metrics


def compute_robustness_metrics(audio: np.ndarray, sr: int, transcript: str) -> Dict:
    """Detect failure modes."""
    metrics = {}

    # Check for repetitions in transcript
    words = transcript.split()
    has_repetition = 0

    for n in [2, 3, 4]:  # n-gram size
        for i in range(len(words) - n * 3):
            ngram = ' '.join(words[i:i+n])
            count = 0
            for j in range(i, len(words) - n + 1):
                if ' '.join(words[j:j+n]) == ngram:
                    count += 1
            if count >= 3:
                has_repetition = 1
                break
        if has_repetition:
            break

    metrics['has_repetition'] = has_repetition

    # has_silence_anomaly is set in evaluate_utterance() using VAD pause data
    # (placeholder so the key is always present)
    metrics['has_silence_anomaly'] = 0

    # Empty/short audio
    duration = len(audio) / sr
    metrics['is_empty_or_short'] = 1 if duration < 0.1 else 0

    return metrics


def evaluate_utterance(
    audio_path: Path,
    entry: Dict,
    gen_metadata: Dict,
    asr_provider: str
) -> Dict:
    """Evaluate one utterance across all metrics."""
    result = {
        'id': entry['id'],
        # Propagate dataset metadata for downstream grouping
        'dataset': entry.get('dataset', 'unknown'),
        'category': entry.get('category', 'unknown'),
        'difficulty': entry.get('difficulty', 'standard'),
    }

    try:
        # Load audio
        audio, sr = sf.read(audio_path)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # ASR transcript
        try:
            transcript = get_asr_transcript(str(audio_path), asr_provider)
        except Exception as e:
            print(f"    ASR failed for {entry['id']}: {e}")
            transcript = ""

        # Intelligibility
        intel_metrics = compute_intelligibility_metrics(entry['text'], transcript)
        result.update(intel_metrics)

        # Semantic distance (round-trip)
        try:
            result['semantic_distance'] = compute_semantic_distance(entry['text'], transcript)
        except Exception:
            result['semantic_distance'] = None

        # Naturalness - UTMOS
        try:
            result['utmos'] = compute_utmos(audio, sr)
        except:
            result['utmos'] = None

        # SCOREQ
        result['scoreq'] = compute_scoreq(audio, sr)

        # NISQA
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, sr)
            nisqa_scores = compute_nisqa(tmp_path)
            result.update(nisqa_scores)
            os.remove(tmp_path)

        # DNSMOS
        dnsmos_scores = compute_dnsmos(audio, sr)
        result.update(dnsmos_scores)

        # Output SNR (reference-free)
        try:
            result['output_snr'] = compute_output_snr(audio, sr)
        except Exception:
            result['output_snr'] = None

        # Reference-based metrics (PESQ, STOI, Speaker Similarity)
        if entry.get('reference_audio_path'):
            try:
                ref_audio, ref_sr = sf.read(entry['reference_audio_path'])
                if ref_audio.ndim > 1:
                    ref_audio = ref_audio.mean(axis=1)

                # PESQ
                result['pesq'] = compute_pesq(audio, sr, ref_audio, ref_sr)

                # STOI
                result['stoi'] = compute_stoi(audio, sr, ref_audio, ref_sr)

                # Speaker similarity
                spk_sim = compute_speaker_similarity(audio, sr, ref_audio, ref_sr)
                result.update(spk_sim)
            except Exception as e:
                print(f"    Reference-based metrics error for {entry['id']}: {e}")
                result['pesq'] = None
                result['stoi'] = None
                result['ecapa_cosine_sim'] = None
                result['resemblyzer_cosine_sim'] = None
        else:
            result['pesq'] = None
            result['stoi'] = None
            result['ecapa_cosine_sim'] = None
            result['resemblyzer_cosine_sim'] = None

        # Prosody (includes pause_count, pause_mean_duration, syllable_rate, duration_ratio,
        #          dynamic_range_db alongside existing metrics)
        prosody_metrics = compute_prosody_metrics(audio, sr, transcript)
        result.update(prosody_metrics)

        # Robustness
        robust_metrics = compute_robustness_metrics(audio, sr, transcript)

        # Silence anomaly — use pause data already computed by prosody
        # Flag if: any single pause > 1.5 s, or mean pause > 1.0 s, or > 5 pauses
        pause_count = prosody_metrics.get('pause_count') or 0
        pause_mean = prosody_metrics.get('pause_mean_duration') or 0.0
        pause_ratio = prosody_metrics.get('pause_ratio') or 0.0
        robust_metrics['has_silence_anomaly'] = int(
            pause_mean > 1.5 or pause_count > 5 or pause_ratio > 0.6
        )

        result.update(robust_metrics)

        # Latency (from generation metadata)
        if entry['id'] in gen_metadata:
            meta = gen_metadata[entry['id']]
            result['ttfa_ms'] = meta.get('ttfa_ms')
            result['rtf'] = meta.get('rtf')
            result['inference_time_ms'] = meta.get('inference_time_ms')
        else:
            result['ttfa_ms'] = None
            result['rtf'] = None
            result['inference_time_ms'] = None

    except Exception as e:
        print(f"    Error evaluating {entry['id']}: {e}")
        # Set all metrics to None
        for key in ['utmos', 'scoreq', 'wer', 'cer', 'pesq', 'stoi',
                    'semantic_distance', 'output_snr']:
            result[key] = None

    return result


def export_to_csv(all_results: List[Dict], output_dir: Path):
    """Export aggregated metrics to CSV format."""
    import csv

    # Aggregate metrics for each model
    csv_rows = []

    for model_results in all_results:
        model_name = model_results['model']
        utterances = model_results['per_utterance']

        if not utterances:
            continue

        # Calculate mean metrics across all utterances
        metrics_sum = {}
        metrics_count = {}

        for utt in utterances:
            for key, value in utt.items():
                if key in ('id', 'dataset', 'category', 'difficulty'):
                    continue

                if value is not None and isinstance(value, (int, float)):
                    if key not in metrics_sum:
                        metrics_sum[key] = 0
                        metrics_count[key] = 0
                    metrics_sum[key] += value
                    metrics_count[key] += 1

        # Calculate averages
        row = {'model': model_name}
        for key in sorted(metrics_sum.keys()):
            if metrics_count[key] > 0:
                row[key] = metrics_sum[key] / metrics_count[key]
            else:
                row[key] = None

        csv_rows.append(row)

    if not csv_rows:
        print("  No data to export to CSV")
        return

    # Get all metric columns
    all_columns = set()
    for row in csv_rows:
        all_columns.update(row.keys())
    all_columns.discard('model')
    all_columns = ['model'] + sorted(all_columns)

    # Write aggregated metrics CSV
    csv_path = output_dir / 'aggregated_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"✓ Saved aggregated CSV to {csv_path}")

    # Also create per-utterance CSV
    per_utt_rows = []
    for model_results in all_results:
        model_name = model_results['model']
        for utt in model_results['per_utterance']:
            row = {'model': model_name}
            row.update(utt)
            per_utt_rows.append(row)

    if per_utt_rows:
        all_columns = set()
        for row in per_utt_rows:
            all_columns.update(row.keys())
        all_columns.discard('model')
        all_columns = ['model'] + sorted(all_columns)

        csv_path = output_dir / 'per_utterance_metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            writer.writerows(per_utt_rows)

        print(f"✓ Saved per-utterance CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate TTS outputs')
    parser.add_argument('--gen-dir', type=str, default='generated_audio',
                       help='Generated audio directory')
    parser.add_argument('--manifest', type=str, default='datasets/manifest.json',
                       help='Utterance manifest')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--asr', type=str, default='whisper',
                       choices=['whisper', 'assemblyai', 'deepgram'],
                       help='ASR provider')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--format', type=str, default='both',
                       choices=['json', 'csv', 'both'],
                       help='Output format (default: both)')
    args = parser.parse_args()

    # Load manifest
    manifest = load_manifest(Path(args.manifest))
    print(f"✓ Loaded {len(manifest)} utterances")

    # Show dataset breakdown
    by_ds: Dict[str, int] = {}
    for e in manifest:
        ds = e.get('dataset', 'unknown')
        by_ds[ds] = by_ds.get(ds, 0) + 1
    if len(by_ds) > 1:
        print("  Datasets:")
        for ds, cnt in sorted(by_ds.items()):
            print(f"    {ds}: {cnt}")

    gen_dir = Path(args.gen_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all models
    model_dirs = [d for d in gen_dir.iterdir() if d.is_dir()]

    all_model_results = []  # Store all results for CSV export

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n▶ Evaluating {model_name}")

        # Load generation metadata
        gen_metadata = load_gen_metadata(model_dir)

        # Evaluate each utterance
        results = {
            'model': model_name,
            'n_utterances': len(manifest),
            'per_utterance': []
        }

        for entry in tqdm(manifest, desc=f"  {model_name}"):
            audio_path = model_dir / f"{entry['id']}.wav"

            if not audio_path.exists():
                continue

            result = evaluate_utterance(audio_path, entry, gen_metadata, args.asr)
            results['per_utterance'].append(result)

        # Save JSON results (if requested)
        if args.format in ['json', 'both']:
            output_path = output_dir / f"{model_name}_metrics.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Saved JSON to {output_path}")

        all_model_results.append(results)

    # Save CSV results (if requested)
    if args.format in ['csv', 'both']:
        export_to_csv(all_model_results, output_dir)

    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
