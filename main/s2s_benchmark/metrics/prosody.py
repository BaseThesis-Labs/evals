"""
Prosody metrics for S2S evaluation.

All computed via parselmouth (Praat wrapper) — CPU only.
DS-WED via HuBERT discrete tokens.

Exposed functions:
    compute_f0_rmse(ref_path, hyp_path)               → float
    compute_pitch_corr(ref_path, hyp_path)            → float
    compute_energy_corr(ref_path, hyp_path)           → float
    compute_duration_ratio(ref_path, hyp_path)        → float
    compute_speaking_rate(audio, sr, text)            → float  syllables/sec
    compute_pause_ratio(audio, sr)                    → float  silent_frames/total
    compute_dswed(ref_path, hyp_path)                 → float  DS-WED (prosody diversity)
    compute_all_prosody(ref_path, hyp_path, audio, sr, text) → dict
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

MODELS: Dict = {}


def _extract_f0(
    path: str, time_step: float = 0.01, min_pitch: float = 75.0, max_pitch: float = 500.0
) -> Tuple[np.ndarray, np.ndarray]:
    import parselmouth  # type: ignore
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch(time_step=time_step, pitch_floor=min_pitch, pitch_ceiling=max_pitch)
    return pitch.xs(), pitch.selected_array["frequency"]


def _extract_intensity(path: str, time_step: float = 0.01) -> np.ndarray:
    import parselmouth  # type: ignore
    snd = parselmouth.Sound(path)
    return snd.to_intensity(time_step=time_step).values.T.flatten()


def _align_arrays(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from dtw import dtw as _dtw  # type: ignore
        aln = _dtw(a.reshape(-1, 1), b.reshape(-1, 1), distance_only=False)
        return a[aln.index1], b[aln.index2]
    except ImportError:
        n = min(len(a), len(b))
        return a[:n], b[:n]


# ─────────────────────────────────────────────────────────────────────────────
# F0 RMSE
# ─────────────────────────────────────────────────────────────────────────────

def compute_f0_rmse(ref_path: str, hyp_path: str) -> Optional[float]:
    """F0 RMSE (Hz) on voiced frames only. Lower = better pitch accuracy."""
    try:
        _, ref_f0 = _extract_f0(ref_path)
        _, hyp_f0 = _extract_f0(hyp_path)
        ref_a, hyp_a = _align_arrays(ref_f0, hyp_f0)
        mask = (ref_a > 0) & (hyp_a > 0)
        if mask.sum() == 0:
            return None
        return float(np.sqrt(np.mean((ref_a[mask] - hyp_a[mask]) ** 2)))
    except Exception as exc:
        print(f"  [prosody] f0_rmse error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Pitch correlation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pitch_corr(ref_path: str, hyp_path: str) -> Optional[float]:
    """Pearson correlation of F0 contours (voiced frames). Range [-1, 1]."""
    try:
        _, ref_f0 = _extract_f0(ref_path)
        _, hyp_f0 = _extract_f0(hyp_path)
        ref_a, hyp_a = _align_arrays(ref_f0, hyp_f0)
        mask = (ref_a > 0) & (hyp_a > 0)
        if mask.sum() < 2:
            return None
        corr = float(np.corrcoef(ref_a[mask], hyp_a[mask])[0, 1])
        return None if np.isnan(corr) else corr
    except Exception as exc:
        print(f"  [prosody] pitch_corr error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Energy correlation
# ─────────────────────────────────────────────────────────────────────────────

def compute_energy_corr(ref_path: str, hyp_path: str) -> Optional[float]:
    """Pearson correlation of frame-level intensity (dB). Range [-1, 1]."""
    try:
        ref_int = _extract_intensity(ref_path)
        hyp_int = _extract_intensity(hyp_path)
        ref_a, hyp_a = _align_arrays(ref_int, hyp_int)
        if len(ref_a) < 2:
            return None
        corr = float(np.corrcoef(ref_a, hyp_a)[0, 1])
        return None if np.isnan(corr) else corr
    except Exception as exc:
        print(f"  [prosody] energy_corr error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Duration ratio
# ─────────────────────────────────────────────────────────────────────────────

def compute_duration_ratio(ref_path: str, hyp_path: str) -> Optional[float]:
    """hyp_duration / ref_duration. Ideal = 1.0."""
    try:
        import soundfile as sf  # type: ignore
        ri = sf.info(ref_path)
        hi = sf.info(hyp_path)
        ref_dur = ri.frames / ri.samplerate
        hyp_dur = hi.frames / hi.samplerate
        return float(hyp_dur / ref_dur) if ref_dur > 0.01 else None
    except Exception as exc:
        print(f"  [prosody] duration_ratio error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Speaking rate  (syllables / second)
# ─────────────────────────────────────────────────────────────────────────────

def compute_speaking_rate(audio: np.ndarray, sr: int, text: str) -> Optional[float]:
    """Estimate speaking rate in syllables per second.

    Syllable count is estimated from the hypothesis text via a vowel-group
    heuristic (fast, CPU-only, no external library required).
    Uses 'syllapy' if installed for a more accurate count.

    Typical conversational English ≈ 3.5–4.5 syl/s.
    """
    if not text or len(audio) == 0 or sr == 0:
        return None
    try:
        total_duration_s = len(audio) / sr
        if total_duration_s < 0.05:
            return None

        # Use voiced duration (exclude silence padding) so the rate reflects
        # actual speech tempo rather than being diluted by trailing silence.
        pause_ratio = compute_pause_ratio(audio, sr)
        if pause_ratio is not None and pause_ratio < 1.0:
            voiced_duration_s = total_duration_s * (1.0 - pause_ratio)
        else:
            voiced_duration_s = total_duration_s

        if voiced_duration_s < 0.05:
            return None

        # Try syllapy first (more accurate)
        try:
            import syllapy  # type: ignore
            words = text.strip().split()
            n_syllables = sum(syllapy.count(w) for w in words if w) or 1
        except ImportError:
            # Vowel-group heuristic fallback
            import re
            n_syllables = max(1, len(re.findall(r"[aeiouyAEIOUY]+", text)))

        return float(n_syllables / voiced_duration_s)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Pause ratio  (fraction of silent frames)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pause_ratio(
    audio: np.ndarray,
    sr: int,
    frame_ms: int = 20,
    threshold_db: float = -40.0,
) -> Optional[float]:
    """Fraction of audio that is silence (RMS energy < threshold_db).

    Uses a simple energy-based VAD (no external library).
    threshold_db = -40 dBFS is a standard near-silence threshold.
    Typical natural speech: 0.10–0.25.  > 0.4 suggests truncation or artefacts.
    """
    if len(audio) == 0 or sr == 0:
        return None
    try:
        frame_len = int(sr * frame_ms / 1000)
        if frame_len == 0:
            return None
        n_frames = len(audio) // frame_len
        if n_frames == 0:
            return None
        frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        threshold_lin = 10.0 ** (threshold_db / 20.0)
        n_silent = int(np.sum(rms < threshold_lin))
        return float(n_silent / n_frames)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DS-WED — Discrete Semantic Weighted Edit Distance
#
# Measures prosody/diversity distance using discrete HuBERT semantic tokens.
# Correlates ~0.7 with human diversity ratings. Lower = closer to reference.
#
# Algorithm:
#   1. Extract HuBERT features (layer 9) → cluster via k-means (k=100)
#   2. Map each frame to its cluster token ID
#   3. Run-length encode to get token sequences
#   4. Compute normalised weighted edit distance (substitution=1, ins/del=0.5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dswed(ref_path: str, hyp_path: str, n_clusters: int = 100) -> Optional[float]:
    """DS-WED: Discrete Semantic Weighted Edit Distance.

    Lower is better for matching reference prosody distributions.
    Practical range ≈ [0, 1] after normalisation.

    Requires: transformers, torch, sklearn
    """
    try:
        tokens_ref = _get_hubert_tokens(ref_path, n_clusters)
        tokens_hyp = _get_hubert_tokens(hyp_path, n_clusters)
        if tokens_ref is None or tokens_hyp is None:
            return None
        # Run-length encode: collapse consecutive identical tokens
        rle_ref = _rle(tokens_ref)
        rle_hyp = _rle(tokens_hyp)
        dist = _weighted_edit_distance(rle_ref, rle_hyp)
        max_len = max(len(rle_ref), len(rle_hyp), 1)
        return float(dist / max_len)
    except Exception as exc:
        print(f"  [prosody] DS-WED error: {exc}")
        return None


def _get_hubert_tokens(path: str, n_clusters: int) -> Optional[List[int]]:
    """Extract discrete HuBERT tokens via k-means clustering."""
    try:
        import torch
        import soundfile as sf
        import librosa
        from transformers import HubertModel, Wav2Vec2FeatureExtractor  # type: ignore
        from sklearn.cluster import MiniBatchKMeans  # type: ignore

        # Load HuBERT
        model_id = "facebook/hubert-base-ls960"
        if "hubert" not in MODELS:
            print("  [prosody] Loading HuBERT for DS-WED …")
            fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            m  = HubertModel.from_pretrained(model_id).cpu().eval()
            MODELS["hubert"] = (fe, m)

        fe, m = MODELS["hubert"]

        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != 16_000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)

        inputs = fe(audio, sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            # Use layer 9 features (standard for semantic unit extraction)
            out = m(**inputs, output_hidden_states=True)
            feats = out.hidden_states[9][0].cpu().numpy()   # (T, 768)

        # Fit or reuse k-means
        km_key = f"kmeans_{n_clusters}"
        if km_key not in MODELS:
            MODELS[km_key] = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
            MODELS[km_key].fit(feats)
        else:
            # Partial fit with new data to keep centroids updated
            MODELS[km_key].partial_fit(feats)

        tokens = MODELS[km_key].predict(feats).tolist()
        return tokens

    except ImportError as exc:
        print(f"  [prosody] DS-WED requires transformers + sklearn: {exc}")
        return None
    except Exception as exc:
        print(f"  [prosody] HuBERT token extraction error: {exc}")
        return None


def _rle(tokens: List[int]) -> List[int]:
    """Run-length encode: collapse consecutive duplicate tokens."""
    if not tokens:
        return []
    result = [tokens[0]]
    for t in tokens[1:]:
        if t != result[-1]:
            result.append(t)
    return result


def _weighted_edit_distance(a: List[int], b: List[int]) -> float:
    """Edit distance with weights: substitution=1.0, insertion/deletion=0.5."""
    m, n = len(a), len(b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * 0.5
    for j in range(n + 1):
        dp[0][j] = j * 0.5
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j]     + 0.5,   # deletion
                    dp[i][j - 1]     + 0.5,   # insertion
                    dp[i - 1][j - 1] + 1.0,   # substitution
                )
    return dp[m][n]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: compute all prosody metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_prosody(
    ref_path: str,
    hyp_path: str,
    include_dswed: bool = False,
    hyp_audio: Optional[np.ndarray] = None,
    hyp_sr: Optional[int] = None,
    hyp_text: Optional[str] = None,
    is_echo: bool = True,
) -> Dict[str, Optional[float]]:
    """Compute all prosody metrics.

    Args:
        ref_path:   path to reference audio (required for F0/energy/duration).
        hyp_path:   path to hypothesis audio (required for F0/energy/duration).
        include_dswed: opt-in DS-WED (requires HuBERT + sklearn, slow).
        hyp_audio:  pre-loaded hypothesis audio array for speaking_rate/pause_ratio.
        hyp_sr:     sample rate of hyp_audio.
        hyp_text:   ASR transcript for speaking_rate estimation.
        is_echo:    if False (generative model), skip duration_ratio.
    """
    result: Dict[str, Optional[float]] = {
        "f0_rmse":        compute_f0_rmse(ref_path, hyp_path),
        "pitch_corr":     compute_pitch_corr(ref_path, hyp_path),
        "energy_corr":    compute_energy_corr(ref_path, hyp_path),
        "duration_ratio": compute_duration_ratio(ref_path, hyp_path) if is_echo else None,
    }

    # Speaking rate and pause ratio (only need hyp audio + text)
    if hyp_audio is not None and hyp_sr is not None:
        result["pause_ratio"] = compute_pause_ratio(hyp_audio, hyp_sr)
        if hyp_text:
            result["speaking_rate"] = compute_speaking_rate(hyp_audio, hyp_sr, hyp_text)

    if include_dswed:
        result["dswed"] = compute_dswed(ref_path, hyp_path)

    return result
