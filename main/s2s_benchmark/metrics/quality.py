"""
Audio quality metrics for S2S evaluation.

Adapted from tts_benchmark/evaluate.py patterns.

Exposed functions:
    compute_utmos(audio, sr) → float
    compute_dnsmos(audio, sr) → dict
    compute_nisqa(audio, sr) → dict   (nisqa_mos + sub-scores; None if NISQA not installed)
    compute_pesq(ref_audio, ref_sr, hyp_audio, hyp_sr) → float
    compute_mcd(ref_audio, ref_sr, hyp_audio, hyp_sr) → float
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

# ── Shared model cache ────────────────────────────────────────────────────────
MODELS: Dict = {}


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa  # type: ignore
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        return audio.mean(axis=-1)
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# UTMOS
# ─────────────────────────────────────────────────────────────────────────────

def compute_utmos(audio: np.ndarray, sr: int) -> Optional[float]:
    """UTMOS (tarepan/SpeechMOS utmos22_strong).  Range ≈ 1–5, higher=better."""
    try:
        import torch  # type: ignore

        if "utmos" not in MODELS:
            print("  [quality] Loading UTMOS model …")
            model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            )
            model = model.cpu().eval()
            MODELS["utmos"] = model

        model = MODELS["utmos"]
        audio = _to_mono(audio)
        audio_16k = _resample(audio, sr, 16_000)
        tensor = torch.from_numpy(audio_16k).float().unsqueeze(0)
        with torch.no_grad():
            score = model(tensor, 16_000)
        return float(score.item())

    except Exception as exc:
        print(f"  [quality] UTMOS error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DNSMOS
# ─────────────────────────────────────────────────────────────────────────────

def compute_dnsmos(audio: np.ndarray, sr: int) -> Dict[str, Optional[float]]:
    """DNSMOS via ONNX sig_bak_ovr model.  Returns dnsmos_sig, dnsmos_bak, dnsmos_ovrl."""
    null = {"dnsmos_sig": None, "dnsmos_bak": None, "dnsmos_ovrl": None}
    try:
        import onnxruntime as ort  # type: ignore

        if "dnsmos" not in MODELS:
            model_path = Path.home() / ".cache" / "dnsmos" / "sig_bak_ovr.onnx"
            if not model_path.exists():
                print("  [quality] DNSMOS ONNX model not found at ~/.cache/dnsmos/sig_bak_ovr.onnx")
                return null
            MODELS["dnsmos"] = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )

        session = MODELS["dnsmos"]
        audio = _to_mono(audio)
        audio_16k = _resample(audio, sr, 16_000).astype(np.float32)

        # DNSMOS requires exactly 9.01 s = 144160 samples
        target_len = 144160
        if len(audio_16k) < target_len:
            audio_16k = np.pad(audio_16k, (0, target_len - len(audio_16k)))
        else:
            audio_16k = audio_16k[:target_len]

        audio_16k = audio_16k / (np.max(np.abs(audio_16k)) + 1e-8)
        inp = audio_16k.reshape(1, -1)

        outputs = session.run(None, {session.get_inputs()[0].name: inp})
        scores = outputs[0][0]  # [sig, bak, ovrl]
        return {
            "dnsmos_sig": float(scores[0]),
            "dnsmos_bak": float(scores[1]),
            "dnsmos_ovrl": float(scores[2]),
        }

    except Exception as exc:
        print(f"  [quality] DNSMOS error: {exc}")
        return null


# ─────────────────────────────────────────────────────────────────────────────
# NISQA — Multidimensional non-intrusive quality prediction
# Sub-scores: Noisiness, Coloration, Discontinuity, Loudness
# ─────────────────────────────────────────────────────────────────────────────

def compute_nisqa(audio: np.ndarray, sr: int) -> dict:
    """NISQA multidimensional speech quality prediction.

    Returns nisqa_mos and four sub-scores.  Returns None values if the NISQA
    package is not installed (pip install nisqa) or model weights are missing.

    Install: pip install nisqa   (requires PyTorch)
    """
    null = {
        "nisqa_mos": None,
        "nisqa_noisiness": None,
        "nisqa_coloration": None,
        "nisqa_discontinuity": None,
        "nisqa_loudness": None,
    }
    try:
        import tempfile, soundfile as sf  # type: ignore
        from nisqa.NISQA_lib import NISQA_model  # type: ignore

        if "nisqa" not in MODELS:
            print("  [quality] Loading NISQA model …")
            MODELS["nisqa"] = NISQA_model()  # loads pretrained weights

        model = MODELS["nisqa"]
        audio_m = _to_mono(audio)
        audio_16k = _resample(audio_m, sr, 48_000).astype(np.float32)  # NISQA uses 48 kHz

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio_16k, 48_000)

        pred = model.predict(tmp_path)
        import os; os.unlink(tmp_path)

        return {
            "nisqa_mos": float(pred.get("mos_pred", 0) or 0) or None,
            "nisqa_noisiness": float(pred.get("noi_pred", 0) or 0) or None,
            "nisqa_coloration": float(pred.get("col_pred", 0) or 0) or None,
            "nisqa_discontinuity": float(pred.get("dis_pred", 0) or 0) or None,
            "nisqa_loudness": float(pred.get("loud_pred", 0) or 0) or None,
        }
    except ImportError:
        return null
    except Exception as exc:
        print(f"  [quality] NISQA error: {exc}")
        return null


# ─────────────────────────────────────────────────────────────────────────────
# PESQ
# ─────────────────────────────────────────────────────────────────────────────

def compute_pesq(
    ref_audio: np.ndarray,
    ref_sr: int,
    hyp_audio: np.ndarray,
    hyp_sr: int,
) -> Optional[float]:
    """PESQ (ITU-T P.862).  Range -0.5–4.5, higher=better.

    Requires a clean reference recording.  Returns None if unavailable.
    """
    try:
        from pesq import pesq  # type: ignore

        ref = _to_mono(ref_audio)
        hyp = _to_mono(hyp_audio)

        # PESQ works at 8 or 16 kHz
        target_sr = 16_000
        ref_r = _resample(ref, ref_sr, target_sr).astype(np.float32)
        hyp_r = _resample(hyp, hyp_sr, target_sr).astype(np.float32)

        # Match lengths
        min_len = min(len(ref_r), len(hyp_r))
        ref_r = ref_r[:min_len]
        hyp_r = hyp_r[:min_len]

        score = pesq(target_sr, ref_r, hyp_r, "wb")
        return float(score)

    except Exception as exc:
        print(f"  [quality] PESQ error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MCD (Mel Cepstral Distortion)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mcd(
    ref_audio: np.ndarray,
    ref_sr: int,
    hyp_audio: np.ndarray,
    hyp_sr: int,
) -> Optional[float]:
    """Mel Cepstral Distortion (dB).  Lower=better.

    Uses DTW-aligned 13-MFCC with Kubichek (1993) formula.
    pymcd was removed because its dtw_sl mode returns values on a different
    scale (~100x higher) than standard 13-MFCC MCD.
    """
    return _mcd_dtw_fallback(ref_audio, ref_sr, hyp_audio, hyp_sr)


def _mcd_dtw_fallback(
    ref_audio: np.ndarray,
    ref_sr: int,
    hyp_audio: np.ndarray,
    hyp_sr: int,
) -> Optional[float]:
    """Pure numpy/librosa MCD with DTW alignment."""
    try:
        import librosa  # type: ignore

        ref = _to_mono(ref_audio)
        hyp = _to_mono(hyp_audio)

        # Extract 13 MFCCs at 22050 Hz
        target_sr = 22_050
        ref_r = _resample(ref, ref_sr, target_sr)
        hyp_r = _resample(hyp, hyp_sr, target_sr)

        ref_mfcc = librosa.feature.mfcc(y=ref_r, sr=target_sr, n_mfcc=13).T
        hyp_mfcc = librosa.feature.mfcc(y=hyp_r, sr=target_sr, n_mfcc=13).T

        # DTW alignment
        try:
            from dtw import dtw as _dtw  # type: ignore
            alignment = _dtw(ref_mfcc, hyp_mfcc, distance_only=False)
            idx_ref = alignment.index1
            idx_hyp = alignment.index2
        except ImportError:
            # No dtw-python; use nearest-neighbor fallback
            min_len = min(len(ref_mfcc), len(hyp_mfcc))
            idx_ref = list(range(min_len))
            idx_hyp = list(range(min_len))

        aligned_ref = ref_mfcc[idx_ref]
        aligned_hyp = hyp_mfcc[idx_hyp]

        # Standard MCD formula (Kubichek 1993):
        #   MCD = (10/ln10) * mean_over_frames[ sqrt(2 * sum_k(c_k1 - c_k2)^2) ]
        # Exclude c0 (energy) — use coefficients 1..K only
        diff = (aligned_ref - aligned_hyp)[:, 1:]  # skip c0
        per_frame = np.sqrt(2.0 * np.sum(diff ** 2, axis=1))
        mcd = (10.0 / np.log(10)) * np.mean(per_frame)
        return float(mcd)

    except Exception as exc:
        print(f"  [quality] MCD DTW fallback error: {exc}")
        return None
