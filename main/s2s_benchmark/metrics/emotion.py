"""
Emotion metrics for S2S evaluation.

Canonical label set: neutral | sad | happy | unknown
  - neutral  : calm, bored, neutral states
  - sad      : sadness, melancholy, depression
  - happy    : happiness, joy, excitement, amusement
  - unknown  : angry, fearful, disgust, contempt, or low-confidence prediction

Every classifier output is mapped to these 4 classes and normalised to a
probability vector summing to 1.0.  The "unknown" bucket absorbs all
emotions that fall outside neutral/sad/happy AND absorbs probability mass
when the classifier is uncertain (max prob < CONFIDENCE_THRESHOLD).

Metrics:
  emotion_match          — binary 1.0 if top canonical labels match
  emotion_sim            — cosine similarity of the 4-dim canonical prob vectors
  esim                   — E-SIM: cosine similarity of AVD (Arousal-Valence-Dominance)

Stored per utterance (not aggregated as scalars):
  ref_emotion            — top canonical label for reference audio
  hyp_emotion            — top canonical label for hypothesis audio
  ref_emotion_probs      — {neutral, sad, happy, unknown} probabilities for ref
  hyp_emotion_probs      — {neutral, sad, happy, unknown} probabilities for hyp

Primary classifier: emotion2vec (iic/emotion2vec_plus_seed, CPU-viable)
Fallback:           speechbrain wav2vec2-IEMOCAP
AVD model:          audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

MODELS: Dict = {}

# ── Canonical 4-class schema ──────────────────────────────────────────────────
CANONICAL_EMOTIONS: List[str] = ["neutral", "sad", "happy", "unknown"]

# If the classifier's top-class confidence is below this threshold, the
# residual probability mass is redistributed to "unknown".
CONFIDENCE_THRESHOLD = 0.40

# Map raw model labels → canonical class.
# Any label not listed here is mapped to "unknown".
EMOTION_MAP: Dict[str, str] = {
    # neutral
    "neutral":      "neutral",
    "calm":         "neutral",
    "boredom":      "neutral",
    "bored":        "neutral",
    "no emotion":   "neutral",
    # happy
    "happy":        "happy",
    "happiness":    "happy",
    "joy":          "happy",
    "joyful":       "happy",
    "excited":      "happy",
    "excitement":   "happy",
    "surprised":    "happy",   # positive surprise → happy bucket
    "amused":       "happy",
    "pleased":      "happy",
    # sad
    "sad":          "sad",
    "sadness":      "sad",
    "depressed":    "sad",
    "melancholy":   "sad",
    # unknown (negative / ambiguous → we cannot claim neutral/sad/happy)
    "angry":        "unknown",
    "anger":        "unknown",
    "fearful":      "unknown",
    "fear":         "unknown",
    "disgust":      "unknown",
    "disgusted":    "unknown",
    "contempt":     "unknown",
    "frustrated":   "unknown",
    "confused":     "unknown",
    "other":        "unknown",
}


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helper: raw probs → canonical 4-dim prob vector
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_to_canonical(raw_probs: Dict[str, float]) -> Dict[str, float]:
    """Map arbitrary emotion probability dict to the 4-class canonical schema.

    Algorithm:
      1. Sum raw probabilities into each canonical bucket using EMOTION_MAP.
         Labels not in EMOTION_MAP are routed to "unknown".
      2. If the winning canonical class probability < CONFIDENCE_THRESHOLD,
         mark all mass as "unknown" (classifier is uncertain).
      3. Normalise so the 4 values sum to 1.0.

    Args:
        raw_probs: {label: probability} from any classifier.

    Returns:
        {"neutral": p, "sad": p, "happy": p, "unknown": p}  summing to 1.0.
    """
    buckets: Dict[str, float] = {e: 0.0 for e in CANONICAL_EMOTIONS}

    for label, prob in raw_probs.items():
        canonical = EMOTION_MAP.get(label.lower().strip(), "unknown")
        buckets[canonical] += float(prob)

    # If the classifier is uncertain, route everything to "unknown"
    top_prob = max(buckets.values())
    if top_prob < CONFIDENCE_THRESHOLD:
        buckets = {"neutral": 0.0, "sad": 0.0, "happy": 0.0, "unknown": 1.0}

    # Normalise
    total = sum(buckets.values())
    if total > 1e-8:
        buckets = {k: v / total for k, v in buckets.items()}
    else:
        buckets = {"neutral": 0.0, "sad": 0.0, "happy": 0.0, "unknown": 1.0}

    return buckets


def _top_label(probs: Dict[str, float]) -> str:
    """Return the canonical label with the highest probability."""
    return max(probs, key=lambda k: probs[k])


# ─────────────────────────────────────────────────────────────────────────────
# emotion2vec (primary categorical classifier)
# ─────────────────────────────────────────────────────────────────────────────

def _predict_emotion_emotion2vec(audio_path: str) -> Optional[Dict]:
    """Predict emotion using FunASR emotion2vec_plus_seed.

    Returns {"label": str, "probabilities": {neutral, sad, happy, unknown}}
    """
    try:
        if "emotion2vec" not in MODELS:
            print("  [emotion] Loading emotion2vec_plus_seed …")
            from funasr import AutoModel  # type: ignore
            MODELS["emotion2vec"] = AutoModel(
                model="iic/emotion2vec_plus_seed", disable_pbar=True
            )
        result = MODELS["emotion2vec"].generate(
            audio_path,
            output_dir=None,
            granularity="utterance",
            extract_embedding=False,
        )
        if not result or not isinstance(result, list):
            return None
        item = result[0]
        labels: List[str] = item.get("labels", [])
        scores: List[float] = item.get("scores", [])
        if not labels or not scores:
            return None

        # emotion2vec_plus_seed returns bilingual labels like "生气/angry"
        # Extract the English part after "/" (or use the full label if no "/")
        raw_probs = {}
        for lbl, sc in zip(labels, scores):
            en_label = lbl.split("/")[-1].lower().strip()
            raw_probs[en_label] = float(sc)
        canonical = _aggregate_to_canonical(raw_probs)
        return {"label": _top_label(canonical), "probabilities": canonical}

    except ImportError:
        print("  [emotion] funasr not installed. pip install funasr modelscope")
        return None
    except Exception as exc:
        print(f"  [emotion] emotion2vec error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SpeechBrain wav2vec2-IEMOCAP (categorical fallback)
# IEMOCAP classes: neutral, happy, sad, angry
# ─────────────────────────────────────────────────────────────────────────────

# IEMOCAP label order from speechbrain's custom_interface
_IEMOCAP_LABELS = ["neutral", "happy", "sad", "angry"]

def _predict_emotion_wav2vec2(audio_path: str) -> Optional[Dict]:
    """Predict emotion using speechbrain wav2vec2-IEMOCAP (4 IEMOCAP classes)."""
    try:
        if "sb_emotion" not in MODELS:
            print("  [emotion] Loading wav2vec2-IEMOCAP …")
            from speechbrain.inference.interfaces import foreign_class  # type: ignore
            MODELS["sb_emotion"] = foreign_class(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                run_opts={"device": "cpu"},
            )
        clf = MODELS["sb_emotion"]

        import torchaudio  # type: ignore
        signal, sr = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        out_prob, score, idx, label = clf.classify_batch(signal)

        n = min(len(_IEMOCAP_LABELS), out_prob.shape[1])
        raw_probs = {_IEMOCAP_LABELS[i]: float(out_prob[0][i]) for i in range(n)}

        # "angry" maps to "unknown" via EMOTION_MAP
        canonical = _aggregate_to_canonical(raw_probs)
        return {"label": _top_label(canonical), "probabilities": canonical}

    except Exception as exc:
        print(f"  [emotion] SpeechBrain emotion error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AVD model — Arousal / Valence / Dominance (for E-SIM, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _predict_avd(audio_path: str) -> Optional[np.ndarray]:
    """Predict [arousal, valence, dominance] using audeering dimensional model."""
    try:
        import torch
        import soundfile as sf
        import librosa

        if "avd_model" not in MODELS:
            print("  [emotion] Loading audeering AVD model …")
            from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification  # type: ignore
            model_id = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            processor = Wav2Vec2Processor.from_pretrained(model_id)
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
            model = model.cpu().eval()
            MODELS["avd_model"] = (processor, model)

        processor, model = MODELS["avd_model"]

        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != 16_000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)

        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits   # (1, 3): arousal, dominance, valence
        avd = logits[0].cpu().numpy()
        return np.array([avd[0], avd[2], avd[1]])  # reorder → [arousal, valence, dominance]

    except Exception as exc:
        print(f"  [emotion] AVD prediction error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def predict_emotion(audio_path: str, use_emotion2vec: bool = True) -> Optional[Dict]:
    """Predict canonical emotion.

    Returns:
        {"label": "neutral"|"sad"|"happy"|"unknown",
         "probabilities": {"neutral": float, "sad": float,
                           "happy": float, "unknown": float}}
    Tries emotion2vec first; falls back to wav2vec2-IEMOCAP.
    """
    if use_emotion2vec:
        result = _predict_emotion_emotion2vec(audio_path)
        if result is not None:
            result["model"] = "emotion2vec"
            return result
    fallback = _predict_emotion_wav2vec2(audio_path)
    if fallback is not None:
        fallback["model"] = "wav2vec2"
    return fallback


def compute_esim(ref_path: str, hyp_path: str) -> Optional[float]:
    """E-SIM: cosine similarity of AVD embeddings. Range [-1, 1]."""
    ref_avd = _predict_avd(ref_path)
    hyp_avd = _predict_avd(hyp_path)
    if ref_avd is None or hyp_avd is None:
        return None
    nr = np.linalg.norm(ref_avd)
    nh = np.linalg.norm(hyp_avd)
    if nr < 1e-8 or nh < 1e-8:
        return None
    return float(np.dot(ref_avd, hyp_avd) / (nr * nh))


def compute_all_emotion(
    ref_path: str,
    hyp_path: str,
    use_emotion2vec: bool = True,
    compute_esim_metric: bool = True,
) -> Dict:
    """Compute all emotion metrics in one call (shares predictions).

    Returns a dict with both scalar metrics (for aggregation) and
    probability dicts (stored in JSON, skipped in numeric aggregation):

        emotion_match        float  1.0 if canonical labels match, else 0.0
        emotion_sim          float  cosine sim of 4-dim probability vectors
        esim                 float  AVD cosine similarity (optional)
        ref_emotion          str    canonical label (neutral/sad/happy/unknown)
        hyp_emotion          str    canonical label
        ref_emotion_probs    dict   {neutral, sad, happy, unknown} → float
        hyp_emotion_probs    dict   {neutral, sad, happy, unknown} → float
    """
    ref_pred = predict_emotion(ref_path, use_emotion2vec)
    hyp_pred = predict_emotion(hyp_path, use_emotion2vec)

    result: Dict = {
        "emotion_match":     None,
        "emotion_sim":       None,
        "esim":              None,
        "ref_emotion":       None,
        "hyp_emotion":       None,
        "ref_emotion_probs": None,
        "hyp_emotion_probs": None,
    }

    if ref_pred and hyp_pred:
        ref_label = ref_pred["label"]
        hyp_label = hyp_pred["label"]
        ref_probs = ref_pred["probabilities"]   # {neutral, sad, happy, unknown}
        hyp_probs = hyp_pred["probabilities"]

        result["ref_emotion"]       = ref_label
        result["hyp_emotion"]       = hyp_label
        result["ref_emotion_probs"] = ref_probs
        result["hyp_emotion_probs"] = hyp_probs
        result["ref_emotion_model"] = ref_pred.get("model")
        result["hyp_emotion_model"] = hyp_pred.get("model")
        result["emotion_match"]     = 1.0 if ref_label == hyp_label else 0.0

        # 4-dim cosine similarity over canonical probability vectors
        rv = np.array([ref_probs.get(e, 0.0) for e in CANONICAL_EMOTIONS])
        hv = np.array([hyp_probs.get(e, 0.0) for e in CANONICAL_EMOTIONS])
        nr, nh = np.linalg.norm(rv), np.linalg.norm(hv)
        if nr > 1e-8 and nh > 1e-8:
            result["emotion_sim"] = float(max(0.0, min(1.0, np.dot(rv, hv) / (nr * nh))))

    if compute_esim_metric:
        result["esim"] = compute_esim(ref_path, hyp_path)

    return result
