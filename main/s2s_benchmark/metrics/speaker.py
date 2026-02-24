"""
Speaker similarity metrics for S2S evaluation.

Exposed functions:
    compute_secs(ref_path, hyp_path)         → float  SECS via WavLM-large (de facto standard)
    compute_sim_wavlm(ref_path, hyp_path)    → float  WavLM-Base+ (CPU-safe alias)
    compute_sim_ecapa(ref_path, hyp_path)    → float  ECAPA-TDNN (optional)
    compute_eer(ref_paths, hyp_paths)        → float  dataset-level EER estimate
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

MODELS: Dict = {}


def _load_audio_16k(path: str) -> np.ndarray:
    import soundfile as sf  # type: ignore
    import librosa          # type: ignore
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != 16_000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
    return audio


def _wavlm_embed(audio: np.ndarray, model_id: str, cache_key: str):
    """Shared embedding extraction for any WavLM XVector model."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector  # type: ignore

    if cache_key not in MODELS:
        print(f"  [speaker] Loading {model_id} …")
        fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        m  = WavLMForXVector.from_pretrained(model_id).cpu().eval()
        MODELS[cache_key] = (fe, m)

    fe, m = MODELS[cache_key]
    inputs = fe(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = m(**inputs).embeddings
    return emb[0]


# ─────────────────────────────────────────────────────────────────────────────
# SECS — de facto standard (Seed-TTS-Eval, WavLM-large)
# ─────────────────────────────────────────────────────────────────────────────

def compute_secs(ref_path: str, hyp_path: str, use_large: bool = True) -> Optional[float]:
    """Speaker Embedding Cosine Similarity via WavLM-large speaker verification.

    SECS = cos(e_ref, e_syn).  De facto standard adopted by Seed-TTS-Eval.

    Model:
      use_large=True  → microsoft/wavlm-large  (~1.26 GB, ~20 s/pair CPU, EER ~3-4%)
      use_large=False → microsoft/wavlm-base-plus-sv (~90 MB, ~5 s/pair CPU)

    Range [-1, 1]; practically [0.3, 1.0] for speech. Higher = more similar.
    """
    model_id  = "microsoft/wavlm-large" if use_large else "microsoft/wavlm-base-plus-sv"
    cache_key = "wavlm_large_sv" if use_large else "wavlm_sv"
    try:
        import torch
        ref_emb = _wavlm_embed(_load_audio_16k(ref_path), model_id, cache_key)
        hyp_emb = _wavlm_embed(_load_audio_16k(hyp_path), model_id, cache_key)
        cos = torch.nn.functional.cosine_similarity(
            ref_emb.unsqueeze(0), hyp_emb.unsqueeze(0)
        )
        return float(torch.clamp(cos, -1.0, 1.0).item())
    except Exception as exc:
        print(f"  [speaker] SECS error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# sim_wavlm — CPU-safe base model (kept as alias for backwards compat)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sim_wavlm(ref_path: str, hyp_path: str) -> Optional[float]:
    """WavLM-Base+ cosine similarity (fast CPU fallback for SECS)."""
    return compute_secs(ref_path, hyp_path, use_large=False)


# ─────────────────────────────────────────────────────────────────────────────
# ECAPA-TDNN (optional, EER ~0.92% on VoxCeleb)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sim_ecapa(ref_path: str, hyp_path: str) -> Optional[float]:
    """ECAPA-TDNN cosine similarity. ~30 s/pair CPU. Enable via include_ecapa: true."""
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        if "ecapa" not in MODELS:
            print("  [speaker] Loading ECAPA-TDNN …")
            MODELS["ecapa"] = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )
        clf = MODELS["ecapa"]

        def _embed(path):
            import torchaudio
            sig, sr = torchaudio.load(path)
            if sig.shape[0] > 1:
                sig = sig.mean(dim=0, keepdim=True)
            return clf.encode_batch(sig).squeeze()

        cos = torch.nn.functional.cosine_similarity(
            _embed(ref_path).unsqueeze(0), _embed(hyp_path).unsqueeze(0)
        )
        return float(cos.item())
    except Exception as exc:
        print(f"  [speaker] ECAPA error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# EER — dataset-level Equal Error Rate estimate
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(
    ref_paths: List[str],
    hyp_paths: List[str],
    use_large: bool = False,
) -> Optional[float]:
    """Estimate EER from genuine/impostor cosine similarity distributions.

    Genuine pairs  : (ref[i], hyp[i])   — same utterance, different rendition
    Impostor pairs : (ref[i], hyp[j])   — different speakers / utterances

    EER is the threshold where FAR == FRR.
    Range [0, 100]%; lower is better.
    WavLM-large achieves ~3-4% EER; ECAPA ~0.92% on VoxCeleb.

    Note: for meaningful EER you need ≥ 100 pairs.
    """
    if len(ref_paths) < 5:
        return None

    model_id  = "microsoft/wavlm-large" if use_large else "microsoft/wavlm-base-plus-sv"
    cache_key = "wavlm_large_sv" if use_large else "wavlm_sv"

    try:
        import torch

        # Extract all embeddings once
        embeddings = []
        for path in ref_paths + hyp_paths:
            emb = _wavlm_embed(_load_audio_16k(path), model_id, cache_key)
            embeddings.append(emb)

        n = len(ref_paths)
        ref_embs = embeddings[:n]
        hyp_embs = embeddings[n:]

        genuine_scores, impostor_scores = [], []

        for i in range(n):
            # Genuine: ref[i] vs hyp[i]
            cos = torch.nn.functional.cosine_similarity(
                ref_embs[i].unsqueeze(0), hyp_embs[i].unsqueeze(0)
            )
            genuine_scores.append(float(cos.item()))

            # Impostor: ref[i] vs hyp[j≠i] (one random impostor per genuine)
            j = (i + 1) % n
            cos_imp = torch.nn.functional.cosine_similarity(
                ref_embs[i].unsqueeze(0), hyp_embs[j].unsqueeze(0)
            )
            impostor_scores.append(float(cos_imp.item()))

        # Sweep thresholds to find EER
        all_scores = genuine_scores + impostor_scores
        labels     = [1] * len(genuine_scores) + [0] * len(impostor_scores)
        thresholds = sorted(set(all_scores))

        best_eer = 1.0
        for thr in thresholds:
            tp = sum(1 for s, l in zip(genuine_scores,  [1]*len(genuine_scores))  if s >= thr)
            fn = len(genuine_scores)  - tp
            fp = sum(1 for s in impostor_scores if s >= thr)
            tn = len(impostor_scores) - fp
            far = fp / max(fp + tn, 1)
            frr = fn / max(fn + tp, 1)
            eer_candidate = (far + frr) / 2
            if abs(far - frr) < abs(best_eer - 0.5) or eer_candidate < best_eer:
                best_eer = eer_candidate

        return float(best_eer * 100)   # return as percentage

    except Exception as exc:
        print(f"  [speaker] EER error: {exc}")
        return None
