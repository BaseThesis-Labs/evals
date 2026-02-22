"""src/evaluation/semascore.py — SeMaScore: semantic similarity × (1 − MER).

Paper: Sasindran et al., Interspeech 2024.
Reuses an existing SentenceTransformer encoder to avoid reloading.
"""
from __future__ import annotations

import numpy as np


def compute_semascore(reference: str, hypothesis: str, encoder) -> float:
    """
    Combine semantic cosine similarity with MER to get a single 0→1 score.
    Higher = better. Captures meaning preservation AND accuracy together.

    Both ref and hyp should already be normalised before calling this function
    (same normalisation as used for WER) so that formatting differences between
    API models and reference text don't pollute the semantic score.
    """
    if not reference.strip():
        return 0.0 if hypothesis.strip() else 1.0
    if not hypothesis.strip():
        return 0.0

    try:
        import jiwer
        # jiwer 3.x: process_words().mer; jiwer 2.x: jiwer.mer()
        out = jiwer.process_words(reference, hypothesis)
        error_rate = float(out.mer)
    except Exception:
        error_rate = 1.0

    try:
        emb = encoder.encode([reference, hypothesis])
        na = np.linalg.norm(emb[0])
        nb = np.linalg.norm(emb[1])
        cos_sim = float(np.dot(emb[0], emb[1]) / (na * nb)) if na > 0 and nb > 0 else 0.0
        cos_sim = max(0.0, cos_sim)
    except Exception:
        cos_sim = 0.0

    return float(cos_sim * (1.0 - error_rate))
