"""
Content-preservation metrics for S2S evaluation.

Integrates with stt_benchmark for WER/BERTScore/HER/FWER when available;
provides standalone fallbacks for all metrics.

Exposed functions:
    compute_wer_cer(ref, hyp) → dict
    compute_bert_score(refs, hyps) → dict
    compute_sem_dist(refs, hyps) → list[float]
    compute_rouge_l(ref, hyp) → float
    compute_asr_details(ref, hyp) → dict
    compute_stt_enriched(ref, hyp) → dict  (mer, wil, her, hallucination_rate, fwer)
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

# ── sys.path: inject stt_benchmark for optional metric re-use ─────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_STT_BENCH = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "stt_benchmark"))
if os.path.isdir(_STT_BENCH) and _STT_BENCH not in sys.path:
    sys.path.insert(0, _STT_BENCH)

# ── Shared model cache ────────────────────────────────────────────────────────
MODELS: Dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# WER / CER  (jiwer)
# ─────────────────────────────────────────────────────────────────────────────

def compute_wer_cer(ref: str, hyp: str) -> Dict[str, Optional[float]]:
    """Compute WER and CER using jiwer.

    Returns dict with keys: wer, cer
    """
    if not ref.strip() or not hyp.strip():
        return {"wer": None, "cer": None}
    try:
        import jiwer  # type: ignore
        wer = jiwer.wer(ref, hyp)
        cer = jiwer.cer(ref, hyp)
        return {"wer": float(wer), "cer": float(cer)}
    except Exception:
        return {"wer": None, "cer": None}


# ─────────────────────────────────────────────────────────────────────────────
# ASR details: insertion / deletion / substitution rates
# ─────────────────────────────────────────────────────────────────────────────

def compute_asr_details(ref: str, hyp: str) -> Dict[str, Optional[float]]:
    """Compute detailed word-level ASR error components via jiwer.process_words().

    Returns:
        insertion_rate, deletion_rate, substitution_rate, word_accuracy
    """
    if not ref.strip() or not hyp.strip():
        return {
            "insertion_rate": None,
            "deletion_rate": None,
            "substitution_rate": None,
            "word_accuracy": None,
        }
    try:
        import jiwer  # type: ignore
        out = jiwer.process_words(ref, hyp)
        n_ref = out.hits + out.deletions + out.substitutions
        if n_ref == 0:
            return {
                "insertion_rate": 0.0,
                "deletion_rate": 0.0,
                "substitution_rate": 0.0,
                "word_accuracy": 1.0,
            }
        ins_rate = out.insertions / n_ref
        del_rate = out.deletions / n_ref
        sub_rate = out.substitutions / n_ref
        word_acc = out.hits / n_ref
        return {
            "insertion_rate": float(ins_rate),
            "deletion_rate": float(del_rate),
            "substitution_rate": float(sub_rate),
            "word_accuracy": float(word_acc),
        }
    except Exception:
        return {
            "insertion_rate": None,
            "deletion_rate": None,
            "substitution_rate": None,
            "word_accuracy": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# STT-benchmark enriched metrics (mer, wil, her, hallucination_rate, fwer)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stt_enriched(ref: str, hyp: str) -> Dict[str, Optional[float]]:
    """Richer ASR metrics sourced from stt_benchmark when available.

    Falls back to jiwer 4.x built-ins if stt_benchmark not on sys.path.

    Returns:
        mer, wil, wip, ser, her, hallucination_rate, fwer

    Note on SER vs MER:
        SER (Sentence Error Rate) — per utterance: 1.0 if ANY alignment error
        exists (substitution, deletion, or insertion), else 0.0.  When averaged
        across utterances this gives the dataset-level SER.
        MER = (S+D+I) / (S+D+I+H) — a fractional ratio, always ≤ 1 and always
        ≥ per-utterance SER.  They are numerically distinct except when a
        sentence is either 100% correct or 100% wrong.
    """
    result: Dict[str, Optional[float]] = {
        "mer": None,
        "wil": None,
        "wip": None,
        "ser": None,
        "her": None,
        "hallucination_rate": None,
        "fwer": None,
    }

    if not ref.strip() or not hyp.strip():
        return result

    # ── MER / WIL / SER / WIP from jiwer 4.x ────────────────────────────────
    try:
        import jiwer  # type: ignore
        result["mer"] = float(jiwer.mer(ref, hyp))
        wil_val = float(jiwer.wil(ref, hyp))
        result["wil"] = wil_val
        result["wip"] = max(0.0, 1.0 - wil_val)   # WIP = 1 − WIL

        # SER: binary per-utterance sentence error flag
        out = jiwer.process_words(ref, hyp)
        has_error = (out.substitutions + out.deletions + out.insertions) > 0
        result["ser"] = 1.0 if has_error else 0.0
    except Exception:
        pass

    # ── HER from stt_benchmark ─────────────────────────────────────────────────
    try:
        from src.evaluation.her import compute_her  # type: ignore
        her_res = compute_her(ref, hyp)
        result["her"] = float(her_res.her)
    except Exception:
        pass

    # ── Hallucination rate: fraction of hyp words not in reference vocab ───────
    # Lower is better (0 = no hallucinated words, 1 = all words hallucinated).
    # Note: this is a loose proxy; identical ref/hyp → 0.0.
    try:
        ref_words = set(ref.lower().split())
        hyp_words = hyp.lower().split()
        if hyp_words:
            hall = sum(1 for w in hyp_words if w not in ref_words) / len(hyp_words)
            result["hallucination_rate"] = float(hall)
        else:
            result["hallucination_rate"] = 0.0
    except Exception:
        pass

    # ── FWER from stt_benchmark ───────────────────────────────────────────────
    try:
        from src.evaluation.formatting_metrics import compute_fwer  # type: ignore
        result["fwer"] = float(compute_fwer(ref, hyp))
    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def compute_bert_score(
    refs: List[str],
    hyps: List[str],
    model_type: str = "distilbert-base-uncased",
) -> Dict[str, Optional[float]]:
    """Compute BERTScore F1 (CPU-friendly distilbert).

    Returns dict with keys: bert_score_f1, bert_score_precision, bert_score_recall
    """
    if not refs or not hyps:
        return {"bert_score_f1": None, "bert_score_precision": None, "bert_score_recall": None}

    # Try stt_benchmark
    try:
        from src.evaluation.semantic_metrics import compute_bert_score as _stt_bert  # type: ignore
        res = _stt_bert(refs=refs, hyps=hyps, model_type=model_type)
        return {
            "bert_score_f1": res.get("mean_f1"),
            "bert_score_precision": res.get("mean_precision"),
            "bert_score_recall": res.get("mean_recall"),
        }
    except ImportError:
        pass

    # Standalone fallback
    try:
        from bert_score import score as _bert_score  # type: ignore
        import math as _math
        P, R, F = _bert_score(
            cands=hyps,
            refs=refs,
            model_type=model_type,
            lang="en",
            verbose=False,
            device="cpu",
        )
        def _safe(t) -> Optional[float]:
            v = float(t.mean().item())
            return None if _math.isnan(v) else v
        return {
            "bert_score_f1": _safe(F),
            "bert_score_precision": _safe(P),
            "bert_score_recall": _safe(R),
        }
    except Exception:
        return {"bert_score_f1": None, "bert_score_precision": None, "bert_score_recall": None}


# ─────────────────────────────────────────────────────────────────────────────
# Semantic distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_sem_dist(
    refs: List[str],
    hyps: List[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Optional[float]]:
    """Cosine distance between sentence embeddings (0=identical, 1=unrelated).

    Tries stt_benchmark first, then standalone sentence-transformers.
    """
    if not refs or not hyps:
        return [None] * max(len(refs), len(hyps))

    # stt_benchmark path
    try:
        from src.evaluation.semantic_metrics import compute_semdist  # type: ignore
        return compute_semdist(refs=refs, hyps=hyps, model_name=model_name)
    except ImportError:
        pass

    # Standalone
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np
        key = f"st_{model_name}"
        if key not in MODELS:
            MODELS[key] = SentenceTransformer(model_name)
        model = MODELS[key]
        ref_embs = model.encode(refs, convert_to_numpy=True, show_progress_bar=False)
        hyp_embs = model.encode(hyps, convert_to_numpy=True, show_progress_bar=False)
        dists = []
        for re, he in zip(ref_embs, hyp_embs):
            cos_sim = float(np.dot(re, he) / (np.linalg.norm(re) * np.linalg.norm(he) + 1e-8))
            dists.append(1.0 - cos_sim)
        return dists
    except Exception:
        return [None] * len(refs)


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE-L
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge_l(ref: str, hyp: str) -> Optional[float]:
    """ROUGE-L F1 score."""
    if not ref.strip() or not hyp.strip():
        return None
    try:
        from rouge_score import rouge_scorer  # type: ignore
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        score = scorer.score(ref, hyp)
        return float(score["rougeL"].fmeasure)
    except Exception:
        return None
