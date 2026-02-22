"""src/evaluation/metrics.py — WER/CER/MER/WIL/WIP/SER via jiwer 4.x."""
from __future__ import annotations

import logging

import numpy as np
import jiwer

from .normalizer import TranscriptNormalizer

log = logging.getLogger(__name__)


def compute_sample_metrics(
    ref: str,
    hyp: str,
    normalizer: TranscriptNormalizer | None = None,
) -> dict:
    """
    Compute all surface-accuracy metrics for one ref/hyp pair.
    Normalizes both texts if a normalizer is provided.
    """
    if normalizer:
        ref = normalizer.normalize(ref)
        hyp = normalizer.normalize(hyp)

    if not ref and not hyp:
        return _zero_metrics()
    if not ref:
        return _inf_metrics()

    word_out = jiwer.process_words(ref, hyp)
    char_out = jiwer.process_characters(ref, hyp)

    hits  = word_out.hits
    subs  = word_out.substitutions
    dels  = word_out.deletions
    ins   = word_out.insertions
    ref_words = hits + subs + dels

    # Word Recognition Rate (complement of WER)
    wrr = hits / ref_words if ref_words > 0 else 0.0

    # Per-type error rates (normalized by ref length)
    sub_rate = subs / ref_words if ref_words > 0 else 0.0
    del_rate = dels / ref_words if ref_words > 0 else 0.0
    ins_rate = ins  / ref_words if ref_words > 0 else 0.0

    # Total error word count
    error_words = subs + dels + ins

    # SER = fraction of sentences with at least one error (binary per sample)
    # When aggregated (mean), this gives % of sentences with any error.
    sentence_has_error = 1.0 if (subs > 0 or dels > 0 or ins > 0) else 0.0

    return {
        "wer":           word_out.wer,
        "cer":           char_out.cer,
        "mer":           word_out.mer,
        "wil":           word_out.wil,
        "wip":           word_out.wip,
        "ser":           sentence_has_error,
        "substitutions": subs,
        "deletions":     dels,
        "insertions":    ins,
        "hits":          hits,
        # New per-sample metrics
        "wrr":           wrr,
        "sub_rate":      sub_rate,
        "del_rate":      del_rate,
        "ins_rate":      ins_rate,
        "error_words":   error_words,
    }


def _zero_metrics() -> dict:
    return dict(wer=0.0, cer=0.0, mer=0.0, wil=0.0, wip=1.0, ser=0.0,
                substitutions=0, deletions=0, insertions=0, hits=0,
                wrr=1.0, sub_rate=0.0, del_rate=0.0, ins_rate=0.0, error_words=0)


def _inf_metrics() -> dict:
    # ser=1.0: a sentence with no reference but a hypothesis is an error
    return dict(wer=float("inf"), cer=float("inf"), mer=1.0, wil=1.0, wip=0.0, ser=1.0,
                substitutions=0, deletions=0, insertions=0, hits=0,
                wrr=0.0, sub_rate=0.0, del_rate=0.0, ins_rate=0.0, error_words=0)


def aggregate_metrics(per_sample: list[dict]) -> dict:
    """
    Micro-average WER (sum errors / sum ref words).
    Mean for all other metrics.
    """
    valid = [m for m in per_sample if np.isfinite(m.get("wer", float("inf")))]
    if not valid:
        return {}

    total_subs = sum(m.get("substitutions", 0) for m in valid)
    total_dels = sum(m.get("deletions",     0) for m in valid)
    total_ins  = sum(m.get("insertions",    0) for m in valid)
    total_hits = sum(m.get("hits",          0) for m in valid)
    total_ref  = total_hits + total_subs + total_dels
    micro_wer  = (total_subs + total_dels + total_ins) / total_ref if total_ref > 0 else float("inf")

    def _mean(key: str) -> float:
        vals = [m[key] for m in valid if key in m and np.isfinite(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "micro_wer":       micro_wer,
        "mean_wer":        _mean("wer"),
        "mean_cer":        _mean("cer"),
        "mean_mer":        _mean("mer"),
        "mean_wil":        _mean("wil"),
        "mean_wip":        _mean("wip"),
        "mean_ser":        _mean("ser"),
        "mean_wrr":        _mean("wrr"),
        "mean_sub_rate":   _mean("sub_rate"),
        "mean_del_rate":   _mean("del_rate"),
        "mean_ins_rate":   _mean("ins_rate"),
        "n_samples":       len(valid),
        "total_ref_words": total_ref,
        "total_error_words": sum(m.get("error_words", 0) for m in valid),
    }
