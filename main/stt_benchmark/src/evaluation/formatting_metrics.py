"""src/evaluation/formatting_metrics.py — F-WER, Punctuation F1, Capitalisation Accuracy."""
from __future__ import annotations

import jiwer

PUNCT_CHARS = {".": "period", ",": "comma", "?": "question", "!": "exclamation"}

# F-WER: lowercase only — NO punctuation removal, NO contraction expansion
_fwer_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def compute_fwer(ref: str, hyp: str) -> float:
    """WER computed WITHOUT removing punctuation. Always >= standard WER."""
    ref_n = _fwer_transform(ref)
    hyp_n = _fwer_transform(hyp)
    if not ref_n:
        return float("inf") if hyp_n else 0.0
    return jiwer.process_words(ref_n, hyp_n).wer


def compute_punctuation_f1(ref: str, hyp: str) -> dict:
    """
    Per-type Punctuation F1: period, comma, question, exclamation.

    macro_f1 edge cases:
      - ref has no punctuation AND hyp has no punctuation → 1.0 (perfect agreement)
      - ref has no punctuation BUT hyp adds punctuation   → 0.0 (false positives)
      - ref has punctuation                               → normal F1 average
    """
    results = {}
    f1_values = []
    total_ref_punct = sum(ref.count(c) for c in PUNCT_CHARS)
    total_hyp_punct = sum(hyp.count(c) for c in PUNCT_CHARS)

    for char, label in PUNCT_CHARS.items():
        ref_c = ref.count(char)
        hyp_c = hyp.count(char)
        tp = min(ref_c, hyp_c)
        fp = max(0, hyp_c - ref_c)
        fn = max(0, ref_c - hyp_c)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[label] = {"precision": p, "recall": r, "f1": f1}
        if ref_c > 0:
            f1_values.append(f1)

    if f1_values:
        macro_f1 = sum(f1_values) / len(f1_values)
    elif total_ref_punct == 0 and total_hyp_punct == 0:
        macro_f1 = 1.0   # both agree: no punctuation expected or produced
    else:
        macro_f1 = 0.0   # ref has no punctuation but hyp incorrectly added some

    return {"per_type": results, "macro_f1": macro_f1}


def compute_capitalization_accuracy(ref: str, hyp: str) -> float:
    """
    Capitalisation accuracy: 1.0 = all aligned words share the same casing pattern, 0.0 = none do.

    Formula: 1 - (n_mismatches / n_ref_words)

    A mismatch is counted only for the min(len_ref, len_hyp) aligned word pairs.
    Un-matched reference words (deletions) are not counted as mismatches, but they
    do reduce accuracy because the denominator is always len(ref_words).
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return float("nan")
    n = min(len(ref_words), len(hyp_words))
    if n == 0:
        return 0.0
    mismatches = sum(1 for r, h in zip(ref_words[:n], hyp_words[:n]) if _casing(r) != _casing(h))
    return 1.0 - (mismatches / len(ref_words))


def _casing(word: str) -> str:
    if word.isupper():  return "upper"
    if word.islower():  return "lower"
    if word.istitle():  return "title"
    return "mixed"
