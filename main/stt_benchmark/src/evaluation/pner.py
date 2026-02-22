"""src/evaluation/pner.py — PNER and Alphanumeric Entity Accuracy.

PNER (Proper-Noun Entity Recognition Rate):
  Extract named entities from reference with spaCy, align ref→hyp via jiwer
  word alignment, compute Jaro-Winkler similarity per entity, average = PNER.

Alphanumeric Accuracy:
  Detect codes / serial-numbers / identifiers (e.g. "A1B2", "V100", "COVID-19")
  in reference, check whether the hypothesis contains the same string (exact,
  after lowercasing).  Fraction correctly reproduced = alphanumeric accuracy.
"""
from __future__ import annotations

import re
import logging
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

_nlp = None


# ── Jaro-Winkler (no external dependency) ─────────────────────────────────────

def _jaro(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if not len1 or not len2:
        return 0.0
    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len2)
        for j in range(lo, hi):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    t = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1
    return (matches / len1 + matches / len2 + (matches - t / 2) / matches) / 3


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler similarity ∈ [0,1].  p = prefix-scaling factor (std = 0.1)."""
    j = _jaro(s1, s2)
    prefix = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            break
        prefix += 1
        if prefix >= 4:
            break
    return j + prefix * p * (1 - j)


# ── spaCy lazy loader ──────────────────────────────────────────────────────────

def _get_nlp(model: str = "en_core_web_sm"):
    global _nlp
    if _nlp is None:
        log.info(f"Loading spaCy '{model}' (lazy)…")
        import spacy
        try:
            _nlp = spacy.load(model)
        except OSError:
            log.warning(f"spaCy model '{model}' not found. Run: python -m spacy download {model}")
            raise
    return _nlp


# ── jiwer alignment helper ─────────────────────────────────────────────────────

def _build_ref_to_hyp_map(ref_words: List[str], hyp_words: List[str]) -> dict:
    """Return {ref_word_idx: [hyp_word_idx, ...]} via jiwer word alignment."""
    import jiwer
    ref_to_hyp: dict[int, list[int]] = {}
    try:
        out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
        for chunk in out.alignments[0]:
            if chunk.type == "equal":
                for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    ref_to_hyp[chunk.ref_start_idx + offset] = [chunk.hyp_start_idx + offset]
            elif chunk.type == "substitute":
                # Map each ref word to the corresponding (possibly off-by-one) hyp word
                for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    hi = chunk.hyp_start_idx + offset
                    if hi < chunk.hyp_end_idx:
                        ref_to_hyp[chunk.ref_start_idx + offset] = [hi]
            # "delete": ref word maps to nothing → leave absent from dict
            # "insert": hyp words with no ref counterpart → ignore for ref→hyp map
    except Exception:
        pass
    return ref_to_hyp


# ── PNER ──────────────────────────────────────────────────────────────────────

def compute_pner(
    ref: str,
    hyp: str,
    spacy_model: str = "en_core_web_sm",
) -> dict:
    """
    Proper-Noun Entity Recognition Rate via Jaro-Winkler alignment.

    For each named entity in the reference:
      1. Locate it in the ref word list.
      2. Map its words to hyp words via jiwer alignment.
      3. Compute Jaro-Winkler(ref_entity_str, hyp_entity_str).

    Returns:
      pner           – mean Jaro-Winkler similarity across all ref entities ∈ [0,1]
      pner_precision – fraction of entities perfectly reproduced (JW ≥ 0.95)
      pner_n         – number of reference named entities
    """
    try:
        nlp = _get_nlp(spacy_model)
    except Exception:
        return {"pner": float("nan"), "pner_precision": float("nan"), "pner_n": 0}

    ref_lower = ref.lower()
    hyp_lower = hyp.lower()

    ref_doc = nlp(ref)
    entities = [(e.text.lower(), e.label_) for e in ref_doc.ents]

    if not entities:
        return {"pner": float("nan"), "pner_precision": float("nan"), "pner_n": 0}

    ref_words = ref_lower.split()
    hyp_words = hyp_lower.split()
    ref_to_hyp = _build_ref_to_hyp_map(ref_words, hyp_words)

    similarities: list[float] = []
    perfect = 0

    for ent_text, _ in entities:
        ent_words = ent_text.split()
        ent_len = len(ent_words)

        # Find entity start position in ref word list
        start_idx = None
        for i in range(len(ref_words) - ent_len + 1):
            if ref_words[i:i + ent_len] == ent_words:
                start_idx = i
                break

        if start_idx is None:
            # Entity text not found as exact token span (tokenization mismatch) — skip
            continue

        # Gather corresponding hyp indices
        hyp_indices: list[int] = []
        for ri in range(start_idx, start_idx + ent_len):
            hyp_indices.extend(ref_to_hyp.get(ri, []))

        hyp_indices = sorted(set(i for i in hyp_indices if i < len(hyp_words)))
        if not hyp_indices:
            similarities.append(0.0)
            continue

        hyp_span = " ".join(hyp_words[i] for i in hyp_indices)
        sim = jaro_winkler(ent_text, hyp_span)
        similarities.append(sim)
        if sim >= 0.95:
            perfect += 1

    if not similarities:
        return {"pner": float("nan"), "pner_precision": float("nan"), "pner_n": len(entities)}

    return {
        "pner":           float(np.mean(similarities)),
        "pner_precision": perfect / len(similarities),
        "pner_n":         len(entities),
    }


# ── Alphanumeric accuracy ──────────────────────────────────────────────────────

# Pattern: tokens that contain at least one digit AND at least one letter,
# OR tokens that are purely numeric (model may drop leading zeros, etc.)
# Catches: "A1B2", "V100", "COVID-19", "10km", "GPT-4", "C3PO"
_ALNUM_PATTERN = re.compile(
    r"\b(?=[A-Za-z0-9\-]*[A-Za-z])(?=[A-Za-z0-9\-]*[0-9])[A-Za-z0-9\-]{2,}\b"
)


def compute_alphanumeric_accuracy(ref: str, hyp: str) -> dict:
    """
    Alphanumeric Entity Accuracy.

    Finds all mixed letter+digit tokens in the reference (e.g. "COVID-19", "A100",
    "V8", "GPT-4") and checks whether each appears verbatim in the hypothesis
    (case-insensitive).  Returns:

      alphanumeric_acc – fraction of ref alphanumeric tokens found in hyp
      alphanumeric_n   – number of ref alphanumeric tokens found
    """
    ref_tokens = _ALNUM_PATTERN.findall(ref.lower())
    if not ref_tokens:
        return {"alphanumeric_acc": float("nan"), "alphanumeric_n": 0}

    hyp_lower = hyp.lower()
    # Deduplicate ref tokens so each unique code is counted once
    unique_tokens = list(dict.fromkeys(ref_tokens))
    hits = sum(1 for tok in unique_tokens if tok in hyp_lower)
    return {
        "alphanumeric_acc": hits / len(unique_tokens),
        "alphanumeric_n":   len(unique_tokens),
    }
