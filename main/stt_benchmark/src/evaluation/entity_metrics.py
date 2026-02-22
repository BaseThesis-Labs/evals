"""src/evaluation/entity_metrics.py — Named Entity F1 via spaCy."""
from __future__ import annotations

import logging
from collections import defaultdict

log = logging.getLogger(__name__)

_nlp = None


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


def compute_entity_f1(ref: str, hyp: str, spacy_model: str = "en_core_web_sm") -> dict:
    """NER F1 between reference and hypothesis text."""
    nlp = _get_nlp(spacy_model)
    ref_ents = [(e.text.lower().strip(), e.label_) for e in nlp(ref).ents]
    hyp_ents = [(e.text.lower().strip(), e.label_) for e in nlp(hyp).ents]

    ref_counts: dict = defaultdict(int)
    hyp_counts: dict = defaultdict(int)
    for e in ref_ents: ref_counts[e] += 1
    for e in hyp_ents: hyp_counts[e] += 1

    tp = sum(min(ref_counts[e], hyp_counts.get(e, 0)) for e in ref_counts)
    fp = sum(max(0, hyp_counts[e] - ref_counts.get(e, 0)) for e in hyp_counts)
    fn = sum(max(0, ref_counts[e] - hyp_counts.get(e, 0)) for e in ref_counts)

    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "precision": p, "recall": r, "f1": f1,
        "n_ref_ents": len(ref_ents),
        "n_hyp_ents": len(hyp_ents),
    }


def compute_entity_and_krr(ref: str, hyp: str, spacy_model: str = "en_core_web_sm") -> dict:
    """
    Compute Entity F1 and spaCy-based Keyword Recall Rate (KRR) in one pass.

    entity_f1 — NER-based F1 over named entities (persons, orgs, locations, etc.)
    krr       — fraction of content keywords (nouns, proper nouns, numbers) from
                the reference that appear anywhere in the hypothesis.

    No manual keyword list required — keywords are extracted automatically via POS tags.
    """
    nlp = _get_nlp(spacy_model)
    ref_doc = nlp(ref)
    hyp_doc = nlp(hyp)

    # ── Entity F1 ─────────────────────────────────────────────────────────────
    ref_entities = {ent.text.lower().strip() for ent in ref_doc.ents}
    hyp_entities = {ent.text.lower().strip() for ent in hyp_doc.ents}

    if not ref_entities:
        entity_f1 = None
    else:
        tp = len(ref_entities & hyp_entities)
        prec = tp / len(hyp_entities) if hyp_entities else 0.0
        rec  = tp / len(ref_entities)
        entity_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # ── KRR — nouns + proper nouns + numbers ─────────────────────────────────
    ref_keywords   = {t.text.lower() for t in ref_doc if t.pos_ in ("NOUN", "PROPN", "NUM")}
    hyp_word_lower = {t.text.lower() for t in hyp_doc}

    if not ref_keywords:
        krr = None
    else:
        krr = len(ref_keywords & hyp_word_lower) / len(ref_keywords)

    return {
        "entity_f1": entity_f1,
        "krr":       krr,
        "entity_n":  len(ref_entities),
        "keyword_n": len(ref_keywords),
    }
