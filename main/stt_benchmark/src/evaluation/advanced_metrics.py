"""src/evaluation/advanced_metrics.py — BLEU, METEOR, PER, Keyword Recall,
Error Severity, SHALLOW hallucination scores, LLM Impact Judge.

All functions return flat dicts suitable for merging into per-sample metrics.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

_PUNCT_RE = re.compile(r'[.!?,;:\-\'"()]')


# ── BLEU ──────────────────────────────────────────────────────────────────────

def compute_bleu(reference: str, hypothesis: str) -> dict:
    """BLEU-1 and BLEU-4 with add-1 smoothing."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not ref_tokens or not hyp_tokens:
            return {"bleu_1": 0.0, "bleu_4": 0.0}
        smooth = SmoothingFunction().method1
        bleu_1 = sentence_bleu(
            [ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth
        )
        bleu_4 = sentence_bleu(
            [ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth
        )
        return {"bleu_1": float(bleu_1), "bleu_4": float(bleu_4)}
    except Exception:
        return {"bleu_1": float("nan"), "bleu_4": float("nan")}


# ── METEOR ────────────────────────────────────────────────────────────────────

def compute_meteor(reference: str, hypothesis: str) -> float:
    """METEOR score — accounts for synonyms and stemming via WordNet."""
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        return float(nltk_meteor([ref_tokens], hyp_tokens))
    except Exception:
        return float("nan")


# ── Punctuation Error Rate ─────────────────────────────────────────────────────

def compute_per(reference: str, hypothesis: str) -> dict:
    """
    Punctuation Error Rate (punct_per) — WER on punctuation token sequences.

    Uses RAW (unnormalized) text so punctuation is preserved.
    Returns 0.0 for datasets without punctuation in the reference (e.g. LibriSpeech ALL-CAPS).

    punct_per = jiwer.wer(ref_punct_sequence, hyp_punct_sequence)
    punct_precision/recall/f1_adv = token-level overlap F1 on the same sequences.
    """
    import jiwer as _jiwer

    def _extract_punct_seq(text: str) -> str:
        return " ".join(_PUNCT_RE.findall(text))

    ref_punct = _extract_punct_seq(reference)
    hyp_punct = _extract_punct_seq(hypothesis)

    if not ref_punct.strip():
        # No punctuation in reference — metric is undefined; return 0
        return {"punct_per": 0.0, "punct_precision": 0.0, "punct_recall": 0.0, "punct_f1_adv": 0.0}

    punct_per = float(_jiwer.wer(ref_punct, hyp_punct))

    # Token-level overlap for precision/recall/F1 (complements the WER view)
    ref_toks = ref_punct.split()
    hyp_toks = hyp_punct.split()
    rc = Counter(ref_toks)
    hc = Counter(hyp_toks)
    tp = sum(min(rc[c], hc.get(c, 0)) for c in rc)
    fp = sum(max(0, hc[c] - rc.get(c, 0)) for c in hc)
    fn = sum(max(0, rc[c] - hc.get(c, 0)) for c in rc)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "punct_per":       punct_per,
        "punct_precision": precision,
        "punct_recall":    recall,
        "punct_f1_adv":    f1,
    }


# ── Keyword Recall Rate ───────────────────────────────────────────────────────

def compute_krr(reference: str, hypothesis: str, keywords: list[str]) -> dict:
    """
    Keyword Recall Rate — fraction of domain-critical terms correctly transcribed.
    keywords: list of strings from evaluation.yaml advanced.keywords.
    """
    if not keywords:
        return {"krr": float("nan"), "keywords_missed": []}
    ref_lower = reference.lower()
    hyp_lower = hypothesis.lower()
    present   = [kw for kw in keywords if kw.lower() in ref_lower]
    if not present:
        return {"krr": 1.0, "keywords_missed": []}
    found  = [kw for kw in present if kw.lower() in hyp_lower]
    missed = [kw for kw in present if kw.lower() not in hyp_lower]
    return {"krr": len(found) / len(present), "keywords_missed": missed}


# ── Error Severity ─────────────────────────────────────────────────────────────

def compute_error_severity(reference: str, hypothesis: str, encoder) -> dict:
    """
    Per-substitution embedding distance — measures HOW BAD each error is.
    0 = minor (cosmetic), 1 = catastrophic (opposite meaning).
    Paper: BioNLP 2023 "Evaluating and Improving ASR: Severity Score".
    """
    try:
        import jiwer
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        if not ref_words or not hyp_words:
            return {"avg_error_severity": float("nan"), "max_error_severity": float("nan")}
        out = jiwer.process_words(reference.lower(), hypothesis.lower())
    except Exception:
        return {"avg_error_severity": float("nan"), "max_error_severity": float("nan")}

    severities: list[float] = []
    for chunk in out.alignments[0]:
        if chunk.type == "substitute":
            ref_w = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
            hyp_w = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
            try:
                emb = encoder.encode([ref_w, hyp_w])
                na = np.linalg.norm(emb[0])
                nb = np.linalg.norm(emb[1])
                dist = 1.0 - float(np.dot(emb[0], emb[1]) / (na * nb)) if na > 0 and nb > 0 else 1.0
                severities.append(max(0.0, dist))
            except Exception:
                severities.append(1.0)
        elif chunk.type in ("delete", "insert"):
            severities.append(1.0)

    if not severities:
        return {"avg_error_severity": 0.0, "max_error_severity": 0.0}
    return {
        "avg_error_severity": float(np.mean(severities)),
        "max_error_severity": float(max(severities)),
    }


# ── SHALLOW Hallucination Scores ──────────────────────────────────────────────

def compute_shallow(reference: str, hypothesis: str) -> dict:
    """
    Lightweight approximation of the SHALLOW 4-dimension hallucination scoring.
    Paper: "Hallucination Benchmark for Speech Foundation Models" (arXiv Oct 2025).

    shallow_sf: Semantic Fabrication   — lower is better
    shallow_pf: Phonetic (safe) errors — higher means errors are phonetically close
    shallow_rl: Repetitive Looping     — lower is better
    shallow_lc: Language Confusion     — lower is better
    """
    try:
        import jiwer
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        ref_set   = set(ref_words)

        # SF: words in hypothesis not present anywhere in reference
        fabricated = [w for w in hyp_words if w not in ref_set]
        sf_rate = len(fabricated) / len(hyp_words) if hyp_words else 0.0

        # PF: phonetic substitutions (share first 2 chars)
        phonetic_subs = 0
        total_subs    = 0
        try:
            out = jiwer.process_words(reference.lower(), hypothesis.lower())
            for chunk in out.alignments[0]:
                if chunk.type == "substitute":
                    total_subs += 1
                    r = ref_words[chunk.ref_start_idx] if chunk.ref_start_idx < len(ref_words) else ""
                    h = hyp_words[chunk.hyp_start_idx] if chunk.hyp_start_idx < len(hyp_words) else ""
                    if len(r) >= 2 and len(h) >= 2 and r[:2] == h[:2]:
                        phonetic_subs += 1
        except Exception:
            pass
        pf_rate = phonetic_subs / total_subs if total_subs > 0 else 0.0

        # RL: repeated bigrams
        bigrams = [f"{hyp_words[i]} {hyp_words[i+1]}" for i in range(len(hyp_words) - 1)]
        if bigrams:
            counts  = Counter(bigrams)
            repeated = sum(c - 1 for c in counts.values() if c > 1)
            rl_rate  = repeated / len(bigrams)
        else:
            rl_rate = 0.0

        # LC: non-ASCII characters
        non_ascii = len(re.findall(r'[^\x00-\x7F]', hypothesis))
        lc_rate   = non_ascii / len(hypothesis) if hypothesis else 0.0

        return {
            "shallow_sf": float(sf_rate),
            "shallow_pf": float(pf_rate),
            "shallow_rl": float(rl_rate),
            "shallow_lc": float(lc_rate),
        }
    except Exception:
        return {
            "shallow_sf": float("nan"),
            "shallow_pf": float("nan"),
            "shallow_rl": float("nan"),
            "shallow_lc": float("nan"),
        }


# ── LLM Impact Judge ──────────────────────────────────────────────────────────

def compute_llm_impact(
    reference: str,
    hypothesis: str,
    client,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    LLM-as-judge: does the ASR error actually change the meaning?
    Paper: "WER is Unaware" (arXiv 2511.16544, Nov 2025).
    Achieves 90% accuracy, Cohen's κ = 0.816 with human experts.
    """
    if not reference.strip():
        return {"impact_label": "none", "impact_score": 0.0}
    if reference.strip().lower() == hypothesis.strip().lower():
        return {"impact_label": "none", "impact_score": 0.0}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": (
                f"Compare these two transcriptions and rate the impact of differences:\n"
                f"Reference: \"{reference}\"\n"
                f"ASR Output: \"{hypothesis}\"\n\n"
                f"Rate the impact on meaning/usability:\n"
                f"- NO_IMPACT: cosmetic differences only (articles, minor formatting)\n"
                f"- MINIMAL_IMPACT: slight meaning change but intent preserved\n"
                f"- SIGNIFICANT_IMPACT: meaning changed or intent corrupted\n\n"
                f"Respond with ONLY one of: NO_IMPACT, MINIMAL_IMPACT, SIGNIFICANT_IMPACT"
            )}],
            max_tokens=20,
            temperature=0,
        )
        label = resp.choices[0].message.content.strip().upper()
        score_map = {
            "NO_IMPACT":          0.0,
            "MINIMAL_IMPACT":     0.5,
            "SIGNIFICANT_IMPACT": 1.0,
        }
        return {
            "impact_label": label.lower().replace("_impact", ""),
            "impact_score": score_map.get(label, 0.5),
        }
    except Exception as e:
        return {"impact_label": "error", "impact_score": float("nan")}


# ── Embedding-based Hallucination Detection ────────────────────────────────────

def compute_embedding_hallucination(
    reference: str,
    hypothesis: str,
    encoder,
    threshold: float = 0.7,
    window_size: int = 3,
) -> dict:
    """
    Flag inserted words whose cosine distance to every reference window
    exceeds `threshold` (default 0.7).  Procedure section §3(d).

    For each word insertion identified by jiwer:
      - Compute its embedding.
      - Compute cosine similarity to all overlapping reference n-gram windows.
      - If max similarity < (1 - threshold), the insertion is semantically
        ungrounded in the reference → hallucination.

    Returns:
      emb_hallucination_rate   – n_hallucinated / n_ref_words
      n_emb_hallucinations     – raw count of hallucinated insertions
    """
    import jiwer

    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    total_ref = len(ref_words)

    if not ref_words or not hyp_words:
        return {"emb_hallucination_rate": 0.0, "n_emb_hallucinations": 0}

    try:
        out = jiwer.process_words(reference.lower(), hypothesis.lower())
    except Exception:
        return {"emb_hallucination_rate": float("nan"), "n_emb_hallucinations": 0}

    # Collect inserted word strings
    insertions: list[str] = []
    for chunk in out.alignments[0]:
        if chunk.type == "insert":
            for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                if i < len(hyp_words):
                    insertions.append(hyp_words[i])

    if not insertions:
        return {"emb_hallucination_rate": 0.0, "n_emb_hallucinations": 0}

    # Build overlapping reference windows
    ref_windows = [
        " ".join(ref_words[i: i + window_size])
        for i in range(max(1, len(ref_words) - window_size + 1))
    ]

    try:
        all_texts = insertions + ref_windows
        embs = encoder.encode(all_texts, show_progress_bar=False, normalize_embeddings=True)
        ins_embs = embs[:len(insertions)]      # shape (n_ins, d)
        win_embs = embs[len(insertions):]      # shape (n_win, d)

        # cos_sim matrix: (n_ins, n_win) — already unit-norm so dot = cosine
        cos_sim_matrix = ins_embs @ win_embs.T   # (n_ins, n_win)
        max_sim_per_ins = cos_sim_matrix.max(axis=1)  # (n_ins,)

        n_hallucinated = int((max_sim_per_ins < (1.0 - threshold)).sum())
    except Exception:
        return {"emb_hallucination_rate": float("nan"), "n_emb_hallucinations": 0}

    return {
        "emb_hallucination_rate": n_hallucinated / total_ref if total_ref > 0 else 0.0,
        "n_emb_hallucinations":   n_hallucinated,
    }


# ── Semantic WER (LLM-judged) ─────────────────────────────────────────────────

def compute_semantic_wer(
    reference: str,
    hypothesis: str,
    client,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """
    LLM-judged Semantic WER — distinguishes meaningful errors from trivial ones.

    For each word-level diff (substitution, insertion, deletion), asks Claude
    whether the difference changes the MEANING of the utterance.

    trivial:    filler words, minor rephrasing, contractions, articles
    meaningful: changed entities, numbers, negation, key facts, intent

    Returns:
        semantic_wer       — meaningful_errors / n_ref_words
        trivial_errors     — count of trivial diffs
        meaningful_errors  — count of meaningful diffs
    """
    import jiwer as _jiwer

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return {"semantic_wer": float("nan"), "trivial_errors": 0, "meaningful_errors": 0}
    if reference.strip().lower() == hypothesis.strip().lower():
        return {"semantic_wer": 0.0, "trivial_errors": 0, "meaningful_errors": 0}

    try:
        out = _jiwer.process_words(reference, hypothesis)
    except Exception:
        return {"semantic_wer": float("nan"), "trivial_errors": 0, "meaningful_errors": 0}

    diffs = []
    for chunk in out.alignments[0]:
        if chunk.type != "equal":
            ref_span = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
            hyp_span = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
            diffs.append({"type": chunk.type, "ref": ref_span, "hyp": hyp_span})

    if not diffs:
        return {"semantic_wer": 0.0, "trivial_errors": 0, "meaningful_errors": 0}

    diff_text = "\n".join(
        f"- {d['type']}: \"{d['ref']}\" → \"{d['hyp']}\"" for d in diffs
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": (
                    f"Reference: \"{reference}\"\n"
                    f"Hypothesis: \"{hypothesis}\"\n\n"
                    f"For each word difference below, reply with ONLY \"meaningful\" or \"trivial\" "
                    f"on each line (one answer per diff, same order).\n"
                    f"trivial = cosmetic (articles, fillers, minor phrasing, contractions).\n"
                    f"meaningful = changes intent, entities, numbers, negation, or key facts.\n\n"
                    f"{diff_text}"
                ),
            }],
        )
        text = response.content[0].text.lower()
        meaningful = text.count("meaningful")
        trivial    = text.count("trivial")
    except Exception:
        return {"semantic_wer": float("nan"), "trivial_errors": 0, "meaningful_errors": 0}

    ref_word_count = len(ref_words)
    return {
        "semantic_wer":     meaningful / ref_word_count if ref_word_count > 0 else float("nan"),
        "trivial_errors":   trivial,
        "meaningful_errors": meaningful,
    }
