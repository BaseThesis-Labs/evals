"""src/evaluation/semantic_metrics.py — SemDist (cosine distance) + BERTScore."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_st_model = None


def _get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    global _st_model
    if _st_model is None:
        log.info(f"Loading sentence-transformers '{model_name}' (lazy)…")
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(model_name)
    return _st_model


def compute_semdist(
    refs: list[str],
    hyps: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[float]:
    """
    Semantic distance per pair: 1 - cosine_similarity(ref_emb, hyp_emb).
    0.0 = identical meaning, 1.0 = completely unrelated.
    """
    model = _get_st_model(model_name)
    embeddings = model.encode(
        refs + hyps, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )
    n = len(refs)
    cos_sim = np.sum(embeddings[:n] * embeddings[n:], axis=1)
    return (1.0 - cos_sim).tolist()


def compute_semdist_single(ref: str, hyp: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    return compute_semdist([ref], [hyp], model_name)[0]


def compute_asd(
    refs: list[str],
    hyps: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[float]:
    """
    Average Semantic Distance (ASD) — word-level semantic distance.

    Complements sentence-level SemDist by operating at word granularity:
      1. Encode each word individually (both ref and hyp words in one batch).
      2. For each ref word, compute cosine similarity to all hyp words.
      3. Per-sentence ASD = mean of (1 - max cosine_sim) over all ref words.

    ASD ≈ 0  → all ref words have a semantically close match in the hyp.
    ASD ≈ 1  → ref words are semantically unrelated to anything in the hyp.

    More sensitive to individual word errors and insertions/deletions than
    sentence-level SemDist; less sensitive to sentence structure.
    """
    model = _get_st_model(model_name)
    results: list[float] = []

    for ref, hyp in zip(refs, hyps):
        ref_words = ref.split()
        hyp_words = hyp.split()

        if not ref_words:
            results.append(float("nan"))
            continue
        if not hyp_words:
            # No hypothesis words — every ref word has distance 1.0
            results.append(1.0)
            continue

        # Encode all words in one batch (efficient)
        all_words = ref_words + hyp_words
        embs = model.encode(all_words, batch_size=128, show_progress_bar=False,
                            normalize_embeddings=True)
        ref_embs = embs[:len(ref_words)]    # (n_ref, d)
        hyp_embs = embs[len(ref_words):]    # (n_hyp, d)

        # Cosine similarity matrix: (n_ref, n_hyp) — already unit-norm so dot = cosine
        sim_matrix = ref_embs @ hyp_embs.T   # (n_ref, n_hyp)
        max_sim = sim_matrix.max(axis=1)      # best match per ref word
        word_distances = 1.0 - np.clip(max_sim, -1.0, 1.0)
        results.append(float(np.mean(word_distances)))

    return results


def _bertscore_direct(
    refs: list[str],
    hyps: list[str],
    model_type: str = "bert-base-uncased",
) -> tuple[list[float], list[float], list[float]]:
    """
    Compute BERTScore P/R/F1 directly via transformers, bypassing the bert_score
    package (which is incompatible with transformers ≥5.x).

    Implements the greedy-matching algorithm from Zhang et al. (2020):
      P = mean of max_r cosine_sim(h_i, r_j) over hyp tokens
      R = mean of max_h cosine_sim(r_j, h_i) over ref tokens
      F1 = 2*P*R / (P+R)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type)
    model.eval()

    P_list, R_list, F_list = [], [], []

    with torch.no_grad():
        for ref, hyp in zip(refs, hyps):
            if not ref.strip() or not hyp.strip():
                P_list.append(0.0); R_list.append(0.0); F_list.append(0.0)
                continue

            r_enc = tokenizer(ref, return_tensors="pt", max_length=512, truncation=True)
            h_enc = tokenizer(hyp, return_tensors="pt", max_length=512, truncation=True)

            r_emb = model(**r_enc).last_hidden_state[0]   # (L_r, d)
            h_emb = model(**h_enc).last_hidden_state[0]   # (L_h, d)

            # Strip [CLS] / [SEP]
            r_emb = r_emb[1:-1] if r_emb.shape[0] > 2 else r_emb
            h_emb = h_emb[1:-1] if h_emb.shape[0] > 2 else h_emb

            if r_emb.shape[0] == 0 or h_emb.shape[0] == 0:
                P_list.append(0.0); R_list.append(0.0); F_list.append(0.0)
                continue

            r_norm = torch.nn.functional.normalize(r_emb, dim=-1)
            h_norm = torch.nn.functional.normalize(h_emb, dim=-1)

            sim = h_norm @ r_norm.T           # (L_h, L_r)
            p   = float(sim.max(dim=1).values.mean())   # precision
            r   = float(sim.max(dim=0).values.mean())   # recall
            f   = 2 * p * r / (p + r) if p + r > 0 else 0.0

            P_list.append(p); R_list.append(r); F_list.append(f)

    return P_list, R_list, F_list


def compute_bert_score(
    refs: list[str],
    hyps: list[str],
    model_type: str = "bert-base-uncased",
    lang: str = "en",
) -> dict:
    """
    BERTScore P/R/F1.

    Strategy (in order):
      1. Try the `bert_score` package  (fast, cached IDF weights).
      2. If that fails (e.g. bert-score 0.3.x + transformers ≥5.x
         incompatibility), fall back to a direct transformers implementation.
      3. If both fail, return NaN lists with a warning.

    Returns: precision, recall, f1 (lists), mean_precision, mean_recall, mean_f1.
    """
    nan_result = {
        "precision":      [float("nan")] * len(refs),
        "recall":         [float("nan")] * len(refs),
        "f1":             [float("nan")] * len(refs),
        "mean_precision": float("nan"),
        "mean_recall":    float("nan"),
        "mean_f1":        float("nan"),
    }
    if not refs:
        return nan_result

    # ── Attempt 1: bert_score package ─────────────────────────────────────────
    try:
        log.info("Computing BERTScore via bert_score package…")
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(hyps, refs, model_type=model_type, lang=lang, verbose=False)
        return {
            "precision":      P.tolist(),
            "recall":         R.tolist(),
            "f1":             F1.tolist(),
            "mean_precision": float(P.mean()),
            "mean_recall":    float(R.mean()),
            "mean_f1":        float(F1.mean()),
        }
    except Exception as e1:
        log.warning(f"bert_score package failed ({e1}). Trying direct transformers fallback…")

    # ── Attempt 2: direct transformers implementation ──────────────────────────
    try:
        log.info(f"Computing BERTScore directly via transformers ({model_type})…")
        P_list, R_list, F_list = _bertscore_direct(refs, hyps, model_type)
        return {
            "precision":      P_list,
            "recall":         R_list,
            "f1":             F_list,
            "mean_precision": float(np.mean(P_list)),
            "mean_recall":    float(np.mean(R_list)),
            "mean_f1":        float(np.mean(F_list)),
        }
    except Exception as e2:
        log.warning(
            f"Direct BERTScore also failed ({e2}). "
            "Returning NaN. Install PyTorch + transformers to enable BERTScore."
        )
        return nan_result
