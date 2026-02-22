"""src/evaluation/hallucination.py — Hallucination rate on silence/noise inputs."""
from __future__ import annotations

import re

REPETITION_PATTERNS = [
    re.compile(r"(\b\w+\b)(\s+\1){4,}", re.IGNORECASE),
    re.compile(r"(.{3,})\1{3,}"),
]

PHANTOM_PHRASES = [
    "thank you for watching", "thanks for watching", "please subscribe",
    "like and share", "for more information", "visit our website",
    "this video", "in this video",
]


def classify_hallucination(text: str) -> str | None:
    """Return hallucination type or None if empty (= no hallucination)."""
    if not text.strip():
        return None
    text_lower = text.lower().strip()
    for pat in REPETITION_PATTERNS:
        if pat.search(text):
            return "repetition"
    for phrase in PHANTOM_PHRASES:
        if phrase in text_lower:
            return "phantom"
    return "fabrication"


def compute_hallucination_metrics(hypotheses: list[str]) -> dict:
    """
    Given hypotheses from SILENT or NOISE inputs (no real speech),
    any non-empty output = hallucination.
    """
    n = len(hypotheses)
    if n == 0:
        return {"hallucination_rate": float("nan"), "n_total": 0}

    type_counts: dict[str, int] = {"repetition": 0, "phantom": 0, "fabrication": 0}
    n_hall = 0
    examples: list[str] = []

    for hyp in hypotheses:
        htype = classify_hallucination(hyp)
        if htype is not None:
            n_hall += 1
            type_counts[htype] += 1
            if len(examples) < 5:
                examples.append(hyp)

    return {
        "hallucination_rate": n_hall / n,
        "n_hallucinated":     n_hall,
        "n_total":            n,
        "type_counts":        type_counts,
        "examples":           examples,
    }
