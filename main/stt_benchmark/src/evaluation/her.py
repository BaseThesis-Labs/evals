"""src/evaluation/her.py — Hallucination Error Rate (HER).

Paper: ACL Findings 2025, "Demystifying Hallucination in Speech Foundation Models".
Classifies each ASR error as phonetic, hallucination, or repetition loop.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HERResult:
    total_errors: int
    phonetic_errors: int
    hallucination_errors: int
    repetition_errors: int
    her: float                       # hallucination_errors / total_ref_words
    insertion_hallucination_rate: float


def compute_her(reference: str, hypothesis: str) -> HERResult:
    """
    Classify ASR errors into:
      - phonetic      : substitution where first 2 chars match (sound-alike)
      - hallucination : fabricated insertion or semantically unrelated substitution
      - repetition    : consecutive duplicate words (Whisper-style looping)
    """
    import jiwer

    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    total_ref = len(ref_words)

    if not ref_words:
        return HERResult(0, 0, 0, 0, 0.0, 0.0)
    if not hyp_words:
        return HERResult(total_ref, 0, 0, 0, 0.0, 0.0)

    try:
        out = jiwer.process_words(reference.lower(), hypothesis.lower())
    except Exception:
        return HERResult(0, 0, 0, 0, 0.0, 0.0)

    # Repetition loops — consecutive duplicate words
    repetition = sum(
        1 for i in range(1, len(hyp_words))
        if hyp_words[i] == hyp_words[i - 1]
    )

    phonetic = 0
    hallucination = 0

    for chunk in out.alignments[0]:
        if chunk.type == "substitute":
            ref_w = ref_words[chunk.ref_start_idx] if chunk.ref_start_idx < len(ref_words) else ""
            hyp_w = hyp_words[chunk.hyp_start_idx] if chunk.hyp_start_idx < len(hyp_words) else ""
            # Heuristic: share first 2 chars → phonetically similar
            if len(ref_w) >= 2 and len(hyp_w) >= 2 and ref_w[:2] == hyp_w[:2]:
                phonetic += 1
            else:
                hallucination += 1
        elif chunk.type == "insert":
            hallucination += chunk.hyp_end_idx - chunk.hyp_start_idx

    total_errors = out.substitutions + out.deletions + out.insertions

    return HERResult(
        total_errors=total_errors,
        phonetic_errors=phonetic,
        hallucination_errors=hallucination,
        repetition_errors=repetition,
        her=hallucination / total_ref if total_ref > 0 else 0.0,
        insertion_hallucination_rate=out.insertions / total_ref if total_ref > 0 else 0.0,
    )
