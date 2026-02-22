"""src/evaluation/normalizer.py — CRITICAL: identical normalisation applied to ref AND hyp."""
from __future__ import annotations

import logging

import jiwer

log = logging.getLogger(__name__)

DEFAULT_FILLERS = ["uh", "um", "hmm", "mhm", "uh-huh", "mm-hmm", "ah", "er"]


class TranscriptNormalizer:
    """
    Apply IDENTICAL text normalisation to both reference and hypothesis.
    Must be used on BOTH sides before any metric computation.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        transforms = [jiwer.Strip(), jiwer.RemoveMultipleSpaces()]

        if cfg.get("lowercase", True):
            transforms.append(jiwer.ToLowerCase())
        if cfg.get("expand_contractions", True):
            transforms.append(jiwer.ExpandCommonEnglishContractions())
        if cfg.get("remove_fillers", True):
            fillers = cfg.get("filler_words", DEFAULT_FILLERS)
            transforms.append(jiwer.RemoveSpecificWords(fillers))
        if cfg.get("remove_punctuation", True):
            transforms.append(jiwer.RemovePunctuation())

        transforms.extend([jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
        self._transform = jiwer.Compose(transforms)

    def normalize(self, text: str) -> str:
        result = self._transform(text)
        # jiwer Compose may return list-of-lists when ReduceToListOfListOfWords is in chain
        if isinstance(result, list):
            if result and isinstance(result[0], list):
                return " ".join(w for sent in result for w in sent)
            return " ".join(str(t) for t in result)
        return str(result).strip()

    def normalize_pair(self, ref: str, hyp: str) -> tuple[str, str]:
        return self.normalize(ref), self.normalize(hyp)

    def normalize_batch(self, texts: list[str]) -> list[str]:
        return [self.normalize(t) for t in texts]


_default: TranscriptNormalizer | None = None


def get_default_normalizer() -> TranscriptNormalizer:
    global _default
    if _default is None:
        _default = TranscriptNormalizer()
    return _default
