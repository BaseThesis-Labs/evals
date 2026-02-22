"""Tests for src/evaluation/semascore.py — SeMaScore formula."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.semascore import compute_semascore


class _FakeEncoder:
    """Deterministic stub encoder — no model downloads required."""

    def __init__(self, embeddings: dict[str, list[float]]):
        self._embs = embeddings

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            if t in self._embs:
                vecs.append(self._embs[t])
            else:
                # Unit vector of 1s by default (cos_sim = 1.0 when both unknown)
                vecs.append([1.0, 0.0])
        return np.array(vecs, dtype=float)


@pytest.fixture
def identical_encoder():
    """Encoder that returns the same embedding for any input — cosine sim = 1."""
    class _Identical:
        def encode(self, texts):
            return np.array([[1.0, 0.0]] * len(texts))
    return _Identical()


@pytest.fixture
def orthogonal_encoder():
    """Encoder that returns orthogonal embeddings — cosine sim = 0."""
    class _Orthogonal:
        def encode(self, texts):
            return np.array([[1.0, 0.0], [0.0, 1.0]])
    return _Orthogonal()


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_ref_empty_hyp(identical_encoder):
    score = compute_semascore("", "", identical_encoder)
    assert score == 1.0


def test_empty_ref_nonempty_hyp(identical_encoder):
    score = compute_semascore("", "something", identical_encoder)
    assert score == 0.0


def test_empty_hyp(identical_encoder):
    score = compute_semascore("hello world", "", identical_encoder)
    assert score == 0.0


# ── Perfect match (WER=0, cos_sim=1) ─────────────────────────────────────────

def test_perfect_match_score_one(identical_encoder):
    # MER = 0 for identical strings → semascore = cos_sim × (1 − 0) = 1.0
    score = compute_semascore("hello world", "hello world", identical_encoder)
    assert score == pytest.approx(1.0, abs=1e-6)


# ── Fully wrong (MER=1) ───────────────────────────────────────────────────────

def test_fully_wrong_mer(identical_encoder):
    # "a b" vs "x y" → MER = 1, semascore = cos_sim × 0 = 0
    score = compute_semascore("a b", "x y", identical_encoder)
    assert score == pytest.approx(0.0, abs=1e-6)


# ── Cosine sim = 0 → score = 0 regardless of MER ─────────────────────────────

def test_orthogonal_embeddings_score_zero(orthogonal_encoder):
    score = compute_semascore("hello", "world", orthogonal_encoder)
    assert score == pytest.approx(0.0, abs=1e-6)


# ── Range: score always in [0, 1] ─────────────────────────────────────────────

@pytest.mark.parametrize("ref,hyp", [
    ("hello world", "hello"),
    ("the quick brown fox", "a slow white dog"),
    ("test", "test sentence here"),
])
def test_score_range(ref, hyp, identical_encoder):
    score = compute_semascore(ref, hyp, identical_encoder)
    assert 0.0 <= score <= 1.0


# ── Formula: score = cos_sim × (1 − MER) ─────────────────────────────────────

def test_formula_with_known_mer():
    """Verify the formula directly: score = cos_sim × (1 − MER)."""
    # ref: "a b c", hyp: "a b x"  →  1 sub / 3 ref = MER ≈ 0.333
    # encoder returns identical vectors → cos_sim = 1.0
    # expected: 1.0 × (1 − 0.333) ≈ 0.667
    class _Identical:
        def encode(self, texts):
            return np.array([[1.0, 0.0]] * len(texts))

    import jiwer
    out = jiwer.process_words("a b c", "a b x")
    expected = float(1.0 * (1.0 - out.mer))
    score = compute_semascore("a b c", "a b x", _Identical())
    assert score == pytest.approx(expected, abs=1e-6)


# ── Encoder failure → graceful fallback ──────────────────────────────────────

def test_encoder_failure_returns_zero():
    class _BrokenEncoder:
        def encode(self, texts):
            raise RuntimeError("model unavailable")

    # Should not raise; cos_sim fallback = 0.0
    score = compute_semascore("hello world", "hello world", _BrokenEncoder())
    assert score == 0.0
