"""Tests for src/evaluation/normalizer.py — TranscriptNormalizer."""
from __future__ import annotations

import pytest

from src.evaluation.normalizer import TranscriptNormalizer, get_default_normalizer


@pytest.fixture
def normalizer():
    return TranscriptNormalizer()


# ── Basic normalization ────────────────────────────────────────────────────────

def test_lowercase(normalizer):
    assert normalizer.normalize("Hello World") == "hello world"


def test_strips_leading_trailing_whitespace(normalizer):
    assert normalizer.normalize("  hello  ") == "hello"


def test_collapses_multiple_spaces(normalizer):
    assert normalizer.normalize("hello   world") == "hello world"


def test_removes_punctuation(normalizer):
    result = normalizer.normalize("Hello, world!")
    assert "," not in result
    assert "!" not in result


def test_removes_fillers(normalizer):
    result = normalizer.normalize("I um need uh help")
    assert "um" not in result
    assert "uh" not in result
    assert "need" in result
    assert "help" in result


def test_expands_contractions(normalizer):
    result = normalizer.normalize("I don't know")
    # "don't" → "do not"
    assert "not" in result
    assert "do" in result


# ── normalize_pair ─────────────────────────────────────────────────────────────

def test_normalize_pair_both(normalizer):
    ref, hyp = normalizer.normalize_pair("Hello, World!", "hello world")
    assert ref == hyp


# ── normalize_batch ────────────────────────────────────────────────────────────

def test_normalize_batch(normalizer):
    results = normalizer.normalize_batch(["Hello!", "World?"])
    assert results == ["hello", "world"]


# ── Empty / edge cases ─────────────────────────────────────────────────────────

def test_empty_string(normalizer):
    assert normalizer.normalize("") == ""


def test_only_punctuation(normalizer):
    assert normalizer.normalize("!!! ???") == ""


def test_only_fillers(normalizer):
    result = normalizer.normalize("uh um hmm")
    assert result == ""


# ── Custom config ──────────────────────────────────────────────────────────────

def test_no_lowercase():
    norm = TranscriptNormalizer({"lowercase": False, "remove_fillers": False,
                                 "remove_punctuation": False, "expand_contractions": False})
    assert norm.normalize("Hello") == "Hello"


def test_no_filler_removal():
    norm = TranscriptNormalizer({"remove_fillers": False, "lowercase": True,
                                 "remove_punctuation": True, "expand_contractions": False})
    result = norm.normalize("I uh need help")
    assert "uh" in result


def test_custom_filler_words():
    norm = TranscriptNormalizer({
        "remove_fillers": True,
        "filler_words": ["basically", "literally"],
        "lowercase": True,
        "remove_punctuation": True,
        "expand_contractions": False,
    })
    result = norm.normalize("I basically literally agree")
    assert "basically" not in result
    assert "literally" not in result
    assert "agree" in result


# ── Singleton default normalizer ──────────────────────────────────────────────

def test_get_default_normalizer_singleton():
    n1 = get_default_normalizer()
    n2 = get_default_normalizer()
    assert n1 is n2
