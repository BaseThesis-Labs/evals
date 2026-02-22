"""Tests for src/evaluation/metrics.py — WER/CER/SER computation."""
from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import compute_sample_metrics, aggregate_metrics


# ── Perfect match ──────────────────────────────────────────────────────────────

def test_perfect_match():
    m = compute_sample_metrics("hello world", "hello world")
    assert m["wer"] == 0.0
    assert m["cer"] == 0.0
    assert m["ser"] == 0.0
    assert m["hits"] == 2
    assert m["substitutions"] == 0
    assert m["deletions"] == 0
    assert m["insertions"] == 0
    assert m["wrr"] == 1.0
    assert m["sub_rate"] == 0.0
    assert m["del_rate"] == 0.0
    assert m["ins_rate"] == 0.0
    assert m["error_words"] == 0


# ── One substitution ──────────────────────────────────────────────────────────

def test_one_substitution():
    m = compute_sample_metrics("hello world", "hello earth")
    assert m["wer"] == pytest.approx(1 / 2)
    assert m["substitutions"] == 1
    assert m["hits"] == 1
    assert m["ser"] == 1.0  # sentence has an error


# ── One deletion ──────────────────────────────────────────────────────────────

def test_one_deletion():
    m = compute_sample_metrics("hello world", "hello")
    assert m["wer"] == pytest.approx(1 / 2)
    assert m["deletions"] == 1


# ── One insertion ─────────────────────────────────────────────────────────────

def test_one_insertion():
    m = compute_sample_metrics("hello", "hello world")
    assert m["wer"] == pytest.approx(1 / 1)
    assert m["insertions"] == 1


# ── SER: binary sentence error flag ──────────────────────────────────────────

def test_ser_no_error():
    m = compute_sample_metrics("test", "test")
    assert m["ser"] == 0.0


def test_ser_with_error():
    m = compute_sample_metrics("test sentence", "test")
    assert m["ser"] == 1.0


# ── Error rate fields ─────────────────────────────────────────────────────────

def test_sub_rate():
    # ref: 4 words, 2 substitutions
    m = compute_sample_metrics("a b c d", "x x c d")
    assert m["sub_rate"] == pytest.approx(2 / 4)


def test_del_rate():
    m = compute_sample_metrics("a b c d", "a b")
    assert m["del_rate"] == pytest.approx(2 / 4)


def test_ins_rate():
    m = compute_sample_metrics("a b", "a b c d")
    assert m["ins_rate"] == pytest.approx(2 / 2)


# ── Empty inputs ──────────────────────────────────────────────────────────────

def test_both_empty():
    m = compute_sample_metrics("", "")
    assert m["wer"] == 0.0
    assert m["ser"] == 0.0


def test_empty_ref():
    m = compute_sample_metrics("", "something")
    assert math.isinf(m["wer"])
    assert m["ser"] == 1.0


def test_empty_hyp():
    m = compute_sample_metrics("hello world", "")
    assert m["wer"] > 0.0


# ── With normalizer ───────────────────────────────────────────────────────────

def test_with_normalizer():
    from src.evaluation.normalizer import TranscriptNormalizer
    norm = TranscriptNormalizer()
    # "Hello, World!" vs "hello world" → after normalization they should match
    m = compute_sample_metrics("Hello, World!", "hello world", normalizer=norm)
    assert m["wer"] == 0.0


# ── aggregate_metrics ─────────────────────────────────────────────────────────

def test_aggregate_micro_wer():
    # sample 1: 2 ref words, 1 error; sample 2: 2 ref words, 0 errors
    # micro_wer = 1 error / 4 ref words = 0.25
    samples = [
        compute_sample_metrics("hello world", "hello earth"),  # 1 sub / 2 ref
        compute_sample_metrics("foo bar", "foo bar"),          # 0 errors / 2 ref
    ]
    agg = aggregate_metrics(samples)
    assert agg["micro_wer"] == pytest.approx(1 / 4)
    assert agg["n_samples"] == 2


def test_aggregate_empty():
    assert aggregate_metrics([]) == {}


def test_aggregate_ignores_infinite_wer():
    samples = [
        compute_sample_metrics("hello", "hello"),
        compute_sample_metrics("", "something"),  # inf WER
    ]
    agg = aggregate_metrics(samples)
    # Only the first sample has finite WER
    assert agg["n_samples"] == 1


def test_aggregate_total_ref_words():
    samples = [
        compute_sample_metrics("a b c", "a b c"),    # 3 ref words
        compute_sample_metrics("d e", "d e"),          # 2 ref words
    ]
    agg = aggregate_metrics(samples)
    assert agg["total_ref_words"] == 5
