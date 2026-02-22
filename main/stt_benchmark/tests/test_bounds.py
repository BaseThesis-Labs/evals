"""Tests for aggregate.py metric-bound normalization (_normalize_metric)."""
from __future__ import annotations

import math

import pytest

# Import the private helper directly — it contains no side effects.
from aggregate import _normalize_metric


# ── Helper to build a bound dict ──────────────────────────────────────────────

def bound(lo, hi, lower_is_better=True):
    return {"min": lo, "max": hi, "lower_is_better": lower_is_better}


# ── lower_is_better metrics (WER, CER, …) ────────────────────────────────────

def test_lower_is_better_best():
    # value at minimum bound → score = 1.0 (best)
    assert _normalize_metric(0.0, bound(0.0, 1.0)) == pytest.approx(1.0)


def test_lower_is_better_worst():
    # value at maximum bound → score = 0.0 (worst)
    assert _normalize_metric(1.0, bound(0.0, 1.0)) == pytest.approx(0.0)


def test_lower_is_better_midpoint():
    assert _normalize_metric(0.5, bound(0.0, 1.0)) == pytest.approx(0.5)


def test_lower_is_better_clamps_below_min():
    # value < lo → clipped to lo → score = 1.0
    assert _normalize_metric(-0.1, bound(0.0, 1.0)) == pytest.approx(1.0)


def test_lower_is_better_clamps_above_max():
    # value > hi → clipped to hi → score = 0.0
    assert _normalize_metric(1.5, bound(0.0, 1.0)) == pytest.approx(0.0)


# ── higher_is_better metrics (SeMaScore, entity_f1, …) ───────────────────────

def test_higher_is_better_best():
    assert _normalize_metric(1.0, bound(0.0, 1.0, lower_is_better=False)) == pytest.approx(1.0)


def test_higher_is_better_worst():
    assert _normalize_metric(0.0, bound(0.0, 1.0, lower_is_better=False)) == pytest.approx(0.0)


def test_higher_is_better_midpoint():
    assert _normalize_metric(0.5, bound(0.0, 1.0, lower_is_better=False)) == pytest.approx(0.5)


# ── RTFx: log-scale normalization ─────────────────────────────────────────────

def test_rtfx_very_fast_model():
    # RTFx = 20 (at log-scale max) → score ≈ 1.0
    score = _normalize_metric(20.0, bound(0.1, 20.0), metric="rtfx")
    assert score == pytest.approx(1.0, abs=0.01)


def test_rtfx_very_slow_model():
    # RTFx ≤ 0.1 (at log-scale min) → score ≈ 0.0
    score = _normalize_metric(0.1, bound(0.1, 20.0), metric="rtfx")
    assert score == pytest.approx(0.0, abs=0.01)


def test_rtfx_intermediate():
    # RTFx > 0.1 and < 20 → score in (0, 1)
    score = _normalize_metric(5.0, bound(0.1, 20.0), metric="rtfx")
    assert 0.0 < score < 1.0


def test_rtfx_clamped_above_max():
    # RTFx way above max → clamped to 1.0
    score = _normalize_metric(1000.0, bound(0.1, 20.0), metric="rtfx")
    assert score == pytest.approx(1.0, abs=0.01)


def test_rtfx_clamped_below_min():
    # RTFx = 0 (can't take log of 0) → clamped to _RTFX_MIN = 0.1 → score ≈ 0
    score = _normalize_metric(0.0, bound(0.1, 20.0), metric="rtfx")
    assert score == pytest.approx(0.0, abs=0.01)


# ── NaN / Inf inputs ──────────────────────────────────────────────────────────

def test_nan_input():
    result = _normalize_metric(float("nan"), bound(0.0, 1.0))
    assert math.isnan(result)


def test_inf_input():
    result = _normalize_metric(float("inf"), bound(0.0, 1.0))
    assert math.isnan(result)


# ── Degenerate bound (min == max) ─────────────────────────────────────────────

def test_degenerate_bound_lower_is_better():
    # hi == lo → norm = 0.0 → result = 1.0 - 0.0 = 1.0 (lower_is_better=True)
    result = _normalize_metric(0.5, bound(0.5, 0.5, lower_is_better=True))
    assert result == 1.0


def test_degenerate_bound_higher_is_better():
    # hi == lo → norm = 0.0 → result = 0.0 (higher_is_better)
    result = _normalize_metric(0.5, bound(0.5, 0.5, lower_is_better=False))
    assert result == 0.0


# ── Typical real-world bounds sanity checks ────────────────────────────────────

@pytest.mark.parametrize("wer,expected_ge,expected_le", [
    (0.0,  0.9, 1.0),   # near-perfect → high score
    (0.05, 0.8, 1.0),   # 5% WER → still high
    (0.5,  0.0, 0.5),   # 50% WER → moderate to low
    (1.0,  0.0, 0.1),   # 100% WER → near zero
])
def test_wer_bound_realistic(wer, expected_ge, expected_le):
    b = bound(0.0, 1.0, lower_is_better=True)
    score = _normalize_metric(wer, b)
    assert expected_ge <= score <= expected_le
