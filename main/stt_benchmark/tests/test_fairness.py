"""Tests for src/evaluation/fairness.py — compute_disaggregated_wer."""
from __future__ import annotations

import math

import pytest

from src.evaluation.fairness import compute_disaggregated_wer


def _sample(speaker_id="spk1", wer=0.1, hits=9, subs=1, dels=0, ins=0):
    return {
        "speaker_id": speaker_id,
        "wer": wer,
        "hits": hits,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
    }


# ── Basic structure ───────────────────────────────────────────────────────────

def test_returns_dict_keyed_by_field():
    samples = [_sample("s1"), _sample("s2")]
    result = compute_disaggregated_wer(samples, group_by=["speaker_id"])
    assert "speaker_id" in result
    assert "groups" in result["speaker_id"]
    assert "wer_gap" in result["speaker_id"]


def test_groups_contain_stats():
    samples = [_sample("spkA"), _sample("spkA"), _sample("spkB")]
    result = compute_disaggregated_wer(samples, group_by=["speaker_id"])
    groups = result["speaker_id"]["groups"]
    assert "spkA" in groups
    assert "spkB" in groups
    for g in groups.values():
        assert "micro_wer" in g
        assert "mean_wer" in g
        assert "std_wer" in g
        assert "n_samples" in g


# ── Micro-WER calculation ─────────────────────────────────────────────────────

def test_micro_wer_correct():
    # spkA: 1 sub / 10 ref words → micro_wer = 0.1
    # spkB: 2 sub / 10 ref words → micro_wer = 0.2
    samples = [
        _sample("spkA", wer=0.1, hits=9, subs=1, dels=0, ins=0),
        _sample("spkB", wer=0.2, hits=8, subs=2, dels=0, ins=0),
    ]
    result = compute_disaggregated_wer(samples, group_by=["speaker_id"])
    groups = result["speaker_id"]["groups"]
    assert groups["spkA"]["micro_wer"] == pytest.approx(1 / 10)
    assert groups["spkB"]["micro_wer"] == pytest.approx(2 / 10)


# ── WER gap ───────────────────────────────────────────────────────────────────

def test_wer_gap_is_max_minus_min():
    samples = [
        _sample("s1", wer=0.1, hits=9, subs=1),
        _sample("s2", wer=0.3, hits=7, subs=3),
    ]
    result = compute_disaggregated_wer(samples, group_by=["speaker_id"])
    gap = result["speaker_id"]["wer_gap"]
    assert gap == pytest.approx(0.3 - 0.1, abs=1e-6)


def test_wer_gap_single_group_is_nan():
    samples = [_sample("s1"), _sample("s1")]
    result = compute_disaggregated_wer(samples, group_by=["speaker_id"])
    assert math.isnan(result["speaker_id"]["wer_gap"])


# ── Multiple group_by fields ──────────────────────────────────────────────────

def test_multiple_group_by_fields():
    samples = [
        {"speaker_id": "s1", "tts_model": "m1", "wer": 0.1,
         "hits": 9, "substitutions": 1, "deletions": 0, "insertions": 0},
        {"speaker_id": "s2", "tts_model": "m2", "wer": 0.2,
         "hits": 8, "substitutions": 2, "deletions": 0, "insertions": 0},
    ]
    result = compute_disaggregated_wer(samples, group_by=["speaker_id", "tts_model"])
    assert "speaker_id" in result
    assert "tts_model" in result


# ── Missing / empty group_by field ───────────────────────────────────────────

def test_missing_field_skipped():
    samples = [{"wer": 0.1, "hits": 9, "substitutions": 1, "deletions": 0, "insertions": 0}]
    result = compute_disaggregated_wer(samples, group_by=["speaker_id"])
    # speaker_id is absent → group is empty → field is skipped
    assert "speaker_id" not in result


# ── Samples with infinite WER are excluded ───────────────────────────────────

def test_infinite_wer_excluded():
    samples = [
        _sample("s1", wer=float("inf"), hits=0, subs=0, dels=0, ins=0),
        _sample("s1", wer=0.2, hits=8, subs=2),
    ]
    result = compute_disaggregated_wer(samples, group_by=["speaker_id"])
    g = result["speaker_id"]["groups"]["s1"]
    assert g["n_samples"] == 1


# ── Default group_by includes speaker_id ─────────────────────────────────────

def test_default_group_by():
    samples = [_sample("a"), _sample("b")]
    result = compute_disaggregated_wer(samples)
    assert "speaker_id" in result


# ── Empty sample list ─────────────────────────────────────────────────────────

def test_empty_samples():
    result = compute_disaggregated_wer([], group_by=["speaker_id"])
    assert result == {}
