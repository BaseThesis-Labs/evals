"""Tests for src/models/_retry.py — with_retry exponential backoff."""
from __future__ import annotations

import time

import pytest

from src.models._retry import with_retry, _is_retryable


# ── _is_retryable ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("msg,expected", [
    ("429 Too Many Requests",  True),
    ("rate limit exceeded",    True),
    ("503 Service Unavailable", True),
    ("connection reset by peer", True),
    ("timeout reading data",   True),
    ("404 Not Found",          False),
    ("invalid API key",        False),
    ("ValueError: bad input",  False),
])
def test_is_retryable(msg, expected):
    assert _is_retryable(Exception(msg)) == expected


# ── with_retry: success on first attempt ─────────────────────────────────────

def test_success_on_first_attempt():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result = with_retry(fn, max_attempts=3, base_delay=0.0)
    assert result == "ok"
    assert len(calls) == 1


# ── with_retry: success after transient failures ──────────────────────────────

def test_retries_on_transient_error():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 rate limit")
        return "ok"

    result = with_retry(fn, max_attempts=3, base_delay=0.0)
    assert result == "ok"
    assert len(calls) == 3


# ── with_retry: non-retryable error is NOT retried ───────────────────────────

def test_no_retry_on_non_retryable_error():
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("invalid API key — not a transient error")

    with pytest.raises(ValueError):
        with_retry(fn, max_attempts=3, base_delay=0.0)
    assert len(calls) == 1


# ── with_retry: exhausts all attempts and re-raises last exception ────────────

def test_raises_after_all_attempts_exhausted():
    calls = []

    def fn():
        calls.append(1)
        raise RuntimeError("503 server error")

    with pytest.raises(RuntimeError, match="503"):
        with_retry(fn, max_attempts=3, base_delay=0.0)
    assert len(calls) == 3


# ── with_retry: single attempt ───────────────────────────────────────────────

def test_max_attempts_one_no_retry():
    calls = []

    def fn():
        calls.append(1)
        raise RuntimeError("429 rate limit")

    with pytest.raises(RuntimeError):
        with_retry(fn, max_attempts=1, base_delay=0.0)
    assert len(calls) == 1


# ── with_retry: model_name appears in log (smoke test via caplog) ─────────────

def test_model_name_in_warning(caplog):
    import logging
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("503 server error")
        return "done"

    with caplog.at_level(logging.WARNING, logger="src.models._retry"):
        result = with_retry(fn, max_attempts=3, base_delay=0.0, model_name="test-model")

    assert result == "done"
    assert any("test-model" in r.message for r in caplog.records)
