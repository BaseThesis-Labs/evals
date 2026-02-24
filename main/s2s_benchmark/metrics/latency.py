"""
Latency metrics for S2S evaluation.

Exposed functions:
    extract_latency_from_result(gen_meta_record) → dict
    compute_latency_percentiles(latency_list) → dict  (P50/P90/P99)
    summarize_latency(records) → dict
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional


def extract_latency_from_result(record: Dict) -> Dict[str, Optional[float]]:
    """Extract latency fields from a gen_meta.jsonl record.

    Args:
        record: dict parsed from one line of gen_meta.jsonl

    Returns:
        dict with: ttfb_ms, e2e_latency_ms, asr_latency_ms, tts_latency_ms, rtf
    """
    def _f(key: str) -> Optional[float]:
        v = record.get(key)
        if v is None:
            return None
        try:
            f = float(v)
            return f if not math.isnan(f) else None
        except (TypeError, ValueError):
            return None

    return {
        "ttfb_ms": _f("ttfb_ms"),
        "e2e_latency_ms": _f("e2e_latency_ms"),
        "asr_latency_ms": _f("asr_latency_ms"),
        "tts_latency_ms": _f("tts_latency_ms"),
        "rtf": _f("rtf"),
    }


def compute_latency_percentiles(
    latency_values: List[Optional[float]],
) -> Dict[str, Optional[float]]:
    """Compute P50, P90, P99 latency percentiles.

    Args:
        latency_values: list of latency measurements (ms); None entries skipped.

    Returns:
        {"p50": float, "p90": float, "p99": float, "mean": float, "n": int}
    """
    import statistics

    valid = sorted(v for v in latency_values if v is not None and not math.isnan(v))
    n = len(valid)
    if n == 0:
        return {"p50": None, "p90": None, "p99": None, "mean": None, "n": 0}

    def _pct(pct: float) -> float:
        idx = (pct / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return valid[lo] + frac * (valid[hi] - valid[lo])

    p50_val = _pct(50)
    p99_val = _pct(99)
    tail_ratio = (p99_val / p50_val) if (p50_val and p50_val > 0) else None

    return {
        "p50": p50_val,
        "p90": _pct(90),
        "p99": p99_val,
        "mean": statistics.mean(valid),
        "n": n,
        "tail_ratio": tail_ratio,
    }


def summarize_latency(records: List[Dict]) -> Dict[str, Dict]:
    """Compute latency percentiles for all latency dimensions across records.

    Args:
        records: list of dicts each containing latency fields.

    Returns:
        {field_name: {"p50": ..., "p90": ..., "p99": ..., "mean": ..., "n": ...}}
    """
    fields = ["ttfb_ms", "e2e_latency_ms", "asr_latency_ms", "tts_latency_ms", "rtf"]
    result = {}
    for field in fields:
        values = [r.get(field) for r in records]
        result[field] = compute_latency_percentiles(values)
    return result
