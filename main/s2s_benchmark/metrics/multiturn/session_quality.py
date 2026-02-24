"""
Session quality aggregator for multi-turn S2S evaluation.

Computes all session-level metrics and returns a flat dict of values.

Exposed functions:
    compute_session_quality(session_result, scenario=None) -> dict
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def compute_session_quality(session_result, scenario=None) -> Dict[str, Optional[float]]:
    """Aggregate all session-level multi-turn metrics into one dict.

    Calls each multi-turn metric function and also computes:
      - avg_turn_latency:    mean e2e_latency_ms across agent turns
      - session_dnsmos_ovrl: mean DNSMOS overall across agent turns
      - session_utmos:       mean UTMOS across agent turns

    Args:
        session_result: SessionResult with .turns, .probe_results, etc.
        scenario:       Optional scenario object (needed for task_completion).

    Returns:
        Flat dict with all multi-turn metric values.
    """
    from metrics.multiturn.context_retention import compute_context_retention
    from metrics.multiturn.consistency import compute_voice_consistency
    from metrics.multiturn.task_completion import compute_task_completion
    from metrics.multiturn.degradation import compute_degradation
    from metrics.multiturn.error_recovery import compute_error_recovery
    from metrics.multiturn.dialogue_coherence import compute_dialogue_coherence

    result: Dict[str, Optional[float]] = {}

    # ── Core multi-turn metrics ───────────────────────────────────────────────
    result["context_retention"] = compute_context_retention(session_result)
    result["voice_consistency"] = compute_voice_consistency(session_result)
    result["error_recovery"] = compute_error_recovery(session_result)
    result["dialogue_coherence"] = compute_dialogue_coherence(session_result)

    # Task completion requires the scenario
    if scenario is not None:
        result["task_completion"] = compute_task_completion(session_result, scenario)
    else:
        result["task_completion"] = None

    # Structured session verdict (LLM judge pass/fail with reasoning)
    if scenario is not None:
        from metrics.multiturn.session_verdict import compute_session_verdict
        verdict = compute_session_verdict(session_result, scenario)
        result.update(verdict)

    # Degradation returns a dict — merge into result
    degradation = compute_degradation(session_result)
    result.update(degradation)

    # ── Per-turn aggregates ───────────────────────────────────────────────────
    latencies, dnsmos_vals, utmos_vals = _compute_turn_aggregates(session_result)

    result["avg_turn_latency"] = (
        float(np.mean(latencies)) if latencies else None
    )
    result["session_dnsmos_ovrl"] = (
        float(np.mean(dnsmos_vals)) if dnsmos_vals else None
    )
    result["session_utmos"] = (
        float(np.mean(utmos_vals)) if utmos_vals else None
    )

    return result


def _compute_turn_aggregates(session_result):
    """Compute per-turn quality values for agent turns.

    Returns:
        (latencies, dnsmos_vals, utmos_vals) — lists of valid float values.
    """
    latencies: List[float] = []
    dnsmos_vals: List[float] = []
    utmos_vals: List[float] = []

    for turn in session_result.turns:
        if turn.role != "agent":
            continue

        # Latency
        if hasattr(turn, "e2e_latency_ms") and turn.e2e_latency_ms is not None:
            latencies.append(float(turn.e2e_latency_ms))

        # Audio quality
        if turn.output_audio_path:
            try:
                import soundfile as sf  # type: ignore
                from metrics.quality import compute_dnsmos, compute_utmos

                audio, sr = sf.read(turn.output_audio_path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=-1)

                dnsmos = compute_dnsmos(audio, sr)
                ovrl = dnsmos.get("dnsmos_ovrl")
                if ovrl is not None:
                    dnsmos_vals.append(ovrl)

                utmos_val = compute_utmos(audio, sr)
                if utmos_val is not None:
                    utmos_vals.append(utmos_val)

            except Exception as exc:
                print(f"  [multiturn] turn aggregate audio error: {exc}")

    return latencies, dnsmos_vals, utmos_vals
