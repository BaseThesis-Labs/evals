"""
Interaction metrics for S2S evaluation.

Metrics:
  tor_up   — TOR↑: fraction of user-speech time where system responds before user finishes
             (system "takes over" / interrupts). Lower is better for turn-taking.
  tor_down — TOR↓: normalised silence gap after user finishes before system responds.
             Higher = system slow to respond. Lower is better.
  response_latency_ms — time from end-of-user-turn to start of system response (ms).

For cascaded adapters (non-streaming), TOR is approximated from:
  - ref_audio_duration_ms: duration of input audio (reference turn)
  - ttfb_ms:               time to first byte from generation record

TOR↑ = min(1, max(0, ref_duration_ms - ttfb_ms) / ref_duration_ms)
        → fraction of user speech time covered by early system response
TOR↓ = min(1, max(0, ttfb_ms - ref_duration_ms) / ref_duration_ms)
        → normalised silence gap after user finishes

Full-duplex TOR (Phase 3) uses real segment timing from streaming harness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _get_audio_duration_ms(path: Optional[str]) -> Optional[float]:
    """Return duration of audio file in milliseconds."""
    if not path or not Path(path).exists():
        return None
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(path)
        return (info.frames / info.samplerate) * 1000.0
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TOR↑ / TOR↓
# ─────────────────────────────────────────────────────────────────────────────

def compute_tor(
    ref_audio_path: Optional[str],
    ttfb_ms: Optional[float],
) -> Dict[str, Optional[float]]:
    """Compute TOR↑ and TOR↓ from reference audio duration and TTFB.

    Args:
        ref_audio_path: path to the input/reference audio (user's speech).
        ttfb_ms:        time-to-first-byte of system response (ms from audio start).

    Returns:
        {"tor_up": float|None, "tor_down": float|None}

    Interpretation:
        tor_up   ∈ [0, 1]: 0 = no early overlap, 1 = system responded at time 0
        tor_down ∈ [0, 1]: 0 = no silence gap, 1 = system took one full user-turn to respond
    """
    ref_dur_ms = _get_audio_duration_ms(ref_audio_path)

    if ref_dur_ms is None or ttfb_ms is None or ref_dur_ms <= 0:
        return {"tor_up": None, "tor_down": None}

    # TOR↑: system started before user finished
    overlap_ms = ref_dur_ms - ttfb_ms          # positive → system was early
    tor_up = max(0.0, min(1.0, overlap_ms / ref_dur_ms))

    # TOR↓: system started after user finished (silence gap)
    silence_ms = ttfb_ms - ref_dur_ms          # positive → gap after user finished
    tor_down = max(0.0, min(1.0, silence_ms / ref_dur_ms))

    return {"tor_up": tor_up, "tor_down": tor_down}


# ─────────────────────────────────────────────────────────────────────────────
# Segment-based TOR (for streaming / full-duplex, Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_turn_overlap_ratio(
    ref_segments: List[Tuple[float, float]],
    hyp_segments: List[Tuple[float, float]],
) -> Optional[float]:
    """Turn Overlap Ratio from explicit segment timing (streaming harness).

    Args:
        ref_segments: list of (start_ms, end_ms) for reference/user speech.
        hyp_segments: list of (start_ms, end_ms) for S2S system output.

    Returns:
        TOR ∈ [0, 1]: fraction of total span where both speakers are active.
        None if inputs are empty.
    """
    if not ref_segments or not hyp_segments:
        return None

    all_ends = [e for _, e in ref_segments] + [e for _, e in hyp_segments]
    total_span = max(all_ends)
    if total_span <= 0:
        return None

    # Build per-ms active sets (simplified: use frame resolution of 10 ms)
    frame_ms = 10.0
    n_frames = int(total_span / frame_ms) + 1

    ref_active = [False] * n_frames
    hyp_active = [False] * n_frames

    for start, end in ref_segments:
        for i in range(int(start / frame_ms), min(n_frames, int(end / frame_ms) + 1)):
            ref_active[i] = True

    for start, end in hyp_segments:
        for i in range(int(start / frame_ms), min(n_frames, int(end / frame_ms) + 1)):
            hyp_active[i] = True

    overlap_frames = sum(1 for i in range(n_frames) if ref_active[i] and hyp_active[i])
    return overlap_frames / n_frames


def compute_response_latency(
    turn_end_ms: Optional[float],
    response_start_ms: Optional[float],
) -> Optional[float]:
    """Time between end of user turn and start of system response (ms).

    Negative = system started responding before user finished (barge-in).
    """
    if turn_end_ms is None or response_start_ms is None:
        return None
    return float(response_start_ms - turn_end_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_interaction(
    record: Dict,
    ref_audio_path: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """Compute all interaction metrics from a generation record.

    Args:
        record:         gen_meta.jsonl entry (has ttfb_ms, turn_end_ms, etc.)
        ref_audio_path: path to input/reference audio (for TOR computation)
    """
    ttfb_ms = record.get("ttfb_ms")
    tor = compute_tor(ref_audio_path, ttfb_ms)

    # response_latency_ms = time the user waits after finishing speaking
    # For cascaded adapters: e2e_latency_ms - ref_audio_duration_ms gives the
    # actual wait time after the user finishes (total pipeline time minus input
    # duration). Falls back to ttfb_ms - ref_dur_ms when e2e is unavailable.
    # Only computed when ref_audio_path is available; otherwise set to None
    # (raw ttfb_ms alone is meaningless as a response latency).
    response_latency_ms: Optional[float] = None
    ref_dur_ms = _get_audio_duration_ms(ref_audio_path)
    if ref_dur_ms is not None:
        e2e_ms = record.get("e2e_latency_ms")
        if e2e_ms is not None:
            response_latency_ms = float(e2e_ms - ref_dur_ms)
        elif ttfb_ms is not None:
            response_latency_ms = float(ttfb_ms - ref_dur_ms)

    return {
        "tor_up": tor["tor_up"],
        "tor_down": tor["tor_down"],
        "response_latency_ms": response_latency_ms,
        "barge_in_count": None,   # Phase 3: requires streaming VAD timing
    }
