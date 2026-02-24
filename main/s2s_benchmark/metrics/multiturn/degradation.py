"""
Quality degradation metric for multi-turn S2S evaluation.

Computes the slope of quality metrics (DNSMOS, UTMOS) and latency
over successive agent turns.  Negative slopes indicate degradation.

Exposed functions:
    compute_degradation(session_result) -> dict
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def compute_degradation(session_result) -> Dict[str, Optional[float]]:
    """Quality metric slopes over agent turns in the session.

    Loads each agent turn's audio via soundfile and computes DNSMOS and UTMOS
    using metrics.quality (which expects (audio_np, sr) not file paths).
    Latency slope is computed from e2e_latency_ms.

    All values are normalized to [0, 1] range before slope fitting so
    slopes are comparable across metrics.

    Args:
        session_result: SessionResult with .turns list of TurnResult objects.

    Returns:
        Dict with keys: dnsmos_slope, utmos_slope, latency_slope, degradation_slope.
        Slopes are per-turn-index change in normalized value.
        Negative = degradation over time.
        degradation_slope is the mean of the three component slopes and maps
        to DIMENSIONS['quality'] for scoring.
    """
    result: Dict[str, Optional[float]] = {
        "dnsmos_slope": None,
        "utmos_slope": None,
        "latency_slope": None,
        "degradation_slope": None,
    }

    try:
        import soundfile as sf  # type: ignore
        from metrics.quality import compute_dnsmos, compute_utmos

        # Collect per-agent-turn quality values
        dnsmos_vals: List[Optional[float]] = []
        utmos_vals: List[Optional[float]] = []
        latency_vals: List[Optional[float]] = []

        for turn in session_result.turns:
            if turn.role != "agent":
                continue

            # Audio quality metrics
            if turn.output_audio_path:
                try:
                    audio, sr = sf.read(turn.output_audio_path, dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=-1)

                    dnsmos = compute_dnsmos(audio, sr)
                    dnsmos_vals.append(dnsmos.get("dnsmos_ovrl"))

                    utmos_val = compute_utmos(audio, sr)
                    utmos_vals.append(utmos_val)
                except Exception:
                    dnsmos_vals.append(None)
                    utmos_vals.append(None)
            else:
                dnsmos_vals.append(None)
                utmos_vals.append(None)

            # Latency
            latency_vals.append(
                turn.e2e_latency_ms if hasattr(turn, "e2e_latency_ms") else None
            )

        # Compute slopes
        result["dnsmos_slope"] = _compute_slope(dnsmos_vals, normalize_range=(1.0, 5.0))
        result["utmos_slope"] = _compute_slope(utmos_vals, normalize_range=(1.0, 5.0))
        result["latency_slope"] = _compute_slope(latency_vals, invert=True)
        # For latency, invert so that increasing latency = negative slope

        # Degradation slope = mean of available component slopes
        slopes = [v for v in [result["dnsmos_slope"], result["utmos_slope"],
                              result["latency_slope"]] if v is not None]
        if slopes:
            result["degradation_slope"] = float(np.mean(slopes))

    except Exception as exc:
        print(f"  [multiturn] degradation error: {exc}")

    return result


def _compute_slope(
    values: List[Optional[float]],
    normalize_range: Optional[tuple] = None,
    invert: bool = False,
) -> Optional[float]:
    """Fit a linear slope to a list of values over turn indices.

    Args:
        values:          list of metric values (None entries skipped).
        normalize_range: (min, max) to normalize values to [0, 1].
                         If None, uses observed min/max.
        invert:          if True, negate values before fitting (for metrics
                         where higher = worse, like latency).

    Returns:
        Slope (change per turn index in normalized units), or None.
    """
    # Filter valid values with their indices
    pairs = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pairs) < 2:
        return None

    indices = np.array([p[0] for p in pairs], dtype=float)
    vals = np.array([p[1] for p in pairs], dtype=float)

    if invert:
        vals = -vals

    # Normalize to [0, 1]
    if normalize_range is not None:
        lo, hi = normalize_range
    else:
        lo, hi = float(vals.min()), float(vals.max())

    if hi - lo > 1e-8:
        vals = (vals - lo) / (hi - lo)
    else:
        return 0.0  # constant values, no degradation

    # Linear fit
    try:
        coeffs = np.polyfit(indices, vals, 1)
        return float(coeffs[0])  # slope
    except Exception:
        return None
