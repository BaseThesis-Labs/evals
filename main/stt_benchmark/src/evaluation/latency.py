"""src/evaluation/latency.py — RTF, RTFx, and latency percentile statistics."""
from __future__ import annotations

import numpy as np


def compute_latency_stats(
    inference_times: list[float],
    audio_durations: list[float],
    percentiles: list[int] | None = None,
) -> dict:
    """Aggregate latency statistics from per-sample timing arrays."""
    if percentiles is None:
        percentiles = [50, 95, 99]

    times = np.array(inference_times, dtype=float)
    durs  = np.array(audio_durations,  dtype=float)
    valid = (times > 0) & (durs > 0)

    rtf  = np.where(valid, times / durs, np.inf)
    rtfx = np.where(valid, durs / times, np.inf)

    fin_rtf  = rtf[np.isfinite(rtf)]
    fin_rtfx = rtfx[np.isfinite(rtfx)]
    pos_times = times[times > 0]

    stats: dict = {
        "mean_rtf":            float(np.mean(fin_rtf))   if len(fin_rtf)  > 0 else float("nan"),
        "mean_rtfx":           float(np.mean(fin_rtfx))  if len(fin_rtfx) > 0 else float("nan"),
        "mean_inference_s":    float(np.mean(pos_times)) if len(pos_times) > 0 else float("nan"),
        "total_audio_min":     float(durs.sum() / 60),
        "total_inference_min": float(times.sum() / 60),
    }
    for p in percentiles:
        stats[f"p{p}_inference_s"] = (
            float(np.percentile(pos_times, p)) if len(pos_times) > 0 else float("nan")
        )
        stats[f"p{p}_rtfx"] = (
            float(np.percentile(fin_rtfx, p)) if len(fin_rtfx) > 0 else float("nan")
        )
    return stats


def compute_rtf_per_sample(inference_time_s: float, audio_duration_s: float) -> dict:
    if audio_duration_s <= 0 or inference_time_s <= 0:
        return {"rtf": float("inf"), "rtfx": float("inf")}
    return {
        "rtf":  inference_time_s / audio_duration_s,
        "rtfx": audio_duration_s / inference_time_s,
    }
