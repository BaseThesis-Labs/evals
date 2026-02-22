"""src/evaluation/snr.py — Signal-to-Noise Ratio estimation from audio files."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_FRAME_MS = 25   # analysis frame size in ms
_HOP_MS   = 10   # hop size in ms


def compute_snr_db(audio_path: str) -> float:
    """
    Estimate Signal-to-Noise Ratio (dB) using a percentile-based energy heuristic.

    Method:
      - Frame the audio at 25 ms / 10 ms hop
      - Compute RMS energy per frame
      - Top 25% of frames  → signal RMS estimate
      - Bottom 25% of frames → noise RMS estimate
      - SNR = 20 × log10(signal_rms / noise_rms)

    Works best on speech audio with voiced/silent segments.
    Voiced frames are captured in the top percentile (signal),
    silent/background frames in the bottom percentile (noise floor).

    Returns:
      SNR in dB.  nan if unreadable or too short.  inf if noise floor ≈ 0.
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception as e:
        log.debug(f"SNR: cannot read {audio_path}: {e}")
        return float("nan")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    frame_size = max(1, int(sr * _FRAME_MS / 1000))
    hop_size   = max(1, int(sr * _HOP_MS  / 1000))

    energies = [
        float(np.sqrt(np.mean(audio[i : i + frame_size] ** 2)))
        for i in range(0, len(audio) - frame_size + 1, hop_size)
    ]

    if len(energies) < 4:
        return float("nan")

    e = np.array(energies)
    signal_rms = float(np.mean(e[e >= np.percentile(e, 75)]))
    noise_rms  = float(np.mean(e[e <= np.percentile(e, 25)]))

    if noise_rms < 1e-10:
        # Near-zero noise floor = very clean studio audio.
        # Return a high finite value (60 dB) rather than inf so downstream
        # aggregation and JSON serialisation work correctly.
        return 60.0

    snr = float(20.0 * np.log10(signal_rms / noise_rms))
    # Clamp to a sane range: real speech SNR is roughly -5 to 60 dB
    return float(np.clip(snr, -5.0, 60.0))
