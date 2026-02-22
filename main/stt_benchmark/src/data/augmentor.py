"""src/data/augmentor.py — SNR-controlled noise injection for robustness testing."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)) + 1e-9)


def _pink_noise(n: int) -> np.ndarray:
    """Generate pink (1/f) noise of length n."""
    f = np.fft.rfftfreq(n)
    f[0] = 1.0
    power = 1.0 / f
    power[0] = 0.0
    phases = np.random.uniform(0, 2 * np.pi, len(power))
    spectrum = np.sqrt(power) * np.exp(1j * phases)
    noise = np.fft.irfft(spectrum, n=n)
    return (noise / (np.abs(noise).max() + 1e-9)).astype(np.float32)


def add_noise_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix clean speech with noise at the given SNR (dB)."""
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[: len(clean)]
    target_noise_rms = _rms(clean) / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / _rms(noise))
    return np.clip(clean + scaled_noise, -1.0, 1.0).astype(np.float32)


def augment_file(
    input_path: str,
    output_path: str | None = None,
    snr_db: float = 10.0,
    noise_type: str = "white",
    sr: int = 16000,
) -> str:
    """
    Load audio, inject noise at snr_db, save result.
    noise_type: "white" | "pink"
    Returns output path.
    """
    import librosa
    clean, _ = librosa.load(input_path, sr=sr, mono=True, dtype=np.float32)

    if noise_type == "pink":
        noise = _pink_noise(len(clean))
    else:
        noise = np.random.randn(len(clean)).astype(np.float32)

    mixed = add_noise_at_snr(clean, noise, snr_db)

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name

    sf.write(output_path, mixed, sr, subtype="PCM_16")
    return output_path


def batch_augment(
    paths: list[str],
    output_dir: str,
    snr_levels: list[float],
    noise_types: list[str] | None = None,
    sr: int = 16000,
) -> dict[tuple[float, str], list[str]]:
    """Augment files at multiple SNR levels. Returns {(snr, noise_type): [paths]}."""
    if noise_types is None:
        noise_types = ["white"]
    out = Path(output_dir)
    results: dict[tuple[float, str], list[str]] = {}

    for snr in snr_levels:
        for ntype in noise_types:
            snr_dir = out / f"snr_{snr}db_{ntype}"
            snr_dir.mkdir(parents=True, exist_ok=True)
            augmented = []
            for p in paths:
                out_path = str(snr_dir / (Path(p).stem + ".wav"))
                augmented.append(augment_file(p, out_path, snr, ntype, sr))
            results[(snr, ntype)] = augmented
            log.info(f"Augmented {len(paths)} files at SNR={snr}dB noise={ntype}")

    return results
