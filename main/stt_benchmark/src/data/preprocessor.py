"""src/data/preprocessor.py — Resample audio to 16 kHz mono."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

TARGET_SR = 16_000
TARGET_CHANNELS = 1


def load_audio(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio, resample to target_sr, convert to mono float32."""
    import librosa
    audio, sr = librosa.load(path, sr=target_sr, mono=True, dtype=np.float32)
    return audio, sr


def preprocess_to_wav(
    input_path: str,
    output_path: str | None = None,
    target_sr: int = TARGET_SR,
    normalize_amplitude: bool = True,
) -> str:
    """Load any audio format, resample to 16 kHz mono, write 16-bit PCM .wav.
    Returns the output path. Creates a temp file if output_path is None.
    """
    audio, _ = load_audio(input_path, target_sr=target_sr)

    if normalize_amplitude:
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.95

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name

    sf.write(output_path, audio, target_sr, subtype="PCM_16")
    log.debug(f"Preprocessed {input_path} → {output_path}")
    return output_path


def needs_preprocessing(path: str, target_sr: int = TARGET_SR) -> bool:
    """Return True if the file is not already 16 kHz mono."""
    try:
        info = sf.info(path)
        return info.samplerate != target_sr or info.channels != TARGET_CHANNELS
    except Exception:
        return True


def batch_preprocess(
    paths: list[str],
    output_dir: str,
    target_sr: int = TARGET_SR,
) -> list[str]:
    """Preprocess a batch of audio files into output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for p in paths:
        out_path = str(out / (Path(p).stem + ".wav"))
        results.append(preprocess_to_wav(p, out_path, target_sr))
    return results
