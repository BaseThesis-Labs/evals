"""src/data/loader.py — AudioSample dataclass, manifest + directory loaders."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

log = logging.getLogger(__name__)


@dataclass
class AudioSample:
    audio_filepath: str
    text: str                    # ground-truth reference transcript
    duration: float = 0.0
    tts_model: str = ""
    case_study: str = ""
    lang: str = "en"
    speaker_id: str = ""
    sample_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.duration and Path(self.audio_filepath).exists():
            try:
                self.duration = sf.info(self.audio_filepath).duration
            except Exception:
                pass
        if not self.sample_id:
            self.sample_id = Path(self.audio_filepath).stem


def _get_duration(path: str) -> float:
    try:
        return sf.info(path).duration
    except Exception:
        return 0.0


def load_manifest(manifest_path: str) -> list[AudioSample]:
    """Load samples from a JSONL manifest file.

    Each line must have at least: audio_filepath, text
    Optional: duration, tts_model, case_study, lang, speaker_id
    """
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples: list[AudioSample] = []
    with open(p) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Skipping malformed line {i} in {manifest_path}: {e}")
                continue

            if "audio_filepath" not in d or "text" not in d:
                log.warning(f"Line {i} missing audio_filepath or text, skipping")
                continue

            duration = d.get("duration", 0.0)
            if not duration:
                duration = _get_duration(d["audio_filepath"])

            reserved = {"audio_filepath", "text", "duration", "tts_model",
                        "case_study", "lang", "speaker_id", "id"}
            samples.append(AudioSample(
                audio_filepath=d["audio_filepath"],
                text=d["text"],
                duration=duration,
                tts_model=d.get("tts_model", ""),
                case_study=d.get("case_study", ""),
                lang=d.get("lang", "en"),
                speaker_id=d.get("speaker_id", ""),
                sample_id=d.get("id", Path(d["audio_filepath"]).stem),
                metadata={k: v for k, v in d.items() if k not in reserved},
            ))

    log.info(f"Loaded {len(samples)} samples from {manifest_path}")
    return samples


def load_wav_directory(audio_dir: str, reference_text: str = "") -> list[AudioSample]:
    """Load all .wav files from a directory.

    Pairs each .wav with a matching .txt file (same stem) if present.
    Falls back to reference_text — pass "" for hallucination testing.
    """
    d = Path(audio_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {audio_dir}")

    samples: list[AudioSample] = []
    for wav in sorted(d.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        ref = txt.read_text(encoding="utf-8").strip() if txt.exists() else reference_text
        samples.append(AudioSample(
            audio_filepath=str(wav),
            text=ref,
            duration=_get_duration(str(wav)),
            sample_id=wav.stem,
        ))

    log.info(f"Loaded {len(samples)} .wav files from {audio_dir}")
    return samples


def validate_samples(samples: list[AudioSample]) -> list[AudioSample]:
    """Drop samples whose audio file does not exist or has zero duration."""
    valid = []
    for s in samples:
        if not Path(s.audio_filepath).exists():
            log.warning(f"Audio missing, skipping: {s.audio_filepath}")
            continue
        if s.duration == 0.0:
            log.warning(f"Zero-duration audio, skipping: {s.audio_filepath}")
            continue
        valid.append(s)
    dropped = len(samples) - len(valid)
    if dropped:
        log.warning(f"Dropped {dropped} invalid samples ({len(valid)} remain)")
    return valid
