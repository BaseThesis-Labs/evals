#!/usr/bin/env python3
"""
generate.py — Download datasets and/or generate TTS test audio, then build manifests.

Usage:
    # Download LibriSpeech test-clean + test-other and build manifests
    python generate.py --source librispeech

    # Download TED-LIUM 3 (HuggingFace)
    python generate.py --source tedlium

    # Download VoxPopuli English (HuggingFace)
    python generate.py --source voxpopuli

    # Point to a local Kincaid46 directory of .wav files
    python generate.py --source kincaid --kincaid-dir /path/to/kincaid46

    # Generate TTS audio from a text corpus using enabled TTS providers
    python generate.py --source tts --text-file datasets/corpus.txt

    # Do everything
    python generate.py --source all
"""
from __future__ import annotations

import json
import logging
import os
import tarfile
import time
import urllib.request
from pathlib import Path

import click
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)

# ── LibriSpeech ────────────────────────────────────────────────────────────────

LIBRISPEECH_URLS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {url} → {dest}")
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=dest.name) as t:
        def reporthook(b, bsize, tsize):
            t.total = tsize
            t.update(b * bsize - t.n)
        urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    return dest


def _flac_to_wav(flac_path: Path, wav_path: Path, target_sr: int = 16000):
    """Convert .flac to 16 kHz mono .wav using librosa."""
    import librosa
    audio, _ = librosa.load(str(flac_path), sr=target_sr, mono=True)
    sf.write(str(wav_path), audio, target_sr, subtype="PCM_16")


def download_librispeech(
    subset: str,
    output_dir: str = "datasets/librispeech",
    convert_to_wav: bool = True,
    max_samples: int | None = None,
) -> str:
    """Download one LibriSpeech subset and build a JSONL manifest. Returns manifest path."""
    url = LIBRISPEECH_URLS[subset]
    out = Path(output_dir)
    tar_path = out / f"{subset}.tar.gz"
    extract_dir = out / "raw"

    if not tar_path.exists():
        _download(url, tar_path)
    else:
        log.info(f"Archive already exists: {tar_path}")

    log.info(f"Extracting {tar_path}…")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_dir)

    # Parse .trans.txt files and collect (flac_path, transcript) pairs
    libri_root = extract_dir / "LibriSpeech" / subset
    pairs: list[tuple[Path, str, str, str]] = []   # (audio, text, speaker_id, chapter_id)

    for trans_file in sorted(libri_root.rglob("*.trans.txt")):
        speaker_id  = trans_file.parent.parent.name
        chapter_id  = trans_file.parent.name
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, *words = line.split()
                text = " ".join(words)
                audio_file = trans_file.parent / f"{utt_id}.flac"
                if audio_file.exists():
                    pairs.append((audio_file, text, speaker_id, chapter_id))

    if max_samples:
        pairs = pairs[:max_samples]

    log.info(f"Found {len(pairs)} utterances in LibriSpeech {subset}")

    # Convert flac → wav and write manifest
    wav_dir = out / subset / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"{subset}_manifest.jsonl"

    with open(manifest_path, "w") as mf:
        for flac_path, text, speaker_id, chapter_id in tqdm(pairs, desc=f"Converting {subset}"):
            wav_path = wav_dir / (flac_path.stem + ".wav")
            if not wav_path.exists():
                if convert_to_wav:
                    _flac_to_wav(flac_path, wav_path)
                else:
                    wav_path = flac_path  # use flac directly

            try:
                duration = sf.info(str(wav_path)).duration
            except Exception:
                duration = 0.0

            row = {
                "audio_filepath": str(wav_path),
                "text":           text,
                "duration":       round(duration, 3),
                "speaker_id":     speaker_id,
                "chapter_id":     chapter_id,
                "subset":         subset,
                "lang":           "en",
            }
            mf.write(json.dumps(row) + "\n")

    log.info(f"Manifest written: {manifest_path} ({len(pairs)} samples)")
    return str(manifest_path)


# ── TED-LIUM 3 (via HuggingFace datasets) ────────────────────────────────────

def _hf_audio_to_array(audio_info: dict) -> tuple:
    """
    Decode a HuggingFace Audio field (decode=False) → (float32 array, sample_rate).
    Works without torchcodec by using soundfile via BytesIO.
    """
    import io
    raw = audio_info.get("bytes")
    if raw:
        audio_arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    else:
        audio_arr, sr = sf.read(audio_info["path"], dtype="float32", always_2d=False)
    if audio_arr.ndim > 1:
        audio_arr = audio_arr.mean(axis=1)
    return audio_arr.astype(np.float32), sr


def download_tedlium(
    output_dir: str = "datasets/tedlium",
    split: str = "test",
    max_samples: int | None = None,
) -> str:
    """
    Stream TED-LIUM 3 from HuggingFace (sanchit-gandhi/tedlium-data).
    Uses streaming + decode=False so only the samples you need are fetched.
    """
    import os, itertools
    os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")
    try:
        from datasets import load_dataset, Audio as HFAudio
    except ImportError:
        log.error("Install datasets: pip install datasets huggingface-hub")
        return ""

    out = Path(output_dir)
    wav_dir = out / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"tedlium_{split}_manifest.jsonl"

    n = max_samples or 500
    log.info(f"Streaming TED-LIUM {split} (up to {n} samples) from sanchit-gandhi/tedlium-data…")

    ds = load_dataset("sanchit-gandhi/tedlium-data", split=split, streaming=True)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000, decode=False))

    rows = []
    fetched = 0
    stream = iter(ds)
    pbar = tqdm(total=n, desc=f"TED-LIUM {split}")
    while len(rows) < n:
        try:
            sample = next(stream)
            fetched += 1
        except StopIteration:
            break

        # Skip inter-segment gaps — they are silence, not real speech
        spk = str(sample.get("speaker_id", ""))
        if spk == "inter_segment_gap":
            continue

        text = str(sample.get("text", "")).strip()
        if not text:
            continue

        try:
            audio_arr, sr = _hf_audio_to_array(sample["audio"])
        except Exception as e:
            log.warning(f"  Skipping sample {fetched}: {e}")
            continue

        if sr != 16000:
            import librosa
            audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
            sr = 16000

        idx = len(rows)
        wav_path = wav_dir / f"tedlium_{split}_{idx:05d}.wav"
        if not wav_path.exists():
            sf.write(str(wav_path), audio_arr, sr, subtype="PCM_16")

        rows.append({
            "audio_filepath": str(wav_path),
            "text":           text,
            "duration":       round(len(audio_arr) / sr, 3),
            "speaker_id":     spk,
            "subset":         f"tedlium_{split}",
            "lang":           "en",
        })
        pbar.update(1)
    pbar.close()

    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    log.info(f"TED-LIUM manifest: {manifest_path} ({len(rows)} samples)")
    return str(manifest_path)


# ── VoxPopuli English (via HuggingFace datasets) ──────────────────────────────

def download_voxpopuli(
    output_dir: str = "datasets/voxpopuli",
    split: str = "test",
    max_samples: int | None = None,
) -> str:
    """
    Stream VoxPopuli English from HuggingFace (facebook/voxpopuli).
    Uses streaming + decode=False so only the samples you need are fetched.
    """
    import os, itertools
    os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")
    try:
        from datasets import load_dataset, Audio as HFAudio
    except ImportError:
        log.error("Install datasets: pip install datasets huggingface-hub")
        return ""

    out = Path(output_dir)
    wav_dir = out / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"voxpopuli_{split}_manifest.jsonl"

    n = max_samples or 500
    log.info(f"Streaming VoxPopuli en {split} (up to {n} samples) from facebook/voxpopuli…")

    ds = load_dataset("facebook/voxpopuli", "en", split=split, streaming=True)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000, decode=False))

    rows = []
    for i, sample in enumerate(tqdm(itertools.islice(ds, n),
                                    desc=f"VoxPopuli {split}", total=n)):
        try:
            audio_arr, sr = _hf_audio_to_array(sample["audio"])
        except Exception as e:
            log.warning(f"  Skipping sample {i}: {e}")
            continue

        if sr != 16000:
            import librosa
            audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
            sr = 16000

        wav_path = wav_dir / f"voxpopuli_{split}_{i:05d}.wav"
        if not wav_path.exists():
            sf.write(str(wav_path), audio_arr, sr, subtype="PCM_16")

        text = sample.get("normalized_text") or sample.get("raw_text", "")
        rows.append({
            "audio_filepath": str(wav_path),
            "text":           str(text).strip(),
            "duration":       round(len(audio_arr) / sr, 3),
            "speaker_id":     str(sample.get("speaker_id", f"vox_{i}")),
            "subset":         f"voxpopuli_{split}",
            "lang":           "en",
        })

    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    log.info(f"VoxPopuli manifest: {manifest_path} ({len(rows)} samples)")
    return str(manifest_path)


# ── Kincaid46 (local directory) ───────────────────────────────────────────────

def build_kincaid_manifest(
    kincaid_dir: str,
    output_dir: str = "datasets/kincaid46",
    max_samples: int | None = None,
) -> str:
    """
    Build a manifest from a local Kincaid46 directory.
    Expected structure:
      kincaid_dir/
        *.wav          — audio files
        *.txt          — matching transcripts (same stem), OR
        transcripts.txt — tab-separated "filename\\ttext" lines
    """
    src = Path(kincaid_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "kincaid46_manifest.jsonl"

    # Try to load transcripts from a bulk file first
    transcripts: dict[str, str] = {}
    bulk_file = src / "transcripts.txt"
    if bulk_file.exists():
        for line in bulk_file.read_text().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                transcripts[parts[0].strip()] = parts[1].strip()
    else:
        # Fall back: one .txt file per .wav with same stem
        for txt_file in src.glob("*.txt"):
            transcripts[txt_file.stem] = txt_file.read_text().strip()

    wav_files = sorted(src.glob("*.wav"))
    if max_samples:
        wav_files = wav_files[:max_samples]

    rows = []
    for wav_path in tqdm(wav_files, desc="Kincaid46"):
        text = transcripts.get(wav_path.stem, "")
        try:
            duration = sf.info(str(wav_path)).duration
        except Exception:
            duration = 0.0
        rows.append({
            "audio_filepath": str(wav_path),
            "text":           text,
            "duration":       round(duration, 3),
            "speaker_id":     "kincaid",
            "subset":         "kincaid46",
            "lang":           "en",
        })

    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    log.info(f"Kincaid46 manifest: {manifest_path} ({len(rows)} samples)")
    return str(manifest_path)


# ── Common Voice (Mozilla) ────────────────────────────────────────────────────

def download_commonvoice(
    output_dir: str = "datasets/commonvoice",
    lang: str = "en",
    split: str = "test",
    max_samples: int | None = None,
) -> str:
    """
    Stream Mozilla Common Voice 17.0 from HuggingFace.
    Preserves age / gender / accent metadata for fairness analysis.
    Requires HuggingFace login + dataset access acceptance:
      huggingface-cli login
    """
    import os
    os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")
    try:
        from datasets import load_dataset, Audio as HFAudio
    except ImportError:
        log.error("Install datasets: pip install datasets huggingface-hub")
        return ""

    out = Path(output_dir)
    wav_dir = out / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"commonvoice_{lang}_{split}_manifest.jsonl"

    n = max_samples or 500
    log.info(
        f"Streaming Common Voice 17.0 '{lang}' {split} "
        f"(up to {n} samples) from mozilla-foundation/common_voice_17_0…"
    )

    try:
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            lang,
            split=split,
            streaming=True,
            trust_remote_code=False,
        )
    except Exception as e:
        log.error(f"Failed to load Common Voice: {e}")
        log.error(
            "Common Voice requires accepting the dataset terms on HuggingFace "
            "and being logged in: huggingface-cli login"
        )
        return ""

    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000, decode=False))

    rows = []
    pbar = tqdm(total=n, desc=f"CommonVoice {lang}/{split}")
    for i, sample in enumerate(ds):
        if len(rows) >= n:
            break

        text = str(sample.get("sentence", "")).strip()
        if not text:
            continue

        try:
            audio_arr, sr = _hf_audio_to_array(sample["audio"])
        except Exception as e:
            log.warning(f"  Skipping sample {i}: {e}")
            continue

        if sr != 16000:
            import librosa
            audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
            sr = 16000

        idx = len(rows)
        wav_path = wav_dir / f"cv_{lang}_{split}_{idx:05d}.wav"
        if not wav_path.exists():
            sf.write(str(wav_path), audio_arr, sr, subtype="PCM_16")

        rows.append({
            "audio_filepath": str(wav_path),
            "text":           text,
            "duration":       round(len(audio_arr) / sr, 3),
            "speaker_id":     str(sample.get("client_id", f"cv_{idx}")),
            # fairness metadata
            "age":            str(sample.get("age", "")),
            "gender":         str(sample.get("gender", "")),
            "accent":         str(sample.get("accents", "")),
            "locale":         str(sample.get("locale", lang)),
            "subset":         f"commonvoice_{lang}_{split}",
            "lang":           lang,
        })
        pbar.update(1)
    pbar.close()

    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    log.info(f"Common Voice manifest: {manifest_path} ({len(rows)} samples)")
    return str(manifest_path)


# ── Earnings22 ────────────────────────────────────────────────────────────────

def download_earnings22(
    output_dir: str = "datasets/earnings22",
    split: str = "test",
    max_samples: int | None = None,
) -> str:
    """
    Stream Earnings22 from HuggingFace (revdotcom/earnings22).
    Accented earnings-call audio; financial jargon domain.
    25% weight in Artificial Analysis AA-WER v2.0.
    """
    import os
    os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")
    try:
        from datasets import load_dataset, Audio as HFAudio
    except ImportError:
        log.error("Install datasets: pip install datasets huggingface-hub")
        return ""

    out = Path(output_dir)
    wav_dir = out / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"earnings22_{split}_manifest.jsonl"

    n = max_samples or 500
    log.info(
        f"Streaming Earnings22 {split} (up to {n} samples) "
        f"from revdotcom/earnings22…"
    )

    try:
        ds = load_dataset("revdotcom/earnings22", split=split, streaming=True)
    except Exception as e:
        log.error(f"Failed to load Earnings22: {e}")
        return ""

    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000, decode=False))

    rows = []
    pbar = tqdm(total=n, desc=f"Earnings22 {split}")
    for i, sample in enumerate(ds):
        if len(rows) >= n:
            break

        # Earnings22 field names: 'sentence' or 'text'
        text = str(
            sample.get("sentence") or sample.get("text") or ""
        ).strip()
        if not text:
            continue

        try:
            audio_arr, sr = _hf_audio_to_array(sample["audio"])
        except Exception as e:
            log.warning(f"  Skipping sample {i}: {e}")
            continue

        if sr != 16000:
            import librosa
            audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
            sr = 16000

        idx = len(rows)
        wav_path = wav_dir / f"earnings22_{split}_{idx:05d}.wav"
        if not wav_path.exists():
            sf.write(str(wav_path), audio_arr, sr, subtype="PCM_16")

        rows.append({
            "audio_filepath": str(wav_path),
            "text":           text,
            "duration":       round(len(audio_arr) / sr, 3),
            "speaker_id":     str(sample.get("speaker_id", f"earn_{idx}")),
            "subset":         f"earnings22_{split}",
            "domain":         "financial",
            "lang":           "en",
        })
        pbar.update(1)
    pbar.close()

    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    log.info(f"Earnings22 manifest: {manifest_path} ({len(rows)} samples)")
    return str(manifest_path)


# ── Silence / Noise generation ────────────────────────────────────────────────

def generate_silence_noise(output_dir: str = "datasets/silence_noise", sr: int = 16000):
    """Generate silence and noise .wav files for hallucination testing."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files_created = []

    # Silences
    for dur in [5, 10, 30]:
        path = out / f"silence_{dur}s.wav"
        audio = np.zeros(int(dur * sr), dtype=np.float32)
        sf.write(str(path), audio, sr, subtype="PCM_16")
        files_created.append(path)

    # White noise
    for dur in [5, 10, 15]:
        path = out / f"white_noise_{dur}s.wav"
        audio = (np.random.randn(int(dur * sr)) * 0.1).astype(np.float32)
        sf.write(str(path), audio, sr, subtype="PCM_16")
        files_created.append(path)

    # 60 Hz hum
    for dur in [10, 30]:
        path = out / f"hum_60hz_{dur}s.wav"
        t = np.linspace(0, dur, int(dur * sr), endpoint=False)
        audio = (np.sin(2 * np.pi * 60 * t) * 0.05).astype(np.float32)
        sf.write(str(path), audio, sr, subtype="PCM_16")
        files_created.append(path)

    log.info(f"Generated {len(files_created)} silence/noise files in {out}")
    return [str(p) for p in files_created]


# ── TTS generation ────────────────────────────────────────────────────────────

SAMPLE_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The early bird catches the worm.",
    "A stitch in time saves nine.",
    "Actions speak louder than words.",
    "Every cloud has a silver lining.",
    "The pen is mightier than the sword.",
    "Time flies like an arrow; fruit flies like a banana.",
    "I think therefore I am.",
    "To infinity and beyond.",
    "Houston, we have a problem.",
    "Elementary, my dear Watson.",
    "May the force be with you.",
    "The only thing we have to fear is fear itself.",
    "Ask not what your country can do for you.",
    "One small step for man, one giant leap for mankind.",
    "In the beginning God created the heavens and the earth.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael.",
    "It is a truth universally acknowledged.",
    "Happy families are all alike; every unhappy family is unhappy in its own way.",
    "The sky above the port was the color of television tuned to a dead channel.",
    "It was a bright cold day in April and the clocks were striking thirteen.",
    "You must be the change you wish to see in the world.",
    "In three words I can sum up everything I have learned about life: it goes on.",
    "Not all those who wander are lost.",
]


def _tts_elevenlabs(text: str, output_path: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
    from elevenlabs import ElevenLabs
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    audio = client.text_to_speech.convert(
        voice_id=voice_id, text=text, model_id="eleven_turbo_v2_5",
        output_format="pcm_16000",
    )
    data = b"".join(audio)
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(output_path, arr, 16000, subtype="PCM_16")


def _tts_openai(text: str, output_path: str, voice: str = "alloy"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.audio.speech.create(model="tts-1", voice=voice, input=text)
    import tempfile, librosa
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name
    audio, _ = librosa.load(tmp_path, sr=16000, mono=True)
    sf.write(output_path, audio, 16000, subtype="PCM_16")
    Path(tmp_path).unlink(missing_ok=True)


TTS_PROVIDERS = {
    "elevenlabs": (_tts_elevenlabs, "ELEVENLABS_API_KEY"),
    "openai":     (_tts_openai,     "OPENAI_API_KEY"),
}


def generate_tts_audio(
    texts: list[str],
    output_dir: str = "datasets/tts_generated",
    providers: list[str] | None = None,
    case_study: str = "general",
) -> str:
    """Generate TTS audio for every text × provider. Returns manifest path."""
    if providers is None:
        providers = [p for p, (_, env) in TTS_PROVIDERS.items() if os.environ.get(env)]

    if not providers:
        log.warning("No TTS API keys set. Skipping TTS generation.")
        return ""

    out = Path(output_dir)
    manifest_path = out / "manifest.jsonl"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for provider in providers:
        fn, env = TTS_PROVIDERS[provider]
        if not os.environ.get(env):
            log.warning(f"Skipping TTS provider '{provider}': {env} not set")
            continue

        audio_dir = out / "audio" / provider
        audio_dir.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(tqdm(texts, desc=f"TTS {provider}")):
            audio_path = audio_dir / f"{provider}_{i:04d}.wav"
            if audio_path.exists():
                log.debug(f"Skipping existing: {audio_path}")
            else:
                try:
                    fn(text, str(audio_path))
                    time.sleep(0.5)   # basic rate limiting
                except Exception as e:
                    log.error(f"TTS {provider} failed on sample {i}: {e}")
                    continue

            try:
                duration = sf.info(str(audio_path)).duration
            except Exception:
                duration = 0.0

            rows.append({
                "audio_filepath": str(audio_path),
                "text":           text,
                "duration":       round(duration, 3),
                "tts_model":      provider,
                "case_study":     case_study,
                "lang":           "en",
            })

    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    log.info(f"TTS manifest: {manifest_path} ({len(rows)} samples)")
    return str(manifest_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--source", "-s",
              type=click.Choice([
                  "librispeech", "tedlium", "voxpopuli", "kincaid",
                  "commonvoice", "earnings22",
                  "tts", "silence", "all",
              ]),
              default="librispeech", show_default=True,
              help="What to generate/download")
@click.option("--subsets", default="test-clean,test-other", show_default=True,
              help="Comma-separated LibriSpeech subsets")
@click.option("--tedlium-split", default="test", show_default=True,
              help="TED-LIUM 3 split: train/validation/test")
@click.option("--voxpopuli-split", default="test", show_default=True,
              help="VoxPopuli split: train/validation/test")
@click.option("--commonvoice-lang", default="en", show_default=True,
              help="Common Voice language code, e.g. en, de, fr, es")
@click.option("--commonvoice-split", default="test", show_default=True,
              help="Common Voice split: train/validation/test/other/invalidated")
@click.option("--earnings22-split", default="test", show_default=True,
              help="Earnings22 split: train/validation/test")
@click.option("--kincaid-dir", default=None,
              help="Path to local Kincaid46 directory (required when --source kincaid)")
@click.option("--output-dir", "-o", default="datasets", show_default=True)
@click.option("--text-file", default=None,
              help="Path to plain-text file (one sentence per line) for TTS. Uses built-in corpus if omitted.")
@click.option("--tts-providers", default="all",
              help="Comma-separated TTS providers or 'all'. Options: elevenlabs, openai")
@click.option("--max-samples", default=None, type=int,
              help="Cap samples per dataset subset (useful for quick tests)")
def main(source, subsets, tedlium_split, voxpopuli_split,
         commonvoice_lang, commonvoice_split, earnings22_split,
         kincaid_dir, output_dir, text_file, tts_providers, max_samples):
    """
    Prepare datasets for the STT benchmark:
      librispeech  → download + convert to 16 kHz wav + build manifest
      tedlium      → download TED-LIUM 3 via HuggingFace
      voxpopuli    → download VoxPopuli EN via HuggingFace
      kincaid      → build manifest from local Kincaid46 directory (--kincaid-dir required)
      commonvoice  → Mozilla Common Voice 17.0 (120+ langs, age/gender/accent metadata)
      earnings22   → Earnings22 accented financial calls (AA-WER benchmark)
      tts          → generate audio from text using TTS APIs + build manifest
      silence      → generate silence/noise files for hallucination testing
      all          → librispeech + tedlium + voxpopuli + commonvoice + earnings22 + tts + silence
    """
    out = Path(output_dir)

    if source in ("librispeech", "all"):
        for subset in [s.strip() for s in subsets.split(",")]:
            download_librispeech(
                subset=subset,
                output_dir=str(out / "librispeech"),
                max_samples=max_samples,
            )

    if source in ("tedlium", "all"):
        download_tedlium(
            output_dir=str(out / "tedlium"),
            split=tedlium_split,
            max_samples=max_samples,
        )

    if source in ("voxpopuli", "all"):
        download_voxpopuli(
            output_dir=str(out / "voxpopuli"),
            split=voxpopuli_split,
            max_samples=max_samples,
        )

    if source == "kincaid":
        if not kincaid_dir:
            log.error("--kincaid-dir is required when --source kincaid")
        else:
            build_kincaid_manifest(
                kincaid_dir=kincaid_dir,
                output_dir=str(out / "kincaid46"),
                max_samples=max_samples,
            )

    if source in ("commonvoice", "all"):
        download_commonvoice(
            output_dir=str(out / "commonvoice"),
            lang=commonvoice_lang,
            split=commonvoice_split,
            max_samples=max_samples,
        )

    if source in ("earnings22", "all"):
        download_earnings22(
            output_dir=str(out / "earnings22"),
            split=earnings22_split,
            max_samples=max_samples,
        )

    if source in ("tts", "all"):
        if text_file:
            texts = Path(text_file).read_text().strip().splitlines()
        else:
            texts = SAMPLE_CORPUS
            log.info(f"Using built-in corpus ({len(texts)} sentences)")

        providers = (
            list(TTS_PROVIDERS.keys()) if tts_providers == "all"
            else [p.strip() for p in tts_providers.split(",")]
        )
        generate_tts_audio(texts, str(out / "tts_generated"), providers)

    if source in ("silence", "all"):
        generate_silence_noise(str(out / "silence_noise"))

    log.info("generate.py complete. Next: python evaluate.py")


if __name__ == "__main__":
    main()
