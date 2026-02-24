#!/usr/bin/env python3
"""
Download additional datasets for S2S benchmark.

Usage:
    python datasets/download_additional.py --dataset tess
    python datasets/download_additional.py --dataset savee
    python datasets/download_additional.py --dataset ravdess
    python datasets/download_additional.py --dataset cmu_arctic
    python datasets/download_additional.py --dataset ljspeech
    python datasets/download_additional.py --dataset all
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TTS_BENCH = _HERE.parent.parent / "tts_benchmark"
if str(_TTS_BENCH) not in sys.path:
    sys.path.insert(0, str(_TTS_BENCH))


# ─────────────────────────────────────────────────────────────────────────────
# TESS — Kaggle ejlok1/toronto-emotional-speech-set-tess
# (HuggingFace mirrors use deprecated dataset scripts — use Kaggle directly)
# Folder structure on Kaggle:
#   TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav
#   Emotion is encoded in the parent folder name after the speaker prefix.
# ─────────────────────────────────────────────────────────────────────────────

# Map Kaggle folder suffixes → canonical emotion labels
_TESS_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "pleasant_surprise": "surprised",
    "sad": "sad",
}


def _tess_emotion_from_folder(folder_name: str) -> str:
    """Extract emotion label from TESS folder name like OAF_angry or YAF_Fear."""
    lower = folder_name.lower()
    # Strip speaker prefix (OAF_ / YAF_)
    for prefix in ("oaf_", "yaf_"):
        if lower.startswith(prefix):
            lower = lower[len(prefix):]
    # Normalise to canonical label
    return _TESS_EMOTION_MAP.get(lower, lower)


def download_tess(output_dir: Path = _HERE / "tess_audio") -> None:
    """Download TESS from Kaggle (primary) or prompt for manual download.

    Kaggle dataset: ejlok1/toronto-emotional-speech-set-tess
    Requires: pip install kagglehub  AND  ~/.kaggle/kaggle.json credentials.

    Output layout: datasets/tess_audio/<emotion>/<file>.wav
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Primary: kagglehub ────────────────────────────────────────────────────
    try:
        import kagglehub  # type: ignore
        print("  [tess] Downloading from Kaggle (ejlok1/toronto-emotional-speech-set-tess) …")
        cache_path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
        src = Path(cache_path)
        wav_files = list(src.rglob("*.wav"))
        if not wav_files:
            raise RuntimeError("No WAV files found in downloaded archive.")

        copied = 0
        for wav in wav_files:
            emotion = _tess_emotion_from_folder(wav.parent.name)
            dest = output_dir / emotion / wav.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav, dest)
            copied += 1

        print(f"  [tess] Done — {copied} files organised into {output_dir}")
        _print_emotion_breakdown(output_dir)
        return

    except ImportError:
        print("  [tess] kagglehub not installed → pip install kagglehub")
    except Exception as exc:
        print(f"  [tess] Kaggle download failed: {exc}")

    # ── Fallback: manual instructions ─────────────────────────────────────────
    print(
        "\n  [tess] Manual download steps:\n"
        "    1. pip install kaggle\n"
        "    2. Get your API key from https://www.kaggle.com/settings → API\n"
        "    3. Save to ~/.kaggle/kaggle.json\n"
        "    4. Run:\n"
        "         kaggle datasets download ejlok1/toronto-emotional-speech-set-tess\n"
        "         unzip toronto-emotional-speech-set-tess.zip -d /tmp/tess_raw\n"
        "    5. Then re-run this script (kagglehub will find it in cache)\n"
        "    OR manually copy WAV files into:\n"
        f"       {output_dir}/<emotion>/<file>.wav\n"
    )


def _print_emotion_breakdown(base_dir: Path) -> None:
    for d in sorted(base_dir.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.wav")))
            print(f"    {d.name}: {n} files")


# ─────────────────────────────────────────────────────────────────────────────
# LJSpeech — extracted from local tarball or direct HTTP download
# keithito/lj_speech on HuggingFace uses a deprecated dataset script;
# we extract directly from the official tar.bz2 instead.
#
# Tarball layout:
#   LJSpeech-1.1/metadata.csv       (pipe-separated: id|text|normalized_text)
#   LJSpeech-1.1/wavs/LJ001-0001.wav
# ─────────────────────────────────────────────────────────────────────────────

# Known locations where the tarball may already exist.
# Include .1 / .2 suffixes — macOS Safari appends these for duplicate downloads.
# We pick the LARGEST file found (partial downloads are much smaller).
_LJSPEECH_TARBALL_CANDIDATES = [
    _TTS_BENCH / "LJSpeech-1.1.tar.bz2",
    _TTS_BENCH / "LJSpeech-1.1.tar.bz2.1",
    _TTS_BENCH / "LJSpeech-1.1.tar.bz2.2",
    _HERE.parent / "LJSpeech-1.1.tar.bz2",
    Path.home() / "Downloads" / "LJSpeech-1.1.tar.bz2",
]
# Full LJSpeech-1.1 is ~2.6 GB; reject anything under 2 GB as incomplete
_LJSPEECH_MIN_BYTES = 2_000_000_000
_LJSPEECH_DOWNLOAD_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def download_ljspeech(
    output_dir: Path = _HERE / "ljspeech_audio",
    n_samples: int = 100,
) -> None:
    """Extract LJSpeech WAVs from local tarball (already in tts_benchmark) or
    download from keithito.com directly — no HuggingFace dataset scripts needed.

    Steps:
      1. Look for an existing LJSpeech-1.1.tar.bz2 in tts_benchmark/
      2. If not found, download ~2.6 GB from keithito.com
      3. Extract, read metadata.csv, copy n_samples WAVs to output_dir
      4. Write datasets/manifests/ljspeech_s2s.json
    """
    import json
    import tarfile
    import urllib.request

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: find existing tarball (pick largest ≥ 2 GB) ─────────────────
    tarball: Optional[Path] = None
    best_size = 0
    for candidate in _LJSPEECH_TARBALL_CANDIDATES:
        if candidate.exists():
            sz = candidate.stat().st_size
            if sz >= _LJSPEECH_MIN_BYTES and sz > best_size:
                tarball = candidate
                best_size = sz
    if tarball:
        print(f"  [ljspeech] Found existing tarball: {tarball} ({best_size // 1_048_576} MB)")

    # ── Step 2: download if missing ──────────────────────────────────────────
    if tarball is None:
        tarball = _HERE / "LJSpeech-1.1.tar.bz2"
        print(f"  [ljspeech] Downloading LJSpeech-1.1 (~2.6 GB) from keithito.com …")
        print(f"             → {tarball}")
        try:
            urllib.request.urlretrieve(
                _LJSPEECH_DOWNLOAD_URL,
                tarball,
                reporthook=_download_progress,
            )
            print()
        except Exception as exc:
            print(f"  [ljspeech] Download failed: {exc}")
            return

    # ── Step 3: extract to a temp location and read metadata ─────────────────
    extract_root = output_dir.parent / "_ljspeech_extracted"
    lj_dir = extract_root / "LJSpeech-1.1"
    metadata_path = lj_dir / "metadata.csv"

    if not metadata_path.exists():
        print(f"  [ljspeech] Extracting {tarball.name} …")
        extract_root.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(tarball, "r:bz2") as tf:
                tf.extractall(extract_root)
        except Exception as exc:
            print(f"  [ljspeech] Extraction failed: {exc}")
            return

    if not metadata_path.exists():
        print(f"  [ljspeech] metadata.csv not found at {metadata_path}")
        return

    # ── Step 4: read metadata and copy WAVs ──────────────────────────────────
    entries = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            if len(entries) >= n_samples:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            uid = parts[0]
            # Use normalized text (col 3) if available, else raw text (col 2)
            text = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else parts[1].strip()
            src_wav = lj_dir / "wavs" / f"{uid}.wav"
            if not src_wav.exists():
                continue

            dst_wav = output_dir / f"{uid}.wav"
            if not dst_wav.exists():
                shutil.copy2(src_wav, dst_wav)

            entries.append({
                "id": f"LJS_{uid}",
                "audio_in_path": str(dst_wav),
                "reference_text": text,
                "speaker_id": "LJSpeech",
                "dataset": "ljspeech",
                "category": "audiobook",
                "emotion": None,
                "difficulty": "standard",
            })

    # ── Step 5: save manifest ─────────────────────────────────────────────────
    manifest_path = _HERE / "manifests" / "ljspeech_s2s.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"  [ljspeech] Done — {len(entries)} files in {output_dir}")
    print(f"  [ljspeech] Manifest → {manifest_path}")


def _download_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        print(f"\r    {pct:.1f}%  ({downloaded // 1_048_576} MB / {total_size // 1_048_576} MB)", end="")


# ── Type hint used inside download_ljspeech ──────────────────────────────────
from typing import Optional  # noqa: E402  (re-import for clarity)


# ─────────────────────────────────────────────────────────────────────────────
# SAVEE — Kaggle ejlok1/surrey-audiovisual-expressed-emotion-savee
# ─────────────────────────────────────────────────────────────────────────────

def download_savee(output_dir: Path = _HERE / "savee_audio") -> None:
    """Download SAVEE via kagglehub or direct URL fallback."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try kagglehub first
    try:
        import kagglehub  # type: ignore
        print("  [savee] Downloading SAVEE via kagglehub …")
        path = kagglehub.dataset_download(
            "ejlok1/surrey-audiovisual-expressed-emotion-savee"
        )
        src = Path(path)
        print(f"  [savee] Downloaded to {src}")

        # Copy WAV files
        wav_files = list(src.rglob("*.wav"))
        for wav in wav_files:
            dest = output_dir / wav.name
            shutil.copy2(wav, dest)
        print(f"  [savee] {len(wav_files)} WAV files copied to {output_dir}")
        return

    except ImportError:
        print("  [savee] kagglehub not found, trying HuggingFace …")
    except Exception as exc:
        print(f"  [savee] kagglehub failed: {exc}, trying HuggingFace …")

    # Fallback: manual instructions (HuggingFace mirror also uses deprecated scripts)
    print(
        "  [savee] Manual download steps:\n"
        "    1. pip install kaggle\n"
        "    2. kaggle datasets download ejlok1/surrey-audiovisual-expressed-emotion-savee\n"
        "    3. unzip and copy WAV files to:\n"
        f"       {output_dir}/\n"
        "    OR: https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAVDESS — Zenodo direct download (official source, no auth required)
# https://zenodo.org/record/1188976
#
# Single zip: Audio_Speech_Actors_01-24.zip (~350 MB, 1440 WAV files)
# Filename encoding: 03-01-06-01-02-01-12.wav
#   field[2] = emotion: 01=neutral 02=calm 03=happy 04=sad
#                       05=angry 06=fearful 07=disgust 08=surprised
#   field[6] = actor id (01-24)
# ─────────────────────────────────────────────────────────────────────────────

_RAVDESS_ZENODO_URL = (
    "https://zenodo.org/records/1188976/files/"
    "Audio_Speech_Actors_01-24.zip?download=1"
)
_RAVDESS_EMOTION_MAP = {
    "01": "neutral", "02": "calm",     "03": "happy",    "04": "sad",
    "05": "angry",   "06": "fearful",  "07": "disgust",  "08": "surprised",
}


def download_ravdess_hf(output_dir: Path = _HERE / "ravdess_audio") -> None:
    """Download RAVDESS speech audio from Zenodo (~350 MB, no login required).

    Falls back to Kaggle (uwrfkaggler/ravdess-emotional-speech-audio) if
    Zenodo is unavailable.
    """
    import urllib.request
    import zipfile

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Check if already downloaded ──────────────────────────────────────────
    existing_wavs = list(output_dir.rglob("*.wav"))
    if len(existing_wavs) >= 1400:
        print(f"  [ravdess] Already have {len(existing_wavs)} files in {output_dir}, skipping.")
        return

    # ── Primary: Zenodo ──────────────────────────────────────────────────────
    zip_path = output_dir.parent / "ravdess_speech.zip"

    if not zip_path.exists():
        print("  [ravdess] Downloading from Zenodo (~350 MB) …")
        try:
            urllib.request.urlretrieve(
                _RAVDESS_ZENODO_URL,
                zip_path,
                reporthook=_download_progress,
            )
            print()
        except Exception as exc:
            print(f"\n  [ravdess] Zenodo download failed: {exc}")
            zip_path.unlink(missing_ok=True)
            _ravdess_kaggle_fallback(output_dir)
            return

    # ── Extract ───────────────────────────────────────────────────────────────
    print("  [ravdess] Extracting …")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
    except Exception as exc:
        print(f"  [ravdess] Extraction failed: {exc}")
        return

    # ── Flatten: move Actor_XX/file.wav → output_dir/file.wav ────────────────
    for actor_dir in output_dir.glob("Actor_*"):
        for wav in actor_dir.glob("*.wav"):
            dest = output_dir / wav.name
            if not dest.exists():
                shutil.move(str(wav), dest)
        try:
            actor_dir.rmdir()
        except OSError:
            pass

    wav_count = len(list(output_dir.glob("*.wav")))
    print(f"  [ravdess] Done — {wav_count} files in {output_dir}")


def _ravdess_kaggle_fallback(output_dir: Path) -> None:
    """Fallback: download RAVDESS from Kaggle."""
    try:
        import kagglehub  # type: ignore
        print("  [ravdess] Trying Kaggle (uwrfkaggler/ravdess-emotional-speech-audio) …")
        path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
        src = Path(path)
        wav_files = list(src.rglob("*.wav"))
        for wav in wav_files:
            dest = output_dir / wav.name
            shutil.copy2(wav, dest)
        print(f"  [ravdess] Done — {len(wav_files)} files via Kaggle")
    except Exception as exc:
        print(f"  [ravdess] Kaggle also failed: {exc}")
        print(
            "  [ravdess] Manual download:\n"
            "    https://zenodo.org/record/1188976\n"
            "    Download 'Audio_Speech_Actors_01-24.zip', extract WAVs to:\n"
            f"    {output_dir}/"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CMU-ARCTIC — reuse tts_benchmark helper
# ─────────────────────────────────────────────────────────────────────────────

def download_cmu_arctic(output_dir: Path = _HERE / "cmu_arctic_audio") -> None:
    """Download CMU-Arctic by reusing tts_benchmark download.py."""
    try:
        from datasets.download import create_cmu_arctic_manifest  # type: ignore
        print("  [cmu_arctic] Delegating to tts_benchmark download.py …")
        create_cmu_arctic_manifest(str(output_dir))
    except ImportError:
        print("  [cmu_arctic] tts_benchmark download.py not found.")
        print("  [cmu_arctic] Falling back to direct download …")
        _download_cmu_arctic_direct(output_dir)


def _download_cmu_arctic_direct(output_dir: Path) -> None:
    """Direct download of a subset of CMU-Arctic speakers."""
    import urllib.request

    SPEAKERS = ["slt", "bdl", "clb", "rms"]
    BASE_URL = "http://www.festvox.org/cmu_arctic/packed/"
    output_dir.mkdir(parents=True, exist_ok=True)

    for spk in SPEAKERS:
        url = f"{BASE_URL}cmu_us_{spk}_arctic.tar.bz2"
        dest = output_dir / f"cmu_us_{spk}_arctic.tar.bz2"
        if not dest.exists():
            print(f"  [cmu_arctic] Downloading {spk} …")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as exc:
                print(f"    ✗ Failed: {exc}")
                continue
        # Extract
        print(f"  [cmu_arctic] Extracting {spk} …")
        os.system(f"tar -xjf {dest} -C {output_dir}")

    wav_count = len(list(output_dir.rglob("*.wav")))
    print(f"  [cmu_arctic] Done — {wav_count} WAV files in {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DOWNLOADERS = {
    "tess": download_tess,
    "savee": download_savee,
    "ravdess": download_ravdess_hf,
    "cmu_arctic": download_cmu_arctic,
    "ljspeech": download_ljspeech,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download S2S benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DOWNLOADERS.keys()) + ["all"],
        required=True,
        help="Dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: datasets/<name>_audio/)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Max samples to download (ljspeech only)",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        targets = list(DOWNLOADERS.keys())
    else:
        targets = [args.dataset]

    for name in targets:
        fn = DOWNLOADERS[name]
        print(f"\n▶ Downloading {name} …")
        if name == "ljspeech":
            out = Path(args.output_dir) / "ljspeech_audio" if args.output_dir else None
            if out:
                fn(out, args.n_samples)
            else:
                fn(n_samples=args.n_samples)
        elif args.output_dir:
            fn(Path(args.output_dir) / f"{name}_audio")
        else:
            fn()

    print("\n✓ All requested datasets downloaded.")


if __name__ == "__main__":
    main()
