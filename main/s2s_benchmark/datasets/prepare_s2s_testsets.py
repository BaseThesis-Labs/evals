#!/usr/bin/env python3
"""
Build S2S evaluation manifests from existing tts_benchmark datasets plus
new emotional/multi-speaker datasets.

Usage:
    python datasets/prepare_s2s_testsets.py \
        --datasets libritts_r ljspeech tess savee ravdess \
        --n-samples 50 \
        --output datasets/manifests/s2s_manifest.json

Supported --datasets values:
    libritts_r    -- from tts_benchmark libritts_r_50.json
    ljspeech      -- from tts_benchmark ljspeech_manifest.json
    vctk          -- from tts_benchmark (if present)
    tess          -- HuggingFace ejlok1/tess (run download_additional.py first)
    savee         -- Kaggle ejlok1/surrey-audiovisual-expressed-emotion-savee
    ravdess       -- HuggingFace narad/ravdess
    crema_d       -- local crema_d_audio/ if present
    dailytalk     -- local dailytalk_audio/ if present
    cmu_arctic    -- reuses tts_benchmark CMU-Arctic downloader
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_S2S_ROOT = _HERE.parent
_TTS_BENCH_DATASETS = _S2S_ROOT.parent / "tts_benchmark" / "datasets"
_MANIFESTS_DIR = _HERE / "manifests"
_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest entry schema
# ─────────────────────────────────────────────────────────────────────────────

def make_entry(
    uid: str,
    audio_in_path: str,
    reference_text: str,
    speaker_id: Optional[str] = None,
    dataset: str = "unknown",
    category: str = "general",
    emotion: Optional[str] = None,
    difficulty: str = "standard",
) -> Dict:
    return {
        "id": uid,
        "audio_in_path": audio_in_path,
        "reference_text": reference_text,
        "speaker_id": speaker_id,
        "dataset": dataset,
        "category": category,
        "emotion": emotion,
        "difficulty": difficulty,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_libritts_r(n_samples: int) -> List[Dict]:
    manifest_path = _TTS_BENCH_DATASETS / "libritts_r_50.json"
    if not manifest_path.exists():
        manifest_path = _TTS_BENCH_DATASETS / "libritts_r_200.json"
    if not manifest_path.exists():
        print(f"  [libritts_r] Manifest not found at {manifest_path}, skipping.")
        return []

    with open(manifest_path) as f:
        entries = json.load(f)

    # Filter entries that have a usable audio file
    valid = [e for e in entries if e.get("reference_audio_path")]
    sampled = _sample(valid, n_samples)
    result = []
    for i, e in enumerate(sampled):
        audio_path = str(_TTS_BENCH_DATASETS.parent / e["reference_audio_path"])
        if not Path(audio_path).exists():
            # Try relative to tts_benchmark datasets dir
            audio_path = str(_TTS_BENCH_DATASETS / e["reference_audio_path"])
        result.append(make_entry(
            uid=f"LIBR_{i+1:04d}",
            audio_in_path=audio_path,
            reference_text=e.get("text", ""),
            speaker_id=e.get("speaker_id"),
            dataset="libritts_r",
            category=e.get("category", "phonetically_balanced"),
            emotion=None,
            difficulty=e.get("difficulty", "standard"),
        ))
    print(f"  [libritts_r] {len(result)} entries")
    return result


def load_ljspeech(n_samples: int) -> List[Dict]:
    manifest_path = _TTS_BENCH_DATASETS / "ljspeech_manifest.json"
    if not manifest_path.exists():
        print(f"  [ljspeech] Manifest not found, skipping.")
        return []
    with open(manifest_path) as f:
        content = f.read().strip()
    if not content or content in ("{}", "[]", "null"):
        print("  [ljspeech] Manifest is empty, skipping.")
        return []
    entries = json.loads(content)
    if not isinstance(entries, list):
        print("  [ljspeech] Unexpected format, skipping.")
        return []

    sampled = _sample(entries, n_samples)
    result = []
    for i, e in enumerate(sampled):
        audio_path = e.get("reference_audio_path") or e.get("audio_path", "")
        if not audio_path:
            continue
        if not Path(audio_path).is_absolute():
            audio_path = str(_TTS_BENCH_DATASETS.parent / audio_path)
        result.append(make_entry(
            uid=f"LJS_{i+1:04d}",
            audio_in_path=audio_path,
            reference_text=e.get("text", ""),
            speaker_id="LJSpeech",
            dataset="ljspeech",
            category="audiobook",
            emotion=None,
            difficulty=e.get("difficulty", "standard"),
        ))
    print(f"  [ljspeech] {len(result)} entries")
    return result


def load_tess(n_samples: int) -> List[Dict]:
    """Load TESS from datasets/tess_audio/ (download_additional.py first)."""
    tess_dir = _HERE / "tess_audio"
    if not tess_dir.exists():
        print("  [tess] datasets/tess_audio/ not found. Run download_additional.py --dataset tess first.")
        return []
    return _load_emotion_dir(
        base_dir=tess_dir,
        dataset="tess",
        id_prefix="TESS",
        n_samples=n_samples,
        emotion_from_dirname=True,
    )


def load_savee(n_samples: int) -> List[Dict]:
    """Load SAVEE from datasets/savee_audio/."""
    savee_dir = _HERE / "savee_audio"
    if not savee_dir.exists():
        print("  [savee] datasets/savee_audio/ not found. Run download_additional.py --dataset savee first.")
        return []
    return _load_emotion_dir(
        base_dir=savee_dir,
        dataset="savee",
        id_prefix="SAVEE",
        n_samples=n_samples,
        emotion_from_filename=True,
    )


def load_ravdess(n_samples: int) -> List[Dict]:
    """Load RAVDESS from datasets/ravdess_audio/."""
    ravdess_dir = _HERE / "ravdess_audio"
    if not ravdess_dir.exists():
        print("  [ravdess] datasets/ravdess_audio/ not found. Run download_additional.py --dataset ravdess first.")
        return []

    # RAVDESS filename encoding: 03-01-06-01-02-01-12.wav
    # position 3 (0-indexed 2) = emotion code
    RAVDESS_EMOTIONS = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
    }
    wav_files = sorted(ravdess_dir.rglob("*.wav"))
    entries = []
    for wav in wav_files:
        parts = wav.stem.split("-")
        emotion_code = parts[2] if len(parts) > 2 else "01"
        emotion = RAVDESS_EMOTIONS.get(emotion_code, "neutral")
        speaker_id = parts[-1] if parts else "unknown"
        entries.append((str(wav), emotion, speaker_id))

    sampled = _sample(entries, n_samples)
    result = []
    for i, (path, emotion, spk) in enumerate(sampled):
        result.append(make_entry(
            uid=f"RAVDESS_{i+1:04d}",
            audio_in_path=path,
            reference_text="",  # no transcript in RAVDESS; use ASR output
            speaker_id=spk,
            dataset="ravdess",
            category="expressive",
            emotion=emotion,
        ))
    print(f"  [ravdess] {len(result)} entries")
    return result


def load_crema_d(n_samples: int) -> List[Dict]:
    crema_dir = _HERE / "crema_d_audio"
    if not crema_dir.exists():
        print("  [crema_d] datasets/crema_d_audio/ not found, skipping.")
        return []
    CREMA_EMOTIONS = {"ANG": "angry", "DIS": "disgust", "FEA": "fearful",
                      "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
    wav_files = sorted(crema_dir.rglob("*.wav"))
    entries = []
    for wav in wav_files:
        parts = wav.stem.split("_")
        emotion_code = parts[2].upper() if len(parts) > 2 else "NEU"
        emotion = CREMA_EMOTIONS.get(emotion_code, "neutral")
        speaker_id = parts[0] if parts else "unknown"
        entries.append((str(wav), emotion, speaker_id))
    sampled = _sample(entries, n_samples)
    result = []
    for i, (path, emotion, spk) in enumerate(sampled):
        result.append(make_entry(
            uid=f"CREMA_{i+1:04d}",
            audio_in_path=path,
            reference_text="",
            speaker_id=spk,
            dataset="crema_d",
            category="expressive",
            emotion=emotion,
        ))
    print(f"  [crema_d] {len(result)} entries")
    return result


def load_cmu_arctic(n_samples: int) -> List[Dict]:
    """Load CMU-Arctic via tts_benchmark helper or local cache."""
    # Try to find locally cached arctic data
    for arctic_dir in [
        _HERE / "cmu_arctic_audio",
        _TTS_BENCH_DATASETS / "cmu_arctic",
    ]:
        if arctic_dir.exists():
            wav_files = sorted(arctic_dir.rglob("*.wav"))
            sampled = _sample(wav_files, n_samples)
            result = []
            for i, wav in enumerate(sampled):
                result.append(make_entry(
                    uid=f"ARCTIC_{i+1:04d}",
                    audio_in_path=str(wav),
                    reference_text="",
                    speaker_id=wav.parent.name,
                    dataset="cmu_arctic",
                    category="phonetically_balanced",
                    emotion=None,
                ))
            print(f"  [cmu_arctic] {len(result)} entries from {arctic_dir}")
            return result

    print("  [cmu_arctic] No local data found. Run download_additional.py --dataset cmu_arctic first.")
    return []


def load_dailytalk(n_samples: int) -> List[Dict]:
    dt_dir = _HERE / "dailytalk_audio"
    if not dt_dir.exists():
        print("  [dailytalk] datasets/dailytalk_audio/ not found, skipping.")
        return []
    wav_files = sorted(dt_dir.rglob("*.wav"))
    sampled = _sample(wav_files, n_samples)
    result = []
    for i, wav in enumerate(sampled):
        result.append(make_entry(
            uid=f"DAILY_{i+1:04d}",
            audio_in_path=str(wav),
            reference_text="",
            speaker_id=wav.parent.name,
            dataset="dailytalk",
            category="conversational",
            emotion=None,
        ))
    print(f"  [dailytalk] {len(result)} entries")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sample(items: list, n: int) -> list:
    if n <= 0 or n >= len(items):
        return items
    return random.sample(items, n)


def _load_emotion_dir(
    base_dir: Path,
    dataset: str,
    id_prefix: str,
    n_samples: int,
    emotion_from_dirname: bool = False,
    emotion_from_filename: bool = False,
) -> List[Dict]:
    wav_files = sorted(base_dir.rglob("*.wav"))
    entries = []
    for wav in wav_files:
        emotion = None
        if emotion_from_dirname:
            emotion = wav.parent.name.lower()
        elif emotion_from_filename:
            # SAVEE: DC_a01.wav → emotion code = 'a' → 'angry'
            SAVEE_MAP = {"a": "angry", "d": "disgust", "f": "fearful",
                         "h": "happy", "n": "neutral", "sa": "sad", "su": "surprised"}
            stem = wav.stem.lower()
            for code in sorted(SAVEE_MAP, key=len, reverse=True):
                if code in stem:
                    emotion = SAVEE_MAP[code]
                    break
        entries.append((str(wav), emotion))

    sampled = _sample(entries, n_samples)
    result = []
    for i, (path, emotion) in enumerate(sampled):
        result.append(make_entry(
            uid=f"{id_prefix}_{i+1:04d}",
            audio_in_path=path,
            reference_text="",
            dataset=dataset,
            category="expressive",
            emotion=emotion,
        ))
    print(f"  [{dataset}] {len(result)} entries")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

LOADERS = {
    "libritts_r": load_libritts_r,
    "ljspeech": load_ljspeech,
    "tess": load_tess,
    "savee": load_savee,
    "ravdess": load_ravdess,
    "crema_d": load_crema_d,
    "cmu_arctic": load_cmu_arctic,
    "dailytalk": load_dailytalk,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare S2S evaluation manifests — one per dataset (mirrors tts_benchmark pattern)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(LOADERS.keys()),
        choices=list(LOADERS.keys()),
        help="Datasets to build (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Max samples per dataset",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_MANIFESTS_DIR),
        help="Directory to write per-dataset manifests (default: datasets/manifests/)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Also write a combined s2s_manifest.json with all datasets merged",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building per-dataset S2S manifests → {out_dir}/")
    print(f"  datasets   : {args.datasets}")
    print(f"  n_samples  : {args.n_samples} per dataset")

    all_entries: List[Dict] = []

    for ds_name in args.datasets:
        loader = LOADERS[ds_name]
        entries = loader(args.n_samples)
        if not entries:
            continue

        # Deduplicate within this dataset
        seen: set[str] = set()
        deduped = []
        for e in entries:
            p = e["audio_in_path"]
            if p not in seen:
                seen.add(p)
                deduped.append(e)

        # Write one manifest per dataset  (e.g. datasets/manifests/libritts_r.json)
        ds_path = out_dir / f"{ds_name}.json"
        with open(ds_path, "w") as f:
            json.dump(deduped, f, indent=2)
        print(f"  ✓ {ds_name:20s} → {ds_path}  ({len(deduped)} entries)")

        all_entries.extend(deduped)

    if args.combine and all_entries:
        combined_path = out_dir / "s2s_manifest.json"
        with open(combined_path, "w") as f:
            json.dump(all_entries, f, indent=2)
        print(f"\n✓ Combined manifest → {combined_path}  ({len(all_entries)} total entries)")

    print(f"\nDone. Run inference per dataset, e.g.:")
    if args.datasets:
        first = args.datasets[0]
        print(f"  python inference/run_s2s_inference.py \\")
        print(f"    --manifest datasets/manifests/{first}.json \\")
        print(f"    --output   s2s_outputs/{first}")


if __name__ == "__main__":
    main()
