#!/usr/bin/env python3
"""
Download and prepare Full-Duplex-Bench v2 manifest for S2S evaluation.

Full-Duplex-Bench (FDB) evaluates full-duplex interaction quality:
  - Barge-in handling (user interrupts agent mid-utterance)
  - Backchanneling (agent produces "mm-hmm", "yeah" during user turn)
  - Turn-overlap detection and recovery
  - Silence handling (appropriate response to user pauses)

Sources tried in order:
  1. HuggingFace: WillHeld/full-duplex-bench  (or similar)
  2. GitHub release archive (if available)
  3. Synthetic fallback: builds a minimal FDB-style manifest from
     existing RAVDESS/TESS audio files for turn-taking annotation.

Usage:
    python datasets/prepare_fullduplex.py \
        --n-samples 50 \
        --out-dir datasets/manifests

Output manifest: datasets/manifests/full_duplex_bench.json
Audio saved to:  datasets/fullduplex_audio/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_S2S_ROOT = _HERE.parent
_AUDIO_DIR = _HERE / "fullduplex_audio"
_MANIFESTS_DIR = _HERE / "manifests"

# Interaction categories and their definitions
CATEGORIES = {
    "barge_in": "User starts speaking before agent finishes",
    "backchannel": "Agent produces backchannel token during user turn",
    "silence_gap": "Gap > 500ms between user end and agent start",
    "overlap": "Simultaneous speech from both parties",
    "clean_turn": "Clean non-overlapping turn exchange (baseline)",
}


def make_fdb_entry(
    uid: str,
    audio_in_path: str,
    reference_text: str,
    category: str,
    barge_in_expected: bool = False,
    overlap_region_s: Optional[list] = None,
    backchannel_expected: bool = False,
    expected_gap_ms: Optional[float] = None,
    speaker_id: Optional[str] = None,
) -> Dict:
    return {
        "id": uid,
        "audio_in_path": audio_in_path,
        "reference_text": reference_text,
        "speaker_id": speaker_id,
        "dataset": "full_duplex_bench",
        "category": category,
        "emotion": None,
        "difficulty": "standard",
        # FDB-specific annotations
        "barge_in_expected": barge_in_expected,
        "backchannel_expected": backchannel_expected,
        "overlap_region_s": overlap_region_s,   # [start_s, end_s] or None
        "expected_gap_ms": expected_gap_ms,       # ideal gap for clean turns
    }


def download_from_hf(n_samples: int, audio_dir: Path) -> Optional[List[Dict]]:
    """Try to load Full-Duplex-Bench from HuggingFace."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return None

    # Try known dataset names
    candidates = [
        "WillHeld/full-duplex-bench",
        "hlt-nlp/FullDuplexBench",
        "speechbrain/full-duplex-bench",
    ]
    for ds_name in candidates:
        try:
            print(f"  Trying HuggingFace: {ds_name} …")
            ds = load_dataset(ds_name, split="test", streaming=True, trust_remote_code=True)

            import soundfile as sf  # type: ignore
            import numpy as np

            entries: List[Dict] = []
            audio_dir.mkdir(parents=True, exist_ok=True)

            for i, sample in enumerate(ds):
                if len(entries) >= n_samples:
                    break

                uid = f"FDB_{i:04d}"
                audio_data = sample.get("audio") or sample.get("speech")
                if audio_data is None:
                    continue

                if isinstance(audio_data, dict):
                    array = np.array(audio_data["array"], dtype=np.float32)
                    sr = int(audio_data["sampling_rate"])
                else:
                    array = np.array(audio_data, dtype=np.float32)
                    sr = sample.get("sampling_rate", 16_000)

                audio_path = str(audio_dir / f"{uid}.wav")
                sf.write(audio_path, array, sr, subtype="PCM_16")

                ref_text = sample.get("text") or sample.get("transcript") or ""
                category = sample.get("category") or "clean_turn"
                barge_in = bool(sample.get("barge_in", False))
                overlap = sample.get("overlap_region_s")
                backchannel = bool(sample.get("backchannel", False))
                gap_ms = sample.get("expected_gap_ms")

                entries.append(make_fdb_entry(
                    uid=uid,
                    audio_in_path=audio_path,
                    reference_text=ref_text.strip(),
                    category=str(category),
                    barge_in_expected=barge_in,
                    overlap_region_s=overlap,
                    backchannel_expected=backchannel,
                    expected_gap_ms=float(gap_ms) if gap_ms is not None else None,
                    speaker_id=sample.get("speaker_id"),
                ))

            if entries:
                print(f"  ✓ Loaded {len(entries)} entries from {ds_name}")
                return entries

        except Exception as e:
            print(f"  Could not load {ds_name}: {e}")
            continue

    return None


def build_synthetic_fdb(n_samples: int, audio_dir: Path) -> List[Dict]:
    """
    Build a synthetic Full-Duplex-Bench manifest from existing local audio.

    Uses RAVDESS, TESS, or SAVEE audio from datasets/ directory and
    annotates them with FDB-style turn-taking metadata.  This creates a
    workable interaction-quality test set even without the official FDB.

    The synthetic manifest provides:
      - 40% clean_turn samples (baseline)
      - 25% barge_in samples (agent should stop on user interruption)
      - 20% silence_gap samples (long user pause, agent shouldn't jump in)
      - 15% overlap samples (simultaneous speech handling)
    """
    import soundfile as sf  # type: ignore
    import numpy as np

    audio_dir.mkdir(parents=True, exist_ok=True)

    # Search for existing audio in s2s_benchmark datasets
    source_dirs = [
        _HERE / "savee_audio",
        _HERE / "tess_audio",
        _HERE / "ravdess_audio",
        _HERE / "cmu_arctic_audio",
        _HERE / "libritts_r_audio",
    ]

    source_files: List[Path] = []
    for d in source_dirs:
        if d.exists():
            source_files.extend(sorted(d.glob("*.wav"))[:200])

    if not source_files:
        # Last resort: use any wav in datasets/
        source_files = sorted(_HERE.glob("**/*.wav"))[:200]

    if not source_files:
        print("  WARNING: No source audio found for synthetic FDB. ")
        print("  Run prepare_s2s_testsets.py first to download audio.")
        # Return a skeleton manifest that users can populate
        skeleton = []
        for i in range(min(n_samples, 20)):
            category = list(CATEGORIES.keys())[i % len(CATEGORIES)]
            skeleton.append(make_fdb_entry(
                uid=f"FDB_SYNTHETIC_{i:04d}",
                audio_in_path="PLACEHOLDER — run prepare_s2s_testsets.py first",
                reference_text="",
                category=category,
                barge_in_expected=(category == "barge_in"),
                backchannel_expected=(category == "backchannel"),
                expected_gap_ms=500.0 if category == "silence_gap" else None,
            ))
        return skeleton

    random.shuffle(source_files)
    source_files = source_files[:n_samples]

    # Category distribution
    category_weights = [
        ("clean_turn", 0.40),
        ("barge_in", 0.25),
        ("silence_gap", 0.20),
        ("overlap", 0.15),
    ]
    cats = []
    for cat, w in category_weights:
        cats.extend([cat] * int(w * n_samples))
    while len(cats) < n_samples:
        cats.append("clean_turn")
    random.shuffle(cats)

    entries: List[Dict] = []
    for i, (src_path, category) in enumerate(zip(source_files, cats)):
        uid = f"FDB_SYNTHETIC_{i:04d}"

        # Copy audio to fullduplex_audio/ dir
        try:
            audio, sr = sf.read(str(src_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
        except Exception:
            continue

        # For barge_in: create a two-segment audio (user speaks partway, pauses)
        # This simulates a scenario where the model should detect the user
        if category == "barge_in" and len(audio) > sr:
            barge_point = int(0.6 * len(audio))
            audio_out = np.concatenate([
                audio[:barge_point],
                np.zeros(int(0.1 * sr)),  # short silence
                audio[barge_point:],
            ])
        elif category == "silence_gap":
            # Insert a long silence in the middle (tests if agent waits)
            mid = len(audio) // 2
            silence = np.zeros(int(0.8 * sr))   # 800ms silence
            audio_out = np.concatenate([audio[:mid], silence, audio[mid:]])
        else:
            audio_out = audio

        dst_path = audio_dir / f"{uid}.wav"
        try:
            sf.write(str(dst_path), audio_out.astype(np.float32), sr, subtype="PCM_16")
        except Exception as exc:
            print(f"  Warning: could not write {uid}: {exc}")
            continue

        # Annotate
        audio_dur = len(audio_out) / sr
        overlap_region = None
        if category == "overlap":
            # Simulate a 300ms overlap starting 70% into the utterance
            start = 0.7 * audio_dur
            overlap_region = [round(start, 2), round(min(start + 0.3, audio_dur), 2)]

        entries.append(make_fdb_entry(
            uid=uid,
            audio_in_path=str(dst_path),
            reference_text=f"[{category} test — {src_path.stem}]",
            category=category,
            barge_in_expected=(category == "barge_in"),
            overlap_region_s=overlap_region,
            backchannel_expected=(category == "backchannel"),
            expected_gap_ms=800.0 if category == "silence_gap" else 200.0,
        ))

    print(f"  ✓ Built {len(entries)} synthetic FDB entries from local audio")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Full-Duplex-Bench S2S manifest")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--out-dir", default=str(_MANIFESTS_DIR))
    parser.add_argument("--audio-dir", default=str(_AUDIO_DIR))
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip HuggingFace download and build synthetic manifest from local audio",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir)

    entries: Optional[List[Dict]] = None

    if not args.synthetic_only:
        entries = download_from_hf(args.n_samples, audio_dir)

    if not entries:
        print("  Falling back to synthetic FDB manifest …")
        entries = build_synthetic_fdb(args.n_samples, audio_dir)

    if not entries:
        print("No entries collected — exiting.")
        sys.exit(1)

    out_path = out_dir / "full_duplex_bench.json"
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"✓ Manifest saved → {out_path} ({len(entries)} entries)")
    print()
    print("Category breakdown:")
    from collections import Counter
    counts = Counter(e["category"] for e in entries)
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<20} {n:>4} entries — {CATEGORIES.get(cat, '')}")
    print()
    print("Next steps:")
    print(f"  python run_integrated.py --manifest {out_path} --model gpt4o_realtime --results results/full_duplex_bench")
    print(f"  python run_integrated.py --manifest {out_path} --model gemini_live    --results results/full_duplex_bench")


if __name__ == "__main__":
    main()
