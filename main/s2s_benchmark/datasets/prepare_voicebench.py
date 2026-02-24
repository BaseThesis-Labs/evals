#!/usr/bin/env python3
"""
Download and prepare VoiceBench manifest for S2S evaluation.

Dataset: hlt-lab/voicebench  (HuggingFace)
Paper:   "VoiceBench: Benchmarking LLM-Based Voice Assistants" (2024)

Available configs (each has audio + prompt fields):
  commoneval  — 200 general-knowledge spoken questions (recommended)
  advbench    — 520 adversarial/safety prompts
  alpacaeval  — 199 instruction-following prompts
  mmsu        — multiple-choice spoken questions
  sd-qa       — spoken-domain QA, accent splits: usa gbr ind_n ind_s irl kenya nga nzl phl zaf
  wildvoice   — in-the-wild voice questions

Usage:
    python datasets/prepare_voicebench.py \
        --n-samples 100 \
        --config commoneval \
        --out-dir datasets/manifests \
        --audio-dir datasets/voicebench_audio

    # Multiple configs at once (all merged):
    python datasets/prepare_voicebench.py --config commoneval alpacaeval --n-samples 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_S2S_ROOT = _HERE.parent
_MANIFESTS_DIR = _HERE / "manifests"
_AUDIO_DIR = _HERE / "voicebench_audio"

# sd-qa splits (accent-diverse)
SDQA_SPLITS = ["usa", "gbr", "ind_n", "ind_s", "irl", "kenya", "nga", "nzl", "phl", "zaf"]


def _extract_audio(raw) -> Optional[tuple]:
    """Extract (numpy_array, sample_rate) from whatever the datasets lib returns.

    Handles:
      - dict {'array': np.ndarray, 'sampling_rate': int}  (standard)
      - AudioDecoder objects (torchcodec, newer datasets versions)
      - raw bytes / memoryview
    """
    import numpy as np

    if isinstance(raw, dict):
        arr = raw.get("array") or raw.get("data")
        sr  = raw.get("sampling_rate") or raw.get("sample_rate") or 16_000
        if arr is not None:
            return np.asarray(arr, dtype=np.float32), int(sr)

    # torchcodec AudioDecoder (datasets >= 3.x)
    cls_name = type(raw).__name__
    if "AudioDecoder" in cls_name or "Audio" in cls_name:
        try:
            frames = raw.get_all_samples()   # returns AudioSamples
            # attribute is .data (torchcodec), shape (channels, T)
            arr = frames.data.numpy()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            return arr.astype(np.float32), int(frames.sample_rate)
        except Exception:
            pass

    # bytes fallback — try soundfile
    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            import io, soundfile as sf
            arr, sr = sf.read(io.BytesIO(bytes(raw)), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            return arr, sr
        except Exception:
            pass

    return None


def load_config_split(
    config: str,
    split: str,
    n_samples: int,
    audio_dir: Path,
    uid_prefix: str,
) -> List[Dict]:
    """Download one config/split from hlt-lab/voicebench and return manifest entries."""
    from datasets import load_dataset  # type: ignore
    import soundfile as sf
    import numpy as np

    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading hlt-lab/voicebench  config={config!r}  split={split!r} …")
    try:
        ds = load_dataset("hlt-lab/voicebench", config, split=split)
    except Exception as e:
        print(f"  ERROR loading {config}/{split}: {e}")
        return []

    entries: List[Dict] = []
    for i, sample in enumerate(ds):
        if len(entries) >= n_samples:
            break

        uid = f"VB_{uid_prefix}_{i:04d}"
        prompt = sample.get("prompt") or sample.get("text") or sample.get("question") or ""

        # Extract audio
        raw_audio = sample.get("audio") or sample.get("speech")
        if raw_audio is None:
            continue

        result = _extract_audio(raw_audio)
        if result is None:
            print(f"  Warning: could not decode audio for {uid} — skipping")
            continue
        arr, sr = result

        audio_path = audio_dir / f"{uid}.wav"
        try:
            sf.write(str(audio_path), arr, sr, subtype="PCM_16")
        except Exception as exc:
            print(f"  Warning: could not save {uid}: {exc}")
            continue

        entries.append({
            "id": uid,
            "audio_in_path": str(audio_path),
            "reference_text": prompt.strip(),
            "speaker_id": sample.get("speaker_id"),
            "dataset": "voicebench",
            "category": config,
            "emotion": None,
            "difficulty": "standard",
            "instruction": "Answer the spoken question clearly and concisely.",
        })

    print(f"  ✓ {len(entries)} entries from {config}/{split}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VoiceBench S2S manifest")
    parser.add_argument(
        "--config", nargs="+",
        default=["commoneval"],
        choices=["commoneval", "advbench", "alpacaeval", "mmsu", "mtbench",
                 "bbh", "ifeval", "openbookqa", "wildvoice", "sd-qa"],
        help="VoiceBench sub-benchmark(s) to download",
    )
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Max samples per config")
    parser.add_argument("--out-dir", default=str(_MANIFESTS_DIR))
    parser.add_argument("--audio-dir", default=str(_AUDIO_DIR))
    parser.add_argument(
        "--sdqa-splits", nargs="+", default=["usa"],
        choices=SDQA_SPLITS,
        help="Which accent splits to use for sd-qa config",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir)

    all_entries: List[Dict] = []

    for config in args.config:
        if config == "sd-qa":
            for accent in args.sdqa_splits:
                entries = load_config_split(
                    config="sd-qa",
                    split=accent,
                    n_samples=args.n_samples,
                    audio_dir=audio_dir,
                    uid_prefix=f"SDQA_{accent.upper()}",
                )
                all_entries.extend(entries)
        else:
            entries = load_config_split(
                config=config,
                split="test",
                n_samples=args.n_samples,
                audio_dir=audio_dir,
                uid_prefix=config.upper(),
            )
            all_entries.extend(entries)

    if not all_entries:
        print("No entries collected — exiting.")
        sys.exit(1)

    # Write one manifest per config (and a combined one if multiple)
    if len(args.config) == 1:
        cfg_name = args.config[0].replace("-", "_")
        out_path = out_dir / f"voicebench_{cfg_name}.json"
        with open(out_path, "w") as f:
            json.dump(all_entries, f, indent=2)
        print(f"\n✓ Manifest → {out_path}  ({len(all_entries)} entries)")
    else:
        # Combined
        out_path = out_dir / "voicebench.json"
        with open(out_path, "w") as f:
            json.dump(all_entries, f, indent=2)
        print(f"\n✓ Combined manifest → {out_path}  ({len(all_entries)} entries)")

        # Also write per-config
        from collections import defaultdict
        by_cat: dict = defaultdict(list)
        for e in all_entries:
            by_cat[e["category"]].append(e)
        for cat, items in by_cat.items():
            p = out_dir / f"voicebench_{cat.replace('-','_')}.json"
            with open(p, "w") as f:
                json.dump(items, f, indent=2)
            print(f"  {cat}: {len(items)} entries → {p}")

    print()
    print("Next step:")
    print(f"  python run_all.py --datasets voicebench_{args.config[0].replace('-','_')} --models cascaded_groq_deepgram ultravox")


if __name__ == "__main__":
    main()
