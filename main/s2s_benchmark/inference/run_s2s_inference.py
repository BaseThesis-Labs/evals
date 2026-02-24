#!/usr/bin/env python3
"""
Run S2S inference for all enabled models.

Usage:
    python inference/run_s2s_inference.py \
        --manifest datasets/manifests/s2s_manifest.json \
        --output   s2s_outputs \
        [--model   cascaded_elevenlabs] \
        [--config  config/eval_config.yaml]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from tqdm import tqdm

# ── Ensure s2s_benchmark root is importable ───────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference.adapters.base import S2SResult  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Config / manifest helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest(path: str) -> List[Dict]:
    """Load JSON or JSONL manifest."""
    with open(path) as f:
        first_char = f.read(1)
    with open(path) as f:
        if first_char == "[":
            return json.load(f)
        # JSONL
        return [json.loads(line) for line in f if line.strip()]


def get_adapter(model_name: str, model_cfg: Dict):
    """Dynamically instantiate an S2S adapter from its class_path."""
    class_path: str = model_cfg["class_path"]
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    config = model_cfg.get("config", {})
    return cls(model_name=model_name, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# Core inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model_name: str,
    model_cfg: Dict,
    manifest: List[Dict],
    output_root: Path,
    skip_existing: bool = True,
) -> None:
    print(f"\n▶ Running S2S inference: {model_name}")

    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    meta_path = model_dir / "gen_meta.jsonl"

    # Build set of already-processed IDs for fast skip-check
    done_ids: set[str] = set()
    if skip_existing and meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("error") is None:
                        done_ids.add(rec["id"])
                except json.JSONDecodeError:
                    pass

    # Rate limiting
    rate_limit_rps: float = model_cfg.get("rate_limit_rps", 0)
    min_interval = 1.0 / rate_limit_rps if rate_limit_rps > 0 else 0.0
    last_request_t = 0.0

    # Initialise adapter
    try:
        adapter = get_adapter(model_name, model_cfg)
    except Exception as exc:
        print(f"✗ Failed to initialise {model_name}: {exc}")
        return

    meta_file = open(meta_path, "a")

    try:
        for entry in tqdm(manifest, desc=model_name):
            utt_id: str = entry["id"]
            audio_in: str = entry.get("audio_in_path", "")
            ref_text: str = entry.get("reference_text", "")

            if skip_existing and utt_id in done_ids:
                continue

            # Enforce rate limit
            if min_interval > 0:
                elapsed = time.time() - last_request_t
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

            try:
                result: S2SResult = adapter.process(
                    audio_in_path=audio_in,
                    reference_text=ref_text,
                    utterance_id=utt_id,
                    output_dir=str(model_dir),
                )
                last_request_t = time.time()

                meta = {
                    "id": utt_id,
                    "audio_out_path": result.audio_out_path,
                    "asr_transcript": result.asr_transcript,
                    "ttfb_ms": result.ttfb_ms,
                    "e2e_latency_ms": result.e2e_latency_ms,
                    "asr_latency_ms": result.asr_latency_ms,
                    "tts_latency_ms": result.tts_latency_ms,
                    "rtf": result.rtf,
                    "error": result.error,
                }

            except Exception as exc:
                last_request_t = time.time()
                print(f"  ✗ Error on {utt_id}: {exc}")
                meta = {
                    "id": utt_id,
                    "audio_out_path": None,
                    "asr_transcript": None,
                    "ttfb_ms": None,
                    "e2e_latency_ms": None,
                    "asr_latency_ms": None,
                    "tts_latency_ms": None,
                    "rtf": None,
                    "error": str(exc),
                }

            meta_file.write(json.dumps(meta) + "\n")
            meta_file.flush()

    finally:
        meta_file.close()
        try:
            adapter.cleanup()
        except Exception:
            pass

    print(f"  ✓ {model_name} done — metadata at {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="S2S inference runner")
    parser.add_argument(
        "--manifest",
        default="datasets/manifests/s2s_manifest.json",
        help="Path to S2S manifest (JSON or JSONL)",
    )
    parser.add_argument(
        "--output", default="s2s_outputs", help="Root output directory"
    )
    parser.add_argument(
        "--config",
        default="config/eval_config.yaml",
        help="Evaluation config YAML",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Run a single model (default: all enabled)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-generate even if output already exists",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = load_manifest(args.manifest)
    print(f"✓ Loaded {len(manifest)} utterances from {args.manifest}")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    models_cfg: Dict = cfg.get("models", {})
    skip_existing = not args.no_skip

    if args.model:
        if args.model not in models_cfg:
            print(f"✗ Model '{args.model}' not found in config")
            sys.exit(1)
        if not models_cfg[args.model].get("enabled", True):
            print(f"✗ Model '{args.model}' is disabled in config")
            sys.exit(1)
        run_inference(args.model, models_cfg[args.model], manifest, output_root, skip_existing)
    else:
        for mname, mcfg in models_cfg.items():
            if not mcfg.get("enabled", True):
                print(f"  ⊘ Skipping disabled model: {mname}")
                continue
            run_inference(mname, mcfg, manifest, output_root, skip_existing)

    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
