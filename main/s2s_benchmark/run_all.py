#!/usr/bin/env python3
"""
run_all.py — run every enabled model on every dataset manifest.

Loads API keys from sibling .env files automatically.
Skips already-completed utterances (resumable).
Logs per-combination status to run_all.log.

Usage:
    python run_all.py [--datasets libritts_r tess ...] [--models cascaded_deepgram ...]
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_EVALS = _ROOT.parent
sys.path.insert(0, str(_ROOT))
_STT = _EVALS / "stt_benchmark"
if _STT.is_dir():
    sys.path.insert(0, str(_STT))

from inference.adapters.base import S2SResult
from pipeline import evaluate_utterance, evaluate_multiturn, load_manifest
from scoring.aggregate import (
    aggregate_model,
    build_leaderboard,
    build_split_leaderboards,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_ROOT / "run_all.log", mode="a"),
    ],
)
log = logging.getLogger("run_all")


# ─────────────────────────────────────────────────────────────────────────────
# Load API keys from sibling .env files
# ─────────────────────────────────────────────────────────────────────────────

def load_env_keys() -> None:
    """Source all sibling benchmark .env files to populate os.environ."""
    env_files = [
        _EVALS / "tts_benchmark" / ".env",
        _EVALS / "stt_benchmark" / ".env",
        _ROOT / ".env",
    ]
    for ef in env_files:
        if not ef.exists():
            continue
        with open(ef) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val

    # Report what we have
    keys_to_check = [
        "DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "ELEVENLABS_API_KEY",
        "GROQ_API_KEY", "ULTRAVOX_API_KEY", "GEMINI_API_KEY",
        "GOOGLE_API_KEY", "OPENAI_API_KEY",
    ]
    log.info("API key status:")
    for k in keys_to_check:
        v = os.environ.get(k, "")
        log.info(f"  {'✓' if v else '✗'}  {k}")


# ─────────────────────────────────────────────────────────────────────────────
# Adapter loader
# ─────────────────────────────────────────────────────────────────────────────

def get_adapter(model_name: str, model_cfg: Dict):
    class_path = model_cfg["class_path"]
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(model_name=model_name, config=model_cfg.get("config", {}))


# ─────────────────────────────────────────────────────────────────────────────
# Single model × dataset run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    model_name: str,
    model_cfg: Dict,
    manifest: List[Dict],
    results_dir: Path,
    cfg_metrics: Dict,
) -> Dict:
    """Run inference + eval for one model on one dataset. Returns aggregate."""
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = results_dir / f"{model_name}_utterances.jsonl"

    # Resume: load already-done IDs
    done_ids: set = set()
    utterance_results: List[Dict] = []
    if metrics_jsonl.exists():
        with open(metrics_jsonl) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec["id"])
                    utterance_results.append(rec)
                except Exception:
                    pass

    remaining = [e for e in manifest if e["id"] not in done_ids]
    log.info(f"  {model_name}: {len(done_ids)} done, {len(remaining)} remaining")

    _mt = model_cfg.get("model_type", "echo")
    if not remaining:
        log.info(f"  {model_name}: already complete — skipping inference")
        agg = aggregate_model(utterance_results, model_type=_mt)
        return agg

    # Load adapter
    try:
        adapter = get_adapter(model_name, model_cfg)
    except Exception as exc:
        log.error(f"  {model_name}: failed to load adapter: {exc}")
        return {}

    with open(metrics_jsonl, "a") as mf:
        with tempfile.TemporaryDirectory(prefix="s2s_wav_") as tmp_wav_dir:
            for entry in tqdm(remaining, desc=f"{model_name}", unit="utt", leave=False):
                utt_id = entry["id"]
                t0 = time.perf_counter()

                # ── Inference ──────────────────────────────────────────────
                try:
                    result: S2SResult = adapter.process(
                        audio_in_path=entry["audio_in_path"],
                        reference_text=entry.get("reference_text", ""),
                        utterance_id=utt_id,
                        output_dir=tmp_wav_dir,
                    )
                except Exception as exc:
                    log.warning(f"    {utt_id}: inference exception: {exc}")
                    rec = {"id": utt_id, "error": str(exc)}
                    mf.write(json.dumps(rec) + "\n")
                    mf.flush()
                    utterance_results.append(rec)
                    continue

                gen_meta = {
                    "id": utt_id,
                    "audio_out_path": result.audio_out_path,
                    "asr_transcript": result.asr_transcript,
                    "ttfb_ms": result.ttfb_ms,
                    "e2e_latency_ms": result.e2e_latency_ms,
                    "asr_latency_ms": result.asr_latency_ms if hasattr(result, "asr_latency_ms") else None,
                    "tts_latency_ms": result.tts_latency_ms if hasattr(result, "tts_latency_ms") else None,
                    "rtf": result.rtf,
                    "error": result.error,
                    "sample_rate": result.sample_rate,
                    "model_type": model_cfg.get("model_type", "echo"),
                }

                if result.error:
                    log.warning(f"    {utt_id}: model error: {result.error}")
                    rec = {**gen_meta}
                    mf.write(json.dumps(rec, default=str) + "\n")
                    mf.flush()
                    utterance_results.append(rec)
                    wav = Path(result.audio_out_path) if result.audio_out_path else None
                    if wav and wav.exists():
                        wav.unlink()
                    continue

                # ── Evaluate ───────────────────────────────────────────────
                try:
                    metrics = evaluate_utterance(
                        entry, gen_meta, Path(tmp_wav_dir), cfg_metrics
                    )
                except Exception as exc:
                    log.warning(f"    {utt_id}: eval exception: {exc}")
                    metrics = {**gen_meta, "eval_error": str(exc)}

                # ── Delete WAV immediately ─────────────────────────────────
                wav = Path(result.audio_out_path) if result.audio_out_path else None
                if wav and wav.exists():
                    try:
                        wav.unlink()
                    except OSError:
                        pass

                elapsed = (time.perf_counter() - t0) * 1000
                metrics["wall_ms"] = round(elapsed, 1)
                mf.write(json.dumps(metrics, default=str) + "\n")
                mf.flush()
                utterance_results.append(metrics)

    try:
        adapter.cleanup()
    except Exception:
        pass

    agg = aggregate_model(utterance_results, model_type=_mt)
    agg_path = results_dir / f"{model_name}_metrics.json"
    with open(agg_path, "w") as f:
        json.dump({"model": model_name, "aggregate": agg}, f, indent=2)

    n_ok = agg.get("n_utterances", 0)
    n_err = agg.get("n_errors", 0)
    log.info(f"  ✓ {model_name}: {n_ok} ok, {n_err} errors")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/eval_config.yaml")
    parser.add_argument("--manifests-dir", default="datasets/manifests")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset names (default: all manifests in manifests-dir)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model names to run (default: all enabled in config)")
    parser.add_argument("--skip-models", nargs="*", default=[],
                        help="Model names to skip")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max utterances per dataset (default: all)")
    parser.add_argument("--mode", choices=["single", "multiturn", "both"],
                        default="single",
                        help="Evaluation mode: single-turn, multiturn agent, or both")
    args = parser.parse_args()

    load_env_keys()

    cfg = yaml.safe_load(open(args.config))
    cfg_metrics = cfg.get("metrics", {})
    models_cfg: Dict[str, Dict] = cfg.get("models", {})

    # Which models to run
    if args.models:
        model_names = args.models
    else:
        model_names = [
            name for name, mcfg in models_cfg.items()
            if mcfg.get("enabled", True) and name not in args.skip_models
        ]
    log.info(f"Models to run: {model_names}")

    # Which datasets to run
    manifests_dir = Path(args.manifests_dir)
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = [
            p.stem for p in sorted(manifests_dir.glob("*.json"))
            if p.stem not in ("s2s_manifest",)  # skip combined manifest
        ]
    log.info(f"Datasets: {dataset_names}")

    results_root = Path(args.results_root)

    # ── Single-turn inference + eval ──────────────────────────────────────────
    if args.mode in ("single", "both"):
        pass  # fall through to existing loop below

    grand_total = len(model_names) * len(dataset_names) if args.mode != "multiturn" else 0
    done = 0

    for ds_name in (dataset_names if args.mode != "multiturn" else []):
        manifest_path = manifests_dir / f"{ds_name}.json"
        if not manifest_path.exists():
            log.warning(f"Manifest not found: {manifest_path} — skipping")
            continue

        manifest = load_manifest(str(manifest_path))
        if args.limit:
            manifest = manifest[:args.limit]
        log.info(f"\n{'='*60}")
        log.info(f"Dataset: {ds_name}  ({len(manifest)} utterances)")
        log.info(f"{'='*60}")

        ds_results_dir = results_root / ds_name
        ds_aggregates: Dict[str, Dict] = {}

        for model_name in model_names:
            if model_name not in models_cfg:
                log.warning(f"  {model_name}: not in config — skipping")
                continue

            log.info(f"\n  [{done+1}/{grand_total}] {model_name} × {ds_name}")
            model_cfg = models_cfg[model_name]

            try:
                agg = run_one(
                    model_name=model_name,
                    model_cfg=model_cfg,
                    manifest=manifest,
                    results_dir=ds_results_dir,
                    cfg_metrics=cfg_metrics,
                )
                if agg:
                    ds_aggregates[model_name] = agg
            except Exception as exc:
                log.error(f"  {model_name} × {ds_name} FAILED: {exc}", exc_info=True)

            done += 1

        # Per-dataset leaderboard — split by model type (echo vs generative)
        if ds_aggregates:
            # Build model_type_map from config
            model_type_map = {
                name: models_cfg.get(name, {}).get("model_type", "echo")
                for name in ds_aggregates
            }

            for use_case in ["balanced", "conversational", "audiobook", "voice_cloning", "expressive"]:
                # Split leaderboards: echo, generative, combined
                split_lbs = build_split_leaderboards(
                    ds_aggregates, model_type_map, use_case=use_case,
                )
                for split_name, lb in split_lbs.items():
                    lb_path = ds_results_dir / f"leaderboard_{use_case}_{split_name}.json"
                    with open(lb_path, "w") as f:
                        json.dump(lb, f, indent=2)

                # Also write a backwards-compatible combined leaderboard
                lb = build_leaderboard(ds_aggregates, use_case=use_case, model_type_map=model_type_map)
                lb_path = ds_results_dir / f"leaderboard_{use_case}.json"
                with open(lb_path, "w") as f:
                    json.dump(lb, f, indent=2)

            log.info(f"\n  ✓ Leaderboards written to {ds_results_dir}/")

    # ── Multi-turn agent evaluation ──────────────────────────────────────────
    if args.mode in ("multiturn", "both"):
        log.info(f"\n{'='*60}")
        log.info("Multi-turn agent evaluation")
        log.info(f"{'='*60}")

        mt_results_dir = results_root / "multiturn"
        mt_aggregates: Dict[str, Dict] = {}

        for model_name in model_names:
            if model_name not in models_cfg:
                continue
            log.info(f"\n  [multiturn] {model_name}")
            try:
                agg = evaluate_multiturn(model_name, models_cfg[model_name], cfg, mt_results_dir)
                if agg:
                    mt_aggregates[model_name] = agg
            except Exception as exc:
                log.error(f"  {model_name} multiturn FAILED: {exc}", exc_info=True)

        if mt_aggregates:
            lb_agent = build_leaderboard(mt_aggregates, use_case="agent")
            lb_path = mt_results_dir / "leaderboard_agent.json"
            mt_results_dir.mkdir(parents=True, exist_ok=True)
            with open(lb_path, "w") as f:
                json.dump(lb_agent, f, indent=2)
            log.info(f"  ✓ Agent leaderboard → {lb_path}")

    log.info(f"\n{'='*60}")
    log.info("All runs complete.")
    log.info(f"Results in: {results_root}/")


if __name__ == "__main__":
    main()
