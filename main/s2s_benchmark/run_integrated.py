#!/usr/bin/env python3
"""
Integrated S2S inference + evaluation — disk-efficient mode.

Generates one utterance, immediately computes all metrics, saves only the
numbers to JSONL, then deletes the WAV. Peak disk use = 1 WAV at a time.

Usage:
    python run_integrated.py \
        --manifest datasets/manifests/libritts_r.json \
        --model    cascaded_deepgram \
        --results  results/libritts_r \
        [--config  config/eval_config.yaml]
"""
from __future__ import annotations

import argparse
import importlib
import json
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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_STT_BENCH = _ROOT.parent / "stt_benchmark"
if _STT_BENCH.is_dir() and str(_STT_BENCH) not in sys.path:
    sys.path.insert(0, str(_STT_BENCH))

from inference.adapters.base import S2SResult          # noqa: E402
from pipeline import evaluate_utterance, load_manifest  # noqa: E402
from scoring.aggregate import aggregate_model, build_leaderboard  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_adapter(model_name: str, model_cfg: Dict):
    class_path: str = model_cfg["class_path"]
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(model_name=model_name, config=model_cfg.get("config", {}))


# ─────────────────────────────────────────────────────────────────────────────
# Core loop
# ─────────────────────────────────────────────────────────────────────────────

def run(
    model_name: str,
    model_cfg: Dict,
    manifest: List[Dict],
    results_dir: Path,
    cfg_metrics: Dict,
    skip_existing: bool = True,
) -> Optional[Dict]:
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = results_dir / f"{model_name}_utterances.jsonl"
    agg_path      = results_dir / f"{model_name}_metrics.json"

    # Load already-completed IDs so we can skip on resume
    done_ids: set[str] = set()
    utterance_results: List[Dict] = []
    if skip_existing and metrics_jsonl.exists():
        with open(metrics_jsonl) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if not rec.get("error"):
                        done_ids.add(rec["id"])
                    utterance_results.append(rec)
                except json.JSONDecodeError:
                    pass
        print(f"  ↩ Resuming — {len(done_ids)} utterances already done")

    # Rate limiting
    rate_limit_rps: float = model_cfg.get("rate_limit_rps", 0)
    min_interval = 1.0 / rate_limit_rps if rate_limit_rps > 0 else 0.0
    last_t = 0.0

    # Initialise adapter
    try:
        adapter = get_adapter(model_name, model_cfg)
    except Exception as exc:
        print(f"✗ Failed to initialise {model_name}: {exc}")
        return None

    print(f"\n▶ {model_name}  ({len(manifest)} utterances, skip_audio=True)")

    metrics_file = open(metrics_jsonl, "a")

    # Use a single temp directory for WAV files — cleaned up automatically
    with tempfile.TemporaryDirectory(prefix="s2s_wav_") as tmp_wav_dir:
        try:
            for entry in tqdm(manifest, desc=model_name):
                utt_id: str = entry["id"]
                if skip_existing and utt_id in done_ids:
                    continue

                audio_in = entry.get("audio_in_path", "")
                ref_text = entry.get("reference_text", "")

                # Rate limit
                if min_interval > 0:
                    elapsed = time.time() - last_t
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)

                # ── Generate ─────────────────────────────────────────────────
                try:
                    result: S2SResult = adapter.process(
                        audio_in_path=audio_in,
                        reference_text=ref_text,
                        utterance_id=utt_id,
                        output_dir=tmp_wav_dir,
                    )
                    last_t = time.time()
                except Exception as exc:
                    last_t = time.time()
                    rec = {"id": utt_id, "error": str(exc)}
                    utterance_results.append(rec)
                    metrics_file.write(json.dumps(rec) + "\n")
                    metrics_file.flush()
                    continue

                if result.error:
                    rec = {"id": utt_id, "error": result.error}
                    utterance_results.append(rec)
                    metrics_file.write(json.dumps(rec) + "\n")
                    metrics_file.flush()
                    continue

                # ── Evaluate immediately ──────────────────────────────────────
                gen_meta = {
                    "id": utt_id,
                    "audio_out_path": result.audio_out_path,
                    "asr_transcript": result.asr_transcript,
                    "ttfb_ms": result.ttfb_ms,
                    "e2e_latency_ms": result.e2e_latency_ms,
                    "asr_latency_ms": result.asr_latency_ms,
                    "tts_latency_ms": result.tts_latency_ms,
                    "rtf": result.rtf,
                    "error": None,
                }

                try:
                    metrics = evaluate_utterance(entry, gen_meta, Path(tmp_wav_dir), cfg_metrics)
                except Exception as exc:
                    metrics = {"id": utt_id, "error": f"eval_failed: {exc}"}

                # Copy latency fields that aren't in evaluate_utterance output
                for k in ("e2e_latency_ms", "asr_latency_ms", "tts_latency_ms", "rtf", "ttfb_ms"):
                    if k not in metrics:
                        metrics[k] = gen_meta.get(k)

                utterance_results.append(metrics)
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()

                # ── Delete WAV immediately ────────────────────────────────────
                wav = Path(result.audio_out_path)
                if wav.exists():
                    wav.unlink()

        finally:
            metrics_file.close()
            try:
                adapter.cleanup()
            except Exception:
                pass

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if not utterance_results:
        print(f"  ⚠ No results for {model_name}")
        return None

    _mt = model_cfg.get("model_type", "echo")
    agg = aggregate_model(utterance_results, model_type=_mt)
    output = {"model": model_name, "utterances": utterance_results, "aggregate": agg}
    with open(agg_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ {model_name} → {agg_path}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated S2S inference + eval (disk-efficient)")
    parser.add_argument("--manifest", default="datasets/manifests/s2s_manifest.json")
    parser.add_argument("--model",    default=None, help="Single model (default: all enabled)")
    parser.add_argument("--results",  default="results", help="Output directory for metrics JSON")
    parser.add_argument("--config",   default="config/eval_config.yaml")
    parser.add_argument("--no-skip",  action="store_true", help="Re-run even if already done")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    cfg_metrics = cfg.get("metrics", {})
    manifest    = load_manifest(args.manifest)
    print(f"✓ Loaded {len(manifest)} utterances from {args.manifest}")

    models_cfg: Dict = cfg.get("models", {})
    results_dir = Path(args.results)
    skip_existing = not args.no_skip

    if args.model:
        if args.model not in models_cfg:
            print(f"✗ Model '{args.model}' not in config"); sys.exit(1)
        model_names = [args.model]
    else:
        model_names = [n for n, c in models_cfg.items() if c.get("enabled", True)]

    model_aggregates: Dict[str, Dict] = {}
    for mname in model_names:
        agg = run(mname, models_cfg[mname], manifest, results_dir, cfg_metrics, skip_existing)
        if agg is not None:
            model_aggregates[mname] = agg

    if not model_aggregates:
        print("No model results to aggregate.")
        return

    # Leaderboards
    for use_case in ["balanced", "conversational", "audiobook", "voice_cloning", "expressive"]:
        lb = build_leaderboard(model_aggregates, use_case=use_case)
        with open(results_dir / f"leaderboard_{use_case}.json", "w") as f:
            json.dump(lb, f, indent=2)

    with open(results_dir / "leaderboard.json", "w") as f:
        json.dump(build_leaderboard(model_aggregates, "balanced"), f, indent=2)

    print(f"\n✓ Done. Results → {results_dir}")


if __name__ == "__main__":
    main()
