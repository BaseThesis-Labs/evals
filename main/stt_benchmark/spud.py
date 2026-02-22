#!/usr/bin/env python3
"""
spud.py — Speech Performance Under Degradation (SPUD).

Applies 6 additive-noise degradation levels (pure AWGN) to a 200-utterance
subset of LibriSpeech test-clean, transcribes each version with all enabled
models, computes SeMaScore per level, and reports AUC (area under the curve).

Degradation levels (Additive White Gaussian Noise, SNR in dB):
  0 → clean  (no noise,  SNR = ∞)
  1 → 40 dB  (barely perceptible)
  2 → 30 dB  (light noise)
  3 → 20 dB  (moderate noise)
  4 → 10 dB  (heavy noise)
  5 →  5 dB  (very heavy noise)

NOTE — Extended SPUD (not yet implemented):
  A future version will add realistic channel degradations at higher levels:
    Level 3-ext : AWGN (SNR=20 dB) + synthetic room impulse response (reverb)
    Level 4-ext : VoIP simulation — downsample to 8 kHz, Opus codec (16 kbps),
                  resample to 16 kHz, AWGN (SNR=10 dB)
    Level 5-ext : AWGN (SNR=5 dB) + reverb + codec compression
  These require pyroomacoustics (reverb) and ffmpeg/opuslib (codec).

Usage:
    python spud.py \\
        --dataset datasets/librispeech/test-clean_manifest.jsonl \\
        --n-utterances 200 \\
        --output-dir analysis/spud

Output:
    analysis/spud/spud_results.json   — SeMaScore per model per level + AUC
    analysis/spud/spud_summary.csv    — one row per model with AUC
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import click
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_evaluation_config, load_models_config
from src.data.loader import load_manifest, validate_samples
from src.evaluation.normalizer import TranscriptNormalizer
from src.evaluation.semascore import compute_semascore
from src.models.factory import create_all_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


# ── SNR levels ────────────────────────────────────────────────────────────────

DEGRADATION_LEVELS = [
    {"level": 0, "snr_db": float("inf"), "label": "clean"},
    {"level": 1, "snr_db": 40,           "label": "snr40"},
    {"level": 2, "snr_db": 30,           "label": "snr30"},
    {"level": 3, "snr_db": 20,           "label": "snr20"},
    {"level": 4, "snr_db": 10,           "label": "snr10"},
    {"level": 5, "snr_db":  5,           "label": "snr05"},
]


def add_awgn(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Additive White Gaussian Noise at target SNR (dB)."""
    if not np.isfinite(snr_db):
        return audio.copy()
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio.copy()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), size=audio.shape)
    degraded = audio + noise
    # Clip to [-1, 1] without normalising (preserves relative loudness)
    return np.clip(degraded, -1.0, 1.0).astype(audio.dtype)


def degrade_audio(src_path: str, snr_db: float, dst_path: str, rng: np.random.Generator) -> None:
    """Load audio, add noise, save to dst_path."""
    audio, sr = sf.read(src_path, dtype="float32")
    degraded = add_awgn(audio, snr_db, rng)
    sf.write(dst_path, degraded, sr)


# ── SeMaScore computation ─────────────────────────────────────────────────────

def compute_semascore_batch(
    refs: list[str],
    hyps: list[str],
    encoder,
) -> float:
    """Mean SeMaScore over a list of (ref, hyp) pairs."""
    import jiwer
    scores = []
    for ref, hyp in zip(refs, hyps):
        scores.append(compute_semascore(ref, hyp, encoder))
    return float(np.mean(scores)) if scores else float("nan")


# ── SPUD AUC ──────────────────────────────────────────────────────────────────

def compute_spud_auc(semascore_per_level: list[float]) -> float:
    """
    Area under the SeMaScore-vs-degradation curve, normalised to [0, 1].

    x-axis: degradation level index (0 = clean … 5 = heaviest)
    y-axis: mean SeMaScore
    AUC computed via trapezoidal rule, normalised by maximum possible area
    (all levels = 1.0 over 5 intervals).
    """
    valid = [v for v in semascore_per_level if np.isfinite(v)]
    if len(valid) < 2:
        return float("nan")
    x = np.arange(len(semascore_per_level), dtype=float)
    y = np.array(semascore_per_level, dtype=float)
    # Replace NaN with 0 for integration
    y = np.where(np.isfinite(y), y, 0.0)
    trapz = getattr(np, "trapezoid", np.trapz)  # numpy ≥2.0 renamed trapz → trapezoid
    auc = float(trapz(y, x))
    max_auc = float(x[-1] - x[0])   # all y=1.0 → area = n_levels - 1
    return auc / max_auc if max_auc > 0 else float("nan")


# ── Main pipeline ─────────────────────────────────────────────────────────────

@click.command()
@click.option("--dataset",       "-d", required=True,
              help="Path to manifest.jsonl (test-clean recommended)")
@click.option("--n-utterances",  "-n", default=200, show_default=True,
              help="Number of utterances to evaluate per level")
@click.option("--output-dir",    "-o", default="analysis/spud", show_default=True)
@click.option("--models-config",       default="configs/models.yaml",    show_default=True)
@click.option("--eval-config",         default="configs/evaluation.yaml", show_default=True)
@click.option("--models", "-m",  default=None,
              help="Comma-separated model names (default: all enabled)")
@click.option("--seed",                default=42, show_default=True)
def main(dataset, n_utterances, output_dir, models_config, eval_config, models, seed):
    """
    SPUD: Speech Performance Under Degradation.

    Transcribes N utterances at 6 noise levels, computes SeMaScore per level
    and AUC to measure robustness under acoustic degradation.
    """
    rng = np.random.default_rng(seed)

    eval_cfg   = load_evaluation_config(eval_config)
    models_cfg = load_models_config(models_config)
    normalizer = TranscriptNormalizer(eval_cfg.normalization.model_dump())

    model_filter = [m.strip() for m in models.split(",")] if models else None
    model_dicts  = [
        m.model_dump(by_alias=True)
        for m in models_cfg.enabled_models()
        if model_filter is None or m.name in model_filter
    ]
    all_models = create_all_models(model_dicts)
    if not all_models:
        log.error("No models loaded — check API keys.")
        return

    # Load and truncate dataset
    samples = validate_samples(load_manifest(dataset))[:n_utterances]
    log.info(f"SPUD: {len(samples)} utterances × {len(DEGRADATION_LEVELS)} levels "
             f"× {len(all_models)} models = "
             f"{len(samples) * len(DEGRADATION_LEVELS) * len(all_models)} transcriptions")

    # Lazy-load semantic encoder (shared across all levels)
    from src.evaluation.semantic_metrics import _get_st_model
    encoder = _get_st_model(eval_cfg.metrics.semantic.semdist_model)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="spud_") as tmpdir:
        tmpdir = Path(tmpdir)

        for model in all_models:
            log.info(f"▶  {model.name}")
            model_results = {"semascore_per_level": [], "levels": []}

            for deg in DEGRADATION_LEVELS:
                level    = deg["level"]
                snr_db   = deg["snr_db"]
                label    = deg["label"]
                log.info(f"   Level {level} ({label}, SNR={snr_db} dB)…")

                refs: list[str] = []
                hyps: list[str] = []

                for sample in tqdm(samples, desc=f"{model.name}/{label}", leave=False):
                    # Degrade audio
                    if np.isfinite(snr_db):
                        deg_path = str(tmpdir / f"{Path(sample.audio_filepath).stem}_{label}.wav")
                        degrade_audio(sample.audio_filepath, snr_db, deg_path, rng)
                        audio_path = deg_path
                    else:
                        audio_path = sample.audio_filepath  # clean

                    try:
                        result = model.transcribe(audio_path, language=sample.lang or "en")
                        hyp = result.text or ""
                    except Exception as e:
                        log.warning(f"  Transcription error: {e}")
                        hyp = ""

                    refs.append(normalizer.normalize(sample.text or ""))
                    hyps.append(normalizer.normalize(hyp))

                mean_sema = compute_semascore_batch(refs, hyps, encoder)
                log.info(f"   Level {level}: SeMaScore = {mean_sema:.4f}")

                model_results["semascore_per_level"].append(mean_sema)
                model_results["levels"].append({
                    "level":      level,
                    "label":      label,
                    "snr_db":     snr_db if np.isfinite(snr_db) else None,
                    "semascore":  mean_sema,
                    "n_samples":  len(refs),
                })

            spud_auc = compute_spud_auc(model_results["semascore_per_level"])
            model_results["spud_auc"] = spud_auc
            log.info(f"   SPUD AUC = {spud_auc:.4f}")
            results[model.name] = model_results

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)   # ensure dir exists before writing
    json_path = out_dir / "spud_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    log.info(f"Results saved: {json_path}")

    # ── Save summary CSV ───────────────────────────────────────────────────────
    csv_path = out_dir / "spud_summary.csv"
    cols = ["model", "spud_auc"] + [d["label"] for d in DEGRADATION_LEVELS]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for model_name, res in sorted(results.items(), key=lambda x: -x[1].get("spud_auc", 0)):
            row = {"model": model_name, "spud_auc": f"{res['spud_auc']:.4f}"}
            for i, deg in enumerate(DEGRADATION_LEVELS):
                sema = res["semascore_per_level"][i] if i < len(res["semascore_per_level"]) else ""
                row[deg["label"]] = f"{sema:.4f}" if np.isfinite(sema) else ""
            writer.writerow(row)
    log.info(f"Summary CSV saved: {csv_path}")

    # ── Print table ────────────────────────────────────────────────────────────
    from rich.table import Table
    from rich.console import Console

    table = Table(title="SPUD Results", show_lines=True)
    table.add_column("Model")
    table.add_column("AUC")
    for deg in DEGRADATION_LEVELS:
        table.add_column(deg["label"])

    for model_name, res in sorted(results.items(), key=lambda x: -x[1].get("spud_auc", 0)):
        row_vals = [
            model_name,
            f"{res['spud_auc']:.4f}",
        ]
        for i, _ in enumerate(DEGRADATION_LEVELS):
            sema = res["semascore_per_level"][i] if i < len(res["semascore_per_level"]) else float("nan")
            row_vals.append(f"{sema:.4f}" if np.isfinite(sema) else "—")
        table.add_row(*row_vals)

    Console().print(table)
    log.info("\nDone. Next: python visualize.py --analysis-dir analysis/spud")


if __name__ == "__main__":
    main()
