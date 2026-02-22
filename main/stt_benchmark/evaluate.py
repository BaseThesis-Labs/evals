#!/usr/bin/env python3
"""
evaluate.py — Run all STT models on a dataset and compute every metric per sample.

Usage:
    python evaluate.py --dataset datasets/librispeech/test-clean_manifest.jsonl
    python evaluate.py --dataset datasets/librispeech/test-clean_manifest.jsonl --models groq-turbo,deepgram
    python evaluate.py --dataset datasets/silence_noise --hallucination-only
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import re

import click
import jiwer
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_evaluation_config, load_models_config
from src.data.loader import AudioSample, load_manifest, load_wav_directory, validate_samples
from src.evaluation.normalizer import TranscriptNormalizer
from src.evaluation.metrics import compute_sample_metrics
from src.evaluation.formatting_metrics import compute_fwer, compute_punctuation_f1, compute_capitalization_accuracy
from src.evaluation.hallucination import compute_hallucination_metrics
from src.evaluation.her import compute_her
from src.evaluation.advanced_metrics import (
    compute_bleu, compute_meteor, compute_per, compute_krr,
    compute_error_severity, compute_shallow, compute_llm_impact,
    compute_embedding_hallucination,
)
from src.evaluation.pner import compute_pner, compute_alphanumeric_accuracy
from src.evaluation.semascore import compute_semascore
from src.evaluation.snr import compute_snr_db
from src.models.factory import create_all_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)


# ── Text normalizers ──────────────────────────────────────────────────────────

def normalize_lcase(text: str) -> str:
    """Lowercase + strip punctuation. Does NOT expand numbers or contractions."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Minimal normalizer for semantic metrics: punctuation removal + lowercase only.
# Avoids filler/contraction expansion that can introduce spurious token changes.
try:
    _sem_norm_fn = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    def _sem_norm(text: str) -> str:
        return _sem_norm_fn(text)
except Exception:
    def _sem_norm(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


# ── Model key registry (same keys as generate.py TTS providers for cross-ref) ─

MODEL_KEYS = {
    "whisper-turbo":   "faster-whisper-large-v3-turbo",
    "whisper-large":   "faster-whisper-large-v3",
    "openai-transcribe": "openai-gpt-4o-transcribe",
    "openai-mini":     "openai-gpt-4o-mini-transcribe",
    "deepgram":        "deepgram-nova-3",
    "assemblyai":      "assemblyai-universal",
    "elevenlabs":      "elevenlabs-scribe_v2",
    "groq-turbo":      "groq-whisper-large-v3-turbo",
    "groq-distil":     "groq-distil-whisper-large-v3-en",
    "google":          "google-chirp_2",
}


# ── Per-sample evaluation ─────────────────────────────────────────────────────

def evaluate_sample(
    ref: str,
    hyp: str,
    normalizer: TranscriptNormalizer,
    inference_time_s: float,
    audio_duration_s: float,
    cost_usd: float | None,
    compute_formatting: bool = True,
    adv_cfg=None,
) -> dict:
    """Compute all per-sample metrics and return as a flat dict."""
    # Surface metrics on normalised text (primary WER)
    m = compute_sample_metrics(ref, hyp, normalizer)

    # Partial normalisation: lowercase + strip punctuation only (no filler removal,
    # no contraction expansion).  Meaningful across all datasets — avoids the ~99%
    # artefact that occurs when refs are ALL-CAPS (LibriSpeech) and truly-raw WER
    # counts every word as wrong due to casing alone.
    ref_lcase = normalize_lcase(ref)
    hyp_lcase = normalize_lcase(hyp)
    if ref_lcase:
        m["wer_lcase"] = float(jiwer.wer(ref_lcase, hyp_lcase))
        m["cer_lcase"] = float(jiwer.cer(ref_lcase, hyp_lcase))
    else:
        m["wer_lcase"] = float("nan")
        m["cer_lcase"] = float("nan")

    # RTF
    m["rtf"]  = inference_time_s / audio_duration_s if audio_duration_s > 0 and inference_time_s > 0 else float("inf")
    m["rtfx"] = audio_duration_s / inference_time_s if inference_time_s > 0 and audio_duration_s > 0 else float("inf")
    m["inference_time_s"]  = inference_time_s
    m["audio_duration_s"]  = audio_duration_s
    m["cost_usd"]          = cost_usd or 0.0

    # Economics / efficiency metrics
    hits = m.get("hits", 0)
    wer  = m.get("wer", float("nan"))
    rtfx = m.get("rtfx", float("nan"))
    _cost = cost_usd or 0.0

    # cost_per_correct_word: $0 for local/free models, NaN if no correct words
    if _cost == 0.0:
        m["cost_per_correct_word"] = 0.0  # local model — free
    elif hits > 0:
        m["cost_per_correct_word"] = _cost / hits
    else:
        m["cost_per_correct_word"] = float("nan")  # no correct words transcribed

    # accuracy_per_dollar: undefined for free models (∞), leave as NaN
    m["accuracy_per_dollar"] = (
        (1.0 - wer) / _cost if _cost > 0 and np.isfinite(wer) else float("nan")
    )
    # NIC = Normalised Inference Cost: accuracy gain per unit of real-time factor
    m["nic"] = (
        (1.0 - wer) * rtfx if np.isfinite(wer) and np.isfinite(rtfx) else float("nan")
    )

    # Formatting metrics (raw text, no normalisation)
    if compute_formatting and ref:
        m["fwer"]               = compute_fwer(ref, hyp)
        pf1                     = compute_punctuation_f1(ref, hyp)
        m["punctuation_f1"]     = pf1["macro_f1"]
        m["capitalization_acc"] = compute_capitalization_accuracy(ref, hyp)
    else:
        m["fwer"] = m["punctuation_f1"] = m["capitalization_acc"] = float("nan")

    # ── Advanced metrics (fast, rule-based, no encoder needed) ─────────────────
    if adv_cfg and adv_cfg.enabled and ref:
        if adv_cfg.her:
            her = compute_her(ref, hyp)
            m["her"]                       = her.her
            m["phonetic_errors"]           = her.phonetic_errors
            m["hallucination_errors"]      = her.hallucination_errors
            m["repetition_errors"]         = her.repetition_errors
            m["insertion_hallucination_rate"] = her.insertion_hallucination_rate

        if adv_cfg.bleu:
            m.update(compute_bleu(ref, hyp))

        if adv_cfg.meteor:
            m["meteor"] = compute_meteor(ref, hyp)

        if adv_cfg.per:
            m.update(compute_per(ref, hyp))

        if adv_cfg.shallow:
            m.update(compute_shallow(ref, hyp))
            # Normalise so sf+pf+rl+lc sum to 1.0 per sample (avoid sum > 1 for verbose models)
            sf = float(m.get("shallow_sf") or 0.0)
            pf = float(m.get("shallow_pf") or 0.0)
            rl = float(m.get("shallow_rl") or 0.0)
            lc = float(m.get("shallow_lc") or 0.0)
            _total_sh = sf + pf + rl + lc
            if _total_sh > 0 and np.isfinite(_total_sh):
                m["shallow_sf"] = sf / _total_sh
                m["shallow_pf"] = pf / _total_sh
                m["shallow_rl"] = rl / _total_sh
                m["shallow_lc"] = lc / _total_sh

        if adv_cfg.keywords:
            m.update(compute_krr(ref, hyp, adv_cfg.keywords))

        # PNER — named entity recognition rate via Jaro-Winkler alignment
        if getattr(adv_cfg, "pner", True):
            try:
                m.update(compute_pner(ref, hyp))
            except Exception:
                m["pner"] = float("nan")
                m["pner_precision"] = float("nan")
                m["pner_n"] = 0

        # Alphanumeric accuracy — codes, IDs, mixed letter+digit tokens
        if getattr(adv_cfg, "alphanumeric", True):
            m.update(compute_alphanumeric_accuracy(ref, hyp))

    return m


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_evaluation(
    samples: list[AudioSample],
    models_cfg_path: str,
    eval_cfg_path: str,
    output_dir: str,
    models_filter: list[str] | None,
    semantic: bool,
    entity: bool,
    hallucination_only: bool,
    recompute_metrics: bool = False,
    semantic_wer: bool = False,
) -> dict[str, str]:
    """Run all models and save per-model JSONL + JSON results. Returns {model_name: metrics_path}."""
    eval_cfg    = load_evaluation_config(eval_cfg_path)
    models_cfg  = load_models_config(models_cfg_path)
    normalizer  = TranscriptNormalizer(eval_cfg.normalization.model_dump())

    # Filter models if requested
    model_dicts = [
        m.model_dump(by_alias=True)
        for m in models_cfg.enabled_models()
        if models_filter is None or m.name in models_filter
    ]

    models = create_all_models(model_dicts)
    if not models:
        log.error("No models available — check API keys in environment.")
        return {}

    out = Path(output_dir)
    trans_dir   = out / "transcriptions"
    metrics_dir = out / "metrics"
    trans_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    result_paths: dict[str, str] = {}

    # ── Hallucination-only mode ────────────────────────────────────────────────
    if hallucination_only:
        for model in models:
            hyps = []
            for s in tqdm(samples, desc=f"[hallucination] {model.name}"):
                r = model.transcribe(s.audio_filepath, language="en")
                hyps.append(r.text)
            hall = compute_hallucination_metrics(hyps)
            path = metrics_dir / f"{model.name}_hallucination.json"
            path.write_text(json.dumps(hall, indent=2))
            log.info(f"  hallucination_rate={hall['hallucination_rate']:.3f} → {path}")
        return {}

    # ── Pre-compute SNR (audio property, model-independent) ───────────────────
    if not recompute_metrics:
        log.info(f"Pre-computing SNR for {len(samples)} audio files…")
        snr_cache: dict[str, float] = {
            s.audio_filepath: compute_snr_db(s.audio_filepath)
            for s in tqdm(samples, desc="SNR", unit="file", leave=False)
        }
    else:
        snr_cache = {}

    # ── Standard evaluation ────────────────────────────────────────────────────
    for model in models:
        trans_path   = trans_dir   / f"{model.name}.jsonl"
        metrics_path = metrics_dir / f"{model.name}.json"

        per_sample_metrics: list[dict] = []
        log.info(f"▶  {model.name}")

        # ── Recompute-only mode: load existing transcriptions, skip API calls ──
        if recompute_metrics and trans_path.exists():
            log.info(f"  --recompute-metrics: loading from {trans_path}")
            rows = [json.loads(l) for l in trans_path.read_text().splitlines() if l.strip()]
            for row in rows:
                if row.get("error") or not row.get("reference"):
                    continue
                m = evaluate_sample(
                    ref=row["reference"],
                    hyp=row.get("hypothesis", ""),
                    normalizer=normalizer,
                    inference_time_s=row.get("inference_time_s", 0.0),
                    audio_duration_s=row.get("audio_duration_s", 0.0),
                    cost_usd=row.get("cost_usd"),
                    adv_cfg=eval_cfg.advanced,
                )
                m["id"]         = row.get("id", "")
                m["speaker_id"] = row.get("speaker_id", "")
                m["tts_model"]  = row.get("tts_model", "")
                m["case_study"] = row.get("case_study", "")
                m["snr_db"]     = row.get("snr_db", float("nan"))
                per_sample_metrics.append(m)
            log.info(f"  Recomputed metrics for {len(per_sample_metrics)} samples (no API calls)")
        elif not recompute_metrics:
            with open(trans_path, "w") as tf:
                for sample in tqdm(samples, desc=model.name, unit="file"):
                    result = model.transcribe(sample.audio_filepath, language=sample.lang or "en")

                    # Save raw transcription
                    trans_row = {
                        "id":               sample.sample_id,
                        "audio_filepath":   sample.audio_filepath,
                        "reference":        sample.text,
                        "hypothesis":       result.text,
                        "inference_time_s": round(result.inference_time_s, 4),
                        "audio_duration_s": round(result.audio_duration_s or sample.duration, 4),
                        "cost_usd":         result.cost_usd,
                        "model":            result.model_name,
                        "speaker_id":       sample.speaker_id,
                        "tts_model":        sample.tts_model,
                        "case_study":       sample.case_study,
                        "error":            result.error,
                        "snr_db":           round(snr_cache.get(sample.audio_filepath, float("nan")), 2),
                    }
                    tf.write(json.dumps(trans_row) + "\n")

                    if result.error:
                        log.warning(f"  ✗ {sample.sample_id}: {result.error}")
                        continue

                    # Per-sample metrics
                    m = evaluate_sample(
                        ref=sample.text,
                        hyp=result.text,
                        normalizer=normalizer,
                        inference_time_s=result.inference_time_s,
                        audio_duration_s=result.audio_duration_s or sample.duration,
                        cost_usd=result.cost_usd,
                        adv_cfg=eval_cfg.advanced,
                    )
                    m["id"]           = sample.sample_id
                    m["speaker_id"]   = sample.speaker_id
                    m["tts_model"]    = sample.tts_model
                    m["case_study"]   = sample.case_study
                    m["snr_db"]       = snr_cache.get(sample.audio_filepath, float("nan"))
                    per_sample_metrics.append(m)

        # ── Semantic metrics (batch, lazy) ─────────────────────────────────────
        if semantic and per_sample_metrics:
            from src.evaluation.semantic_metrics import compute_semdist, compute_bert_score, compute_asd
            # Build both refs and hyps from the JSONL so they always stay aligned,
            # even when the manifest has more samples than were actually evaluated.
            refs = []
            hyps_raw = []
            with open(trans_path) as tf:
                for line in tf:
                    d = json.loads(line)
                    if not d.get("error") and d.get("reference"):
                        refs.append(normalizer.normalize(d.get("reference", "")))
                        hyps_raw.append(normalizer.normalize(d.get("hypothesis", "")))

            # Apply semantic normalizer: remove punctuation + lowercase only.
            # Full normalizer (filler removal, contractions) can introduce token changes
            # that inflate/deflate semantic distance spuriously.
            refs_sem     = [_sem_norm(r) for r in refs]
            hyps_raw_sem = [_sem_norm(h) for h in hyps_raw]

            if refs_sem and hyps_raw_sem and len(refs_sem) == len(hyps_raw_sem):
                log.info(f"  Computing SemDist…")
                dists = compute_semdist(refs_sem, hyps_raw_sem)
                for i, m in enumerate(per_sample_metrics):
                    m["semdist"] = dists[i] if i < len(dists) else float("nan")

                log.info(f"  Computing ASD (word-level semantic distance)…")
                asd_vals = compute_asd(refs_sem, hyps_raw_sem)
                for i, m in enumerate(per_sample_metrics):
                    m["asd"] = asd_vals[i] if i < len(asd_vals) else float("nan")

                log.info(f"  Computing BERTScore…")
                bs = compute_bert_score(
                    refs_sem, hyps_raw_sem,
                    model_type=eval_cfg.metrics.semantic.bertscore_model,
                )
                for i, m in enumerate(per_sample_metrics):
                    m["bertscore_f1"] = bs["f1"][i] if i < len(bs["f1"]) else float("nan")

        # ── SeMaScore + Error Severity + Embedding Hallucination (reuse encoder) ─
        adv = eval_cfg.advanced
        _need_encoder = (adv.semascore or adv.error_severity
                         or getattr(adv, "embedding_hallucination", True))
        if adv.enabled and per_sample_metrics and _need_encoder:
            from src.evaluation.semantic_metrics import _get_st_model
            _encoder = _get_st_model(eval_cfg.metrics.semantic.semdist_model)

            valid_rows = [r for r in [json.loads(l) for l in open(trans_path)] if not r.get("error") and r.get("reference")]

            if adv.semascore:
                log.info(f"  Computing SeMaScore…")
                for i, m in enumerate(per_sample_metrics):
                    if i < len(valid_rows):
                        # Use minimal semantic normalizer (punct removal + lowercase only)
                        # so formatting differences don't inflate semantic distance.
                        r_text = _sem_norm(valid_rows[i].get("reference", ""))
                        h_text = _sem_norm(valid_rows[i].get("hypothesis", ""))
                        try:
                            m["semascore"] = compute_semascore(r_text, h_text, _encoder)
                        except Exception:
                            m["semascore"] = float("nan")

            if adv.error_severity:
                log.info(f"  Computing Error Severity…")
                for i, m in enumerate(per_sample_metrics):
                    if i < len(valid_rows):
                        r_text = valid_rows[i].get("reference", "")
                        h_text = valid_rows[i].get("hypothesis", "")
                        try:
                            m.update(compute_error_severity(r_text, h_text, _encoder))
                        except Exception:
                            m["avg_error_severity"] = float("nan")
                            m["max_error_severity"] = float("nan")

            # Embedding-based hallucination (§3d): flag insertions whose
            # embedding distance from every reference window > threshold 0.7
            if getattr(adv, "embedding_hallucination", True):
                log.info(f"  Computing Embedding Hallucination…")
                for i, m in enumerate(per_sample_metrics):
                    if i < len(valid_rows):
                        r_text = normalizer.normalize(valid_rows[i].get("reference", ""))
                        h_text = normalizer.normalize(valid_rows[i].get("hypothesis", ""))
                        try:
                            m.update(compute_embedding_hallucination(r_text, h_text, _encoder))
                        except Exception:
                            m["emb_hallucination_rate"] = float("nan")
                            m["n_emb_hallucinations"]   = 0

        # ── LLM Impact Judge (optional, per-sample) ────────────────────────────
        if adv.enabled and adv.llm_judge and per_sample_metrics:
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI
                llm_client = OpenAI(api_key=openai_key)
                log.info(f"  Computing LLM Impact scores…")
                valid_rows = [r for r in [json.loads(l) for l in open(trans_path)] if not r.get("error") and r.get("reference")]
                for i, m in enumerate(per_sample_metrics):
                    if i < len(valid_rows):
                        try:
                            m.update(compute_llm_impact(
                                valid_rows[i].get("reference", ""),
                                valid_rows[i].get("hypothesis", ""),
                                llm_client,
                                model=adv.llm_judge_model,
                            ))
                        except Exception:
                            m["impact_label"] = "error"
                            m["impact_score"]  = float("nan")
            else:
                log.warning("  LLM judge enabled but OPENAI_API_KEY not set — skipping")

        # ── Semantic WER (LLM-judged, optional) ───────────────────────────────
        if semantic_wer and per_sample_metrics:
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_key:
                import anthropic as _anthropic
                from src.evaluation.advanced_metrics import compute_semantic_wer
                _anth_client = _anthropic.Anthropic(api_key=anthropic_key)
                log.info(f"  Computing Semantic WER (LLM-judged via Claude)…")
                valid_rows = [r for r in [json.loads(l) for l in open(trans_path)]
                              if not r.get("error") and r.get("reference")]
                for i, m in enumerate(per_sample_metrics):
                    if i < len(valid_rows):
                        try:
                            m.update(compute_semantic_wer(
                                valid_rows[i].get("reference", ""),
                                valid_rows[i].get("hypothesis", ""),
                                _anth_client,
                            ))
                        except Exception:
                            m["semantic_wer"]      = float("nan")
                            m["trivial_errors"]    = 0
                            m["meaningful_errors"] = 0
            else:
                log.warning("  --semantic-wer requested but ANTHROPIC_API_KEY not set — skipping")

        # ── Entity metrics (per-sample, lazy) ─────────────────────────────────
        if entity and per_sample_metrics:
            from src.evaluation.entity_metrics import compute_entity_and_krr
            log.info(f"  Computing Entity F1 + KRR (spaCy)…")
            with open(trans_path) as tf:
                trans_rows = [json.loads(l) for l in tf if l.strip()]
            valid_trans_rows = [r for r in trans_rows if not r.get("error") and r.get("reference")]
            for i, m in enumerate(per_sample_metrics):
                if i < len(valid_trans_rows):
                    row = valid_trans_rows[i]
                    try:
                        ent = compute_entity_and_krr(row["reference"], row.get("hypothesis", ""))
                        m["entity_f1"] = ent["entity_f1"]
                        m["krr"]       = ent["krr"]
                        m["entity_n"]  = ent["entity_n"]
                        m["keyword_n"] = ent["keyword_n"]
                    except Exception:
                        m["entity_f1"] = float("nan")
                        m["krr"]       = float("nan")

        # ── Fairness: disaggregated WER by group ──────────────────────────────
        fairness: dict = {}
        if per_sample_metrics:
            from src.evaluation.fairness import compute_disaggregated_wer
            # speaker_id is always available; tts_model / case_study are set for
            # TTS-pipeline datasets and silently skipped (empty groups) for others.
            fairness = compute_disaggregated_wer(
                per_sample_metrics,
                group_by=["speaker_id", "tts_model", "case_study"],
            )

        # Save per-model metrics JSON
        metrics_out = {
            "model":          model.name,
            "schema_version": 2,
            "n_samples":      len(per_sample_metrics),
            "fairness":       fairness,
            "per_sample":     per_sample_metrics,
        }
        metrics_path.write_text(json.dumps(metrics_out, indent=2))
        log.info(f"✓  {model.name} → {metrics_path}")
        result_paths[model.name] = str(metrics_path)

    return result_paths


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--dataset", "-d", required=True,
              help="Path to manifest.jsonl or directory of .wav files")
@click.option("--models-config",   default="configs/models.yaml",   show_default=True)
@click.option("--eval-config",     default="configs/evaluation.yaml", show_default=True)
@click.option("--output-dir",  "-o", default="results",              show_default=True)
@click.option("--models", "-m", default=None,
              help="Comma-separated model names to run (default: all enabled)")
@click.option("--semantic/--no-semantic", default=True,
              help="Compute SemDist + BERTScore (slow first run)")
@click.option("--entity/--no-entity", default=False,
              help="Compute NER Entity F1 via spaCy (requires en_core_web_sm)")
@click.option("--hallucination-only", is_flag=True, default=False,
              help="Only run hallucination test (no reference required)")
@click.option("--max-samples", default=None, type=int,
              help="Truncate dataset to N samples (for quick smoke tests)")
@click.option("--recompute-metrics", is_flag=True, default=False,
              help="Skip API calls; recompute metrics from existing transcription JSONLs")
@click.option("--semantic-wer", is_flag=True, default=False,
              help="Compute LLM-judged Semantic WER via Claude (requires ANTHROPIC_API_KEY)")
def main(dataset, models_config, eval_config, output_dir, models,
         semantic, entity, hallucination_only, max_samples, recompute_metrics, semantic_wer):
    """
    Run all enabled STT models on the dataset and compute metrics.
    Saves: results/transcriptions/{model}.jsonl  (raw hypotheses)
           results/metrics/{model}.json          (per-sample metrics)

    Next step: python aggregate.py
    """
    # Load dataset
    p = Path(dataset)
    if p.is_dir():
        samples = load_wav_directory(str(p), reference_text="")
    elif p.suffix == ".jsonl":
        samples = load_manifest(str(p))
    else:
        raise click.BadParameter(f"Dataset must be a .jsonl file or directory: {dataset}")

    samples = validate_samples(samples)
    if max_samples:
        samples = samples[:max_samples]
        log.info(f"Truncated to {len(samples)} samples")

    if not samples:
        log.error("No valid samples found.")
        return

    log.info(f"Dataset: {len(samples)} samples | "
             f"Total audio: {sum(s.duration for s in samples)/60:.1f} min")

    models_filter = [m.strip() for m in models.split(",")] if models else None

    run_evaluation(
        samples=samples,
        models_cfg_path=models_config,
        eval_cfg_path=eval_config,
        output_dir=output_dir,
        models_filter=models_filter,
        semantic=semantic,
        entity=entity,
        hallucination_only=hallucination_only,
        recompute_metrics=recompute_metrics,
        semantic_wer=semantic_wer,
    )

    log.info(f"\nDone. Results saved to: {output_dir}/")
    log.info("Next step: python aggregate.py")


if __name__ == "__main__":
    main()
