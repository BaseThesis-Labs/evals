#!/usr/bin/env python3
"""
aggregate.py — Load per-model metrics, compute aggregate scores with use-case weighting,
run significance tests, and save analysis/leaderboard.json.

Usage:
    python aggregate.py
    python aggregate.py --metrics-dir results/metrics --case-study balanced
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import click
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_case_studies_config, load_evaluation_config
from src.evaluation.statistical import (
    blockwise_bootstrap_ci, wilcoxon_test,
    mapsswe_test, bonferroni_correct, all_pairs_mapsswe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)


# ── Dimension → metric mapping ────────────────────────────────────────────────

DIMENSION_METRICS = {
    "intelligibility": ["wer", "cer", "ser"],
    # bleu_4/meteor/wrr/mer removed — high correlation with WER, add noise not signal
    "semantic":        ["semdist", "asd", "semascore"],
    # bertscore_f1 removed — 0.98-0.99 range, cannot discriminate models
    "latency":         ["rtfx", "nic"],
    "formatting":      ["fwer", "punctuation_f1", "punct_per"],
    # capitalization_acc excluded: near-0 on ALL-CAPS refs regardless of model quality
    "hallucination":   ["hallucination_rate", "her", "shallow_sf", "shallow_rl", "shallow_lc"],
    "entity":          ["entity_f1", "krr"],
    "safety":          ["avg_error_severity", "max_error_severity", "impact_score"],
}

# Columns dropped from ALL aggregate CSV outputs (useless / redundant / always-zero).
# They remain in the per-sample JSON files but are excluded from leaderboard/total CSVs.
COLUMNS_TO_DROP: set[str] = {
    "mean_wer_raw", "mean_cer_raw",    # ~99% on LibriSpeech ALL-CAPS — meaningless
    "mean_mer",                        # ≈ macro_WER at <5% error rates
    "mean_wil",                        # r > 0.99 with WER
    "mean_wip",                        # = 1 - WIL by definition
    "mean_wrr",                        # ≈ 1 - WER
    "mean_bleu_1",                     # MT metric, ranks same as WER for ASR
    "mean_bleu_4",                     # MT metric
    "mean_meteor",                     # MT metric
    "mean_bertscore_f1",               # 0.98-0.99, cannot discriminate
    "mean_punct_per",                  # always 0 on no-punct refs; keep punct_per in per-sample
}

# Keys whose per-sample values should NOT generate mean_/total_ aggregate columns.
# (They stay in per-sample JSON but are excluded from dynamic CSV scan.)
_KEYS_TO_SKIP_AGGREGATE = {"wer_raw", "cer_raw", "mer", "wil", "wip", "wrr",
                            "bleu_1", "bleu_4", "meteor", "bertscore_f1"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_mean(values: list) -> float:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _normalize_metric(value: float, bound: dict, metric: str = "") -> float:
    """
    Map raw metric value to [0, 1] where 1 = best.

    RTFx uses log-scale normalization because linear scale compresses fast models:
    e.g. RTFx=5 (API) vs RTFx=8 (local turbo) look nearly identical on [0, 200]
    but are meaningfully different on log scale.
    """
    if not np.isfinite(value):
        return float("nan")

    # Log-scale for RTFx (latency/speed metric)
    if metric == "rtfx":
        _RTFX_MIN, _RTFX_MAX = 0.1, 20.0
        log_val = np.log10(max(value, _RTFX_MIN))
        log_min = np.log10(_RTFX_MIN)
        log_max = np.log10(_RTFX_MAX)
        return float(np.clip((log_val - log_min) / (log_max - log_min), 0.0, 1.0))

    lo, hi = bound["min"], bound["max"]
    clipped = max(lo, min(hi, value))
    norm = (clipped - lo) / (hi - lo) if hi > lo else 0.0
    return (1.0 - norm) if bound["lower_is_better"] else norm


def _compute_dimension_score(
    per_sample: list[dict],
    dimension: str,
    metric_bounds: dict,
) -> float:
    """Average normalised score across all metrics in a dimension."""
    metrics = DIMENSION_METRICS.get(dimension, [])
    scores = []
    for metric in metrics:
        vals = [s.get(metric) for s in per_sample if s.get(metric) is not None]
        if not vals:
            continue
        mean_val = _safe_mean(vals)
        bound = metric_bounds.get(metric)
        if bound is None or not np.isfinite(mean_val):
            continue
        norm = _normalize_metric(
            mean_val,
            bound.model_dump() if hasattr(bound, "model_dump") else bound,
            metric=metric,
        )
        if np.isfinite(norm):
            scores.append(norm)
    return float(np.mean(scores)) if scores else float("nan")


# ── Load + aggregate one model ────────────────────────────────────────────────

def aggregate_model(
    metrics_path: Path,
    case_studies_cfg,
    eval_cfg,
    bootstrap_iters: int = 10000,
) -> dict:
    """Load a model's metrics JSON and compute all aggregate statistics."""
    data = json.loads(metrics_path.read_text())
    model_name   = data["model"]
    per_sample   = data.get("per_sample", [])
    fairness     = data.get("fairness", {})
    log.info(f"Aggregating {model_name} ({len(per_sample)} samples)")

    metric_bounds = {k: v.model_dump() for k, v in case_studies_cfg.metric_bounds.items()}
    schema_version = data.get("schema_version", 1)

    # ── Backfill derived metrics missing from old JSON files ───────────────────
    # schema_version >= 2: evaluate.py now computes all derived metrics at run
    # time, so these derivations are only needed for result files produced by an
    # older version of the pipeline (schema_version == 1 / missing).
    if schema_version < 2:
        for s in per_sample:
            if "punct_per" not in s and "per" in s:
                s["punct_per"] = s["per"]   # backward-compat rename
            ref_w = s.get("hits", 0) + s.get("substitutions", 0) + s.get("deletions", 0)
            if s.get("wrr") is None:
                s["wrr"] = s.get("hits", 0) / ref_w if ref_w > 0 else 1.0
            if s.get("sub_rate") is None:
                s["sub_rate"] = s.get("substitutions", 0) / ref_w if ref_w > 0 else 0.0
            if s.get("del_rate") is None:
                s["del_rate"] = s.get("deletions", 0) / ref_w if ref_w > 0 else 0.0
            if s.get("ins_rate") is None:
                s["ins_rate"] = s.get("insertions", 0) / ref_w if ref_w > 0 else 0.0
            if s.get("error_words") is None:
                s["error_words"] = (
                    s.get("substitutions", 0) + s.get("deletions", 0) + s.get("insertions", 0)
                )

            # SER fix: old JSONs have ser = mer (wrong). Recompute as binary error flag.
            subs_s = s.get("substitutions", 0)
            dels_s = s.get("deletions", 0)
            ins_s  = s.get("insertions", 0)
            stored_ser = s.get("ser")
            stored_wer = s.get("wer", float("nan"))
            # Detect old bug: ser was set equal to mer (≈ wer) for erroneous samples
            if stored_ser is not None and np.isfinite(float(stored_ser)):
                if abs(float(stored_ser) - float(stored_wer if np.isfinite(float(stored_wer)) else 0)) < 1e-9:
                    s["ser"] = 1.0 if (subs_s > 0 or dels_s > 0 or ins_s > 0) else 0.0

            # Economics: cost_per_correct_word, accuracy_per_dollar, nic
            if s.get("cost_per_correct_word") is None:
                _cost = s.get("cost_usd", 0.0) or 0.0
                hits  = s.get("hits", 0)
                if _cost == 0.0:
                    s["cost_per_correct_word"] = 0.0
                elif hits > 0:
                    s["cost_per_correct_word"] = _cost / hits
                else:
                    s["cost_per_correct_word"] = float("nan")
            if s.get("accuracy_per_dollar") is None:
                _cost = s.get("cost_usd", 0.0) or 0.0
                wer   = s.get("wer", float("nan"))
                s["accuracy_per_dollar"] = (
                    (1.0 - wer) / _cost
                    if _cost > 0 and np.isfinite(wer)
                    else float("nan")
                )
            if s.get("nic") is None:
                wer  = s.get("wer",  float("nan"))
                rtfx = s.get("rtfx", float("nan"))
                s["nic"] = (
                    (1.0 - wer) * rtfx
                    if np.isfinite(wer) and np.isfinite(rtfx)
                    else float("nan")
                )

    # ── Recompute SNR from audio files if all values are missing ──────────────
    snr_valid = [s["snr_db"] for s in per_sample
                 if s.get("snr_db") is not None and np.isfinite(float(s.get("snr_db", float("nan"))))]
    if not snr_valid and per_sample:
        trans_path = metrics_path.parent.parent / "transcriptions" / f"{model_name}.jsonl"
        if trans_path.exists():
            try:
                from src.evaluation.snr import compute_snr_db
                trans_rows = [json.loads(l) for l in trans_path.read_text().splitlines() if l.strip()]
                valid_trans = [r for r in trans_rows if not r.get("error") and r.get("audio_filepath")]
                snr_cache: dict[str, float] = {}
                for i, s in enumerate(per_sample):
                    if i < len(valid_trans):
                        fp = valid_trans[i]["audio_filepath"]
                        if fp not in snr_cache:
                            snr_cache[fp] = compute_snr_db(fp)
                        s["snr_db"] = snr_cache[fp]
                log.info(f"  Recomputed SNR for {len(per_sample)} samples from audio files")
            except Exception as e:
                log.warning(f"  Could not recompute SNR: {e}")

    # (semascore recompute moved to end of backfill section — now uses _sem_norm)

    # ── Recompute punctuation_f1 from transcription JSONL if all values are NaN ─
    # Old JSON files store NaN because the formatting code wasn't run at that time.
    pf1_valid = [
        s["punctuation_f1"] for s in per_sample
        if s.get("punctuation_f1") is not None and np.isfinite(float(s.get("punctuation_f1", float("nan"))))
    ]
    if not pf1_valid and per_sample:
        trans_path = metrics_path.parent.parent / "transcriptions" / f"{model_name}.jsonl"
        if trans_path.exists():
            try:
                from src.evaluation.formatting_metrics import compute_punctuation_f1
                trans_rows = [json.loads(l) for l in trans_path.read_text().splitlines() if l.strip()]
                valid_trans = [r for r in trans_rows if not r.get("error") and r.get("reference")]
                for i, s in enumerate(per_sample):
                    if i < len(valid_trans):
                        pf1 = compute_punctuation_f1(
                            valid_trans[i]["reference"],
                            valid_trans[i].get("hypothesis", ""),
                        )
                        s["punctuation_f1"] = pf1["macro_f1"]
                log.info(f"  Recomputed punctuation_f1 for {len(per_sample)} samples from {trans_path.name}")
            except Exception as e:
                log.warning(f"  Could not recompute punctuation_f1: {e}")

    # ── Backfill wer_lcase / cer_lcase (missing from pre-fix runs) ─────────────
    lcase_valid = [
        s.get("wer_lcase") for s in per_sample
        if s.get("wer_lcase") is not None and np.isfinite(float(s.get("wer_lcase", float("nan"))))
    ]
    if not lcase_valid and per_sample:
        trans_path = metrics_path.parent.parent / "transcriptions" / f"{model_name}.jsonl"
        if trans_path.exists():
            try:
                import re as _re
                import jiwer as _jiwer

                def _norm_lcase(text: str) -> str:
                    text = text.lower()
                    text = _re.sub(r"[^\w\s]", "", text)
                    return _re.sub(r"\s+", " ", text).strip()

                trans_rows = [json.loads(l) for l in trans_path.read_text().splitlines() if l.strip()]
                valid_trans = [r for r in trans_rows if not r.get("error") and r.get("reference")]
                for i, s in enumerate(per_sample):
                    if i < len(valid_trans):
                        rl = _norm_lcase(valid_trans[i].get("reference", ""))
                        hl = _norm_lcase(valid_trans[i].get("hypothesis", ""))
                        s["wer_lcase"] = float(_jiwer.wer(rl, hl)) if rl else float("nan")
                        s["cer_lcase"] = float(_jiwer.cer(rl, hl)) if rl else float("nan")
                log.info(f"  Backfilled wer_lcase/cer_lcase for {len(per_sample)} samples")
            except Exception as e:
                log.warning(f"  Could not backfill wer_lcase: {e}")

    # ── Recompute capitalization_acc if values look like old error-rate format ──
    # Old formula returned match_fraction (≈ 0.005-0.012 for TTS data).
    # New formula returns 1 - mismatch_fraction (≈ 0.988-0.995 for same data).
    # Heuristic: if mean < 0.2, assume old format and recompute from JSONL.
    cap_vals = [
        float(s["capitalization_acc"]) for s in per_sample
        if s.get("capitalization_acc") is not None
        and np.isfinite(float(s.get("capitalization_acc", float("nan"))))
    ]
    if cap_vals and float(np.mean(cap_vals)) < 0.2:
        trans_path = metrics_path.parent.parent / "transcriptions" / f"{model_name}.jsonl"
        if trans_path.exists():
            try:
                from src.evaluation.formatting_metrics import compute_capitalization_accuracy
                trans_rows = [json.loads(l) for l in trans_path.read_text().splitlines() if l.strip()]
                valid_trans = [r for r in trans_rows if not r.get("error") and r.get("reference")]
                for i, s in enumerate(per_sample):
                    if i < len(valid_trans):
                        s["capitalization_acc"] = compute_capitalization_accuracy(
                            valid_trans[i]["reference"],
                            valid_trans[i].get("hypothesis", ""),
                        )
                log.info(f"  Recomputed capitalization_acc (new formula) for {len(per_sample)} samples")
            except Exception as e:
                log.warning(f"  Could not recompute capitalization_acc: {e}")

    # ── Normalise SHALLOW scores so sf+pf+rl+lc sum to 1.0 per sample ──────────
    for s in per_sample:
        sf = float(s.get("shallow_sf") or 0.0)
        pf = float(s.get("shallow_pf") or 0.0)
        rl = float(s.get("shallow_rl") or 0.0)
        lc = float(s.get("shallow_lc") or 0.0)
        if all(np.isfinite(v) for v in [sf, pf, rl, lc]):
            _sh_total = sf + pf + rl + lc
            if _sh_total > 1.05 and _sh_total > 0:  # >1 means old unnormalised data
                s["shallow_sf"] = sf / _sh_total
                s["shallow_pf"] = pf / _sh_total
                s["shallow_rl"] = rl / _sh_total
                s["shallow_lc"] = lc / _sh_total

    # ── Recompute semascore if values look broken (all-zero or majority-zero) ────
    # Triggers on: all zeros (complete failure) OR >50 % zeros (partial failure)
    # OR mean < 0.05 (implausibly low — semascore for any real speech is ≥ 0.1).
    semascore_vals = [float(s.get("semascore", 0.0)) for s in per_sample
                      if s.get("semascore") is not None and np.isfinite(float(s.get("semascore", 0.0)))]
    _zero_frac = (sum(1 for v in semascore_vals if v == 0.0) / len(semascore_vals)
                  if semascore_vals else 0.0)
    _needs_recompute = semascore_vals and (
        _zero_frac > 0.5 or float(np.mean(semascore_vals)) < 0.05
    )
    if _needs_recompute:
        trans_path = metrics_path.parent.parent / "transcriptions" / f"{model_name}.jsonl"
        if trans_path.exists():
            try:
                import re as _re2
                import jiwer as _jiwer2
                from src.evaluation.semascore import compute_semascore
                from src.evaluation.semantic_metrics import _get_st_model

                def _sem_norm_agg(text: str) -> str:
                    """Punct removal + lowercase (same as evaluate.py BUG4 fix)."""
                    try:
                        _fn = _jiwer2.Compose([
                            _jiwer2.RemovePunctuation(), _jiwer2.ToLowerCase(),
                            _jiwer2.RemoveMultipleSpaces(), _jiwer2.Strip(),
                        ])
                        return _fn(text)
                    except Exception:
                        return _re2.sub(r"\s+", " ", _re2.sub(r"[^\w\s]", "", text.lower())).strip()

                _encoder = _get_st_model(eval_cfg.metrics.semantic.semdist_model)
                trans_rows = [json.loads(l) for l in trans_path.read_text().splitlines() if l.strip()]
                valid_trans = [r for r in trans_rows if not r.get("error") and r.get("reference")]
                for i, s in enumerate(per_sample):
                    if i < len(valid_trans):
                        r_text = _sem_norm_agg(valid_trans[i].get("reference", ""))
                        h_text = _sem_norm_agg(valid_trans[i].get("hypothesis", ""))
                        s["semascore"] = compute_semascore(r_text, h_text, _encoder)
                log.info(f"  Recomputed semascore for {len(per_sample)} samples")
            except Exception as e:
                log.warning(f"  Could not recompute semascore: {e}")

    # ── Raw aggregates ─────────────────────────────────────────────────────────
    def _agg(key: str) -> float:
        return _safe_mean([s.get(key) for s in per_sample if s.get(key) is not None])

    # Micro-average WER
    total_subs = sum(s.get("substitutions", 0) for s in per_sample)
    total_dels = sum(s.get("deletions",     0) for s in per_sample)
    total_ins  = sum(s.get("insertions",    0) for s in per_sample)
    total_hits = sum(s.get("hits",          0) for s in per_sample)
    total_ref  = total_hits + total_subs + total_dels
    micro_wer  = (total_subs + total_dels + total_ins) / total_ref if total_ref > 0 else float("nan")

    raw = {
        # ── Primary WER metrics ──────────────────────────────────────────────────
        # micro_wer    = (ΣS+ΣD+ΣI) / Σref_words  — standard benchmark micro-avg
        # mean_wer_norm = per-sample WER on fully-normalised text, then averaged
        # mean_wer_lcase = lowercase+strip-punct only (no filler removal) — partial norm
        # macro_wer    = alias for mean_wer_norm (kept for backward compat)
        "micro_wer":        micro_wer,
        "mean_wer":         micro_wer,           # backward-compat alias = micro_wer
        "mean_wer_norm":    _agg("wer"),
        "mean_wer_lcase":   _agg("wer_lcase"),
        "macro_wer":        _agg("wer"),
        "mean_cer":         _agg("cer"),
        "mean_cer_norm":    _agg("cer"),
        "mean_cer_lcase":   _agg("cer_lcase"),
        "mean_ser":         _agg("ser"),
        "mean_fwer":        _agg("fwer"),
        "mean_punctuation_f1":     _agg("punctuation_f1"),
        "mean_capitalization_acc": _agg("capitalization_acc"),
        # ── Semantic ─────────────────────────────────────────────────────────────
        "mean_semdist":     _agg("semdist"),
        "mean_asd":         _agg("asd"),          # ASD: word-level semantic distance
        "mean_semascore":   _agg("semascore"),
        "mean_semantic_wer":_agg("semantic_wer"),
        # bertscore_f1 kept for per-sample JSON; excluded from leaderboard via COLUMNS_TO_DROP
        # ── Entity / keywords ────────────────────────────────────────────────────
        "mean_entity_f1":   _agg("entity_f1"),
        "mean_krr":         _agg("krr"),
        # ── Latency / speed ──────────────────────────────────────────────────────
        "mean_rtfx":        _agg("rtfx"),
        "mean_rtf":         _agg("rtf"),
        "mean_inference_s": _agg("inference_time_s"),
        "total_cost_usd":   sum(s.get("cost_usd", 0.0) for s in per_sample),
        # ── Hallucination / error quality ─────────────────────────────────────────
        "mean_her":                _agg("her"),
        "mean_avg_error_severity": _agg("avg_error_severity"),
        "mean_max_error_severity": _agg("max_error_severity"),
        "mean_impact_score":       _agg("impact_score"),
        "mean_shallow_sf":         _agg("shallow_sf"),
        "mean_shallow_pf":         _agg("shallow_pf"),
        "mean_shallow_rl":         _agg("shallow_rl"),
        "mean_shallow_lc":         _agg("shallow_lc"),
        # ── PNER & alphanumeric ────────────────────────────────────────────────────
        "mean_pner":               _agg("pner"),
        "mean_pner_precision":     _agg("pner_precision"),
        "mean_alphanumeric_acc":   _agg("alphanumeric_acc"),
        # ── Embedding hallucination ───────────────────────────────────────────────
        "mean_emb_hallucination_rate": _agg("emb_hallucination_rate"),
        "total_emb_hallucinations":    sum(s.get("n_emb_hallucinations", 0) for s in per_sample),
        # ── Error breakdown (Tier 2/3 detail) ────────────────────────────────────
        "mean_sub_rate":  _agg("sub_rate"),
        "mean_del_rate":  _agg("del_rate"),
        "mean_ins_rate":  _agg("ins_rate"),
        "total_error_words": total_subs + total_dels + total_ins,
        # ── Economics ─────────────────────────────────────────────────────────────
        "mean_cost_per_correct_word": _agg("cost_per_correct_word"),
        "mean_accuracy_per_dollar":   _agg("accuracy_per_dollar"),
        "mean_nic":                   _agg("nic"),
        # ── Audio quality ─────────────────────────────────────────────────────────
        "mean_snr_db": _agg("snr_db"),
    }

    # Cost per hour + total duration
    total_audio_s    = sum(s.get("audio_duration_s", 0.0) for s in per_sample)
    total_inference_s = sum(s.get("inference_time_s", 0.0) for s in per_sample)
    total_audio_min  = total_audio_s / 60
    raw["total_audio_s"]  = total_audio_s
    raw["total_audio_hr"] = total_audio_s / 3600
    raw["cost_per_hour_usd"] = (
        raw["total_cost_usd"] / total_audio_min * 60
        if total_audio_min > 0 else float("nan")
    )
    # Micro-average RTF and RTFx (total_audio / total_inference) so that
    # mean_rtf × mean_rtfx = 1.0 by construction, consistent with micro_wer.
    raw["mean_rtf"]  = total_inference_s / total_audio_s if total_audio_s > 0 else float("nan")
    raw["mean_rtfx"] = total_audio_s / total_inference_s if total_inference_s > 0 else float("nan")

    # ── Bootstrap CI on WER ────────────────────────────────────────────────────
    wer_vals = [s["wer"] for s in per_sample if np.isfinite(s.get("wer", float("nan")))]
    block_field = eval_cfg.statistical.block_field
    wer_ci_lo, wer_ci_hi = blockwise_bootstrap_ci(
        per_sample, metric="wer", block_field=block_field,
        n_iter=bootstrap_iters, seed=eval_cfg.statistical.seed,
    )

    # ── Dimension scores ───────────────────────────────────────────────────────
    dimension_scores = {
        dim: _compute_dimension_score(per_sample, dim, metric_bounds)
        for dim in DIMENSION_METRICS
    }

    # ── Use-case composite scores ──────────────────────────────────────────────
    composite: dict[str, float] = {}
    for cs_name, cs_cfg in case_studies_cfg.case_studies.items():
        w = cs_cfg.weights
        score = 0.0
        total_w = 0.0
        for dim in DIMENSION_METRICS:
            weight = getattr(w, dim, 0.0)
            dim_score = dimension_scores.get(dim, float("nan"))
            if weight > 0 and np.isfinite(dim_score):
                score   += weight * dim_score
                total_w += weight
        composite[cs_name] = score / total_w if total_w > 0 else float("nan")

    # ── WER by SNR bucket ──────────────────────────────────────────────────────
    bins   = [(-np.inf, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, np.inf)]
    labels = ["<5 dB", "5-10", "10-15", "15-20", "20-25", ">25 dB"]
    wer_by_snr: dict[str, float] = {}
    for (lo, hi), label in zip(bins, labels):
        wers = [
            s["wer"] for s in per_sample
            if s.get("snr_db") is not None
            and np.isfinite(float(s.get("snr_db", float("nan"))))
            and lo <= float(s["snr_db"]) < hi
            and np.isfinite(s.get("wer", float("nan")))
        ]
        wer_by_snr[label] = float(np.mean(wers)) if wers else float("nan")

    return {
        "model":             model_name,
        "n_samples":         len(per_sample),
        "raw":               raw,
        "wer_ci":            {"lo": wer_ci_lo, "hi": wer_ci_hi},
        "dimension_scores":  dimension_scores,
        "composite_scores":  composite,
        "wer_by_snr":        wer_by_snr,
        "fairness":          fairness,
        "per_sample":        per_sample,   # kept for significance tests
    }


# ── Significance matrix ───────────────────────────────────────────────────────

def compute_significance_matrix(all_results: dict[str, dict]) -> dict:
    """
    Pairwise MAPSSWE significance tests (NIST-standard for ASR) with
    Bonferroni correction for multiple comparisons.
    Also includes Wilcoxon for reference.
    """
    model_wers: dict[str, list[float]] = {
        name: [s["wer"] for s in res["per_sample"] if np.isfinite(s.get("wer", float("nan")))]
        for name, res in all_results.items()
    }

    # All-pairs MAPSSWE with Bonferroni
    pairs_list = all_pairs_mapsswe(model_wers, alpha=0.05)
    pairs_by_key = {(r["model_a"], r["model_b"]): r for r in pairs_list}

    model_names = sorted(all_results.keys())
    matrix: dict[str, dict] = {}
    for a in model_names:
        matrix[a] = {}
        for b in model_names:
            if a == b:
                matrix[a][b] = None
                continue
            key = (a, b) if (a, b) in pairs_by_key else (b, a)
            mapsswe = pairs_by_key.get(key)
            wilcoxon = wilcoxon_test(model_wers[a], model_wers[b])
            matrix[a][b] = {
                "mapsswe":  mapsswe,
                "wilcoxon": wilcoxon,
            }
    return matrix


# ── AAEF — Architecture-Aware Efficiency Frontier ─────────────────────────────

def compute_aaef(
    all_results: dict[str, dict],
    quality_metric: str = "mean_wer",
    efficiency_metric: str = "mean_rtfx",
) -> dict:
    """
    Architecture-Aware Efficiency Frontier.

    Q = 1 − WER (quality score, higher = better).
    E = RTFx or 1/cost_per_hour (efficiency score, higher = better).

    Steps:
      1. Compute Q and E for each model.
      2. Normalize to [0,1] via min-max.
      3. Identify Pareto-optimal set (no model dominates on both Q and E).
      4. For non-Pareto models, compute perpendicular distance to frontier.
      5. Return all data for plotting.
    """
    records = []
    for name, res in all_results.items():
        raw = res.get("raw", {})
        q_raw = raw.get(quality_metric, float("nan"))
        e_raw = raw.get(efficiency_metric, float("nan"))

        # Q = 1 - WER (higher is better)
        q = 1.0 - q_raw if np.isfinite(q_raw) else float("nan")
        # E: RTFx is already "higher = better"; cost_per_hour needs inverting
        if efficiency_metric in ("mean_rtfx",):
            e = e_raw if np.isfinite(e_raw) else float("nan")
        else:
            e = 1.0 / e_raw if (np.isfinite(e_raw) and e_raw > 0) else float("nan")

        records.append({"model": name, "q": q, "e": e})

    valid = [r for r in records if np.isfinite(r["q"]) and np.isfinite(r["e"])]
    if len(valid) < 2:
        return {"frontier": [], "models": records, "quality_metric": quality_metric,
                "efficiency_metric": efficiency_metric}

    q_vals = np.array([r["q"] for r in valid])
    e_vals = np.array([r["e"] for r in valid])

    # Min-max normalise
    q_min, q_max = q_vals.min(), q_vals.max()
    e_min, e_max = e_vals.min(), e_vals.max()
    q_norm = (q_vals - q_min) / (q_max - q_min + 1e-12)
    e_norm = (e_vals - e_min) / (e_max - e_min + 1e-12)

    for i, r in enumerate(valid):
        r["q_norm"] = float(q_norm[i])
        r["e_norm"] = float(e_norm[i])

    # Pareto frontier — model i is dominated if there exists j with q_j >= q_i AND e_j >= e_i
    is_pareto = []
    for i in range(len(valid)):
        dominated = False
        for j in range(len(valid)):
            if i == j:
                continue
            if q_norm[j] >= q_norm[i] and e_norm[j] >= e_norm[i] and \
               (q_norm[j] > q_norm[i] or e_norm[j] > e_norm[i]):
                dominated = True
                break
        is_pareto.append(not dominated)

    for i, r in enumerate(valid):
        r["pareto"] = is_pareto[i]

    # Sort Pareto frontier by efficiency (x-axis)
    frontier_pts = sorted(
        [(r["e_norm"], r["q_norm"]) for i, r in enumerate(valid) if is_pareto[i]],
        key=lambda pt: pt[0]
    )

    # Perpendicular distance to the Pareto frontier (piecewise linear)
    def _point_to_segment_dist(px, py, ax, ay, bx, by) -> float:
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return float(np.hypot(px - ax, py - ay))
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx**2 + dy**2)))
        return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))

    for i, r in enumerate(valid):
        if is_pareto[i]:
            r["inefficiency"] = 0.0
        else:
            px, py = r["e_norm"], r["q_norm"]
            min_dist = float("inf")
            for k in range(len(frontier_pts) - 1):
                d = _point_to_segment_dist(
                    px, py,
                    frontier_pts[k][0], frontier_pts[k][1],
                    frontier_pts[k + 1][0], frontier_pts[k + 1][1],
                )
                min_dist = min(min_dist, d)
            if len(frontier_pts) == 1:
                min_dist = float(np.hypot(px - frontier_pts[0][0], py - frontier_pts[0][1]))
            r["inefficiency"] = min_dist

    return {
        "quality_metric":    quality_metric,
        "efficiency_metric": efficiency_metric,
        "frontier_pts":      frontier_pts,
        "models":            valid + [r for r in records if not np.isfinite(r["q"]) or not np.isfinite(r["e"])],
    }


# ── Leaderboard builder ───────────────────────────────────────────────────────

def build_leaderboard(all_results: dict[str, dict], case_study: str = "balanced") -> list[dict]:
    """Return models sorted by composite score for the given case study."""
    rows = []
    for model_name, res in all_results.items():
        raw = res["raw"]
        rows.append({
            "model":             model_name,
            "composite_score":   res["composite_scores"].get(case_study, float("nan")),
            "micro_wer":         raw.get("micro_wer",         float("nan")),
            "mean_wer":          raw.get("mean_wer",          float("nan")),
            "mean_cer":          raw.get("mean_cer",          float("nan")),
            "mean_semdist":      raw.get("mean_semdist",      float("nan")),
            "mean_bertscore_f1": raw.get("mean_bertscore_f1", float("nan")),
            "mean_rtfx":         raw.get("mean_rtfx",         float("nan")),
            "wer_ci_lo":         res["wer_ci"]["lo"],
            "wer_ci_hi":         res["wer_ci"]["hi"],
            "cost_per_hour_usd": raw.get("cost_per_hour_usd", float("nan")),
            "dimension_scores":  res["dimension_scores"],
            "composite_scores":  res["composite_scores"],
        })
    rows.sort(key=lambda r: (
        -r["composite_score"] if np.isfinite(r["composite_score"]) else -float("inf")
    ))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


# ── Total-metrics CSV export ──────────────────────────────────────────────────

# Metrics where the total (sum) across samples is more meaningful than the mean
_SUM_KEYS = {
    "substitutions", "deletions", "insertions", "hits", "error_words",
    "phonetic_errors", "hallucination_errors", "repetition_errors",
    "alphanumeric_n", "pner_n", "n_emb_hallucinations",
}

# Metrics that are already totals in the raw aggregate dict
_RAW_TOTAL_KEYS = {
    "total_cost_usd", "total_audio_s", "total_audio_hr",
    "total_error_words", "cost_per_hour_usd",
}

# Column order for total_metrics.csv — Tier 1 → Tier 2 → Tier 3.
# Columns in COLUMNS_TO_DROP are filtered out at write-time.
_TOTAL_COL_ORDER = [
    # ── TIER 1 — Main leaderboard ─────────────────────────────────────────────
    "rank", "model", "n_samples",
    "micro_wer", "macro_wer", "mean_cer", "mean_fwer", "mean_wer_lcase",
    "mean_ser", "mean_semascore", "mean_semdist",
    "mean_her", "mean_entity_f1", "mean_pner",
    "mean_rtfx", "cost_per_hour_usd",
    "score_balanced", "score_conversational_ai", "score_audiobook",
    "wer_ci_lo", "wer_ci_hi",
    # ── TIER 2 — Model detail page ────────────────────────────────────────────
    "mean_cer_norm", "mean_wer_norm", "mean_wer_lcase", "mean_cer_lcase",
    "mean_sub_rate", "mean_del_rate", "mean_ins_rate",
    "mean_punctuation_f1", "mean_capitalization_acc",
    "mean_punct_precision", "mean_punct_recall", "mean_punct_f1_adv",
    "mean_pner_precision", "mean_krr", "mean_semantic_wer", "mean_asd",
    "mean_avg_error_severity", "mean_max_error_severity",
    "mean_shallow_sf", "mean_shallow_pf", "mean_shallow_rl", "mean_shallow_lc",
    "mean_rtf", "mean_inference_s",
    "mean_cost_per_correct_word", "mean_accuracy_per_dollar", "mean_nic",
    "dim_intelligibility", "dim_semantic", "dim_latency",
    "dim_formatting", "dim_hallucination", "dim_entity", "dim_safety",
    # ── TIER 3 — Full download (all remaining) ────────────────────────────────
    "mean_wer", "macro_wer",      # backward-compat aliases
    "total_substitutions", "total_deletions", "total_insertions",
    "total_hits", "total_error_words",
    "mean_her", "total_phonetic_errors", "total_hallucination_errors",
    "total_repetition_errors", "mean_insertion_hallucination_rate",
    "mean_emb_hallucination_rate", "total_emb_hallucinations",
    "mean_impact_score",
    "total_pner_n", "mean_alphanumeric_acc", "mean_alphanumeric_n", "total_alphanumeric_n",
    "total_cost_usd", "total_audio_s", "total_audio_hr",
    "score_voice_cloning_qa", "score_low_latency",
    "mean_snr_db",
    # Kept for completeness but redundant/dropped from leaderboard:
    "mean_wil", "mean_wip", "mean_wrr", "mean_mer",
    "mean_bleu_1", "mean_bleu_4", "mean_meteor", "mean_bertscore_f1",
]


def _write_total_metrics_csv(all_results: dict[str, dict], case_study: str, path: Path) -> None:
    """
    Write total_metrics.csv — every metric for every model, ranked by WER.
    Covers:
      - All raw aggregates (mean_* for rates, total_* for counts)
      - All per-sample metrics aggregated dynamically (catches any new metrics)
      - Use-case composite scores
      - Dimension scores
      - Bootstrap WER CI
    Nothing is skipped.
    """
    if not all_results:
        return

    rows = []
    for model_name, res in all_results.items():
        per_sample = res.get("per_sample", [])
        raw        = res.get("raw", {})

        row: dict = {
            "model":    model_name,
            "n_samples": res.get("n_samples", len(per_sample)),
        }

        # ── Fixed raw aggregates ──────────────────────────────────────────────
        _keep_as_is = ("mean_", "total_", "micro_", "macro_", "cost_")
        for k, v in raw.items():
            col = k if k in _RAW_TOTAL_KEYS else (
                k if k.startswith(_keep_as_is) else f"mean_{k}"
            )
            row[col] = _fmt(v)

        # micro_wer lives in raw as "micro_wer"
        row["micro_wer"] = _fmt(raw.get("micro_wer"))

        # ── Dynamic per-sample aggregates (catches every metric automatically) ─
        # For any numeric key not already covered, compute mean and (if a count) total.
        # Skip keys in _KEYS_TO_SKIP_AGGREGATE — their mean_* would land in COLUMNS_TO_DROP.
        if per_sample:
            all_keys: set[str] = set()
            for s in per_sample:
                for k, v in s.items():
                    if isinstance(v, (int, float)) and k not in ("id",) \
                            and k not in _KEYS_TO_SKIP_AGGREGATE:
                        all_keys.add(k)

            for k in sorted(all_keys):
                mean_col  = f"mean_{k}"
                total_col = f"total_{k}"

                # Don't generate aggregate columns that would be dropped anyway
                if mean_col in COLUMNS_TO_DROP or total_col in COLUMNS_TO_DROP:
                    continue

                values = [s[k] for s in per_sample
                          if k in s and s[k] is not None
                          and np.isfinite(float(s[k]))]

                if values:
                    mean_v = float(np.mean(values))
                    sum_v  = float(np.sum(values))
                else:
                    mean_v = float("nan")
                    sum_v  = float("nan")

                if mean_col not in row or row[mean_col] == "":
                    row[mean_col] = _fmt(mean_v)
                if k in _SUM_KEYS and (total_col not in row or row[total_col] == ""):
                    row[total_col] = _fmt(sum_v)

        # ── Use-case composite scores ─────────────────────────────────────────
        for cs_name, score in res.get("composite_scores", {}).items():
            row[f"score_{cs_name}"] = _fmt(score)

        # ── Dimension scores ──────────────────────────────────────────────────
        for dim, score in res.get("dimension_scores", {}).items():
            row[f"dim_{dim}"] = _fmt(score)

        # ── Bootstrap CI ─────────────────────────────────────────────────────
        ci = res.get("wer_ci", {})
        row["wer_ci_lo"] = _fmt(ci.get("lo"))
        row["wer_ci_hi"] = _fmt(ci.get("hi"))

        rows.append(row)

    # Sort by mean_wer ascending (best model first)
    rows.sort(key=lambda r: (
        float(r["mean_wer"]) if r.get("mean_wer") not in (None, "", "nan") else float("inf")
    ))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    # Build column list: priority order first, then any remaining keys alphabetically.
    # Exclude COLUMNS_TO_DROP from the output entirely.
    all_cols = list(dict.fromkeys(
        [c for c in _TOTAL_COL_ORDER if c not in COLUMNS_TO_DROP]
        + sorted(k for k in rows[0] if k not in _TOTAL_COL_ORDER and k not in COLUMNS_TO_DROP)
    ))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v) -> str:
    """Format a numeric value for CSV: blank for NaN/None, rounded float otherwise."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return ""
    # Keep enough precision for small values (e.g. cost, error rates)
    if abs(f) == 0:
        return "0"
    if abs(f) < 0.001:
        return f"{f:.6f}"
    if abs(f) < 1:
        return f"{f:.4f}"
    return f"{f:.4f}"


# ── Leaderboard CSV export ─────────────────────────────────────────────────────

def _write_leaderboard_csv(leaderboard: list[dict], path: Path) -> None:
    """
    Flatten leaderboard rows to a wide CSV.
    Nested dicts (dimension_scores, composite_scores) are expanded as
    dim_<name> and cs_<name> columns.
    """
    if not leaderboard:
        return

    def _flatten(row: dict) -> dict:
        flat: dict = {}
        for k, v in row.items():
            if isinstance(v, dict):
                prefix = "dim" if k == "dimension_scores" else "cs"
                for sub_k, sub_v in v.items():
                    flat[f"{prefix}_{sub_k}"] = round(sub_v, 6) if isinstance(sub_v, float) and np.isfinite(sub_v) else (sub_v if sub_v is not None else "")
            else:
                flat[k] = round(v, 6) if isinstance(v, float) and np.isfinite(v) else (v if v is not None else "")
        return flat

    flat_rows = [_flatten(r) for r in leaderboard]

    # Tier 1 column order for the leaderboard (main benchmark table).
    # Composite scores are prefixed cs_ after flattening.
    _TIER1 = [
        "rank", "model", "n_samples",
        "micro_wer", "macro_wer", "mean_cer", "mean_fwer", "mean_wer_lcase",
        "mean_ser", "mean_semascore", "mean_semdist",
        "mean_her", "mean_entity_f1", "mean_pner",
        "mean_rtfx", "cost_per_hour_usd",
        "cs_balanced", "cs_conversational_ai", "cs_audiobook",
        "composite_score",
        "wer_ci_lo", "wer_ci_hi",
    ]
    all_keys = list(dict.fromkeys(
        [c for c in _TIER1 if c in flat_rows[0]]
        + sorted(k for k in flat_rows[0] if k.startswith("dim_"))
        + sorted(k for k in flat_rows[0] if k.startswith("cs_") and k not in _TIER1)
        + [k for k in flat_rows[0]
           if k not in _TIER1 and not k.startswith("dim_") and not k.startswith("cs_")]
    ))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--metrics-dir",    "-m", default="results/metrics",    show_default=True)
@click.option("--output-dir",     "-o", default="analysis",            show_default=True)
@click.option("--case-study",     "-cs", default="balanced",           show_default=True,
              help="Primary use case for ranking (balanced, conversational_ai, audiobook, voice_cloning_qa, low_latency)")
@click.option("--case-studies-config", default="configs/case_studies.yaml", show_default=True)
@click.option("--eval-config",         default="configs/evaluation.yaml",   show_default=True)
@click.option("--bootstrap-iters",     default=10000, show_default=True,
              help="Bootstrap iterations for WER CI (10 k = NIST/ACL standard)")
@click.option("--no-significance", is_flag=True, default=False,
              help="Skip pairwise significance tests (faster)")
def main(metrics_dir, output_dir, case_study, case_studies_config,
         eval_config, bootstrap_iters, no_significance):
    """
    Aggregate per-model metrics → use-case weighted scores → leaderboard.
    Saves: analysis/leaderboard.json
    Next step: python visualize.py
    """
    metrics_path = Path(metrics_dir)
    out_dir      = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(metrics_path.glob("*.json"))
    if not metric_files:
        log.error(f"No .json files found in {metrics_dir}. Run evaluate.py first.")
        return

    cs_cfg   = load_case_studies_config(case_studies_config)
    eval_cfg = load_evaluation_config(eval_config)

    # ── Aggregate each model ───────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    for mf in metric_files:
        if "_hallucination" in mf.stem:
            continue   # handled separately
        try:
            res = aggregate_model(mf, cs_cfg, eval_cfg, bootstrap_iters)
            all_results[res["model"]] = res
        except Exception as e:
            log.error(f"Failed to aggregate {mf.name}: {e}")

    if not all_results:
        log.error("No models aggregated successfully.")
        return

    # ── Significance matrix ────────────────────────────────────────────────────
    significance = {}
    if not no_significance and len(all_results) >= 2:
        log.info("Computing pairwise significance tests…")
        significance = compute_significance_matrix(all_results)

    # ── Build leaderboard ──────────────────────────────────────────────────────
    leaderboard = build_leaderboard(all_results, case_study)

    # ── Save leaderboard.json ──────────────────────────────────────────────────
    # Strip per_sample from the saved output (too large for leaderboard)
    leaderboard_clean = []
    for row in leaderboard:
        r = {k: v for k, v in row.items()}
        leaderboard_clean.append(r)

    # ── AAEF (Architecture-Aware Efficiency Frontier) ─────────────────────────
    log.info("Computing AAEF Pareto frontier…")
    aaef_speed = compute_aaef(all_results, quality_metric="mean_wer", efficiency_metric="mean_rtfx")
    aaef_cost  = compute_aaef(all_results, quality_metric="mean_wer", efficiency_metric="cost_per_hour_usd")

    # Attach per-model AAEF data to leaderboard
    aaef_by_model: dict[str, dict] = {}
    for entry in aaef_speed["models"]:
        aaef_by_model[entry["model"]] = {
            "q_norm":        entry.get("q_norm"),
            "e_norm_speed":  entry.get("e_norm"),
            "pareto_speed":  entry.get("pareto"),
            "inefficiency_speed": entry.get("inefficiency"),
        }
    for entry in aaef_cost["models"]:
        name = entry["model"]
        aaef_by_model.setdefault(name, {}).update({
            "e_norm_cost":   entry.get("e_norm"),
            "pareto_cost":   entry.get("pareto"),
            "inefficiency_cost": entry.get("inefficiency"),
        })

    output = {
        "config": {
            "primary_case_study":  case_study,
            "n_models":            len(all_results),
            "n_samples":           max((r.get("n_samples", 0) for r in all_results.values()), default=0),
            "bootstrap_iters":     bootstrap_iters,
            # Scoring weights for each use-case profile (published for reproducibility)
            "scoring_weights": {
                cs_name: {
                    dim: getattr(cs_cfg.weights, dim, 0.0)
                    for dim in DIMENSION_METRICS
                }
                for cs_name, cs_cfg in cs_cfg.case_studies.items()
            },
            "dimension_metrics": DIMENSION_METRICS,
        },
        "leaderboard":  leaderboard_clean,
        "significance": significance,
        "aaef": {
            "speed": aaef_speed,
            "cost":  aaef_cost,
        },
        "all_models": {
            model: {k: v for k, v in res.items() if k != "per_sample"}
            for model, res in all_results.items()
        },
    }

    lb_path = out_dir / "leaderboard.json"
    lb_path.write_text(json.dumps(output, indent=2, default=str))
    log.info(f"Leaderboard saved: {lb_path}")

    # ── Save leaderboard.csv ───────────────────────────────────────────────────
    csv_path = out_dir / "leaderboard.csv"
    _write_leaderboard_csv(leaderboard_clean, csv_path)
    log.info(f"Leaderboard CSV saved: {csv_path}")

    # ── Save total_metrics.csv (every metric, every model, nothing skipped) ───
    total_csv_path = out_dir / "total_metrics.csv"
    _write_total_metrics_csv(all_results, case_study, total_csv_path)
    log.info(f"Total metrics CSV saved: {total_csv_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    from rich.table import Table
    from rich.console import Console

    def _pct(v):  return f"{v*100:.2f}%" if np.isfinite(v) else "—"
    def _f(v, d=2): return f"{v:.{d}f}"  if np.isfinite(v) else "—"

    table = Table(title=f"STT Leaderboard — {case_study}", show_lines=True)
    for col in ["Rank", "Model", "WER%", "CER%", "SemDist", "RTFx", "Score", "$/hr"]:
        table.add_column(col)

    for row in leaderboard_clean:
        table.add_row(
            str(row["rank"]),
            row["model"],
            _pct(row["micro_wer"]),
            _pct(row["mean_cer"]),
            _f(row["mean_semdist"]),
            _f(row["mean_rtfx"]),
            _f(row["composite_score"]),
            f"${_f(row['cost_per_hour_usd'])}",
        )

    Console().print(table)
    log.info("Next step: python visualize.py")


if __name__ == "__main__":
    main()
