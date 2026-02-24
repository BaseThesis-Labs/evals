"""
Score aggregation: per-utterance metrics → dimension scores → composite scores.

Mirrors tts_benchmark/aggregate.py patterns but adapted for S2S dimensions.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from scoring.normalize import normalize_metric

# ── Dimension → metric mapping ────────────────────────────────────────────────
DIMENSIONS: Dict[str, List[str]] = {
    # NOTE: Each metric should appear in ONE dimension only to avoid double-counting.
    "content": ["wer", "cer", "mer", "wil", "bert_score_f1", "sem_dist", "rouge_l"],
    "asr_quality": [
        # Error rates (lower=better)
        "insertion_rate", "deletion_rate", "substitution_rate",
        # Sentence-level error flag: SER≠MER — SER is binary per sentence
        "ser",
        # Accuracy / information metrics (higher=better)
        "word_accuracy", "wip",
        # Hallucination
        "her", "hallucination_rate", "fwer",
    ],
    # secs (WavLM-large) is the primary; sim_wavlm (WavLM-Base+) is CPU alias
    # voice_consistency is the multi-turn equivalent (WavLM cosine sim across turns)
    # persona_drift: cosine drift of speaker embedding across multi-turn sessions
    "speaker": ["secs", "sim_wavlm", "sim_ecapa", "pitch_corr", "eer",
                 "voice_consistency", "persona_drift"],
    "quality": [
        "utmos", "pesq", "dnsmos_ovrl", "mcd",
        "nisqa_mos", "nisqa_noisiness", "nisqa_coloration",
        "nisqa_discontinuity", "nisqa_loudness",
        # Multi-turn session-level quality
        "session_utmos", "session_dnsmos_ovrl", "degradation_slope",
    ],
    "prosody": [
        "f0_rmse", "energy_corr", "duration_ratio", "dswed",
        "speaking_rate", "pause_ratio", "speaking_rate_ratio",
    ],
    "emotion": ["emotion_match", "emotion_sim", "esim"],
    "latency": ["ttfb_ms", "rtf", "e2e_latency_ms", "asr_latency_ms", "rtfx", "avg_turn_latency"],
    "response_quality": [
        "judge_overall", "judge_coherence", "judge_relevance",
        "judge_helpfulness", "judge_safety", "judge_naturalness",
        "instruction_follow", "safety_refusal",
    ],
    "interaction": ["tor_up", "tor_down"],
    # ── Multi-turn / agent dimensions ─────────────────────────────────────
    "task_completion": ["task_completion", "session_verdict"],
    "context_retention": ["context_retention", "factual_consistency"],
    "dialogue_coherence": ["dialogue_coherence"],
    "error_recovery": ["error_recovery", "error_recovery_rate"],
}

USE_CASES = ["conversational", "audiobook", "voice_cloning", "expressive", "balanced", "agent"]

# Which composites are valid per model type.
# Echo models repeat input — they have no response quality, no agent capability.
_VALID_COMPOSITES: Dict[str, set] = {
    "echo": {"audiobook", "voice_cloning", "expressive", "balanced"},
    "generative": {"conversational", "audiobook", "voice_cloning", "expressive", "balanced", "agent"},
}

_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "config" / "weights"


def _load_weights(use_case: str) -> Optional[Dict]:
    path = _WEIGHTS_DIR / f"{use_case}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return None
    return sum(valid) / len(valid)


# ─────────────────────────────────────────────────────────────────────────────
# Per-utterance dimension scores
# ─────────────────────────────────────────────────────────────────────────────

def score_utterance_dimensions(
    metrics: Dict[str, Optional[float]],
    weights_cfg: Optional[Dict] = None,
) -> Dict[str, Optional[float]]:
    """Compute per-dimension [0,1] score for a single utterance.

    Args:
        metrics: raw metric values for one utterance.
        weights_cfg: optional per-metric weights from a use-case YAML.

    Returns:
        dict of dimension_name → normalized score (or None if no data).
    """
    dim_scores: Dict[str, Optional[float]] = {}
    for dim, metric_list in DIMENSIONS.items():
        # Per-metric weights
        metric_weights: Dict[str, float] = {}
        if weights_cfg and "metric_weights" in weights_cfg:
            metric_weights = weights_cfg["metric_weights"].get(dim, {})

        norm_values: List[float] = []
        total_w = 0.0

        for m in metric_list:
            raw = metrics.get(m)
            norm = normalize_metric(raw, m) if raw is not None else None
            if norm is None:
                continue
            w = metric_weights.get(m, 1.0)
            norm_values.append(norm * w)
            total_w += w

        if not norm_values:
            dim_scores[dim] = None
        else:
            dim_scores[dim] = sum(norm_values) / total_w

    return dim_scores


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate across all utterances
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_model(
    utterances: List[Dict],
    model_type: str = "generative",
) -> Dict:
    """Aggregate a list of per-utterance metric dicts.

    Args:
        utterances: list of per-utterance metric dicts.
        model_type: "echo" or "generative" — controls which composites are valid.

    Returns:
        {
          "raw_means": {metric: mean, ...},
          "raw_stds":  {metric: std, ...},
          "dimensions": {dim: mean_score, ...},
          "composites": {use_case: composite_score, ...},
          "n_utterances": int,
          "n_errors": int,
        }
    """
    # Filter errors
    valid = [u for u in utterances if not u.get("error")]
    n_errors = len(utterances) - len(valid)

    # ── Raw means / stds ──────────────────────────────────────────────────────
    # Exclude non-metric bookkeeping fields from aggregation
    _SKIP_FIELDS = {"id", "error", "utterance_id", "wall_ms", "model_type",
                    "audio_out_path", "asr_transcript", "sample_rate", "eval_error"}
    all_metrics: Dict[str, List[float]] = {}
    for utt in valid:
        for k, v in utt.items():
            if k in _SKIP_FIELDS or v is None:
                continue
            try:
                fv = float(v)
                if not math.isnan(fv):
                    all_metrics.setdefault(k, []).append(fv)
            except (TypeError, ValueError):
                pass

    raw_means = {m: _safe_mean(vals) for m, vals in all_metrics.items()}
    raw_stds = {
        m: (
            math.sqrt(
                sum((v - raw_means[m]) ** 2 for v in vals) / len(vals)
            )
            if len(vals) > 1
            else 0.0
        )
        for m, vals in all_metrics.items()
    }

    # ── Dimension scores ──────────────────────────────────────────────────────
    per_utt_dims: List[Dict[str, Optional[float]]] = [
        score_utterance_dimensions(u) for u in valid
    ]
    all_dim_names = list(DIMENSIONS.keys())
    dimensions = {
        dim: _safe_mean([ud.get(dim) for ud in per_utt_dims])
        for dim in all_dim_names
    }

    # ── Composite scores ──────────────────────────────────────────────────────
    valid_composites = _VALID_COMPOSITES.get(model_type, _VALID_COMPOSITES["generative"])
    composites: Dict[str, Optional[float]] = {}
    for uc in USE_CASES:
        if uc in valid_composites:
            composites[uc] = compute_composite_score(dimensions, uc)
        else:
            composites[uc] = None

    return {
        "raw_means": raw_means,
        "raw_stds": raw_stds,
        "dimensions": dimensions,
        "composites": composites,
        "n_utterances": len(valid),
        "n_errors": n_errors,
    }


def compute_composite_score(
    dimensions: Dict[str, Optional[float]],
    use_case: str,
) -> Optional[float]:
    """Weighted combination of dimension scores for a use case.

    Weights are loaded from config/weights/{use_case}.yaml.
    Missing dimensions are renormalized.
    """
    cfg = _load_weights(use_case)
    if cfg is None:
        return None

    dim_weights: Dict[str, float] = cfg.get("dimension_weights", {})
    if not dim_weights:
        return None

    # Agent use case requires all 4 multi-turn dimensions to be present;
    # a partial score would be misleading.
    if use_case == "agent":
        required_dims = {"task_completion", "context_retention", "dialogue_coherence", "error_recovery"}
        for rd in required_dims:
            if dimensions.get(rd) is None:
                return None

    weighted_sum = 0.0
    total_w = 0.0
    for dim, w in dim_weights.items():
        score = dimensions.get(dim)
        if score is None:
            continue
        weighted_sum += score * w
        total_w += w

    if total_w == 0:
        return None
    return weighted_sum / total_w


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

def build_leaderboard(
    model_results: Dict[str, Dict],
    use_case: str = "balanced",
    model_type_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Sort models by composite score for a given use case.

    Args:
        model_results: {model_name: aggregate_model() output, ...}
        use_case: which composite score to rank by.
        model_type_map: optional {model_name: "echo"|"generative"} to include in output.

    Returns:
        List of dicts sorted by score descending.
    """
    rows = []
    for model_name, res in model_results.items():
        score = res.get("composites", {}).get(use_case)
        row = {
            "model": model_name,
            "composite_score": score,
            "dimensions": res.get("dimensions", {}),
            "n_utterances": res.get("n_utterances", 0),
            "n_errors": res.get("n_errors", 0),
        }
        if model_type_map:
            row["model_type"] = model_type_map.get(model_name, "unknown")
        rows.append(row)
    rows.sort(key=lambda r: (r["composite_score"] or -1), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


# Dimensions valid for all model types (echo and generative alike)
_UNIVERSAL_DIMENSIONS = {"quality", "speaker", "prosody", "latency", "emotion", "interaction"}

# Dimensions specific to multi-turn agent evaluation
_MULTITURN_DIMENSIONS = {
    "task_completion", "context_retention", "dialogue_coherence", "error_recovery",
}


def build_split_leaderboards(
    model_results: Dict[str, Dict],
    model_type_map: Dict[str, str],
    use_case: str = "balanced",
) -> Dict[str, List[Dict]]:
    """Build separate leaderboards for echo and generative models.

    Returns:
        {
          "echo": [...],
          "generative": [...],
          "combined": [...]   # ranked only on universally-valid dimensions
        }
    """
    echo_results = {m: r for m, r in model_results.items()
                    if model_type_map.get(m) == "echo"}
    gen_results = {m: r for m, r in model_results.items()
                   if model_type_map.get(m) == "generative"}

    echo_lb = build_leaderboard(echo_results, use_case, model_type_map) if echo_results else []
    gen_lb = build_leaderboard(gen_results, use_case, model_type_map) if gen_results else []

    # Combined leaderboard: re-score using only universal dimensions
    combined_rows = []
    for model_name, res in model_results.items():
        dims = res.get("dimensions", {})
        universal_dims = {k: v for k, v in dims.items() if k in _UNIVERSAL_DIMENSIONS}
        score = compute_composite_score(universal_dims, use_case)
        combined_rows.append({
            "model": model_name,
            "composite_score": score,
            "dimensions": universal_dims,
            "model_type": model_type_map.get(model_name, "unknown"),
            "n_utterances": res.get("n_utterances", 0),
            "n_errors": res.get("n_errors", 0),
        })
    combined_rows.sort(key=lambda r: (r["composite_score"] or -1), reverse=True)
    for rank, row in enumerate(combined_rows, start=1):
        row["rank"] = rank

    return {
        "echo": echo_lb,
        "generative": gen_lb,
        "combined": combined_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-turn session aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_multiturn_sessions(
    session_metrics_list: List[Dict],
    model_type: str = "generative",
) -> Dict:
    """Aggregate metrics across multiple multi-turn sessions for one model.

    Args:
        session_metrics_list: list of dicts, each from compute_session_quality().
        model_type: "echo" or "generative" — controls which composites are valid.

    Returns:
        Same structure as aggregate_model(), with multi-turn dimensions included.
    """
    if not session_metrics_list:
        return {
            "raw_means": {},
            "raw_stds": {},
            "dimensions": {},
            "composites": {},
            "n_sessions": 0,
            "n_errors": 0,
        }

    # Collect all numeric metrics across sessions
    all_metrics: Dict[str, List[float]] = {}
    _SKIP_SESSION_FIELDS = {
        "scenario_id", "model_name", "error",
        "session_verdict_reasoning", "session_verdict_failure_reason",
    }
    n_errors = 0
    for sm in session_metrics_list:
        if sm.get("error"):
            n_errors += 1
            continue
        for k, v in sm.items():
            if k in _SKIP_SESSION_FIELDS:
                continue
            try:
                fv = float(v)
                if not math.isnan(fv):
                    all_metrics.setdefault(k, []).append(fv)
            except (TypeError, ValueError):
                pass

    raw_means = {m: _safe_mean(vals) for m, vals in all_metrics.items()}
    raw_stds = {
        m: (
            math.sqrt(sum((v - raw_means[m]) ** 2 for v in vals) / len(vals))
            if len(vals) > 1 else 0.0
        )
        for m, vals in all_metrics.items()
    }

    # Dimension scores — normalize each session's metrics then average
    per_session_dims: List[Dict[str, Optional[float]]] = []
    for sm in session_metrics_list:
        if sm.get("error"):
            continue
        per_session_dims.append(score_utterance_dimensions(sm))

    all_dim_names = list(DIMENSIONS.keys())
    dimensions = {
        dim: _safe_mean([sd.get(dim) for sd in per_session_dims])
        for dim in all_dim_names
    }

    # Composite scores
    valid_composites = _VALID_COMPOSITES.get(model_type, _VALID_COMPOSITES["generative"])
    composites: Dict[str, Optional[float]] = {}
    for uc in USE_CASES:
        if uc in valid_composites:
            composites[uc] = compute_composite_score(dimensions, uc)
        else:
            composites[uc] = None

    return {
        "raw_means": raw_means,
        "raw_stds": raw_stds,
        "dimensions": dimensions,
        "composites": composites,
        "n_sessions": len(session_metrics_list) - n_errors,
        "n_errors": n_errors,
    }
