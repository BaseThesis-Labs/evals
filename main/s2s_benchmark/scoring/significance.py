"""
Statistical significance testing for S2S benchmark comparisons.

Ported from tts_benchmark/aggregate.py.

Exposed functions:
    wilcoxon_test(scores_a, scores_b) → dict
    bonferroni_correct(p_values, n_comparisons) → list[float]
    pairwise_significance(model_scores) → dict
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple


def wilcoxon_test(
    scores_a: List[Optional[float]],
    scores_b: List[Optional[float]],
) -> Dict:
    """Wilcoxon signed-rank test for paired metric scores.

    Args:
        scores_a: per-utterance scores for model A.
        scores_b: per-utterance scores for model B.

    Returns:
        dict with keys: statistic, p_value, significant (α=0.05), n_pairs
    """
    try:
        from scipy.stats import wilcoxon  # type: ignore

        # Build aligned pairs (both non-None)
        pairs = [
            (a, b)
            for a, b in zip(scores_a, scores_b)
            if a is not None and b is not None and not math.isnan(a) and not math.isnan(b)
        ]
        if len(pairs) < 10:
            return {"statistic": None, "p_value": None, "significant": None, "n_pairs": len(pairs)}

        a_vals = [p[0] for p in pairs]
        b_vals = [p[1] for p in pairs]

        # Skip if all differences are zero
        diffs = [a - b for a, b in zip(a_vals, b_vals)]
        if all(d == 0 for d in diffs):
            return {"statistic": 0.0, "p_value": 1.0, "significant": False, "n_pairs": len(pairs)}

        stat, p_val = wilcoxon(a_vals, b_vals, alternative="two-sided")
        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "significant": float(p_val) < 0.05,
            "n_pairs": len(pairs),
        }

    except ImportError:
        return {"statistic": None, "p_value": None, "significant": None, "n_pairs": 0,
                "error": "scipy not installed"}
    except Exception as exc:
        return {"statistic": None, "p_value": None, "significant": None, "n_pairs": 0,
                "error": str(exc)}


def bonferroni_correct(
    p_values: List[Optional[float]],
    n_comparisons: Optional[int] = None,
) -> List[Optional[float]]:
    """Apply Bonferroni correction to a list of p-values.

    Args:
        p_values: raw p-values (None entries returned as None).
        n_comparisons: number of comparisons (default: len(p_values)).

    Returns:
        Corrected p-values (capped at 1.0).
    """
    n = n_comparisons or len([p for p in p_values if p is not None])
    if n == 0:
        return p_values
    return [
        min(1.0, p * n) if p is not None else None
        for p in p_values
    ]


def pairwise_significance(
    model_scores: Dict[str, List[Optional[float]]],
    metric: str = "composite",
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """Run Wilcoxon tests for all model pairs with Bonferroni correction.

    Args:
        model_scores: {model_name: [per-utterance scores], ...}
        metric:       name label for the metric being compared.
        alpha:        significance level.

    Returns:
        {
          "model_a vs model_b": {
            "statistic": float,
            "p_value_raw": float,
            "p_value_corrected": float,
            "significant": bool,
            "n_pairs": int,
          },
          ...
        }
    """
    model_names = list(model_scores.keys())
    pairs = list(combinations(model_names, 2))
    n_comparisons = len(pairs)

    raw_results: List[Dict] = []
    raw_p_values: List[Optional[float]] = []

    for a, b in pairs:
        res = wilcoxon_test(model_scores[a], model_scores[b])
        raw_results.append(res)
        raw_p_values.append(res.get("p_value"))

    corrected_p = bonferroni_correct(raw_p_values, n_comparisons)

    output: Dict[str, Dict] = {}
    for (a, b), res, p_corr in zip(pairs, raw_results, corrected_p):
        key = f"{a} vs {b}"
        output[key] = {
            "metric": metric,
            "statistic": res.get("statistic"),
            "p_value_raw": res.get("p_value"),
            "p_value_corrected": p_corr,
            "significant": (p_corr < alpha) if p_corr is not None else None,
            "n_pairs": res.get("n_pairs", 0),
        }

    return output
