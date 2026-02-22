"""src/evaluation/statistical.py — Bootstrap CI and paired significance tests."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def bootstrap_wer_ci(
    wer_values: list[float],
    n_iter: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Non-parametric bootstrap CI for mean WER. Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    arr = np.array([v for v in wer_values if np.isfinite(v)])
    if len(arr) < 2:
        val = float(arr.mean()) if len(arr) == 1 else float("nan")
        return val, val
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_iter)])
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def blockwise_bootstrap_ci(
    per_sample: list[dict],
    metric: str = "wer",
    block_field: str = "speaker_id",
    n_iter: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Block-wise bootstrap — resample whole speakers to respect within-speaker correlation.

    Falls back to sample-level bootstrap when there are fewer than 5 distinct speaker
    blocks (e.g. single-speaker datasets), because resampling 1–4 blocks gives a
    degenerate CI (lo = hi = grand mean).
    """
    rng = np.random.default_rng(seed)
    blocks: dict[str, list[float]] = defaultdict(list)
    for s in per_sample:
        val = s.get(metric)
        if val is not None and np.isfinite(val):
            blocks[str(s.get(block_field, "unknown"))].append(val)

    block_list = list(blocks.values())
    if not block_list:
        return float("nan"), float("nan")

    # Fall back to sample-level bootstrap when too few blocks to resample meaningfully
    if len(block_list) < 5:
        arr = np.array([v for blk in block_list for v in blk])
        if len(arr) < 2:
            val = float(arr.mean()) if len(arr) == 1 else float("nan")
            return val, val
        means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                          for _ in range(n_iter)])
        alpha = (1 - confidence) / 2
        return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))

    means = []
    for _ in range(n_iter):
        idx = rng.choice(len(block_list), size=len(block_list), replace=True)
        vals = [v for i in idx for v in block_list[i]]
        means.append(np.mean(vals))

    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_significance_test(wers_a: list[float], wers_b: list[float]) -> dict:
    """Paired t-test between two models on identical samples."""
    from scipy import stats
    a = np.array([v for v in wers_a if np.isfinite(v)])
    b = np.array([v for v in wers_b if np.isfinite(v)])
    n = min(len(a), len(b))
    if n < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}
    res = stats.ttest_rel(a[:n], b[:n])
    p = float(res.pvalue)
    return {
        "statistic":   float(res.statistic),
        "p_value":     p,
        "significant": p < 0.05,
        "direction":   "a_better" if res.statistic < 0 else "b_better",
        "n_pairs":     n,
    }


def wilcoxon_test(wers_a: list[float], wers_b: list[float]) -> dict:
    """Non-parametric Wilcoxon signed-rank test (more robust than t-test)."""
    from scipy import stats
    a = np.array([v for v in wers_a if np.isfinite(v)])
    b = np.array([v for v in wers_b if np.isfinite(v)])
    n = min(len(a), len(b))
    if n < 10:
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}
    res = stats.wilcoxon(a[:n], b[:n], alternative="two-sided")
    p = float(res.pvalue)
    return {"statistic": float(res.statistic), "p_value": p, "significant": p < 0.05, "n_pairs": n}


def mapsswe_test(wers_a: list[float], wers_b: list[float]) -> dict:
    """
    MAPSSWE (Matched-pairs Sentence-Segment Word Error) significance test.

    The NIST sc_stats MAPSSWE test is equivalent to a sign test on per-utterance
    WER differences.  For each utterance u, compute d_u = WER_A(u) - WER_B(u).
    Discard ties (d_u = 0).  Among non-tied utterances, let N+ = count(d_u < 0)
    (A beats B).  Under H0: N+ ~ Binomial(N, 0.5).

    Z = (N+ - N/2) / sqrt(N/4)  →  two-tailed p via normal approximation.

    Returns raw p_value (call bonferroni_correct() for multiple-comparison adjustment).
    """
    a = np.array([v for v in wers_a if np.isfinite(v)])
    b = np.array([v for v in wers_b if np.isfinite(v)])
    n = min(len(a), len(b))
    if n < 4:
        return {
            "test": "MAPSSWE",
            "n_pairs": n, "n_nontied": 0,
            "n_a_better": 0, "n_b_better": 0,
            "z_statistic": float("nan"), "p_value": float("nan"),
            "significant_raw": False,
            "direction": "insufficient_data",
        }

    diff = a[:n] - b[:n]
    non_tied = diff[diff != 0.0]
    N = len(non_tied)
    if N == 0:
        return {
            "test": "MAPSSWE",
            "n_pairs": n, "n_nontied": 0,
            "n_a_better": 0, "n_b_better": 0,
            "z_statistic": 0.0, "p_value": 1.0,
            "significant_raw": False,
            "direction": "tie",
        }

    n_a_better = int((non_tied < 0).sum())   # A has lower WER
    n_b_better = int((non_tied > 0).sum())

    # Sign-test statistic with continuity correction
    n_plus = max(n_a_better, n_b_better)
    z = (n_plus - 0.5 - N / 2) / np.sqrt(N / 4)

    from scipy import stats
    p = float(2 * stats.norm.sf(abs(z)))  # two-tailed

    return {
        "test": "MAPSSWE",
        "n_pairs":      n,
        "n_nontied":    N,
        "n_a_better":   n_a_better,
        "n_b_better":   n_b_better,
        "z_statistic":  float(z),
        "p_value":      p,
        "significant_raw": p < 0.05,
        "direction": "a_better" if n_a_better > n_b_better else "b_better",
    }


def bonferroni_correct(
    results: list[dict],
    p_key: str = "p_value",
    alpha: float = 0.05,
) -> list[dict]:
    """
    Apply Bonferroni correction to a list of test result dicts.
    Adds 'p_value_bonferroni' and 'significant_bonferroni' keys.
    """
    m = len(results)
    if m == 0:
        return results
    for r in results:
        p_raw = r.get(p_key, float("nan"))
        p_corr = min(1.0, p_raw * m) if np.isfinite(p_raw) else float("nan")
        r["p_value_bonferroni"] = p_corr
        r["significant_bonferroni"] = p_corr < alpha if np.isfinite(p_corr) else False
    return results


def all_pairs_mapsswe(
    model_wers: dict[str, list[float]],
    alpha: float = 0.05,
) -> list[dict]:
    """
    Run MAPSSWE for every pair of models and apply Bonferroni correction.

    Args:
        model_wers: {model_name: [per_sample_wer, ...]}
        alpha:      family-wise error rate

    Returns:
        List of dicts, one per model pair, with Bonferroni-adjusted p-values.
    """
    models = sorted(model_wers.keys())
    pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]

    results = []
    for a, b in pairs:
        res = mapsswe_test(model_wers[a], model_wers[b])
        res["model_a"] = a
        res["model_b"] = b
        results.append(res)

    bonferroni_correct(results, p_key="p_value", alpha=alpha)
    return results
