"""src/evaluation/fairness.py — Disaggregated WER by metadata groups."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def compute_disaggregated_wer(
    per_sample: list[dict],
    group_by: list[str] | None = None,
) -> dict:
    """
    Micro-average WER broken down by each group_by field.
    Returns {field: {group_value: {micro_wer, mean_wer, std_wer, n_samples}, wer_gap}}.
    """
    if group_by is None:
        group_by = ["speaker_id", "tts_model", "case_study"]

    results: dict = {}
    for field in group_by:
        groups: dict[str, list[dict]] = defaultdict(list)
        for s in per_sample:
            val = s.get(field, "")
            if val:
                groups[str(val)].append(s)
        if not groups:
            continue

        field_results: dict = {}
        for group_val, samples in sorted(groups.items()):
            valid = [s for s in samples if np.isfinite(s.get("wer", float("inf")))]
            if not valid:
                continue
            total_subs = sum(s.get("substitutions", 0) for s in valid)
            total_dels = sum(s.get("deletions",     0) for s in valid)
            total_ins  = sum(s.get("insertions",    0) for s in valid)
            total_hits = sum(s.get("hits",          0) for s in valid)
            total_ref  = total_hits + total_subs + total_dels
            micro_wer  = (total_subs + total_dels + total_ins) / total_ref if total_ref > 0 else float("inf")
            wers = [s["wer"] for s in valid]
            field_results[group_val] = {
                "micro_wer": micro_wer,
                "mean_wer":  float(np.mean(wers)),
                "std_wer":   float(np.std(wers)),
                "n_samples": len(valid),
            }

        finite_wers = [v["micro_wer"] for v in field_results.values() if np.isfinite(v["micro_wer"])]
        gap = (max(finite_wers) - min(finite_wers)) if len(finite_wers) >= 2 else float("nan")
        results[field] = {"groups": field_results, "wer_gap": gap}

    return results
