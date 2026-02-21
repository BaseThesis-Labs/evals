#!/usr/bin/env python3
"""Aggregate metrics and compute rankings, including cross-dataset comparison."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from scipy import stats


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# ── Normalization bounds ───────────────────────────────────────────────────────
# Format: (floor, ceiling, direction)  direction ∈ {'higher', 'lower'}
METRIC_BOUNDS = {
    # Naturalness — predicted MOS scores
    'utmos':                (1.0,   5.0,     'higher'),
    'scoreq':               (1.0,   5.0,     'higher'),
    'nisqa_mos':            (1.0,   5.0,     'higher'),
    'nisqa_noisiness':      (1.0,   5.0,     'higher'),
    'nisqa_coloration':     (1.0,   5.0,     'higher'),
    'nisqa_discontinuity':  (1.0,   5.0,     'higher'),
    'nisqa_loudness':       (1.0,   5.0,     'higher'),
    'dnsmos_sig':           (1.0,   5.0,     'higher'),
    'dnsmos_bak':           (1.0,   5.0,     'higher'),
    'dnsmos_ovrl':          (1.0,   5.0,     'higher'),
    # Signal quality — reference-based
    'pesq':                 (-0.5,  4.5,     'higher'),  # ITU-T P.862
    'stoi':                 (0.0,   1.0,     'higher'),  # Short-Time Objective Intelligibility
    # Reference-free signal quality
    'output_snr':           (0.0,   40.0,    'higher'),  # dB, higher = cleaner
    'dynamic_range_db':     (0.0,   40.0,    'higher'),  # dB, higher = more expressive
    # Intelligibility
    'wer':                  (0.0,   1.0,     'lower'),
    'cer':                  (0.0,   1.0,     'lower'),
    'asr_mismatch':         (0.0,   1.0,     'lower'),
    'asr_mismatch_rate':    (0.0,   1.0,     'lower'),
    'word_skip_rate':       (0.0,   1.0,     'lower'),
    'insertion_rate':       (0.0,   1.0,     'lower'),
    'substitution_rate':    (0.0,   1.0,     'lower'),
    'semantic_distance':    (0.0,   1.0,     'lower'),  # 0 = identical meaning
    # Speaker similarity
    'ecapa_cosine_sim':     (0.0,   1.0,     'higher'),
    'resemblyzer_cosine_sim': (0.0, 1.0,    'higher'),
    # Robustness
    'has_repetition':       (0.0,   1.0,     'lower'),
    'has_silence_anomaly':  (0.0,   1.0,     'lower'),
    'is_empty_or_short':    (0.0,   1.0,     'lower'),
    # Latency
    'ttfa_ms':              (0.0,   5000.0,  'lower'),
    'rtf':                  (0.0,   10.0,    'lower'),
    'inference_time_ms':    (0.0,   30000.0, 'lower'),
    # Prosody — energy & voice quality
    'energy_mean':          (55.0,  85.0,    'higher'),  # dB; appropriate loudness
    'energy_std':           (0.0,   20.0,    'higher'),  # dB std; more variation = expressive
    'hnr':                  (5.0,   25.0,    'higher'),  # dB; higher = cleaner harmonic voice
    # Prosody — timing
    'pause_count':          (0.0,   15.0,    'lower'),   # fewer unnatural pauses = better
    'pause_mean_duration':  (0.0,   2.0,     'lower'),   # shorter pauses = more natural
    # speaking_rate, syllable_rate, jitter, shimmer, f0_range,
    # pause_ratio, duration_ratio use special normalizers (see normalize_metric)
}


# ── Dimension groupings ────────────────────────────────────────────────────────
DIMENSIONS = {
    'naturalness': [
        'utmos', 'scoreq', 'nisqa_mos', 'dnsmos_ovrl',
        'output_snr',                           # reference-free signal quality
        'pesq', 'stoi',                         # reference-based (None when no ref)
    ],
    'intelligibility': [
        'wer', 'cer', 'asr_mismatch', 'word_skip_rate',
        'semantic_distance',                    # round-trip semantic distance
    ],
    'speaker_similarity': [
        'ecapa_cosine_sim',
        'resemblyzer_cosine_sim',
    ],
    'prosody': [
        # Pitch
        'f0_range', 'jitter', 'shimmer', 'hnr',
        # Timing / rhythm
        'pause_ratio', 'speaking_rate', 'syllable_rate',
        'pause_count', 'pause_mean_duration',
        'duration_ratio',
        # Energy
        'energy_mean', 'energy_std', 'dynamic_range_db',
    ],
    'robustness': [
        'has_repetition', 'has_silence_anomaly', 'insertion_rate',
    ],
    'latency': [
        'rtf',  # Only RTF — ttfa_ms includes network overhead for API models
    ],
}


# ── Use-case composite weights ─────────────────────────────────────────────────
USE_CASES = {
    'conversational_ai': {
        'naturalness': 0.20,
        'intelligibility': 0.25,
        'speaker_similarity': 0.0,
        'prosody': 0.10,
        'robustness': 0.15,
        'latency': 0.30
    },
    'audiobook': {
        'naturalness': 0.35,
        'intelligibility': 0.15,
        'speaker_similarity': 0.10,
        'prosody': 0.25,
        'robustness': 0.15,
        'latency': 0.0
    },
    'voice_cloning': {
        'naturalness': 0.20,
        'intelligibility': 0.15,
        'speaker_similarity': 0.40,
        'prosody': 0.05,
        'robustness': 0.10,
        'latency': 0.10
    },
    'low_latency': {
        'naturalness': 0.15,
        'intelligibility': 0.20,
        'speaker_similarity': 0.0,
        'prosody': 0.05,
        'robustness': 0.15,
        'latency': 0.45
    },
    'balanced': {
        'naturalness': 0.20,
        'intelligibility': 0.20,
        'speaker_similarity': 0.15,
        'prosody': 0.10,
        'robustness': 0.15,
        'latency': 0.20
    },
}


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize_metric(value: float, metric: str) -> Optional[float]:
    """Normalize a raw metric value to [0, 1]."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # Special-case normalizers for target-based or non-linear metrics
    if metric == 'pause_ratio':
        # Ideal ~0.15; penalise deviation in either direction
        return 1.0 - min(abs(value - 0.15) / 0.85, 1.0)
    elif metric == 'f0_range':
        # Higher pitch range = more expressive; cap at 200 Hz
        return min(value / 200.0, 1.0)
    elif metric == 'speaking_rate':
        # Ideal ~3.0 wps; 0 or 6+ wps scores 0.0
        return 1.0 - min(abs(value - 3.0) / 3.0, 1.0)
    elif metric == 'syllable_rate':
        # Ideal ~4.5 sps (= 3 wps × 1.5 syl/word); 0 or 9+ sps = 0.0
        return 1.0 - min(abs(value - 4.5) / 4.5, 1.0)
    elif metric == 'duration_ratio':
        # Ideal ~1.0 (actual = expected); penalise deviation
        return 1.0 - min(abs(value - 1.0) / 1.0, 1.0)
    elif metric == 'jitter':
        # Target ~0.005 (0.5%); too low = robotic, too high = broken/unstable
        ideal = 0.005
        if value <= ideal:
            return value / ideal
        else:
            return max(0.0, 1.0 - (value - ideal) / 0.025)
    elif metric == 'shimmer':
        # Target ~0.04 (4%); same shape as jitter
        ideal = 0.04
        if value <= ideal:
            return value / ideal
        else:
            return max(0.0, 1.0 - (value - ideal) / 0.10)

    if metric not in METRIC_BOUNDS:
        return value  # Unknown metric — return as-is

    floor, ceiling, direction = METRIC_BOUNDS[metric]

    # Clip and normalize to [0, 1]
    normalized = (np.clip(value, floor, ceiling) - floor) / (ceiling - floor)

    # Invert if lower is better
    if direction == 'lower':
        normalized = 1.0 - normalized

    return float(normalized)


def compute_dimension_score(utterances: List[Dict], metrics: List[str]) -> Optional[float]:
    """Compute average normalized score for a dimension."""
    all_scores = []

    for utt in utterances:
        for metric in metrics:
            value = utt.get(metric)
            if value is not None:
                normalized = normalize_metric(value, metric)
                if normalized is not None:
                    all_scores.append(normalized)

    if not all_scores:
        return None

    return float(np.mean(all_scores))


def compute_composite_score(dimensions: Dict, use_case: str) -> Optional[float]:
    """Compute weighted composite score for a use case."""
    weights = USE_CASES[use_case]
    score = 0.0
    total_weight = 0.0

    for dim_name, weight in weights.items():
        if weight > 0 and dimensions.get(dim_name) is not None:
            score += dimensions[dim_name] * weight
            total_weight += weight

    if total_weight == 0:
        return None

    # Renormalize if some dimensions are missing
    return score / total_weight


def compute_significance(model_results: Dict) -> Dict:
    """Compute statistical significance between models using Wilcoxon test."""
    from itertools import combinations

    model_names = list(model_results.keys())
    significance = {}

    # Get UTMOS scores for each model
    utmos_scores = {}
    for model in model_names:
        scores = []
        for utt in model_results[model]['per_utterance']:
            if utt.get('utmos') is not None:
                scores.append(utt['utmos'])
        utmos_scores[model] = scores

    # Pairwise comparisons
    for model1, model2 in combinations(model_names, 2):
        scores1 = utmos_scores[model1]
        scores2 = utmos_scores[model2]

        if len(scores1) > 5 and len(scores2) > 5:
            # Wilcoxon signed-rank test (paired)
            min_len = min(len(scores1), len(scores2))
            statistic, p_value = stats.wilcoxon(
                scores1[:min_len],
                scores2[:min_len],
                alternative='two-sided'
            )

            # Bonferroni correction
            n_comparisons = len(list(combinations(model_names, 2)))
            alpha = 0.05 / n_comparisons

            significance[f"{model1}_vs_{model2}"] = {
                'p_value': float(p_value),
                'significant_005': bool(p_value < alpha),
                'n_pairs': min_len,
                'effect_size_r': None  # Could compute rank-biserial r
            }

    return significance


def aggregate_model(result_data: Dict) -> Dict:
    """Aggregate metrics for one model (across all utterances)."""
    utterances = result_data['per_utterance']

    # Raw means and stds
    raw_means = {}
    raw_stds = {}

    all_metrics = set()
    for utt in utterances:
        all_metrics.update(utt.keys())
    # Exclude non-numeric metadata fields
    all_metrics -= {'id', 'dataset', 'category', 'difficulty'}

    for metric in all_metrics:
        values = [utt.get(metric) for utt in utterances
                  if utt.get(metric) is not None and isinstance(utt.get(metric), (int, float))]
        if values:
            raw_means[metric] = float(np.mean(values))
            raw_stds[metric] = float(np.std(values))

    # Dimension scores
    dimensions = {}
    for dim_name, dim_metrics in DIMENSIONS.items():
        score = compute_dimension_score(utterances, dim_metrics)
        if score is not None:
            dimensions[dim_name] = score

    # Composite scores
    composites = {}
    for use_case in USE_CASES.keys():
        score = compute_composite_score(dimensions, use_case)
        if score is not None:
            composites[use_case] = score

    return {
        'raw_means': raw_means,
        'raw_stds': raw_stds,
        'dimensions': dimensions,
        'composites': composites,
    }


# ── Dataset-level analysis ─────────────────────────────────────────────────────

def aggregate_by_dataset(model_results: Dict) -> Dict:
    """Aggregate metrics grouped by dataset tag for each model.

    Returns:
        {
          model_name: {
            dataset_name: {
              'n_utterances': int,
              'dimensions': {dim: score},
              'composites': {use_case: score},
              'raw_means': {metric: mean},
            }
          }
        }
    """
    result = {}

    for model_name, model_data in model_results.items():
        result[model_name] = {}

        # Group utterances by dataset
        by_ds: Dict[str, List[Dict]] = {}
        for utt in model_data['per_utterance']:
            ds = utt.get('dataset', 'unknown')
            by_ds.setdefault(ds, []).append(utt)

        for ds_name, utterances in by_ds.items():
            # Raw means for this dataset subset
            all_metrics = set()
            for utt in utterances:
                all_metrics.update(utt.keys())
            all_metrics -= {'id', 'dataset', 'category', 'difficulty'}

            raw_means: Dict[str, float] = {}
            for metric in all_metrics:
                values = [utt.get(metric) for utt in utterances
                          if utt.get(metric) is not None
                          and isinstance(utt.get(metric), (int, float))]
                if values:
                    raw_means[metric] = float(np.mean(values))

            # Dimension scores
            dims: Dict[str, float] = {}
            for dim_name, dim_metrics in DIMENSIONS.items():
                score = compute_dimension_score(utterances, dim_metrics)
                if score is not None:
                    dims[dim_name] = score

            # Composite scores
            composites: Dict[str, float] = {}
            for use_case in USE_CASES:
                score = compute_composite_score(dims, use_case)
                if score is not None:
                    composites[use_case] = score

            result[model_name][ds_name] = {
                'n_utterances': len(utterances),
                'dimensions': dims,
                'composites': composites,
                'raw_means': raw_means,
            }

    return result


def compute_model_consistency(by_dataset: Dict) -> Dict:
    """Compute how consistent each model is across datasets.

    Uses the balanced composite score across datasets.
    Returns:
        {
          model_name: {
            'mean': float,          # average balanced score
            'std': float,           # std dev across datasets (lower = more consistent)
            'min': float,
            'max': float,
            'range': float,         # max - min
            'n_datasets': int,
            'consistency_score': float,  # 0 = very inconsistent, 1 = perfectly consistent
            'per_dataset': {ds: balanced_score}
          }
        }
    """
    consistency = {}

    for model_name, dataset_scores in by_dataset.items():
        per_ds: Dict[str, float] = {}
        for ds_name, scores in dataset_scores.items():
            bal = scores['composites'].get('balanced')
            if bal is not None:
                per_ds[ds_name] = bal

        if len(per_ds) < 2:
            # Need at least 2 datasets to measure consistency
            consistency[model_name] = {
                'mean': per_ds[list(per_ds.keys())[0]] if per_ds else None,
                'std': 0.0,
                'min': per_ds[list(per_ds.keys())[0]] if per_ds else None,
                'max': per_ds[list(per_ds.keys())[0]] if per_ds else None,
                'range': 0.0,
                'n_datasets': len(per_ds),
                'consistency_score': 1.0 if per_ds else None,
                'per_dataset': per_ds,
            }
            continue

        vals = list(per_ds.values())
        std = float(np.std(vals))

        # Consistency score: 1.0 if std=0, decreases as std grows
        # Max expected std across wildly different datasets ≈ 0.3
        consistency_score = float(max(0.0, 1.0 - std / 0.3))

        consistency[model_name] = {
            'mean': float(np.mean(vals)),
            'std': std,
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'range': float(np.max(vals) - np.min(vals)),
            'n_datasets': len(per_ds),
            'consistency_score': consistency_score,
            'per_dataset': per_ds,
        }

    return consistency


def compute_dataset_difficulty(by_dataset: Dict) -> Dict:
    """Rank datasets by difficulty (lower avg balanced score = harder).

    Returns:
        {
          dataset_name: {
            'avg_balanced': float,   # averaged over all models
            'difficulty_rank': int,  # 1 = hardest
            'model_scores': {model: balanced_score}
          }
        }
    """
    # Collect balanced scores per dataset across all models
    ds_scores: Dict[str, Dict[str, float]] = {}

    for model_name, dataset_scores in by_dataset.items():
        for ds_name, scores in dataset_scores.items():
            bal = scores['composites'].get('balanced')
            if bal is not None:
                ds_scores.setdefault(ds_name, {})[model_name] = bal

    # Compute average per dataset
    difficulty = {}
    for ds_name, model_scores in ds_scores.items():
        avg = float(np.mean(list(model_scores.values())))
        difficulty[ds_name] = {
            'avg_balanced': avg,
            'model_scores': model_scores,
        }

    # Rank (1 = hardest = lowest avg score)
    ranked = sorted(difficulty.keys(), key=lambda d: difficulty[d]['avg_balanced'])
    for rank, ds_name in enumerate(ranked, 1):
        difficulty[ds_name]['difficulty_rank'] = rank

    return difficulty


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Aggregate evaluation results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory')
    parser.add_argument('--output', type=str, default='analysis/leaderboard.json',
                       help='Output file')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    result_files = list(results_dir.glob('*_metrics.json'))

    print(f"✓ Found {len(result_files)} model result files")

    # Load all results
    model_results = {}
    for file_path in result_files:
        with open(file_path) as f:
            data = json.load(f)
            model_name = data['model']
            model_results[model_name] = data

    # Aggregate each model
    aggregated = {}
    for model_name, result_data in model_results.items():
        print(f"▶ Aggregating {model_name}")
        aggregated[model_name] = aggregate_model(result_data)

    # Compute rankings
    rankings = {}
    for use_case in USE_CASES.keys():
        scores = []
        for model_name, data in aggregated.items():
            score = data['composites'].get(use_case)
            if score is not None:
                scores.append((model_name, score))

        # Sort descending
        scores.sort(key=lambda x: x[1], reverse=True)
        rankings[use_case] = [name for name, score in scores]

    # Compute significance
    print("▶ Computing statistical significance...")
    significance = compute_significance(model_results)

    # ── Dataset comparison analysis ──────────────────────────────────────────
    has_multi_dataset = any(
        utt.get('dataset') and utt['dataset'] != 'unknown'
        for model_data in model_results.values()
        for utt in model_data['per_utterance']
    )

    by_dataset = {}
    consistency = {}
    dataset_difficulty = {}

    if has_multi_dataset:
        print("▶ Computing dataset comparison analysis...")
        by_dataset = aggregate_by_dataset(model_results)
        consistency = compute_model_consistency(by_dataset)
        dataset_difficulty = compute_dataset_difficulty(by_dataset)

        # Dataset count summary
        all_datasets = set()
        for model_ds in by_dataset.values():
            all_datasets.update(model_ds.keys())
        print(f"  Found {len(all_datasets)} dataset(s): {', '.join(sorted(all_datasets))}")

    # ── Build leaderboard ────────────────────────────────────────────────────
    leaderboard = {
        'config': {
            'dataset': 'multi' if has_multi_dataset else 'seed_tts_eval',
            'n_models': len(model_results),
            'n_utterances': len(next(iter(model_results.values()))['per_utterance']) if model_results else 0,
            'n_active_metrics': 44,  # Updated count
            'has_dataset_comparison': has_multi_dataset,
        },
        'rankings': rankings,
        'models': aggregated,
        'significance': significance,
    }

    if has_multi_dataset:
        leaderboard['by_dataset'] = by_dataset
        leaderboard['model_consistency'] = consistency
        leaderboard['dataset_difficulty'] = dataset_difficulty

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert NumPy types to native Python types for JSON serialization
    leaderboard = convert_numpy_types(leaderboard)

    with open(output_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    print(f"\n✓ Saved leaderboard to {output_path}")

    # Print summary
    print("\n" + "="*50)
    print("RANKINGS (Balanced)")
    print("="*50)
    for i, model in enumerate(rankings['balanced'], 1):
        score = aggregated[model]['composites']['balanced']
        print(f"{i}. {model:15s}  {score:.3f}")

    if has_multi_dataset and consistency:
        print("\n" + "="*50)
        print("MODEL CONSISTENCY ACROSS DATASETS")
        print("="*50)
        sorted_models = sorted(
            consistency.items(),
            key=lambda x: x[1].get('consistency_score', 0) or 0,
            reverse=True
        )
        for model, c in sorted_models:
            cs = c.get('consistency_score')
            std = c.get('std', 0)
            n = c.get('n_datasets', 0)
            if cs is not None:
                print(f"  {model:15s}  consistency={cs:.3f}  std={std:.3f}  n_datasets={n}")

        print("\n" + "="*50)
        print("DATASET DIFFICULTY (1 = hardest)")
        print("="*50)
        for ds, info in sorted(dataset_difficulty.items(),
                                key=lambda x: x[1]['difficulty_rank']):
            print(f"  #{info['difficulty_rank']}  {ds:40s}  avg_balanced={info['avg_balanced']:.3f}")


if __name__ == '__main__':
    main()
