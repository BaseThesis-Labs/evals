#!/usr/bin/env python3
"""Generate visualizations from leaderboard.
All charts are fully dynamic — they use whatever models appear in
leaderboard.json, ordered by the balanced ranking.
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#dee2e6',
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.color': '#dee2e6',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.6,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
})

# Palette — scales to any number of models
_CMAP = plt.cm.get_cmap('tab10')


def model_colors(model_names: list) -> dict:
    """Return a colour per model, consistent across all charts."""
    return {m: _CMAP(i % 10) for i, m in enumerate(model_names)}


def ranked_models(data: dict) -> list:
    """Models that have real evaluation data, in balanced-ranking order.

    Filters out any model whose composite scores are all zero/None —
    these are failed installs (e.g. piper, xtts) that ended up in the
    leaderboard with empty metrics.
    """
    ranked = list(data['rankings'].get('balanced', []))
    # Append any model present in data but missing from the ranking
    for m in data['models']:
        if m not in ranked:
            ranked.append(m)

    def _has_real_data(model: str) -> bool:
        composites = data['models'][model].get('composites', {})
        return any(v is not None and v > 0 for v in composites.values())

    return [m for m in ranked if _has_real_data(m)]


def result_models(data: dict, results_dir: Path) -> list:
    """All models that have a result JSON on disk, ordered by leaderboard rank.

    This includes models with result files that were not yet re-aggregated
    into the current leaderboard (e.g. hume, lmnt added after the last
    aggregate.py run).
    """
    leaderboard_order = ranked_models(data)
    on_disk = {
        f.stem.replace('_metrics', '')
        for f in results_dir.glob('*_metrics.json')
    }
    # Leaderboard models first (in rank order), then extras alphabetically
    ordered = [m for m in leaderboard_order if m in on_disk]
    extras  = sorted(m for m in on_disk if m not in leaderboard_order)
    return ordered + extras


# ── Normalisation (mirrors aggregate.py) ──────────────────────────────────────
_BOUNDS = {
    'utmos':                  (1.0,  5.0,     'higher'),
    'dnsmos_sig':             (1.0,  5.0,     'higher'),
    'dnsmos_bak':             (1.0,  5.0,     'higher'),
    'dnsmos_ovrl':            (1.0,  5.0,     'higher'),
    'wer':                    (0.0,  1.0,     'lower'),
    'cer':                    (0.0,  1.0,     'lower'),
    'mer':                    (0.0,  1.0,     'lower'),
    'word_skip_rate':         (0.0,  1.0,     'lower'),
    'insertion_rate':         (0.0,  1.0,     'lower'),
    'substitution_rate':      (0.0,  1.0,     'lower'),
    'resemblyzer_cosine_sim': (0.0,  1.0,     'higher'),
    'ecapa_cosine_sim':       (0.0,  1.0,     'higher'),
    'has_repetition':         (0.0,  1.0,     'lower'),
    'has_silence_anomaly':    (0.0,  1.0,     'lower'),
    'is_empty_or_short':      (0.0,  1.0,     'lower'),
    'rtf':                    (0.0,  10.0,    'lower'),
    'ttfa_ms':                (0.0,  5000.0,  'lower'),
    'inference_time_ms':      (0.0,  30000.0, 'lower'),
    # Prosody — energy & voice quality
    'energy_mean':            (55.0, 85.0,    'higher'),
    'energy_std':             (0.0,  20.0,    'higher'),
    'hnr':                    (5.0,  25.0,    'higher'),
}


def _normalize(value, metric: str) -> float:
    """Normalise a raw metric value to [0, 1]."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    # Special cases
    if metric == 'pause_ratio':
        return 1.0 - min(abs(value - 0.15) / 0.85, 1.0)
    if metric == 'f0_range':
        return min(value / 200.0, 1.0)
    if metric == 'speaking_rate':
        return 1.0 - min(abs(value - 3.0) / 3.0, 1.0)
    if metric == 'jitter':
        ideal = 0.005
        return (value / ideal) if value <= ideal else max(0.0, 1.0 - (value - ideal) / 0.025)
    if metric == 'shimmer':
        ideal = 0.04
        return (value / ideal) if value <= ideal else max(0.0, 1.0 - (value - ideal) / 0.10)
    if metric in _BOUNDS:
        lo, hi, direction = _BOUNDS[metric]
        norm = (np.clip(value, lo, hi) - lo) / (hi - lo)
        return (1.0 - norm) if direction == 'lower' else norm
    return float(value)


def _available_metrics(models_data: dict, candidates: list) -> list:
    """Return only metrics where at least one model has a real value."""
    out = []
    for m in candidates:
        for md in models_data.values():
            v = md['raw_means'].get(m)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                out.append(m)
                break
    return out


# ── Chart 1: Leaderboard ──────────────────────────────────────────────────────
def plot_leaderboard(data: dict, output_dir: Path):
    models = ranked_models(data)
    scores = [data['models'][m]['composites'].get('balanced', 0) for m in models]
    colors = model_colors(models)

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.65)))

    bars = ax.barh(models, scores,
                   color=[colors[m] for m in models],
                   edgecolor='white', linewidth=0.8, height=0.6)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.008, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 1.08)
    ax.set_xlabel('Balanced Composite Score  [0 – 1]')
    ax.set_title('TTS Model Leaderboard — Balanced Score')
    ax.invert_yaxis()
    ax.axvline(0, color='#adb5bd', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / '01_leaderboard.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Chart 2: Radar ────────────────────────────────────────────────────────────
def plot_radar(data: dict, output_dir: Path):
    dims = ['naturalness', 'intelligibility', 'speaker_similarity',
            'prosody', 'robustness', 'latency']
    dim_labels = ['Naturalness', 'Intelligibility', 'Spk\nSimilarity',
                  'Prosody', 'Robustness', 'Latency']

    models = ranked_models(data)
    colors = model_colors(models)

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))
    ax.set_facecolor('#f8f9fa')

    for model in models:
        dim_scores = data['models'][model].get('dimensions', {})
        vals = [dim_scores.get(d, 0) or 0 for d in dims] + [dim_scores.get(dims[0], 0) or 0]
        ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=colors[model])
        ax.fill(angles, vals, alpha=0.07, color=colors[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=7, color='#6c757d')
    ax.set_title('Dimension Scores by Model', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.12))

    plt.tight_layout()
    plt.savefig(output_dir / '02_radar.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Chart 3: Metrics Heatmap ─────────────────────────────────────────────────
def plot_heatmap(data: dict, output_dir: Path):
    models = ranked_models(data)
    models_data = data['models']

    # Candidate metrics in display order — only show what's actually populated
    candidates = [
        'utmos', 'dnsmos_ovrl', 'dnsmos_sig',
        'wer', 'cer', 'word_skip_rate',
        'resemblyzer_cosine_sim',
        'f0_range', 'pause_ratio', 'speaking_rate',
        'energy_mean', 'energy_std',
        'jitter', 'shimmer', 'hnr',
        'rtf', 'ttfa_ms',
    ]
    metrics = _available_metrics(models_data, candidates)

    pretty = {
        'utmos': 'UTMOS', 'dnsmos_ovrl': 'DNSMOS\nOverall',
        'dnsmos_sig': 'DNSMOS\nSig', 'wer': 'WER',
        'cer': 'CER', 'word_skip_rate': 'Skip\nRate',
        'resemblyzer_cosine_sim': 'Spk Sim\n(Resemblyzer)',
        'f0_range': 'F0 Range', 'pause_ratio': 'Pause\nRatio',
        'speaking_rate': 'Speaking\nRate',
        'energy_mean': 'Energy\nMean', 'energy_std': 'Energy\nStd',
        'jitter': 'Jitter', 'shimmer': 'Shimmer', 'hnr': 'HNR',
        'rtf': 'RTF', 'ttfa_ms': 'TTFA (ms)',
    }

    matrix = np.full((len(models), len(metrics)), np.nan)
    raw_matrix = np.full((len(models), len(metrics)), np.nan)

    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            raw = models_data[model]['raw_means'].get(metric)
            if raw is not None:
                raw_matrix[i, j] = raw
                matrix[i, j] = _normalize(raw, metric)

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 0.85), max(4, len(models) * 0.8)))

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.RdYlGn
    cmap.set_bad('#e9ecef')
    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([pretty.get(m, m) for m in metrics], rotation=35, ha='right')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    # Annotate with raw values
    for i in range(len(models)):
        for j in range(len(metrics)):
            raw = raw_matrix[i, j]
            if not np.isnan(raw):
                norm = matrix[i, j]
                txt_color = 'white' if (norm < 0.25 or norm > 0.82) else 'black'
                # Format: small numbers as %.3f, large as %.1f
                label = f'{raw:.1f}' if abs(raw) >= 10 else f'{raw:.3f}'
                ax.text(j, i, label, ha='center', va='center',
                        color=txt_color, fontsize=7.5, fontweight='bold')
            else:
                ax.text(j, i, '—', ha='center', va='center',
                        color='#adb5bd', fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Normalised Score  [0 = worst, 1 = best]', fontsize=9)

    ax.set_title('Raw Metric Values (colour = normalised score)')
    plt.tight_layout()
    plt.savefig(output_dir / '03_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Chart 4: Use-Case Composites ──────────────────────────────────────────────
def plot_use_cases(data: dict, output_dir: Path):
    models = ranked_models(data)
    colors = model_colors(models)
    use_cases = ['conversational_ai', 'audiobook', 'voice_cloning',
                 'low_latency', 'balanced']
    uc_labels = ['Conversational\nAI', 'Audiobook', 'Voice\nCloning',
                 'Low\nLatency', 'Balanced']

    n_uc = len(use_cases)
    n_m = len(models)
    width = 0.8 / n_m
    x = np.arange(n_uc)

    fig, ax = plt.subplots(figsize=(13, 6))

    for idx, model in enumerate(models):
        scores = [data['models'][model]['composites'].get(uc, 0) or 0 for uc in use_cases]
        offset = (idx - (n_m - 1) / 2) * width
        bars = ax.bar(x + offset, scores, width * 0.92,
                      color=colors[model], label=model,
                      edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(uc_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Composite Score  [0 – 1]')
    ax.set_title('Model Performance by Use Case')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.axhline(y=0, color='#adb5bd', linewidth=0.6)

    plt.tight_layout()
    plt.savefig(output_dir / '04_use_cases.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Chart 5: Dimension Breakdown ─────────────────────────────────────────────
def plot_dimension_breakdown(data: dict, output_dir: Path):
    """Stacked view of all 6 dimension scores per model."""
    models = ranked_models(data)
    colors_map = model_colors(models)
    dims = ['naturalness', 'intelligibility', 'speaker_similarity',
            'prosody', 'robustness', 'latency']
    dim_labels = ['Naturalness', 'Intelligibility', 'Spk Similarity',
                  'Prosody', 'Robustness', 'Latency']

    dim_colors = ['#e63946', '#457b9d', '#8338ec', '#2ec4b6', '#f4a261', '#6d6875']

    x = np.arange(len(models))
    bar_w = 0.13

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 6))

    for j, (dim, dlabel, dcol) in enumerate(zip(dims, dim_labels, dim_colors)):
        scores = [data['models'][m].get('dimensions', {}).get(dim, 0) or 0 for m in models]
        offset = (j - (len(dims) - 1) / 2) * bar_w
        ax.bar(x + offset, scores, bar_w * 0.9, label=dlabel,
               color=dcol, edgecolor='white', linewidth=0.4, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Dimension Score  [0 – 1]')
    ax.set_title('Dimension Breakdown per Model')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    ax.axhline(1.0, color='#adb5bd', linewidth=0.6, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / '05_dimension_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Chart 6: Prosody Dashboard ────────────────────────────────────────────────
def plot_prosody_dashboard(data: dict, output_dir: Path):
    """Show all prosody sub-metrics as grouped bars with raw values."""
    models = ranked_models(data)
    colors_map = model_colors(models)
    models_data = data['models']

    prosody_metrics = [
        ('f0_range',      'F0 Range (Hz)',       None),
        ('pause_ratio',   'Pause Ratio',          None),
        ('speaking_rate', 'Speaking Rate (wps)',  None),
        ('energy_mean',   'Energy Mean (dB)',     None),
        ('energy_std',    'Energy Std (dB)',      None),
        ('jitter',        'Jitter',               None),
        ('shimmer',       'Shimmer',              None),
        ('hnr',           'HNR (dB)',             None),
    ]
    # Filter to only metrics with at least one value
    available = [(k, lbl, _) for k, lbl, _ in prosody_metrics
                 if any(models_data[m]['raw_means'].get(k) is not None for m in models)]

    if not available:
        print('  ⚠ Skipping prosody dashboard (no prosody data)')
        return

    n_metrics = len(available)
    ncols = min(4, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.array(axes).flatten()

    for idx, (metric, label, _) in enumerate(available):
        ax = axes[idx]
        raw_vals = [models_data[m]['raw_means'].get(metric) for m in models]
        norm_vals = [_normalize(v, metric) if v is not None else np.nan for v in raw_vals]

        bar_colors = [colors_map[m] for m in models]
        bars = ax.bar(models, norm_vals, color=bar_colors,
                      edgecolor='white', linewidth=0.5, width=0.6)

        # Annotate with raw value
        for bar, raw in zip(bars, raw_vals):
            if raw is not None and not np.isnan(bar.get_height()):
                label_str = f'{raw:.2f}' if abs(raw) < 100 else f'{raw:.0f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        label_str, ha='center', va='bottom', fontsize=7.5)

        ax.set_ylim(0, 1.15)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xticklabels(models, rotation=25, ha='right', fontsize=8)
        ax.axhline(1.0, color='#adb5bd', linewidth=0.5, linestyle='--')

    # Hide unused subplots
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle('Prosody Sub-Metrics  (bar height = normalised score, label = raw value)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / '06_prosody_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Chart 7: WER Distribution ─────────────────────────────────────────────────
def plot_wer_distribution(data: dict, results_dir: Path, output_dir: Path):
    models = result_models(data, results_dir)
    wer_data = {}
    for model in models:
        f = results_dir / f'{model}_metrics.json'
        if f.exists():
            with open(f) as fh:
                res = json.load(fh)
            vals = [u['wer'] for u in res['per_utterance'] if u.get('wer') is not None]
            if vals:
                wer_data[model] = vals

    if not wer_data:
        print('  ⚠ Skipping WER distribution (no data)')
        return

    ordered = [m for m in models if m in wer_data]
    colors_map = model_colors(models)

    fig, ax = plt.subplots(figsize=(max(7, len(ordered) * 1.2), 5))

    bp = ax.boxplot([wer_data[m] for m in ordered],
                    patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))

    for patch, model in zip(bp['boxes'], ordered):
        patch.set_facecolor(colors_map[model])
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(ordered) + 1))
    ax.set_xticklabels(ordered, rotation=20, ha='right')
    ax.set_ylabel('Word Error Rate (WER)')
    ax.set_title('WER Distribution per Model (per-utterance)')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / '07_wer_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Chart 8: UTMOS Distribution ──────────────────────────────────────────────
def plot_utmos_distribution(data: dict, results_dir: Path, output_dir: Path):
    models = result_models(data, results_dir)
    utmos_data = {}
    for model in models:
        f = results_dir / f'{model}_metrics.json'
        if f.exists():
            with open(f) as fh:
                res = json.load(fh)
            vals = [u['utmos'] for u in res['per_utterance'] if u.get('utmos') is not None]
            if vals:
                utmos_data[model] = vals

    if not utmos_data:
        print('  ⚠ Skipping UTMOS distribution (no data)')
        return

    ordered = [m for m in models if m in utmos_data]
    colors_map = model_colors(models)

    fig, ax = plt.subplots(figsize=(max(7, len(ordered) * 1.2), 5))

    positions = range(1, len(ordered) + 1)
    parts = ax.violinplot([utmos_data[m] for m in ordered],
                          positions=positions,
                          showmeans=True, showmedians=True)

    for i, (pc, model) in enumerate(zip(parts['bodies'], ordered)):
        pc.set_facecolor(colors_map[model])
        pc.set_alpha(0.75)

    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('#333333')

    ax.set_xticks(positions)
    ax.set_xticklabels(ordered, rotation=20, ha='right')
    ax.set_ylabel('UTMOS Score')
    ax.set_ylim(1, 5)
    ax.set_title('UTMOS Distribution per Model (per-utterance)')
    ax.axhline(y=3, color='#adb5bd', linewidth=0.8, linestyle='--', label='MOS 3.0')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / '08_utmos_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Chart 9: Dataset Comparison Heatmap ──────────────────────────────────────
def plot_dataset_comparison_heatmap(data: dict, output_dir: Path):
    """Heatmap: rows = models, cols = datasets, cells = balanced score.

    Shows at a glance how each model performs across the different evaluation
    datasets — and how consistent (or inconsistent) they are.
    """
    by_dataset = data.get('by_dataset')
    if not by_dataset:
        print('  ⚠ Skipping dataset comparison heatmap (no by_dataset data)')
        return

    models = ranked_models(data)
    # Collect all dataset names that appear for at least one model
    all_datasets = sorted({
        ds
        for model_ds in by_dataset.values()
        for ds in model_ds.keys()
    })

    if not all_datasets or len(all_datasets) < 2:
        print('  ⚠ Skipping dataset comparison heatmap (fewer than 2 datasets)')
        return

    matrix = np.full((len(models), len(all_datasets)), np.nan)
    for i, model in enumerate(models):
        for j, ds in enumerate(all_datasets):
            score = by_dataset.get(model, {}).get(ds, {}).get('composites', {}).get('balanced')
            if score is not None:
                matrix[i, j] = score

    fig, ax = plt.subplots(figsize=(max(8, len(all_datasets) * 1.4), max(4, len(models) * 0.8)))

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.RdYlGn
    cmap.set_bad('#e9ecef')
    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Pretty dataset labels
    ds_labels = [ds.replace('_', '\n') for ds in all_datasets]
    ax.set_xticks(range(len(all_datasets)))
    ax.set_xticklabels(ds_labels, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    # Annotate with score values
    for i in range(len(models)):
        for j in range(len(all_datasets)):
            v = matrix[i, j]
            if not np.isnan(v):
                txt_color = 'white' if (v < 0.30 or v > 0.82) else 'black'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        color=txt_color, fontsize=8, fontweight='bold')
            else:
                ax.text(j, i, '—', ha='center', va='center',
                        color='#adb5bd', fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Balanced Composite Score  [0 – 1]', fontsize=9)

    ax.set_title('Model × Dataset Performance  (Balanced Score)', pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / '09_dataset_comparison_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Chart 10: Model Consistency Across Datasets ───────────────────────────────
def plot_model_consistency(data: dict, output_dir: Path):
    """Bar chart of model consistency scores (and std dev) across datasets.

    A consistent model performs similarly regardless of the input dataset.
    Inconsistent models may excel on one dataset but fail on another.
    """
    consistency = data.get('model_consistency')
    if not consistency:
        print('  ⚠ Skipping consistency chart (no consistency data)')
        return

    models = ranked_models(data)
    models = [m for m in models if m in consistency]

    if not models:
        return

    cons_scores = [consistency[m].get('consistency_score', 0) or 0 for m in models]
    std_vals = [consistency[m].get('std', 0) or 0 for m in models]
    mean_vals = [consistency[m].get('mean', 0) or 0 for m in models]

    colors = model_colors(models)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(models) * 0.7)))

    # Left: Consistency score (higher = more consistent)
    ax = axes[0]
    bars = ax.barh(models, cons_scores,
                   color=[colors[m] for m in models],
                   edgecolor='white', linewidth=0.8, height=0.6)
    for bar, score in zip(bars, cons_scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Consistency Score  [0 = variable, 1 = consistent]')
    ax.set_title('Cross-Dataset Consistency')
    ax.invert_yaxis()
    ax.axvline(0.7, color='#28a745', linewidth=1.0, linestyle='--', alpha=0.6, label='Good ≥ 0.7')
    ax.legend(fontsize=8)

    # Right: Mean score ± std across datasets
    ax = axes[1]
    x = np.arange(len(models))
    bars = ax.barh(x, mean_vals, xerr=std_vals,
                   color=[colors[m] for m in models],
                   edgecolor='white', linewidth=0.8, height=0.6,
                   error_kw=dict(elinewidth=1.5, ecolor='#495057', capsize=4))
    ax.set_yticks(x)
    ax.set_yticklabels(models)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Mean ± Std  Balanced Score across Datasets')
    ax.set_title('Mean Score & Variability per Model')
    ax.invert_yaxis()

    fig.suptitle('Model Consistency Across Evaluation Datasets',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '10_model_consistency.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Chart 11: Per-Dataset Leaderboard (Small Multiples) ──────────────────────
def plot_per_dataset_leaderboard(data: dict, output_dir: Path):
    """One leaderboard bar chart per dataset, arranged as small multiples.

    Lets you compare which model wins on each dataset type.
    """
    by_dataset = data.get('by_dataset')
    if not by_dataset:
        print('  ⚠ Skipping per-dataset leaderboard (no by_dataset data)')
        return

    all_datasets = sorted({
        ds
        for model_ds in by_dataset.values()
        for ds in model_ds.keys()
    })

    if len(all_datasets) < 2:
        print('  ⚠ Skipping per-dataset leaderboard (fewer than 2 datasets)')
        return

    models_all = ranked_models(data)
    colors = model_colors(models_all)

    ncols = min(3, len(all_datasets))
    nrows = int(np.ceil(len(all_datasets) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * max(3.5, len(models_all) * 0.55)))
    axes_flat = np.array(axes).flatten()

    for idx, ds_name in enumerate(all_datasets):
        ax = axes_flat[idx]

        # Collect scores for this dataset, filter models that have data
        model_scores = []
        for m in models_all:
            score = by_dataset.get(m, {}).get(ds_name, {}).get('composites', {}).get('balanced')
            if score is not None:
                model_scores.append((m, score))

        if not model_scores:
            ax.set_visible(False)
            continue

        # Sort by score descending
        model_scores.sort(key=lambda x: x[1], reverse=True)
        names, scores = zip(*model_scores)

        bars = ax.barh(list(names), list(scores),
                       color=[colors[m] for m in names],
                       edgecolor='white', linewidth=0.6, height=0.65)

        for bar, score in zip(bars, scores):
            ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{score:.3f}', va='center', fontsize=8)

        n_utt = by_dataset.get(list(names)[0], {}).get(ds_name, {}).get('n_utterances', '?')
        ax.set_title(f'{ds_name}\n(n={n_utt})', fontsize=9, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.invert_yaxis()
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for ax in axes_flat[len(all_datasets):]:
        ax.set_visible(False)

    fig.suptitle('Per-Dataset Leaderboard  (Balanced Composite Score)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / '11_per_dataset_leaderboard.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Chart 12: Dataset Difficulty ─────────────────────────────────────────────
def plot_dataset_difficulty(data: dict, output_dir: Path):
    """Bar chart ranking datasets by difficulty.

    Difficulty = average balanced score across all models (lower = harder).
    Also overlays model-level scores as scatter points for context.
    """
    difficulty = data.get('dataset_difficulty')
    if not difficulty:
        print('  ⚠ Skipping dataset difficulty chart (no difficulty data)')
        return

    if len(difficulty) < 2:
        return

    # Sort datasets: hardest first (lowest avg score)
    sorted_ds = sorted(difficulty.items(), key=lambda x: x[1]['avg_balanced'])
    ds_names, ds_info = zip(*sorted_ds)

    avg_scores = [info['avg_balanced'] for info in ds_info]
    diff_ranks = [info['difficulty_rank'] for info in ds_info]

    models_all = ranked_models(data)
    colors = model_colors(models_all)

    fig, ax = plt.subplots(figsize=(10, max(4, len(ds_names) * 0.75)))

    # Bar for average score
    y = np.arange(len(ds_names))
    bars = ax.barh(y, avg_scores, color='#6c757d', alpha=0.5,
                   edgecolor='white', height=0.5, label='Avg across models')

    # Scatter individual model scores
    for m in models_all:
        model_scores_for_ds = []
        for ds_name in ds_names:
            s = difficulty[ds_name]['model_scores'].get(m)
            model_scores_for_ds.append(s)

        # Only plot if model has at least some scores
        plot_y = [y[i] for i, s in enumerate(model_scores_for_ds) if s is not None]
        plot_x = [s for s in model_scores_for_ds if s is not None]
        if plot_y:
            ax.scatter(plot_x, plot_y, color=colors[m], s=35, zorder=5,
                       label=m, alpha=0.85)

    # Annotate bars
    for bar, score in zip(bars, avg_scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', fontsize=8, color='#495057')

    ax.set_yticks(y)
    ax.set_yticklabels([
        f'#{r}  {n.replace("_", " ")}' for n, r in zip(ds_names, diff_ranks)
    ], fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Balanced Composite Score  [lower = harder dataset]')
    ax.set_title('Dataset Difficulty Ranking\n(#1 = hardest)')
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, ncol=1)
    ax.axvline(0.5, color='#adb5bd', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / '12_dataset_difficulty.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Generate TTS benchmark visualizations')
    parser.add_argument('--input',       default='analysis/leaderboard.json')
    parser.add_argument('--output',      default='analysis/charts')
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    output_dir  = Path(args.output)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ranked_models(data)
    print(f'▶ {len(models)} models: {", ".join(models)}')

    has_dataset_comparison = data.get('config', {}).get('has_dataset_comparison', False)
    print(f'▶ Dataset comparison: {"yes" if has_dataset_comparison else "no (single dataset)"}')
    print('▶ Generating charts...')

    charts = [
        # Core charts (always generated)
        ('01_leaderboard.png',       lambda: plot_leaderboard(data, output_dir)),
        ('02_radar.png',             lambda: plot_radar(data, output_dir)),
        ('03_heatmap.png',           lambda: plot_heatmap(data, output_dir)),
        ('04_use_cases.png',         lambda: plot_use_cases(data, output_dir)),
        ('05_dimension_breakdown.png', lambda: plot_dimension_breakdown(data, output_dir)),
        ('06_prosody_dashboard.png', lambda: plot_prosody_dashboard(data, output_dir)),
        ('07_wer_distribution.png',  lambda: plot_wer_distribution(data, results_dir, output_dir)),
        ('08_utmos_distribution.png', lambda: plot_utmos_distribution(data, results_dir, output_dir)),
        # Dataset comparison charts (generated when multiple datasets present)
        ('09_dataset_comparison_heatmap.png', lambda: plot_dataset_comparison_heatmap(data, output_dir)),
        ('10_model_consistency.png',          lambda: plot_model_consistency(data, output_dir)),
        ('11_per_dataset_leaderboard.png',    lambda: plot_per_dataset_leaderboard(data, output_dir)),
        ('12_dataset_difficulty.png',         lambda: plot_dataset_difficulty(data, output_dir)),
    ]

    for name, fn in charts:
        try:
            fn()
            print(f'  ✓ {name}')
        except Exception as e:
            print(f'  ✗ {name}  — {e}')

    print(f'\n✓ Charts saved to {output_dir}/')


if __name__ == '__main__':
    main()
