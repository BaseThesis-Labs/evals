#!/usr/bin/env python3
"""
compare_datasets.py — Cross-dataset STT benchmark comparison.

Loads leaderboard.json files from multiple dataset analysis directories
and produces publication-quality comparison charts in the style of the
OpenAI Whisper paper (arXiv 2212.04356):
  - Multi-panel metric comparisons
  - Audio hours vs WER scatter
  - Model ranking bump chart
  - Error type breakdown per dataset
  - Dataset difficulty heatmap
  - Model consistency (WER range across datasets)
  - Dataset profile radar
  - Composite scorecard

Usage:
    # Auto-discover all analysis sub-directories
    python compare_datasets.py --base-dir analysis

    # Comma-separated explicit directories
    python compare_datasets.py -b analysis/ted,analysis/lib,analysis/vox

    # Explicit with custom names
    python compare_datasets.py --dirs "LibriSpeech-clean:analysis/lib,TED-LIUM:analysis/ted,VoxPopuli:analysis/vox"

    # Save charts to a custom folder
    python compare_datasets.py --base-dir analysis --output comparison_charts

Charts produced:
  CD1  panel_metrics.png       — Multi-panel WER/CER/RTFx/SemDist (Whisper Fig 3 style)
  CD2  hours_vs_wer.png        — Audio hours vs WER per model per dataset
  CD3  bump_ranks.png          — Model ranking bump chart across datasets
  CD4  wer_relative.png        — WER relative to easiest dataset (difficulty multiplier)
  CD5  error_composition.png   — Sub / Del / Ins breakdown per dataset
  CD6  dataset_scorecard.png   — Datasets × models WER scorecard heatmap
  CD7  model_consistency.png   — WER range (min→max) across datasets per model
  CD8  dataset_radar.png       — Dataset profile radar (normalised metrics)
  CD9  metric_scatter.png      — WER vs SemDist scatter (per dataset, per model)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)

# ── Publication-quality palette (Whisper-paper inspired) ──────────────────────
_PALETTE = [
    "#4C72B0",  # steel blue
    "#DD8452",  # warm orange
    "#55A868",  # sage green
    "#C44E52",  # muted red
    "#8172B3",  # lavender
    "#937860",  # taupe
    "#DA8BC3",  # pink
    "#8C8C8C",  # neutral grey
    "#CCB974",  # sand
    "#64B5CD",  # sky blue
]

_RG_CMAP = LinearSegmentedColormap.from_list("rg", ["#C44E52", "#FFFFFF", "#55A868"], N=256)


def _colors(n: int) -> list:
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")


def _save(fig: plt.Figure, path: Path, name: str):
    fig.patch.set_facecolor("white")
    fig.savefig(path / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {name}")


# ── Data loading ──────────────────────────────────────────────────────────────

def _infer_name(path: str) -> str:
    parts = Path(path).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _discover_dirs(base_dir: str) -> list[str]:
    base = Path(base_dir)
    return [str(f.parent) for f in sorted(base.rglob("leaderboard.json"))]


def load_leaderboards(dir_specs: list[str]) -> dict[str, dict]:
    """Returns {dataset_name: leaderboard_json_dict}."""
    result: dict[str, dict] = {}
    for spec in dir_specs:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            path = spec
            name = _infer_name(spec)

        lb_path = Path(path) / "leaderboard.json"
        if not lb_path.exists():
            log.warning(f"  No leaderboard.json in {path} — skipping '{name}'")
            continue
        try:
            data = json.loads(lb_path.read_text())
            result[name] = data
            log.info(f"  Loaded '{name}' — {len(data.get('leaderboard', []))} models")
        except Exception as e:
            log.error(f"  Failed to load {lb_path}: {e}")
    return result


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _get_raw(datasets: dict, ds_name: str, model: str, key: str) -> float:
    """Safely pull a raw metric value from all_models."""
    v = (datasets[ds_name]
         .get("all_models", {})
         .get(model, {})
         .get("raw", {})
         .get(key))
    try:
        f = float(v)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _get_lb(datasets: dict, ds_name: str, model: str, key: str) -> float:
    """Pull a top-level leaderboard row value."""
    for row in datasets[ds_name].get("leaderboard", []):
        if row["model"] == model:
            v = row.get(key)
            try:
                f = float(v)
                return f if np.isfinite(f) else float("nan")
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def _all_models(datasets: dict) -> list[str]:
    """Sorted union of all model names across datasets."""
    names = set()
    for data in datasets.values():
        for row in data.get("leaderboard", []):
            names.add(row["model"])
    return sorted(names)


def _all_datasets(datasets: dict) -> list[str]:
    return list(datasets.keys())


# ── CD1: Multi-panel metric comparison (Whisper Fig 3 style) ─────────────────

def chart_panel_metrics(datasets: dict, out: Path):
    """
    Four-panel figure (WER / CER / RTFx / SemDist), each panel a horizontal
    grouped bar chart.  One bar group per model; one bar per dataset.
    White background, no top/right spines, values annotated on bars.
    """
    metrics = [
        ("micro_wer",    "WER (%)",       True,  100),
        ("mean_cer",     "CER (%)",       True,  100),
        ("mean_rtfx",    "RTFx  ↑",       False, 1),
        ("mean_semdist", "SemDist  ↓",    True,  1),
    ]

    ds_names = _all_datasets(datasets)
    models   = _all_models(datasets)
    n_ds     = len(ds_names)
    n_models = len(models)
    ds_cols  = _colors(n_ds)
    ds_color = dict(zip(ds_names, ds_cols))

    bar_h  = 0.7 / n_ds
    y_base = np.arange(n_models)

    fig, axes = plt.subplots(1, 4, figsize=(22, max(5, n_models * 0.7 + 2)))
    fig.patch.set_facecolor("white")

    for ax, (metric_key, label, lower, mult) in zip(axes, metrics):
        _despine(ax)
        for i, ds in enumerate(ds_names):
            vals = [_get_raw(datasets, ds, m, metric_key) * mult for m in models]
            offsets = y_base + (i - (n_ds - 1) / 2) * bar_h
            for j, (val, yo) in enumerate(zip(vals, offsets)):
                if not np.isfinite(val):
                    continue
                ax.barh(yo, val, height=bar_h * 0.88, color=ds_color[ds],
                        label=ds if j == 0 else "_nolegend_", zorder=3)
                ax.text(val + max(vals) * 0.02, yo, f"{val:.1f}",
                        va="center", fontsize=6.5, color="#333333")

        ax.set_yticks(y_base)
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlim(left=0)
        ax.axvline(0, color="#dddddd", linewidth=0.6)

    # Single shared legend
    handles = [mpatches.Patch(color=ds_color[ds], label=ds) for ds in ds_names]
    fig.legend(handles=handles, title="Dataset", loc="lower center",
               ncol=min(n_ds, 5), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Model Performance Across Datasets", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, out, "CD1_panel_metrics.png")


# ── CD2: Audio hours vs WER ────────────────────────────────────────────────────

def chart_hours_vs_wer(datasets: dict, out: Path):
    """
    Scatter + line chart: X = total audio hours in the dataset,
    Y = WER % for that model on that dataset.
    One coloured line per model connecting datasets left→right by duration.
    Datasets labelled on the X axis.
    """
    models  = _all_models(datasets)
    m_cols  = _colors(len(models))
    m_color = dict(zip(models, m_cols))

    # Build (dataset, model, hours, wer) rows
    rows = []
    for ds_name, data in datasets.items():
        all_m = data.get("all_models", {})
        n_samples = data.get("config", {}).get("n_samples", 0)
        for model in models:
            raw = all_m.get(model, {}).get("raw", {})
            wer = raw.get("mean_wer", float("nan"))
            hrs = raw.get("total_audio_hr", float("nan"))
            # fallback: estimate from n_samples × ~5s avg utterance
            if not np.isfinite(hrs) and n_samples:
                hrs = n_samples * 5 / 3600
            if np.isfinite(wer) and np.isfinite(hrs) and hrs > 0:
                rows.append({"dataset": ds_name, "model": model,
                             "hours": hrs, "wer": wer * 100})

    if not rows:
        log.info("  Skipping CD2 — no hours/wer data")
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    _despine(ax)

    # One line per model
    for model in models:
        mdf = df[df["model"] == model].sort_values("hours")
        if mdf.empty:
            continue
        col = m_color[model]
        ax.plot(mdf["hours"], mdf["wer"], color=col, linewidth=2.0,
                marker="o", markersize=7, label=model, zorder=3)
        # Label each point with dataset name
        for _, row in mdf.iterrows():
            ax.annotate(row["dataset"],
                        (row["hours"], row["wer"]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color=col)

    ax.set_xlabel("Total Audio Duration (hours)", fontsize=11)
    ax.set_ylabel("Mean WER (%)", fontsize=11)
    ax.set_title("Audio Hours vs WER  — each point is one dataset",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, frameon=False)
    plt.tight_layout()
    _save(fig, out, "CD2_hours_vs_wer.png")


# ── CD3: Ranking bump chart ────────────────────────────────────────────────────

def chart_bump_ranks(datasets: dict, out: Path):
    """
    Bump chart: X = dataset, Y = rank (1=best).
    Each model is a coloured line. Parallel lines = stable ranking.
    Crossing lines = ranking reversal between datasets.
    """
    ds_names = _all_datasets(datasets)
    models   = _all_models(datasets)
    if len(ds_names) < 2:
        log.info("  Skipping CD3 — need ≥ 2 datasets for bump chart")
        return

    # Rank models per dataset by WER (ascending = lower WER = better rank 1)
    rank_df = pd.DataFrame(index=models, columns=ds_names, dtype=float)
    for ds in ds_names:
        wers = {}
        for m in models:
            v = _get_raw(datasets, ds, m, "mean_wer")
            if np.isfinite(v):
                wers[m] = v
        if not wers:
            continue
        sorted_m = sorted(wers, key=wers.get)
        for rank, m in enumerate(sorted_m, 1):
            rank_df.loc[m, ds] = float(rank)

    m_cols  = _colors(len(models))
    m_color = dict(zip(models, m_cols))
    x_pos   = list(range(len(ds_names)))

    fig, ax = plt.subplots(figsize=(max(9, len(ds_names) * 2.2), max(5, len(models) * 0.75)))
    _despine(ax)
    ax.spines["left"].set_visible(False)

    max_rank = len(models)

    for model in models:
        ranks = [rank_df.loc[model, ds] for ds in ds_names]
        col   = m_color[model]
        valid = [(x, r) for x, r in zip(x_pos, ranks) if np.isfinite(r)]
        if len(valid) < 1:
            continue
        xs, rs = zip(*valid)

        ax.plot(xs, rs, color=col, linewidth=2.5, marker="o",
                markersize=11, label=model, zorder=3, solid_capstyle="round")

        # Rank label inside circle
        for x, r in zip(xs, rs):
            ax.text(x, r, str(int(r)), ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=4)

        # Model label on left and right edges
        ax.text(-0.25, valid[0][1], model, ha="right", va="center",
                fontsize=8, color=col, fontweight="bold")
        ax.text(x_pos[-1] + 0.25, valid[-1][1], model, ha="left", va="center",
                fontsize=8, color=col, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ds_names, fontsize=10, fontweight="bold")
    ax.set_yticks(range(1, max_rank + 1))
    ax.set_yticklabels([f"#{i}" for i in range(1, max_rank + 1)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-1.2, len(ds_names) - 1 + 1.2)
    ax.set_title("Model Ranking Across Datasets  (1 = best WER)",
                 fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, out, "CD3_bump_ranks.png")


# ── CD4: WER relative to easiest dataset ──────────────────────────────────────

def chart_wer_relative(datasets: dict, out: Path):
    """
    Horizontal grouped bar chart showing each model's WER on each dataset
    divided by its WER on its best (lowest WER) dataset.
    A bar of 1.0 = best performance; bars > 1.0 show difficulty penalty.
    """
    ds_names = _all_datasets(datasets)
    models   = _all_models(datasets)
    ds_cols  = _colors(len(ds_names))
    ds_color = dict(zip(ds_names, ds_cols))

    # Raw WER matrix
    wer_mat = {}
    for m in models:
        row = {}
        for ds in ds_names:
            v = _get_raw(datasets, ds, m, "mean_wer")
            if np.isfinite(v):
                row[ds] = v * 100
        wer_mat[m] = row

    # Normalise by best (min WER) per model
    rows_plot = []
    for m in models:
        if not wer_mat[m]:
            continue
        best = min(wer_mat[m].values())
        for ds, wer in wer_mat[m].items():
            rows_plot.append({
                "model": m, "dataset": ds,
                "relative_wer": wer / best if best > 0 else float("nan"),
                "abs_wer": wer,
            })

    if not rows_plot:
        log.info("  Skipping CD4 — no WER data")
        return

    df = pd.DataFrame(rows_plot)
    bar_h  = 0.7 / len(ds_names)
    y_base = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(11, max(5, len(models) * 0.8 + 2)))
    _despine(ax)

    for i, ds in enumerate(ds_names):
        sub = df[df["dataset"] == ds].set_index("model")
        vals = [sub.loc[m, "relative_wer"] if m in sub.index else float("nan")
                for m in models]
        abs_vals = [sub.loc[m, "abs_wer"] if m in sub.index else float("nan")
                    for m in models]
        offsets = y_base + (i - (len(ds_names) - 1) / 2) * bar_h

        for j, (val, aval, yo) in enumerate(zip(vals, abs_vals, offsets)):
            if not np.isfinite(val):
                continue
            color = ds_color[ds]
            # Shade darker if harder
            alpha = min(1.0, 0.4 + 0.6 * (val - 1.0) / max(df["relative_wer"].max() - 1.0, 0.01))
            ax.barh(yo, val, height=bar_h * 0.88, color=color,
                    alpha=min(1.0, alpha + 0.3),
                    label=ds if j == 0 else "_nolegend_", zorder=3)
            ax.text(val + 0.02, yo, f"×{val:.2f}  ({aval:.1f}%)",
                    va="center", fontsize=6.5, color="#333333")

    ax.axvline(1.0, color="#555555", linewidth=1.2, linestyle="--", zorder=5)
    ax.text(1.01, len(models) - 0.5, "baseline (best dataset)",
            fontsize=7.5, color="#555555", va="top")

    ax.set_yticks(y_base)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Relative WER  (1.0 = model's best dataset)", fontsize=11)
    ax.set_title("Dataset Difficulty Multiplier per Model  (higher = harder dataset)",
                 fontsize=12, fontweight="bold")

    handles = [mpatches.Patch(color=ds_color[ds], label=ds) for ds in ds_names]
    ax.legend(handles=handles, title="Dataset", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8, frameon=False)
    ax.set_xlim(left=0)
    plt.tight_layout()
    _save(fig, out, "CD4_wer_relative.png")


# ── CD5: Error type composition per dataset ────────────────────────────────────

def chart_error_composition(datasets: dict, out: Path):
    """
    One subplot per dataset.  Each subplot: stacked horizontal bars
    (Sub / Del / Ins) per model.  Shows how error type mix changes
    across datasets and models.
    """
    ds_names = _all_datasets(datasets)
    models   = _all_models(datasets)
    n_ds     = len(ds_names)
    ncols    = min(n_ds, 3)
    nrows    = (n_ds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 6.5, nrows * max(3, len(models) * 0.6 + 1.5)),
                              squeeze=False)
    fig.patch.set_facecolor("white")

    comp_colors = {"Substitution": "#C44E52", "Deletion": "#DD8452", "Insertion": "#4C72B0"}
    components  = list(comp_colors.keys())
    keys        = ["mean_sub_rate", "mean_del_rate", "mean_ins_rate"]

    for idx, ds_name in enumerate(ds_names):
        ax = axes[idx // ncols][idx % ncols]
        _despine(ax)

        rows = []
        for m in models:
            vals = [_get_raw(datasets, ds_name, m, k) * 100 for k in keys]
            if any(np.isfinite(v) for v in vals):
                rows.append({
                    "model": m,
                    "Substitution": vals[0] if np.isfinite(vals[0]) else 0.0,
                    "Deletion":     vals[1] if np.isfinite(vals[1]) else 0.0,
                    "Insertion":    vals[2] if np.isfinite(vals[2]) else 0.0,
                })

        if not rows:
            ax.set_visible(False)
            continue

        df = pd.DataFrame(rows)
        total_wer = df[components].sum(axis=1)
        df = df.assign(total=total_wer).sort_values("total")

        lefts = np.zeros(len(df))
        for comp in components:
            vals = df[comp].values
            bars = ax.barh(df["model"], vals, left=lefts,
                           color=comp_colors[comp], label=comp, height=0.55, zorder=3)
            for bar, val, left in zip(bars, vals, lefts):
                if val > 0.4:
                    ax.text(left + val / 2, bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%", ha="center", va="center",
                            fontsize=6.5, color="white", fontweight="bold")
            lefts = lefts + vals

        # Total WER label at end
        for i, (total, model) in enumerate(zip(df["total"], df["model"])):
            ax.text(total + 0.2, i, f"{total:.1f}%", va="center", fontsize=7.5, color="#333333")

        ax.set_title(ds_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Error rate (% of ref words)", fontsize=9)
        ax.set_xlim(left=0)

    # Hide empty subplots
    for idx in range(n_ds, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend
    handles = [mpatches.Patch(color=c, label=comp) for comp, c in comp_colors.items()]
    fig.legend(handles=handles, title="Error type", loc="lower center",
               ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Error Type Composition by Dataset  (Sub / Del / Ins)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out, "CD5_error_composition.png")


# ── CD6: Dataset × model WER scorecard ────────────────────────────────────────

def chart_dataset_scorecard(datasets: dict, out: Path):
    """
    Heatmap scorecard: rows = models, columns = datasets.
    Cell colour = WER rank (green=best, red=worst).
    WER % annotated in each cell.
    Includes a 'Mean' column on the right.
    """
    ds_names = _all_datasets(datasets)
    models   = _all_models(datasets)

    wer_data = pd.DataFrame(index=models, columns=ds_names, dtype=float)
    for ds in ds_names:
        for m in models:
            v = _get_raw(datasets, ds, m, "mean_wer")
            wer_data.loc[m, ds] = v * 100 if np.isfinite(v) else float("nan")

    wer_data["Mean"] = wer_data[ds_names].mean(axis=1, skipna=True)
    wer_data = wer_data.sort_values("Mean")
    all_cols = ds_names + ["Mean"]

    _map = getattr(wer_data, "map", None) or getattr(wer_data, "applymap")
    annot = _map(lambda v: f"{v:.1f}%" if np.isfinite(v) else "—")

    # Rank-normalise per column for colour
    rank_data = wer_data.copy()
    for col in all_cols:
        col_vals = wer_data[col].dropna()
        if col_vals.empty:
            rank_data[col] = 0.5
            continue
        lo, hi = col_vals.min(), col_vals.max()
        if hi > lo:
            rank_data[col] = 1.0 - (wer_data[col] - lo) / (hi - lo)
        else:
            rank_data[col] = 1.0

    fig, ax = plt.subplots(figsize=(max(8, len(all_cols) * 1.6),
                                     max(4, len(models) * 0.65)))
    fig.patch.set_facecolor("white")

    rank_arr = rank_data[all_cols].values.astype(float)
    rank_arr = np.where(np.isfinite(rank_arr), rank_arr, 0.5)

    im = ax.imshow(rank_arr, cmap=_RG_CMAP, aspect="auto", vmin=0, vmax=1)

    for r in range(len(models)):
        for c, col in enumerate(all_cols):
            txt = annot.iloc[r][col]
            rank = rank_arr[r, c]
            text_col = "white" if rank < 0.2 or rank > 0.82 else "#222222"
            weight = "bold" if col == "Mean" else "normal"
            ax.text(c, r, txt, ha="center", va="center", fontsize=8.5,
                    color=text_col, fontweight=weight)

    ax.set_xticks(range(len(all_cols)))
    ax.set_xticklabels(all_cols, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(wer_data.index, fontsize=9)

    # Divider before Mean column
    ax.axvline(len(ds_names) - 0.5, color="white", linewidth=2.5)

    ax.set_xticks(np.arange(-0.5, len(all_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title("WER (%) Scorecard  — green = lower WER (better)",
                 fontsize=12, fontweight="bold", pad=10)

    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Relative rank", fontsize=8)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["worst", "mid", "best"])

    plt.tight_layout()
    _save(fig, out, "CD6_dataset_scorecard.png")


# ── CD7: Model consistency (WER range across datasets) ────────────────────────

def chart_model_consistency(datasets: dict, out: Path):
    """
    Horizontal range-bar chart: each model shows its min WER (best dataset),
    max WER (hardest dataset), and mean WER across all datasets.
    A narrow range = consistent model; wide range = sensitive to domain.
    """
    models = _all_models(datasets)
    rows   = []
    for m in models:
        wers = []
        for ds in datasets:
            v = _get_raw(datasets, ds, m, "mean_wer")
            if np.isfinite(v):
                wers.append(v * 100)
        if len(wers) < 1:
            continue
        rows.append({
            "model": m,
            "min":   min(wers),
            "max":   max(wers),
            "mean":  np.mean(wers),
            "range": max(wers) - min(wers),
        })

    if not rows:
        log.info("  Skipping CD7 — no data")
        return

    df = pd.DataFrame(rows).sort_values("mean")
    colors = _colors(len(df))

    fig, ax = plt.subplots(figsize=(11, max(4, len(df) * 0.7 + 1.5)))
    _despine(ax)

    for i, (_, row) in enumerate(df.iterrows()):
        col = colors[i]
        # Range bar
        ax.plot([row["min"], row["max"]], [i, i],
                color=col, linewidth=5, solid_capstyle="round",
                alpha=0.35, zorder=2)
        # Min cap
        ax.plot(row["min"], i, "|", color=col, markersize=14,
                markeredgewidth=2.5, zorder=3)
        # Max cap
        ax.plot(row["max"], i, "|", color=col, markersize=14,
                markeredgewidth=2.5, zorder=3)
        # Mean dot
        ax.plot(row["mean"], i, "o", color=col, markersize=10,
                zorder=4, markeredgecolor="white", markeredgewidth=1.5)

        # Labels
        ax.text(row["min"] - 0.4, i, f"{row['min']:.1f}%",
                ha="right", va="center", fontsize=7.5, color=col)
        ax.text(row["max"] + 0.4, i, f"{row['max']:.1f}%",
                ha="left", va="center", fontsize=7.5, color=col)
        ax.text(row["mean"], i + 0.32, f"μ={row['mean']:.1f}%",
                ha="center", va="bottom", fontsize=6.5, color="#444444")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model"], fontsize=9)
    ax.set_xlabel("WER (%)  —  |: min/max,  ●: mean across datasets", fontsize=10)
    ax.set_title("Model Consistency Across Datasets  (narrow range = more robust)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(left=0)
    plt.tight_layout()
    _save(fig, out, "CD7_model_consistency.png")


# ── CD8: Dataset profile radar ────────────────────────────────────────────────

def chart_dataset_radar(datasets: dict, out: Path):
    """
    Radar (spider) chart: each dataset is one polygon.
    Axes = average WER, CER, RTFx, SemDist, HER across all models
    (each normalised 0→1 where 1 = best).
    Shows the "profile" of each dataset — harder? faster? more semantic drift?
    """
    metric_specs = [
        ("mean_wer",    "WER",     True),
        ("mean_cer",    "CER",     True),
        ("mean_rtfx",   "RTFx",    False),
        ("mean_semdist","SemDist", True),
        ("mean_her",    "HER",     True),
        ("mean_mer",    "MER",     True),
    ]

    ds_names = _all_datasets(datasets)
    models   = _all_models(datasets)

    # Average each metric across models per dataset
    ds_vals: dict[str, list[float]] = {}
    for ds in ds_names:
        vals = []
        for key, _, _ in metric_specs:
            model_vals = [_get_raw(datasets, ds, m, key) for m in models]
            finite = [v for v in model_vals if np.isfinite(v)]
            vals.append(float(np.mean(finite)) if finite else float("nan"))
        ds_vals[ds] = vals

    # Normalise across datasets per metric (0=worst, 1=best)
    n_metrics = len(metric_specs)
    norm_vals: dict[str, list[float]] = {ds: [0.5] * n_metrics for ds in ds_names}
    for mi, (_, _, lower_is_better) in enumerate(metric_specs):
        raw_col = [ds_vals[ds][mi] for ds in ds_names]
        finite  = [v for v in raw_col if np.isfinite(v)]
        if not finite:
            continue
        lo, hi = min(finite), max(finite)
        for ds in ds_names:
            v = ds_vals[ds][mi]
            if not np.isfinite(v):
                norm_vals[ds][mi] = 0.5
                continue
            if hi > lo:
                norm = (v - lo) / (hi - lo)
            else:
                norm = 0.5
            norm_vals[ds][mi] = (1.0 - norm) if lower_is_better else norm

    labels  = [m[1] for m in metric_specs]
    N       = n_metrics
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ds_colors = _colors(len(ds_names))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    for i, ds in enumerate(ds_names):
        vals  = norm_vals[ds] + norm_vals[ds][:1]
        col   = ds_colors[i]
        ax.plot(angles, vals, color=col, linewidth=2.2, label=ds, zorder=3)
        ax.fill(angles, vals, color=col, alpha=0.07)
        ax.scatter(angles[:-1], norm_vals[ds], color=col, s=45, zorder=4)

    # Grid rings
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [r] * (N + 1), color="#cccccc", linewidth=0.5, zorder=1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_visible(False)
    ax.set_title("Dataset Profile Radar  (outer = better, averaged across models)",
                 fontsize=12, fontweight="bold", pad=22)
    ax.legend(bbox_to_anchor=(1.3, 1.1), loc="upper left", fontsize=9,
              frameon=False, title="Dataset")
    plt.tight_layout()
    _save(fig, out, "CD8_dataset_radar.png")


# ── CD9: WER vs SemDist scatter ───────────────────────────────────────────────

def chart_metric_scatter(datasets: dict, out: Path):
    """
    Scatter plot: X = WER (%), Y = SemDist.
    One point per (model, dataset). Colour = dataset, marker shape = model.
    Shows whether models that have low WER also have low semantic distance,
    or whether some models trade one for the other.
    """
    ds_names  = _all_datasets(datasets)
    models    = _all_models(datasets)
    ds_colors = _colors(len(ds_names))
    ds_color  = dict(zip(ds_names, ds_colors))
    markers   = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]
    m_marker  = {m: markers[i % len(markers)] for i, m in enumerate(models)}

    rows = []
    for ds in ds_names:
        for m in models:
            wer = _get_raw(datasets, ds, m, "mean_wer")
            sem = _get_raw(datasets, ds, m, "mean_semdist")
            if np.isfinite(wer) and np.isfinite(sem):
                rows.append({"dataset": ds, "model": m,
                             "wer": wer * 100, "semdist": sem})

    if not rows:
        log.info("  Skipping CD9 — no WER/SemDist data")
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 7))
    _despine(ax)

    for ds in ds_names:
        for m in models:
            sub = df[(df["dataset"] == ds) & (df["model"] == m)]
            if sub.empty:
                continue
            ax.scatter(sub["wer"], sub["semdist"],
                       color=ds_color[ds], marker=m_marker[m],
                       s=90, zorder=3, linewidths=0.8,
                       edgecolors="white")
            ax.annotate(f"{m}\n({ds})",
                        (sub["wer"].values[0], sub["semdist"].values[0]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6, color="#444444")

    # Legend: datasets by colour
    ds_handles = [mpatches.Patch(color=ds_color[ds], label=ds) for ds in ds_names]
    # Legend: models by marker
    m_handles  = [plt.Line2D([0], [0], marker=m_marker[m], color="grey",
                              linestyle="None", markersize=7, label=m) for m in models]

    leg1 = ax.legend(handles=ds_handles, title="Dataset",
                     bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=m_handles, title="Model",
              bbox_to_anchor=(1.01, 0.45), loc="upper left", fontsize=8, frameon=False)

    ax.set_xlabel("Mean WER (%)  ← better", fontsize=11)
    ax.set_ylabel("Mean SemDist  ← better", fontsize=11)
    ax.set_title("WER vs Semantic Distance  per Model × Dataset\n"
                 "Bottom-left corner = best models",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, out, "CD9_metric_scatter.png")


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--base-dir", "-b", default=None,
              help="Comma-separated base directories to auto-discover leaderboard.json files")
@click.option("--dirs", "-d", default=None,
              help="Comma-separated analysis dirs, optionally 'name:path' format")
@click.option("--output", "-o", default="comparison_charts", show_default=True,
              help="Output directory for comparison charts")
def main(base_dir, dirs, output):
    """
    Cross-dataset STT benchmark comparison (Whisper-paper style).

    Examples:
      python compare_datasets.py -b analysis/lib,analysis/ted,analysis/vox
      python compare_datasets.py --dirs analysis/librispeech/test-clean,analysis/tedlium/tedlium_test
      python compare_datasets.py --dirs "clean:analysis/lib,TED-LIUM:analysis/ted"
    """
    if not base_dir and not dirs:
        raise click.UsageError("Provide --base-dir (-b) or --dirs (-d)")

    dir_specs: list[str] = []
    if base_dir:
        for b in [s.strip() for s in base_dir.split(",") if s.strip()]:
            discovered = _discover_dirs(b)
            if discovered:
                dir_specs += discovered
            else:
                dir_specs.append(b)
    if dirs:
        dir_specs += [s.strip() for s in dirs.split(",")]

    if not dir_specs:
        log.error("No directories found.")
        return

    datasets = load_leaderboards(dir_specs)
    if not datasets:
        log.error("No leaderboard.json files loaded. Run aggregate.py first.")
        return

    if len(datasets) < 2:
        log.warning(f"Only {len(datasets)} dataset — comparisons need ≥ 2. "
                    "Charts will still be generated.")

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"Comparing {len(datasets)} datasets → {out}/")

    chart_panel_metrics(datasets, out)
    chart_hours_vs_wer(datasets, out)
    chart_bump_ranks(datasets, out)
    chart_wer_relative(datasets, out)
    chart_error_composition(datasets, out)
    chart_dataset_scorecard(datasets, out)
    chart_model_consistency(datasets, out)
    chart_dataset_radar(datasets, out)
    chart_metric_scatter(datasets, out)

    log.info(f"\n9 comparison charts saved to {out}/")


if __name__ == "__main__":
    main()
