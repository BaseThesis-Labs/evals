#!/usr/bin/env python3
"""
visualize.py — Generate all charts from analysis/leaderboard.json + results/metrics/.

Usage:
    python visualize.py
    python visualize.py --leaderboard analysis/leaderboard.json --output analysis/charts

Charts produced (per dataset):
  01  wer_comparison.png         — WER bar chart with 95% CI
  02  accuracy_vs_speed.png      — WER vs RTFx scatter
  03  cost_vs_wer.png            — Cost/hr vs WER Pareto
  04  use_case_scores.png        — Use-case composite score heatmap
  05  dimension_breakdown.png    — Stacked dimension score bar
  06  wer_boxplot.png            — Per-sample WER distribution
  07  semdist_violin.png         — Semantic distance violin
  08  speaker_heatmap.png        — Speaker × model WER heatmap
  09  wer_vs_duration.png        — WER vs audio duration scatter (trend lines)
  10  metrics_grid.png           — All metrics side-by-side bar grid
  11  snr_vs_wer.png             — Noise robustness: WER per SNR bucket

  Whisper-paper-style charts:
  W1  whisper_grouped_wer.png    — Grouped horizontal WER bars (model × condition)
  W2  whisper_error_stack.png    — Stacked substitution/deletion/insertion bars
  W3  whisper_radar.png          — Radar/spider chart of normalised dimension scores
  W4  whisper_wer_cdf.png        — Cumulative WER distribution (CDF) per model
  W5  whisper_scorecard.png      — Metrics × models scorecard heatmap
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = "RdYlGn_r"

# Metrics shown in the all-metrics grid
# (leaderboard_raw_key, display_label, lower_is_better, multiply_for_display)
GRID_METRICS = [
    ("micro_wer",            "WER (%)",        True,  100),
    ("mean_cer",             "CER (%)",        True,  100),
    ("mean_wrr",             "WRR (%)",        False, 100),
    ("mean_rtfx",            "RTFx",           False, 1),
    ("mean_semdist",         "SemDist",        True,  1),
    ("mean_bertscore_f1",    "BERTScore F1",   False, 1),
    ("mean_fwer",            "FWER (%)",       True,  100),
    ("mean_punctuation_f1",  "Punct. F1",      False, 1),
    ("mean_bleu_4",          "BLEU-4",         False, 1),
    ("mean_meteor",          "METEOR",         False, 1),
    ("mean_her",             "HER",            True,  1),
    ("mean_sub_rate",        "Sub Rate (%)",   True,  100),
    ("mean_del_rate",        "Del Rate (%)",   True,  100),
    ("mean_ins_rate",        "Ins Rate (%)",   True,  100),
    ("cost_per_hour_usd",    "Cost/hr ($)",    True,  1),
    ("mean_nic",             "NIC",            False, 1),
    ("mean_snr_db",          "Mean SNR (dB)",  False, 1),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v*100:.1f}%" if np.isfinite(v) else "—"


def _save(fig: plt.Figure, path: Path, name: str):
    fig.savefig(path / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {name}")


def _load_per_sample(metrics_dir: Path) -> dict[str, list[dict]]:
    """Load per_sample arrays from each model's JSON file."""
    data: dict[str, list[dict]] = {}
    for f in sorted(metrics_dir.glob("*.json")):
        if "_hallucination" in f.stem:
            continue
        try:
            d = json.loads(f.read_text())
            data[d["model"]] = d.get("per_sample", [])
        except Exception as e:
            log.warning(f"  Could not load {f.name}: {e}")
    return data


def _model_palette(models: list[str]) -> dict[str, tuple]:
    colors = sns.color_palette("tab10", len(models))
    return {m: colors[i] for i, m in enumerate(models)}


# ── Chart 01: WER comparison bar ──────────────────────────────────────────────

def chart_wer_comparison(lb: list[dict], out: Path):
    df = pd.DataFrame(lb)
    df = df[df["micro_wer"].apply(np.isfinite)].sort_values("micro_wer")
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.55)))
    colors = sns.color_palette(PALETTE, len(df))
    ax.barh(df["model"], df["micro_wer"] * 100, color=colors, edgecolor="white")

    ci_lo = np.clip((df["micro_wer"] - df["wer_ci_lo"].fillna(df["micro_wer"])) * 100, 0, None)
    ci_hi = np.clip((df["wer_ci_hi"].fillna(df["micro_wer"]) - df["micro_wer"]) * 100, 0, None)
    ax.errorbar(
        df["micro_wer"] * 100, range(len(df)),
        xerr=[ci_lo, ci_hi], fmt="none", color="black", capsize=3, lw=1,
    )
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["micro_wer"] * 100 + 0.3, i, _pct(row["micro_wer"]),
                va="center", fontsize=9)

    ax.set_xlabel("WER (%)")
    ax.set_title("Word Error Rate by Model (lower is better, 95% CI)")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    _save(fig, out, "01_wer_comparison.png")


# ── Chart 02: Accuracy vs Speed scatter ───────────────────────────────────────

def chart_accuracy_vs_speed(lb: list[dict], out: Path):
    df = pd.DataFrame(lb)
    df = df[df["micro_wer"].apply(np.isfinite) & df["mean_rtfx"].apply(np.isfinite)]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["mean_rtfx"], df["micro_wer"] * 100,
               s=120, c=range(len(df)), cmap="tab10", zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["mean_rtfx"], row["micro_wer"] * 100),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)
    ax.set_xlabel("Mean RTFx  (higher = faster than real-time)")
    ax.set_ylabel("WER (%)")
    ax.set_title("Accuracy vs Speed")
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, label="Real-time (RTFx=1)")
    ax.legend(fontsize=8)
    _save(fig, out, "02_accuracy_vs_speed.png")


# ── Chart 03: Cost vs WER Pareto ──────────────────────────────────────────────

def chart_cost_vs_wer(lb: list[dict], out: Path):
    df = pd.DataFrame(lb)
    df = df[
        df["micro_wer"].apply(np.isfinite) &
        df["cost_per_hour_usd"].apply(lambda v: np.isfinite(v) and v > 0)
    ]
    if df.empty:
        log.info("  Skipping cost chart — no models with cost data")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["cost_per_hour_usd"], df["micro_wer"] * 100, s=120, color="steelblue", zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["cost_per_hour_usd"], row["micro_wer"] * 100),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)
    ax.set_xlabel("Cost per hour of audio (USD)")
    ax.set_ylabel("WER (%)")
    ax.set_title("Cost vs Accuracy — Pareto Frontier")
    _save(fig, out, "03_cost_vs_wer.png")


# ── Chart 04: Use-case composite score heatmap ────────────────────────────────

def chart_use_case_scores(lb: list[dict], out: Path):
    rows = []
    for entry in lb:
        cs = entry.get("composite_scores", {})
        for cs_name, score in cs.items():
            if np.isfinite(score):
                rows.append({"model": entry["model"], "use_case": cs_name, "score": score})
    if not rows:
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="model", columns="use_case", values="score")

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4), max(5, len(pivot) * 0.7)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                ax=ax, linewidths=0.5, vmin=0, vmax=1,
                cbar_kws={"label": "Composite Score (0–1, higher=better)"})
    ax.set_title("Use-Case Composite Scores by Model")
    ax.set_xlabel("Use Case")
    ax.set_ylabel("Model")
    plt.xticks(rotation=25, ha="right")
    _save(fig, out, "04_use_case_scores.png")


# ── Chart 05: Dimension breakdown stacked bar ─────────────────────────────────

def chart_dimension_breakdown(lb: list[dict], out: Path):
    dims = ["intelligibility", "semantic", "latency", "formatting", "hallucination", "entity"]
    rows = []
    for entry in lb:
        ds = entry.get("dimension_scores", {})
        row = {"model": entry["model"]}
        for d in dims:
            row[d] = ds.get(d, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model").fillna(0)
    if df.empty:
        return

    colors = sns.color_palette("Set2", len(dims))
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.65)))
    df.plot(kind="barh", stacked=True, ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Cumulative dimension score (0–1 per dimension)")
    ax.set_title("Dimension Score Breakdown by Model")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    _save(fig, out, "05_dimension_breakdown.png")


# ── Chart 06: WER box plots ────────────────────────────────────────────────────

def chart_wer_boxplot(per_sample_all: dict[str, list[dict]], out: Path):
    rows = []
    for model, samples in per_sample_all.items():
        for s in samples:
            wer = s.get("wer")
            if wer is not None and np.isfinite(wer):
                rows.append({"model": model, "wer": wer})
    if not rows:
        return

    df = pd.DataFrame(rows)
    order = df.groupby("model")["wer"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, max(5, len(order) * 0.7)))
    sns.boxplot(data=df, y="model", x="wer", order=order, ax=ax, hue="model",
                palette=PALETTE, orient="h", flierprops={"marker": ".", "markersize": 3},
                legend=False)
    ax.set_xlabel("WER per sample")
    ax.set_title("WER Distribution by Model")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    _save(fig, out, "06_wer_boxplot.png")


# ── Chart 07: SemDist violin ───────────────────────────────────────────────────

def chart_semdist_violin(per_sample_all: dict[str, list[dict]], out: Path):
    rows = []
    for model, samples in per_sample_all.items():
        for s in samples:
            sd = s.get("semdist")
            if sd is not None and np.isfinite(sd):
                rows.append({"model": model, "semdist": sd})
    if not rows:
        log.info("  Skipping SemDist violin — no semdist values found")
        return

    df = pd.DataFrame(rows)
    order = df.groupby("model")["semdist"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, max(5, len(order) * 0.7)))
    sns.violinplot(data=df, y="model", x="semdist", order=order, ax=ax, hue="model",
                   palette="Blues", orient="h", cut=0, legend=False)
    ax.set_xlabel("Semantic Distance (0=identical, 1=unrelated)")
    ax.set_title("Semantic Distance Distribution by Model")
    _save(fig, out, "07_semdist_violin.png")


# ── Chart 08: Speaker WER heatmap ─────────────────────────────────────────────

def chart_speaker_heatmap(per_sample_all: dict[str, list[dict]], out: Path, max_speakers: int = 20):
    rows = []
    for model, samples in per_sample_all.items():
        for s in samples:
            spk = s.get("speaker_id", "")
            wer = s.get("wer")
            if spk and wer is not None and np.isfinite(wer):
                rows.append({"model": model, "speaker_id": spk, "wer": wer})
    if not rows:
        log.info("  Skipping speaker heatmap — no speaker_id data")
        return

    df = pd.DataFrame(rows)
    top_spk = df["speaker_id"].value_counts().head(max_speakers).index.tolist()
    df = df[df["speaker_id"].isin(top_spk)]
    pivot = df.groupby(["model", "speaker_id"])["wer"].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.6), max(5, len(pivot) * 0.6)))
    sns.heatmap(pivot * 100, annot=True, fmt=".0f", cmap=PALETTE,
                ax=ax, linewidths=0.3, cbar_kws={"label": "WER (%)"})
    ax.set_title(f"WER (%) by STT Model × Speaker (top {max_speakers})")
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("STT Model")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    _save(fig, out, "08_speaker_heatmap.png")


# ── Chart 09: WER vs Audio Duration scatter ───────────────────────────────────

def chart_wer_vs_duration(per_sample_all: dict[str, list[dict]], out: Path):
    """
    Scatter plot of WER (%) vs audio duration per sample.
    One series per model with a regression trend line.
    Subtitle shows total audio hours and sample count.
    """
    rows = []
    for model, samples in per_sample_all.items():
        for s in samples:
            wer = s.get("wer")
            dur = s.get("audio_duration_s")
            if wer is not None and dur is not None and np.isfinite(wer) and np.isfinite(dur):
                rows.append({"model": model, "duration_s": dur, "wer_pct": wer * 100})
    if not rows:
        log.info("  Skipping WER-vs-duration — no data")
        return

    df = pd.DataFrame(rows)
    models = sorted(df["model"].unique())
    pal = _model_palette(models)

    total_hrs = df["duration_s"].sum() / 3600 / max(len(models), 1)
    n_samples = df.groupby("model").size().max()

    fig, ax = plt.subplots(figsize=(12, 7))

    for model in models:
        mdf = df[df["model"] == model].copy()
        ax.scatter(mdf["duration_s"], mdf["wer_pct"],
                   alpha=0.30, s=18, color=pal[model], label=model, zorder=2)
        if len(mdf) >= 3:
            z = np.polyfit(mdf["duration_s"], mdf["wer_pct"], 1)
            xline = np.linspace(mdf["duration_s"].min(), mdf["duration_s"].max(), 200)
            ax.plot(xline, np.polyval(z, xline), color=pal[model], linewidth=2.0, alpha=0.85)

    ax.set_xlabel("Audio Duration (seconds)", fontsize=11)
    ax.set_ylabel("WER (%)", fontsize=11)
    ax.set_title(
        f"WER vs Audio Duration  |  ~{total_hrs:.2f} hrs · {n_samples} samples/model",
        fontsize=12,
    )
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title="Model")
    ax.set_ylim(bottom=0)
    _save(fig, out, "09_wer_vs_duration.png")


# ── Chart 10: All-metrics comparison grid ────────────────────────────────────

def chart_metrics_grid(all_models_data: dict[str, dict], out: Path):
    """
    Grid of horizontal bar charts — one subplot per metric, all models compared.
    Color: green = best performing model, red = worst.
    """
    if not all_models_data:
        return

    # Build per-model raw dict
    rows = []
    for model_name, res in all_models_data.items():
        raw = res.get("raw", {})
        row = {"model": model_name}
        for key, _, _, _ in GRID_METRICS:
            row[key] = raw.get(key, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")

    # Only keep metrics with at least one finite value
    available = [
        (k, lbl, lib, mult) for k, lbl, lib, mult in GRID_METRICS
        if k in df.columns and df[k].apply(lambda v: np.isfinite(float(v)) if v is not None else False).any()
    ]
    if not available:
        log.info("  Skipping metrics grid — no finite values found")
        return

    ncols = 4
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.2))
    axes_flat = np.array(axes).flatten()

    models_list = list(df.index)

    for ax_idx, (key, label, lower_is_better, mult) in enumerate(available):
        ax = axes_flat[ax_idx]
        vals = df[key].dropna().astype(float) * mult

        # Sort: best value on top
        sorted_vals = vals.sort_values(ascending=lower_is_better)
        n = len(sorted_vals)

        # Color gradient: green (best) → red (worst)
        grad = sns.color_palette("RdYlGn", n)
        colors = list(reversed(grad))  # best=green at top

        bars = ax.barh(range(n), sorted_vals.values, color=colors, edgecolor="white", height=0.65)
        ax.set_yticks(range(n))
        ax.set_yticklabels(sorted_vals.index, fontsize=7)

        # Annotate bars
        x_max = sorted_vals.values.max() if len(sorted_vals) else 1
        for i, (bar, val) in enumerate(zip(bars, sorted_vals.values)):
            if np.isfinite(val):
                ax.text(val + x_max * 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", va="center", fontsize=6.5)

        direction = "↓ better" if lower_is_better else "↑ better"
        ax.set_title(f"{label}  ({direction})", fontsize=8.5, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7)
        ax.margins(x=0.20)

    # Hide unused subplots
    for ax_idx in range(len(available), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle("All Metrics — Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out, "10_metrics_grid.png")


# ── Chart 11: SNR vs WER (noise robustness) ──────────────────────────────────

def chart_snr_vs_wer(per_sample_all: dict[str, list[dict]], out: Path):
    """
    Line chart: mean WER per SNR dB bucket, one line per model.
    Shows how much WER degrades as audio gets noisier.
    """
    rows = []
    for model, samples in per_sample_all.items():
        for s in samples:
            wer = s.get("wer")
            snr = s.get("snr_db")
            if wer is None or snr is None:
                continue
            try:
                snr_f = float(snr)
                wer_f = float(wer)
            except (TypeError, ValueError):
                continue
            if np.isfinite(snr_f) and np.isfinite(wer_f):
                rows.append({"model": model, "snr_db": snr_f, "wer": wer_f * 100})

    if not rows:
        log.info("  Skipping SNR chart — no snr_db values found")
        return

    df = pd.DataFrame(rows)
    bins   = [-np.inf, 5, 10, 15, 20, 25, np.inf]
    blabels = ["<5 dB", "5–10", "10–15", "15–20", "20–25", ">25 dB"]
    df["bucket"] = pd.cut(df["snr_db"], bins=bins, labels=blabels)

    pivot = (
        df.groupby(["bucket", "model"])["wer"]
        .mean()
        .unstack("model")
        .reindex(blabels)
    )

    models = list(pivot.columns)
    if not models:
        return

    pal = _model_palette(models)
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in models:
        series = pivot[model]
        valid = series.dropna()
        if valid.empty:
            continue
        idx = [blabels.index(b) for b in valid.index]
        ax.plot(idx, valid.values, marker="o", label=model,
                color=pal[model], linewidth=2.2, markersize=7)

    ax.set_xticks(range(len(blabels)))
    ax.set_xticklabels(blabels, fontsize=9)
    ax.set_xlabel("Signal-to-Noise Ratio bucket", fontsize=11)
    ax.set_ylabel("Mean WER (%)", fontsize=11)
    ax.set_title("Noise Robustness: WER vs SNR  (right = cleaner audio)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title="Model")
    ax.set_ylim(bottom=0)
    _save(fig, out, "11_snr_vs_wer.png")


# ── Whisper-paper-style colour palette ───────────────────────────────────────
# Distinct, publication-friendly colours (not RdYlGn)
_W_PALETTE = [
    "#4C72B0",  # steel blue
    "#DD8452",  # warm orange
    "#55A868",  # sage green
    "#C44E52",  # muted red
    "#8172B3",  # lavender
    "#937860",  # taupe
    "#DA8BC3",  # pink
    "#8C8C8C",  # neutral grey
    "#CCB974",  # sand
    "#64B5CD",  # sky
]


def _w_colors(n: int) -> list:
    return [_W_PALETTE[i % len(_W_PALETTE)] for i in range(n)]


# ── Chart W1: Whisper-style grouped horizontal WER bars ───────────────────────

def chart_whisper_grouped_wer(per_sample_all: dict[str, list[dict]], lb: list[dict], out: Path):
    """
    Whisper-paper Figure 3 style:
    Horizontal grouped bar chart — one group per audio-duration bucket
    (short / medium / long), one bar per model.
    Shows how each model behaves as utterance length increases.
    Background: white, no top/right spines, clean labels on bars.
    """
    rows = []
    for model, samples in per_sample_all.items():
        for s in samples:
            wer = s.get("wer")
            dur = s.get("audio_duration_s")
            if wer is None or dur is None or not np.isfinite(wer) or not np.isfinite(dur):
                continue
            if dur < 5:
                bucket = "Short (<5 s)"
            elif dur < 15:
                bucket = "Medium (5–15 s)"
            else:
                bucket = "Long (>15 s)"
            rows.append({"model": model, "bucket": bucket, "wer": wer * 100})

    buckets_order = ["Short (<5 s)", "Medium (5–15 s)", "Long (>15 s)"]

    # Fall back to overall WER from leaderboard if no per-sample duration data
    if not rows:
        rows = [{"model": r["model"], "bucket": "Overall", "wer": r["micro_wer"] * 100}
                for r in lb if np.isfinite(r.get("micro_wer", float("nan")))]
        buckets_order = ["Overall"]

    if not rows:
        return

    df = pd.DataFrame(rows)
    models = sorted(df["model"].unique())
    buckets = [b for b in buckets_order if b in df["bucket"].unique()]

    n_models  = len(models)
    n_buckets = len(buckets)
    colors    = _w_colors(n_models)
    model_col = {m: colors[i] for i, m in enumerate(models)}

    bar_h   = 0.75 / n_models
    y_base  = np.arange(n_buckets)

    fig, ax = plt.subplots(figsize=(10, max(4, n_buckets * 1.4 + 1)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, model in enumerate(models):
        mdf   = df[df["model"] == model]
        means = [mdf[mdf["bucket"] == b]["wer"].mean() if b in mdf["bucket"].values else float("nan")
                 for b in buckets]
        offsets = y_base + (i - (n_models - 1) / 2) * bar_h
        for j, (val, yo) in enumerate(zip(means, offsets)):
            if not np.isfinite(val):
                continue
            ax.barh(yo, val, height=bar_h * 0.88, color=model_col[model],
                    label=model if j == 0 else "_nolegend_", zorder=3)
            ax.text(val + 0.25, yo, f"{val:.1f}%", va="center", fontsize=7.5,
                    color="#333333")

    ax.set_yticks(y_base)
    ax.set_yticklabels(buckets, fontsize=10)
    ax.set_xlabel("Word Error Rate (%)", fontsize=11)
    ax.set_title("WER by Utterance Length  (Whisper-paper style)", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0, color="#cccccc", linewidth=0.8)
    ax.set_xlim(left=0)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8,
              frameon=False)
    plt.tight_layout()
    _save(fig, out, "W1_whisper_grouped_wer.png")


# ── Chart W2: Stacked error-type bars (Sub / Del / Ins) ──────────────────────

def chart_whisper_error_stack(all_models_data: dict[str, dict], out: Path):
    """
    Whisper-paper error decomposition style:
    Horizontal stacked bar — substitution / deletion / insertion rates per model.
    Models sorted by total WER ascending.
    """
    if not all_models_data:
        return

    rows = []
    for model, res in all_models_data.items():
        raw = res.get("raw", {})
        sub = raw.get("mean_sub_rate", float("nan"))
        del_ = raw.get("mean_del_rate", float("nan"))
        ins  = raw.get("mean_ins_rate", float("nan"))
        wer  = raw.get("mean_wer",      float("nan"))
        if any(not np.isfinite(v) for v in [sub, del_, ins]):
            continue
        rows.append({"model": model, "Substitution": sub * 100,
                     "Deletion": del_ * 100, "Insertion": ins * 100, "wer": wer})

    if not rows:
        log.info("  Skipping W2 — no sub/del/ins rate data")
        return

    df = pd.DataFrame(rows).sort_values("wer")
    components = ["Substitution", "Deletion", "Insertion"]
    colors     = ["#C44E52", "#DD8452", "#4C72B0"]  # red, orange, blue

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.65)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    lefts = np.zeros(len(df))
    for comp, col in zip(components, colors):
        vals = df[comp].values
        bars = ax.barh(df["model"], vals, left=lefts, color=col,
                       label=comp, height=0.55, zorder=3)
        for bar, val, left in zip(bars, vals, lefts):
            if val > 0.3:
                ax.text(left + val / 2, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", ha="center",
                        fontsize=7, color="white", fontweight="bold")
        lefts = lefts + vals

    # Total WER label at end of each bar
    for i, (_, row) in enumerate(df.iterrows()):
        total = row["Substitution"] + row["Deletion"] + row["Insertion"]
        ax.text(total + 0.3, i, f"WER {total:.1f}%", va="center", fontsize=8, color="#333333")

    ax.set_xlabel("Error Rate (% of reference words)", fontsize=11)
    ax.set_title("Error Type Breakdown: Substitution / Deletion / Insertion", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    ax.legend(title="Error type", loc="lower right", fontsize=9, frameon=False)
    plt.tight_layout()
    _save(fig, out, "W2_whisper_error_stack.png")


# ── Chart W3: Radar/spider chart of dimension scores ─────────────────────────

def chart_whisper_radar(lb: list[dict], out: Path):
    """
    Whisper-paper multi-axis style:
    Radar/spider chart showing normalised dimension scores per model.
    Each axis = one evaluation dimension; higher = better for all axes.
    """
    dims = ["intelligibility", "semantic", "latency", "formatting", "hallucination", "entity"]
    dim_labels = ["Intelligibility", "Semantic", "Latency", "Formatting", "Hallucination\nResistance", "Entity\nAccuracy"]

    rows = []
    for entry in lb:
        ds = entry.get("dimension_scores", {})
        vals = [ds.get(d, float("nan")) for d in dims]
        if all(not np.isfinite(v) for v in vals):
            continue
        rows.append({"model": entry["model"], "vals": [max(0.0, v) if np.isfinite(v) else 0.0 for v in vals]})

    if not rows:
        log.info("  Skipping W3 — no dimension score data")
        return

    N      = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the loop

    colors = _w_colors(len(rows))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    for idx, row in enumerate(rows):
        vals   = row["vals"] + row["vals"][:1]
        col    = colors[idx]
        ax.plot(angles, vals, color=col, linewidth=2.2, label=row["model"], zorder=3)
        ax.fill(angles, vals, color=col, alpha=0.08)
        # Mark points
        ax.scatter(angles[:-1], row["vals"], color=col, s=55, zorder=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    # Concentric grid rings
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [r] * (N + 1), color="#cccccc", linewidth=0.6, zorder=1)
        ax.text(angles[0], r + 0.03, f"{r:.2f}", fontsize=7, color="#888888", ha="center")

    ax.spines["polar"].set_visible(False)
    ax.set_title("Dimension Score Radar  (higher = better on all axes)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(bbox_to_anchor=(1.25, 1.05), loc="upper left", fontsize=9, frameon=False, title="Model")
    plt.tight_layout()
    _save(fig, out, "W3_whisper_radar.png")


# ── Chart W4: WER cumulative distribution (CDF) ──────────────────────────────

def chart_whisper_wer_cdf(per_sample_all: dict[str, list[dict]], out: Path):
    """
    Whisper-paper robustness style:
    Empirical CDF of per-sample WER per model.
    X: WER (%), Y: fraction of samples at or below that WER.
    A model with the curve shifted left = better overall.
    """
    rows = []
    for model, samples in per_sample_all.items():
        for s in samples:
            wer = s.get("wer")
            if wer is not None and np.isfinite(wer):
                rows.append({"model": model, "wer": wer * 100})

    if not rows:
        log.info("  Skipping W4 — no per-sample WER data")
        return

    df     = pd.DataFrame(rows)
    models = sorted(df["model"].unique())
    colors = _w_colors(len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for model, col in zip(models, colors):
        wers = np.sort(df[df["model"] == model]["wer"].values)
        cdf  = np.arange(1, len(wers) + 1) / len(wers)
        ax.plot(wers, cdf, color=col, linewidth=2.2, label=model)
        # Mark median
        med = np.median(wers)
        med_cdf = cdf[np.searchsorted(wers, med)]
        ax.plot(med, med_cdf, "o", color=col, markersize=7, zorder=5)
        ax.axvline(med, color=col, linewidth=0.6, linestyle=":", alpha=0.55)

    ax.set_xlabel("Word Error Rate (%)", fontsize=11)
    ax.set_ylabel("Fraction of samples ≤ WER", fontsize=11)
    ax.set_title("Cumulative WER Distribution  (left = better; ● = median)", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0.5, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.text(ax.get_xlim()[1] * 0.98, 0.51, "50th pct", ha="right", fontsize=8, color="#888888")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, frameon=False)
    plt.tight_layout()
    _save(fig, out, "W4_whisper_wer_cdf.png")


# ── Chart W5: Metrics scorecard heatmap ──────────────────────────────────────

def chart_whisper_scorecard(all_models_data: dict[str, dict], out: Path):
    """
    Whisper-paper table-as-figure style:
    Heatmap with metrics as rows, models as columns.
    Cell colour = relative rank (green=best, red=worst).
    Raw values annotated in each cell.
    """
    SCORECARD_METRICS = [
        ("mean_wer",               "WER (%)",          True,  100,   ".1f"),
        ("mean_cer",               "CER (%)",          True,  100,   ".1f"),
        ("mean_wrr",               "WRR (%)",          False, 100,   ".1f"),
        ("mean_semdist",           "SemDist",          True,  1,     ".3f"),
        ("mean_bertscore_f1",      "BERTScore F1",     False, 1,     ".3f"),
        ("mean_bleu_4",            "BLEU-4",           False, 1,     ".3f"),
        ("mean_meteor",            "METEOR",           False, 1,     ".3f"),
        ("mean_fwer",              "FWER (%)",         True,  100,   ".1f"),
        ("mean_punctuation_f1",    "Punct. F1",        False, 1,     ".3f"),
        ("mean_rtfx",              "RTFx",             False, 1,     ".1f"),
        ("mean_her",               "HER",              True,  1,     ".3f"),
        ("mean_sub_rate",          "Sub Rate (%)",     True,  100,   ".1f"),
        ("mean_del_rate",          "Del Rate (%)",     True,  100,   ".1f"),
        ("mean_ins_rate",          "Ins Rate (%)",     True,  100,   ".1f"),
        ("cost_per_hour_usd",      "Cost/hr ($)",      True,  1,     ".3f"),
    ]

    if not all_models_data:
        return

    models = list(all_models_data.keys())
    metric_labels = []
    data_matrix   = []   # shape: [n_metrics, n_models]
    rank_matrix   = []   # normalized rank 0→1 where 1 = best

    for key, label, lower_is_better, mult, fmt in SCORECARD_METRICS:
        vals = []
        for model in models:
            raw = all_models_data[model].get("raw", {})
            v = raw.get(key, float("nan"))
            vals.append(float(v) * mult if np.isfinite(float(v) if v is not None else float("nan")) else float("nan"))

        finite_vals = [v for v in vals if np.isfinite(v)]
        if not finite_vals:
            continue

        # Rank: best model gets 1.0, worst gets 0.0
        sorted_v = sorted(finite_vals, reverse=not lower_is_better)
        def _rank(v):
            if not np.isfinite(v):
                return float("nan")
            if len(sorted_v) == 1:
                return 1.0
            pos = sorted_v.index(v) if v in sorted_v else 0
            return 1.0 - pos / (len(sorted_v) - 1)

        ranks = [_rank(v) for v in vals]
        metric_labels.append(label)
        data_matrix.append(vals)
        rank_matrix.append(ranks)

    if not metric_labels:
        log.info("  Skipping W5 — no finite metric data")
        return

    annot = np.array([[
        f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
        if np.isfinite(v) else "—"
        for v in row
    ] for row in data_matrix], dtype=object)

    rank_arr = np.array([[r if np.isfinite(r) else 0.5 for r in row] for row in rank_matrix])

    n_metrics = len(metric_labels)
    n_models  = len(models)

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.6), max(6, n_metrics * 0.55)))
    fig.patch.set_facecolor("white")

    # Custom diverging colormap: red→white→green
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#C44E52", "#FFFFFF", "#55A868"], N=256
    )

    im = ax.imshow(rank_arr, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotate cells
    for r in range(n_metrics):
        for c in range(n_models):
            txt = annot[r, c]
            rank = rank_arr[r, c]
            text_col = "white" if rank < 0.2 or rank > 0.8 else "#222222"
            ax.text(c, r, txt, ha="center", va="center", fontsize=8,
                    color=text_col, fontweight="bold" if rank > 0.85 else "normal")

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(metric_labels, fontsize=9)
    ax.set_title("Model Scorecard  (green = best, red = worst per metric)",
                 fontsize=12, fontweight="bold", pad=12)

    # Thin grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_models, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_metrics, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label("Relative rank (1 = best)", fontsize=8)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["worst", "mid", "best"])

    plt.tight_layout()
    _save(fig, out, "W5_whisper_scorecard.png")


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--leaderboard", "-l", default="analysis/leaderboard.json", show_default=True)
@click.option("--metrics-dir", "-m", default="results/metrics",            show_default=True)
@click.option("--output",      "-o", default="analysis/charts",            show_default=True)
def main(leaderboard, metrics_dir, output):
    """
    Generate 16 publication-quality charts from leaderboard + per-sample metrics.

    Per-dataset charts saved to --output/:
      01 wer_comparison      07 semdist_violin
      02 accuracy_vs_speed   08 speaker_heatmap
      03 cost_vs_wer         09 wer_vs_duration  (scatter + trend)
      04 use_case_scores     10 metrics_grid      (all metrics, all models)
      05 dimension_breakdown 11 snr_vs_wer        (noise robustness)
      06 wer_boxplot

      Whisper-paper style:
      W1 whisper_grouped_wer  (grouped horizontal WER by utterance length)
      W2 whisper_error_stack  (stacked sub/del/ins breakdown)
      W3 whisper_radar        (radar chart of dimension scores)
      W4 whisper_wer_cdf      (cumulative WER distribution)
      W5 whisper_scorecard    (metrics × models heatmap scorecard)

    For cross-dataset comparison run: python compare_datasets.py
    """
    lb_path = Path(leaderboard)
    if not lb_path.exists():
        log.error(f"Leaderboard not found: {lb_path}. Run aggregate.py first.")
        return

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    lb_data         = json.loads(lb_path.read_text())
    lb              = lb_data["leaderboard"]
    all_models_data = lb_data.get("all_models", {})
    per_sample_all  = _load_per_sample(Path(metrics_dir))

    log.info(f"Generating charts for {len(lb)} models → {out}/")

    # ── Original 11 charts ────────────────────────────────────────────────────
    chart_wer_comparison(lb, out)
    chart_accuracy_vs_speed(lb, out)
    chart_cost_vs_wer(lb, out)
    chart_use_case_scores(lb, out)
    chart_dimension_breakdown(lb, out)
    chart_wer_boxplot(per_sample_all, out)
    chart_semdist_violin(per_sample_all, out)
    chart_speaker_heatmap(per_sample_all, out)
    chart_wer_vs_duration(per_sample_all, out)
    chart_metrics_grid(all_models_data, out)
    chart_snr_vs_wer(per_sample_all, out)

    # ── Whisper-paper-style charts ────────────────────────────────────────────
    log.info("Generating Whisper-paper-style charts…")
    chart_whisper_grouped_wer(per_sample_all, lb, out)
    chart_whisper_error_stack(all_models_data, out)
    chart_whisper_radar(lb, out)
    chart_whisper_wer_cdf(per_sample_all, out)
    chart_whisper_scorecard(all_models_data, out)

    log.info(f"\nAll 16 charts saved to {out}/")
    log.info("For cross-dataset comparison run: python compare_datasets.py")


if __name__ == "__main__":
    main()
