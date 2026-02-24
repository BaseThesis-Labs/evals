#!/usr/bin/env python3
"""
Generate results/report.md with Markdown leaderboard table + dimension breakdown.

Usage:
    python report.py \
        [--results results] \
        [--output  results/report.md] \
        [--use-case balanced]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: Optional[float], decimals: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def _pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def _load_json(path: Path) -> Optional[dict | list]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Table builders
# ─────────────────────────────────────────────────────────────────────────────

def _leaderboard_table(leaderboard: List[Dict], use_case: str) -> str:
    """Render a Markdown leaderboard table."""
    lines = [
        f"## Leaderboard — {use_case.replace('_', ' ').title()}",
        "",
        "| Rank | Model | Composite Score | Utterances | Errors |",
        "|------|-------|----------------|-----------|--------|",
    ]
    for row in leaderboard:
        score_str = _fmt(row.get("composite_score"), 4)
        lines.append(
            f"| {row.get('rank', '?')} "
            f"| `{row['model']}` "
            f"| {score_str} "
            f"| {row.get('n_utterances', '?')} "
            f"| {row.get('n_errors', '?')} |"
        )
    return "\n".join(lines)


def _dimension_table(leaderboard: List[Dict]) -> str:
    """Render a per-dimension breakdown table."""
    all_dims = set()
    for row in leaderboard:
        all_dims.update(row.get("dimensions", {}).keys())
    dims = sorted(all_dims)
    if not dims:
        return ""

    header = "| Model | " + " | ".join(d.replace("_", " ").title() for d in dims) + " |"
    sep = "|---|" + "---|" * len(dims)

    lines = [
        "## Dimension Scores",
        "",
        header,
        sep,
    ]
    for row in leaderboard:
        dim_vals = row.get("dimensions", {})
        cells = " | ".join(_fmt(dim_vals.get(d)) for d in dims)
        lines.append(f"| `{row['model']}` | {cells} |")

    return "\n".join(lines)


def _raw_metrics_table(model_results: Dict[str, Dict], metrics: List[str]) -> str:
    """Render a raw metric means table."""
    header = "| Model | " + " | ".join(m for m in metrics) + " |"
    sep = "|---|" + "---|" * len(metrics)
    lines = [
        "## Raw Metric Means",
        "",
        header,
        sep,
    ]
    for model_name, agg in model_results.items():
        raw = agg.get("aggregate", {}).get("raw_means", {})
        cells = " | ".join(_fmt(raw.get(m)) for m in metrics)
        lines.append(f"| `{model_name}` | {cells} |")
    return "\n".join(lines)


def _latency_table(model_results: Dict[str, Dict]) -> str:
    """Render latency percentile table."""
    lines = [
        "## Latency Percentiles (ms)",
        "",
        "| Model | e2e P50 | e2e P90 | e2e P99 | TTFB P50 | ASR P50 |",
        "|-------|---------|---------|---------|----------|---------|",
    ]
    for model_name, agg in model_results.items():
        lp = agg.get("aggregate", {}).get("latency_percentiles", {})
        e2e = lp.get("e2e_latency_ms", {})
        ttfb = lp.get("ttfb_ms", {})
        asr = lp.get("asr_latency_ms", {})
        lines.append(
            f"| `{model_name}` "
            f"| {_fmt(e2e.get('p50'), 0)} "
            f"| {_fmt(e2e.get('p90'), 0)} "
            f"| {_fmt(e2e.get('p99'), 0)} "
            f"| {_fmt(ttfb.get('p50'), 0)} "
            f"| {_fmt(asr.get('p50'), 0)} |"
        )
    return "\n".join(lines)


def _composites_table(model_results: Dict[str, Dict], use_cases: List[str]) -> str:
    """Composite scores across all use cases."""
    header = "| Model | " + " | ".join(uc.replace("_", " ").title() for uc in use_cases) + " |"
    sep = "|---|" + "---|" * len(use_cases)
    lines = [
        "## Composite Scores by Use Case",
        "",
        header,
        sep,
    ]
    for model_name, agg in model_results.items():
        composites = agg.get("aggregate", {}).get("composites", {})
        cells = " | ".join(_fmt(composites.get(uc), 4) for uc in use_cases)
        lines.append(f"| `{model_name}` | {cells} |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

def build_report(results_dir: Path, use_case: str = "balanced") -> str:
    """Read all {model}_metrics.json files and build the full report."""
    import datetime

    # Load model results
    model_results: Dict[str, Dict] = {}
    for p in sorted(results_dir.glob("*_metrics.json")):
        model_name = p.stem.replace("_metrics", "")
        data = _load_json(p)
        if data:
            model_results[model_name] = data

    if not model_results:
        return "# S2S Benchmark Report\n\n_No results found._\n"

    # Load leaderboard for primary use case
    lb_path = results_dir / f"leaderboard_{use_case}.json"
    if not lb_path.exists():
        lb_path = results_dir / "leaderboard.json"
    leaderboard = _load_json(lb_path) or []

    sections: List[str] = []

    # ── Title ─────────────────────────────────────────────────────────────────
    sections.append(
        f"# S2S Benchmark Report\n\n"
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"Models evaluated: {len(model_results)}  \n"
        f"Primary use case: **{use_case.replace('_', ' ').title()}**\n"
    )

    # ── Leaderboard ───────────────────────────────────────────────────────────
    if leaderboard:
        sections.append(_leaderboard_table(leaderboard, use_case))

    # ── Dimension breakdown ───────────────────────────────────────────────────
    if leaderboard:
        sections.append(_dimension_table(leaderboard))

    # ── Composite scores across all use cases ─────────────────────────────────
    use_cases = ["conversational", "audiobook", "voice_cloning", "expressive", "balanced"]
    sections.append(_composites_table(model_results, use_cases))

    # ── Key raw metrics ───────────────────────────────────────────────────────
    key_metrics = ["wer", "cer", "utmos", "sim_wavlm", "f0_rmse", "emotion_match", "pesq", "mcd"]
    sections.append(_raw_metrics_table(model_results, key_metrics))

    # ── Latency ───────────────────────────────────────────────────────────────
    sections.append(_latency_table(model_results))

    # ── Per-model summaries ───────────────────────────────────────────────────
    sections.append("## Per-Model Summaries")
    for model_name, data in model_results.items():
        agg = data.get("aggregate", {})
        n_utt = agg.get("n_utterances", 0)
        n_err = agg.get("n_errors", 0)
        raw = agg.get("raw_means", {})
        composites = agg.get("composites", {})

        summary_lines = [
            f"### `{model_name}`",
            "",
            f"- Utterances: {n_utt} evaluated, {n_err} errors",
            f"- WER: {_fmt(raw.get('wer'))} | CER: {_fmt(raw.get('cer'))} | BERTScore F1: {_fmt(raw.get('bert_score_f1'))}",
            f"- UTMOS: {_fmt(raw.get('utmos'))} | DNSMOS: {_fmt(raw.get('dnsmos_ovrl'))} | PESQ: {_fmt(raw.get('pesq'))}",
            f"- Speaker Sim (WavLM): {_fmt(raw.get('sim_wavlm'))}",
            f"- F0 RMSE: {_fmt(raw.get('f0_rmse'))} Hz | Pitch Corr: {_fmt(raw.get('pitch_corr'))}",
            f"- Emotion Match: {_pct(raw.get('emotion_match'))} | Emotion Sim: {_fmt(raw.get('emotion_sim'))}",
            f"- Balanced composite: **{_fmt(composites.get('balanced'), 4)}**",
            "",
        ]
        sections.append("\n".join(summary_lines))

    # ── Methodology ───────────────────────────────────────────────────────────
    sections.append(
        "## Methodology\n\n"
        "| Dimension | Key Metrics |\n"
        "|-----------|-------------|\n"
        "| Content | WER, CER, BERTScore F1, ROUGE-L, SemDist |\n"
        "| ASR Quality | Insertion/Deletion/Substitution Rate, HER, Hallucination Rate |\n"
        "| Audio Quality | UTMOS, DNSMOS, PESQ, MCD |\n"
        "| Speaker Similarity | WavLM-Base+ cosine, ECAPA-TDNN (optional) |\n"
        "| Prosody | F0 RMSE, Pitch Corr, Energy Corr, Duration Ratio |\n"
        "| Emotion | emotion2vec (iic/emotion2vec_plus_seed) match + similarity |\n"
        "| Latency | TTFB, E2E latency, ASR latency, RTF |\n\n"
        "Cascaded adapters: Whisper-base (CPU) → TTS client (ElevenLabs / Cartesia / Deepgram).\n"
    )

    return "\n\n---\n\n".join(sections) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate S2S benchmark report")
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument("--output", default="results/report.md", help="Output Markdown path")
    parser.add_argument("--use-case", default="balanced", help="Primary use case for leaderboard")
    args = parser.parse_args()

    results_dir = Path(args.results)
    report = build_report(results_dir, args.use_case)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(f"✓ Report written to {out_path}")


if __name__ == "__main__":
    main()
