#!/usr/bin/env python3
"""
export_csv.py — Export all S2S benchmark results to CSV files.

Reads every {model}_metrics.json from results/{dataset}/ and produces:
  results/all_metrics.csv   — one row per model × dataset (all raw means)
  results/leaderboard.csv   — composite scores sorted by balanced score
  results/dimensions.csv    — per-dimension scores per model × dataset
  results/utterances.csv    — every utterance (use --include-utterances)

Usage:
    python export_csv.py [--results-root results] [--include-utterances]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_all_results(results_root: Path):
    rows = []
    for ds_dir in sorted(results_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        for mfile in sorted(ds_dir.glob("*_metrics.json")):
            model = mfile.stem.replace("_metrics", "")
            try:
                data = json.loads(mfile.read_text())
            except Exception:
                continue
            agg = data.get("aggregate", data)
            rows.append((dataset, model, agg))
    return rows


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_all_metrics(rows, out_path: Path):
    all_keys = sorted({k for _, _, agg in rows for k in agg.get("raw_means", {})})
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "n_utterances", "n_errors"] + all_keys)
        for dataset, model, agg in rows:
            raw = agg.get("raw_means", {})
            w.writerow([dataset, model, agg.get("n_utterances", 0), agg.get("n_errors", 0)]
                       + [_fmt(raw.get(k)) for k in all_keys])
    print(f"  ✓ {out_path}  ({len(rows)} rows)")


def write_leaderboard(rows, out_path: Path):
    def _key(r):
        v = r[2].get("composites", {}).get("balanced")
        return v if v is not None else -1.0

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "n_utterances", "n_errors",
                    "score_balanced", "score_conversational", "score_audiobook",
                    "score_voice_cloning", "score_expressive"])
        for dataset, model, agg in sorted(rows, key=_key, reverse=True):
            c = agg.get("composites", {})
            w.writerow([dataset, model, agg.get("n_utterances", 0), agg.get("n_errors", 0),
                        _fmt(c.get("balanced")), _fmt(c.get("conversational")),
                        _fmt(c.get("audiobook")), _fmt(c.get("voice_cloning")),
                        _fmt(c.get("expressive"))])
    print(f"  ✓ {out_path}  ({len(rows)} rows, sorted by balanced score)")


def write_dimensions(rows, out_path: Path):
    dim_keys = ["content", "asr_quality", "speaker", "quality",
                "prosody", "emotion", "latency", "response_quality"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "n_utterances", "n_errors"] + dim_keys)
        for dataset, model, agg in rows:
            dims = agg.get("dimensions", {})
            w.writerow([dataset, model, agg.get("n_utterances", 0), agg.get("n_errors", 0)]
                       + [_fmt(dims.get(k)) for k in dim_keys])
    print(f"  ✓ {out_path}  ({len(rows)} rows)")


def write_utterances(results_root: Path, out_path: Path):
    all_keys: set = set()
    all_rows = []
    for ds_dir in sorted(results_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        for jfile in sorted(ds_dir.glob("*_utterances.jsonl")):
            model = jfile.stem.replace("_utterances", "")
            for line in jfile.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rec["dataset"] = dataset
                rec["model"] = model
                all_keys.update(rec.keys())
                all_rows.append(rec)

    cols = ["dataset", "model", "id"] + sorted(
        k for k in all_keys if k not in ("dataset", "model", "id"))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        for rec in all_rows:
            w.writerow({k: _fmt(rec.get(k)) for k in cols})
    print(f"  ✓ {out_path}  ({len(all_rows)} utterances)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--include-utterances", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"Results directory not found: {results_root}")
        return

    rows = load_all_results(results_root)
    if not rows:
        print("No *_metrics.json files found.")
        return

    print(f"Found {len(rows)} model × dataset results\n")
    write_all_metrics(rows, results_root / "all_metrics.csv")
    write_leaderboard(rows, results_root / "leaderboard.csv")
    write_dimensions(rows, results_root / "dimensions.csv")
    if args.include_utterances:
        write_utterances(results_root, results_root / "utterances.csv")

    print(f"\nAll CSV files written to {results_root}/")


if __name__ == "__main__":
    main()
