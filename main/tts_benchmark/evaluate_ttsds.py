#!/usr/bin/env python3
"""Run TTSDS distributional evaluation across all TTS models.

TTSDS measures how closely TTS output matches real speech distributions
across 5 factors: General, Environment, Intelligibility, Prosody, Speaker.
It uses 2-Wasserstein distance between feature distributions (HuBERT,
wav2vec2, ECAPA-TDNN embeddings) rather than per-utterance scoring.

Usage:
  python evaluate_ttsds.py
  python evaluate_ttsds.py --gen-dir ttsds_gen_audio --reference-dir datasets/ttsds_reference
  python evaluate_ttsds.py --models kokoro elevenlabs cartesia
"""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path


def _get_ttsds_version() -> str:
    try:
        return importlib.metadata.version("ttsds")
    except Exception:
        return "unknown"


def parse_aggregated_results(aggregated_df, model_dirs):
    """Parse a TTSDS aggregated DataFrame into our output schema."""
    FACTOR_ALIASES = {
        "general":         ["general"],
        "environment":     ["environment", "env"],
        "intelligibility": ["intelligibility", "intel"],
        "prosody":         ["prosody"],
        "speaker":         ["speaker"],
    }

    models_output = {}

    try:
        import pandas as pd
    except ImportError:
        pd = None

    for name, _ in model_dirs:
        factor_scores = {}

        if aggregated_df is not None and pd is not None:
            try:
                if isinstance(aggregated_df, pd.DataFrame):
                    # Try to filter rows belonging to this model
                    if "dataset" in aggregated_df.columns:
                        model_rows = aggregated_df[aggregated_df["dataset"] == name]
                    else:
                        model_rows = aggregated_df

                    if not model_rows.empty:
                        for factor, aliases in FACTOR_ALIASES.items():
                            for alias in aliases:
                                # Find a column whose name contains the alias
                                cols = [
                                    c for c in aggregated_df.columns
                                    if alias.lower() in c.lower()
                                ]
                                if cols:
                                    val = model_rows[cols[0]].mean()
                                    if pd.notna(val):
                                        factor_scores[factor] = float(val)
                                        break
            except Exception as e:
                print(f"  ⚠ Could not parse aggregated results for '{name}': {e}")

        available = [v for v in factor_scores.values() if v is not None]
        overall = float(sum(available) / len(available)) if available else None

        models_output[name] = {
            "general":         factor_scores.get("general"),
            "environment":     factor_scores.get("environment"),
            "intelligibility": factor_scores.get("intelligibility"),
            "prosody":         factor_scores.get("prosody"),
            "speaker":         factor_scores.get("speaker"),
            "overall":         overall,
        }

    return models_output


def main():
    parser = argparse.ArgumentParser(
        description="Run TTSDS distributional evaluation across TTS models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models in ttsds_gen_audio/ (default):
  python evaluate_ttsds.py

  # Custom directories:
  python evaluate_ttsds.py \\
      --gen-dir ttsds_gen_audio \\
      --reference-dir datasets/ttsds_reference \\
      --output ttsds_results/ttsds_scores.json

  # Evaluate specific models only:
  python evaluate_ttsds.py --models kokoro elevenlabs cartesia
""",
    )
    parser.add_argument(
        "--gen-dir",
        type=str,
        default="ttsds_gen_audio",
        help="Directory containing model subdirectories of generated audio "
             "(default: ttsds_gen_audio)",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="datasets/ttsds_reference",
        help="Directory of real reference speech WAVs "
             "(default: datasets/ttsds_reference)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ttsds_results/ttsds_scores.json",
        help="Output JSON file path (default: ttsds_results/ttsds_scores.json)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model names to evaluate "
             "(default: all subdirectories found in --gen-dir)",
    )
    args = parser.parse_args()

    gen_dir = Path(args.gen_dir)
    reference_dir = Path(args.reference_dir)
    output_path = Path(args.output)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not gen_dir.exists():
        print(f"✗ Generated audio directory not found: {gen_dir}")
        print(f"  Run: python generate.py --manifest datasets/ttsds_manifest.json "
              f"--output {gen_dir}")
        sys.exit(1)

    if not reference_dir.exists():
        print(f"✗ Reference audio directory not found: {reference_dir}")
        print("  Run: python datasets/download.py --dataset ttsds")
        sys.exit(1)

    ref_wavs = list(reference_dir.glob("*.wav"))
    if not ref_wavs:
        print(f"✗ No WAV files found in reference directory: {reference_dir}")
        print("  Run: python datasets/download.py --dataset ttsds")
        sys.exit(1)

    # ── Discover model directories ────────────────────────────────────────────
    if args.models:
        model_dirs = []
        for name in args.models:
            d = gen_dir / name
            if not d.exists():
                print(f"  ⚠ Model directory not found: {d}, skipping.")
            else:
                model_dirs.append((name, d))
    else:
        model_dirs = [
            (d.name, d)
            for d in sorted(gen_dir.iterdir())
            if d.is_dir()
        ]

    if not model_dirs:
        print(f"✗ No model directories found in {gen_dir}")
        sys.exit(1)

    model_names = [n for n, _ in model_dirs]
    print(f"▶ Evaluating {len(model_dirs)} model(s): {model_names}")
    print(f"▶ Reference dir: {reference_dir} ({len(ref_wavs)} WAVs)")

    # ── Import TTSDS ──────────────────────────────────────────────────────────
    try:
        from ttsds import BenchmarkSuite
        from ttsds.util.dataset import DirectoryDataset
    except ImportError:
        print("✗ ttsds not installed. Run: pip install ttsds")
        sys.exit(1)

    # ── Build dataset objects ─────────────────────────────────────────────────
    reference = DirectoryDataset(str(reference_dir), name="reference")
    datasets = [DirectoryDataset(str(d), name=name) for name, d in model_dirs]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = str(output_path.parent / "raw.csv")

    # ── Run benchmark ─────────────────────────────────────────────────────────
    print(f"▶ Running BenchmarkSuite (this may take a while)...")
    print(f"  Raw CSV will be written to: {raw_csv}")

    suite = BenchmarkSuite(
        datasets=datasets,
        reference_datasets=[reference],
        write_to_file=raw_csv,
        skip_errors=True,
    )
    results = suite.run()

    # ── Aggregate results ─────────────────────────────────────────────────────
    print("▶ Aggregating results...")
    try:
        aggregated_df = suite.get_aggregated_results()
    except AttributeError:
        # Fallback: some versions expose results directly
        aggregated_df = results

    models_output = parse_aggregated_results(aggregated_df, model_dirs)

    # ── Write output JSON ─────────────────────────────────────────────────────
    output_data = {
        "ttsds_version": _get_ttsds_version(),
        "reference_dir": str(reference_dir),
        "models": models_output,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ TTSDS scores saved to {output_path}")
    print(f"  Raw CSV:        {raw_csv}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TTSDS SCORES")
    print("=" * 60)
    header = f"  {'Model':<20}  {'General':>8}  {'Env':>8}  {'Intel':>8}  {'Prosody':>8}  {'Speaker':>8}  {'Overall':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    sorted_models = sorted(
        models_output.items(),
        key=lambda x: x[1].get("overall") or 0,
        reverse=True,
    )
    for model, scores in sorted_models:
        def fmt(v):
            return f"{v:.3f}" if v is not None else "  N/A"

        print(
            f"  {model:<20}  "
            f"{fmt(scores.get('general')):>8}  "
            f"{fmt(scores.get('environment')):>8}  "
            f"{fmt(scores.get('intelligibility')):>8}  "
            f"{fmt(scores.get('prosody')):>8}  "
            f"{fmt(scores.get('speaker')):>8}  "
            f"{fmt(scores.get('overall')):>8}"
        )


if __name__ == "__main__":
    main()
