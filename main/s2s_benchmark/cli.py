#!/usr/bin/env python3
"""
CLI entry point for S2S benchmark.

Unified command-line interface that dispatches to the appropriate pipeline
mode (single-turn eval, multi-turn agent eval, or full run).

Usage:
    # Single-turn evaluation only
    python cli.py eval --manifest datasets/manifests/s2s_manifest.json

    # Multi-turn agent evaluation only
    python cli.py eval --mode multiturn

    # Both single-turn and multi-turn
    python cli.py eval --mode both

    # Full inference + evaluation run (all models × datasets)
    python cli.py run-all --mode both

    # Run specific models
    python cli.py run-all --models gpt4o_realtime gemini_live --mode multiturn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation pipeline (single-turn, multiturn, or both)."""
    from pipeline import main as pipeline_main
    # Build sys.argv for pipeline.main() which uses argparse internally
    argv = [
        "--manifest", args.manifest,
        "--outputs", args.outputs,
        "--results", args.results,
        "--config", args.config,
        "--mode", args.mode,
    ]
    if args.model:
        argv += ["--model", args.model]
    if args.no_skip:
        argv.append("--no-skip")

    sys.argv = ["pipeline.py"] + argv
    pipeline_main()


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run full inference + evaluation across all datasets and models."""
    from run_all import main as run_all_main
    argv = [
        "--config", args.config,
        "--manifests-dir", args.manifests_dir,
        "--results-root", args.results_root,
        "--mode", args.mode,
    ]
    if args.models:
        argv += ["--models"] + args.models
    if args.datasets:
        argv += ["--datasets"] + args.datasets
    if args.skip_models:
        argv += ["--skip-models"] + args.skip_models
    if args.limit:
        argv += ["--limit", str(args.limit)]

    sys.argv = ["run_all.py"] + argv
    run_all_main()


def cmd_scenarios(args: argparse.Namespace) -> None:
    """List available multi-turn scenarios."""
    from datasets.multiturn.scenario_builder import load_all_scenarios
    scenarios = load_all_scenarios(args.scenarios_dir)
    if not scenarios:
        print(f"No scenarios found in {args.scenarios_dir}")
        return
    print(f"Found {len(scenarios)} scenarios in {args.scenarios_dir}:\n")
    for s in scenarios:
        n_turns = len(s.user_turns)
        n_probes = len(s.context_probes)
        print(f"  {s.scenario_id:<30s}  category={s.category:<20s}  "
              f"turns={n_turns}  probes={n_probes}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="s2s_benchmark",
        description="Voice Arena S2S & Agent Evaluation Pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── eval ──────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Run evaluation pipeline")
    p_eval.add_argument("--manifest", default="datasets/manifests/s2s_manifest.json")
    p_eval.add_argument("--outputs", default="s2s_outputs")
    p_eval.add_argument("--results", default="results")
    p_eval.add_argument("--config", default="config/eval_config.yaml")
    p_eval.add_argument("--model", default=None)
    p_eval.add_argument("--mode", choices=["single", "multiturn", "both"], default="single")
    p_eval.add_argument("--no-skip", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    # ── run-all ───────────────────────────────────────────────────────────
    p_run = sub.add_parser("run-all", help="Full inference + evaluation run")
    p_run.add_argument("--config", default="config/eval_config.yaml")
    p_run.add_argument("--manifests-dir", default="datasets/manifests")
    p_run.add_argument("--results-root", default="results")
    p_run.add_argument("--models", nargs="*", default=None)
    p_run.add_argument("--datasets", nargs="*", default=None)
    p_run.add_argument("--skip-models", nargs="*", default=[])
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--mode", choices=["single", "multiturn", "both"], default="single")
    p_run.set_defaults(func=cmd_run_all)

    # ── scenarios ─────────────────────────────────────────────────────────
    p_sc = sub.add_parser("scenarios", help="List available multi-turn scenarios")
    p_sc.add_argument("--scenarios-dir", default="datasets/multiturn/scenarios")
    p_sc.set_defaults(func=cmd_scenarios)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
