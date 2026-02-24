#!/usr/bin/env python3
"""Re-aggregate multiturn result files using updated DIMENSIONS mapping.

Usage:
    python scripts/reaggregate_multiturn.py [results_dir]

Reads each *_multiturn.json, re-computes dimensions/composites from
session-level metrics, and overwrites the aggregate section in-place.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scoring.aggregate import aggregate_multiturn_sessions  # noqa: E402


def reaggregate(results_dir: Path):
    for path in sorted(results_dir.glob("*_multiturn.json")):
        with open(path) as f:
            data = json.load(f)

        sessions = data.get("sessions", [])
        if not sessions:
            print(f"  SKIP {path.name}: no sessions")
            continue

        old_agent = data.get("aggregate", {}).get("composites", {}).get("agent")
        new_agg = aggregate_multiturn_sessions(sessions)
        data["aggregate"] = new_agg
        new_agent = new_agg.get("composites", {}).get("agent")

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  {path.name}: agent {old_agent} → {new_agent}")
        dims = new_agg.get("dimensions", {})
        for d in ("speaker", "quality", "latency", "error_recovery", "task_completion",
                   "context_retention", "dialogue_coherence"):
            print(f"    {d}: {dims.get(d)}")


if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "full_run" / "multiturn"
    print(f"Re-aggregating multiturn results in {results_dir}")
    reaggregate(results_dir)
    print("Done.")
