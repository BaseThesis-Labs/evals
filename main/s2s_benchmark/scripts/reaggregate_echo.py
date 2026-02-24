#!/usr/bin/env python3
"""Re-aggregate echo model results with judge metrics nulled out.

For echo models, judge_* and instruction_follow metrics are semantically
meaningless (the model repeats input, not generates responses). This script:
1. Reads each echo model's utterance-level JSONL
2. Nulls out judge_* and instruction_follow fields
3. Re-runs aggregate_model() with the cleaned data
4. Overwrites the metrics.json with corrected dimensions/composites

Usage:
    python scripts/reaggregate_echo.py [results_root]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scoring.aggregate import aggregate_model  # noqa: E402

ECHO_MODELS = {"cascaded_cartesia", "cascaded_deepgram", "cascaded_elevenlabs"}
JUDGE_FIELDS = {
    "judge_coherence", "judge_relevance", "judge_helpfulness",
    "judge_safety", "judge_naturalness", "judge_overall", "judge_reasoning",
    "instruction_follow", "safety_refusal",
}


def clean_and_reaggregate(results_root: Path):
    datasets = [d for d in results_root.iterdir() if d.is_dir() and d.name != "multiturn"]

    for dataset_dir in sorted(datasets):
        for model in ECHO_MODELS:
            utterances_path = dataset_dir / f"{model}_utterances.jsonl"
            metrics_path = dataset_dir / f"{model}_metrics.json"

            if not utterances_path.exists() or not metrics_path.exists():
                continue

            # Read utterances
            utterances = []
            with open(utterances_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        utterances.append(json.loads(line))

            # Null out judge fields
            n_nulled = 0
            for utt in utterances:
                for field in JUDGE_FIELDS:
                    if field in utt and utt[field] is not None:
                        utt[field] = None
                        n_nulled += 1

            if n_nulled == 0:
                print(f"  SKIP {dataset_dir.name}/{model}: no judge fields to null")
                continue

            # Write cleaned utterances back
            with open(utterances_path, "w") as f:
                for utt in utterances:
                    f.write(json.dumps(utt) + "\n")

            # Re-aggregate
            new_agg = aggregate_model(utterances, model_type="echo")

            # Read existing metrics.json and update aggregate
            with open(metrics_path) as f:
                data = json.load(f)

            old_rq = data["aggregate"]["dimensions"].get("response_quality")
            data["aggregate"] = new_agg

            new_rq = new_agg["dimensions"].get("response_quality")
            old_balanced = None  # will be overwritten
            new_balanced = new_agg["composites"].get("balanced")

            with open(metrics_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"  {dataset_dir.name}/{model}: nulled {n_nulled} judge values")
            print(f"    response_quality: {old_rq} → {new_rq}")
            print(f"    balanced: {new_balanced}")


if __name__ == "__main__":
    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "full_run"
    print(f"Re-aggregating echo models in {results_root}")
    clean_and_reaggregate(results_root)
    print("Done.")
