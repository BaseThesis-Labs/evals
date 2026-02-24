#!/usr/bin/env python3
"""
Quick validation: transcribe existing probe response audio and compare
against expected keywords.  Confirms whether low context_retention scores
are from brittle keyword matching or genuine failures.

Usage:
    cd s2s_benchmark
    python scripts/validate_probes.py [--model ultravox]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.multiturn.scenario_builder import load_all_scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ultravox", help="Model to inspect")
    parser.add_argument(
        "--scenarios-dir",
        default=str(ROOT / "datasets" / "multiturn" / "scenarios"),
    )
    parser.add_argument(
        "--audio-dir",
        default=str(ROOT / "results" / "multiturn" / "audio"),
    )
    args = parser.parse_args()

    # Load scenarios
    scenarios = load_all_scenarios(args.scenarios_dir)
    print(f"Loaded {len(scenarios)} scenarios\n")

    # Load Whisper
    import whisper  # type: ignore

    print("Loading Whisper-base …")
    model = whisper.load_model("base", device="cpu")

    total_probes = 0
    keyword_pass = 0
    semantic_pass = 0  # would-pass with semantic understanding (manual)

    for sc in scenarios:
        if not sc.context_probes:
            continue

        print(f"\n{'='*70}")
        print(f"Scenario: {sc.scenario_id} ({sc.category})")
        print(f"{'='*70}")

        for probe in sc.context_probes:
            total_probes += 1
            turn_idx = probe.inject_after_turn

            # Find the probe response audio
            # Pattern: {audio_dir}/{scenario_id}/{model}/{scenario_id}_probe{NNN}.wav
            probe_audio = (
                Path(args.audio_dir)
                / sc.scenario_id
                / args.model
                / f"{sc.scenario_id}_probe{turn_idx:03d}.wav"
            )

            if not probe_audio.exists():
                print(f"\n  Probe after turn {turn_idx}: AUDIO NOT FOUND")
                print(f"    Path: {probe_audio}")
                continue

            # Transcribe
            result = model.transcribe(str(probe_audio), language="en", fp16=False)
            response_text = result["text"].strip()

            # Keyword matching (current method)
            expected_lower = [e.lower() for e in probe.expected_contains]
            response_lower = response_text.lower()
            found = [e for e in expected_lower if e in response_lower]
            passed = len(found) > 0

            if passed:
                keyword_pass += 1

            status = "PASS (keyword)" if passed else "FAIL (keyword)"
            print(f"\n  Probe after turn {turn_idx}: {status}")
            print(f"    Question:  {probe.probe_text}")
            print(f"    Expected:  {probe.expected_contains}")
            print(f"    Agent said: \"{response_text}\"")
            print(f"    Keywords found: {found if found else 'NONE'}")

            if not passed:
                print(f"    >>> MANUAL CHECK: Does this response demonstrate context retention?")

    print(f"\n{'='*70}")
    print(f"SUMMARY for model={args.model}")
    print(f"{'='*70}")
    print(f"  Total probes:       {total_probes}")
    print(f"  Keyword matches:    {keyword_pass}/{total_probes}")
    print(f"  Keyword pass rate:  {keyword_pass/total_probes:.1%}" if total_probes else "  N/A")
    print(f"\n  Review the FAIL cases above to determine if an LLM judge would pass them.")


if __name__ == "__main__":
    main()
