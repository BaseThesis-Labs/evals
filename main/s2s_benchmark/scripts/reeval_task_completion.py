#!/usr/bin/env python3
"""
Re-evaluate task completion using Gemini LLM judge (with retry logic).

Reads existing agent response audio, transcribes with Whisper, then
re-evaluates task completion using the LLM judge from task_completion.py.
Updates the multiturn JSON result files and recomputes aggregates.

Usage:
    cd s2s_benchmark
    python3 scripts/reeval_task_completion.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from datasets.multiturn.scenario_builder import load_all_scenarios, Scenario
from metrics.multiturn.task_completion import _call_llm
from metrics.multiturn.session_verdict import evaluate_session_verdict
from scoring.aggregate import aggregate_multiturn_sessions

SCENARIOS_DIR = ROOT / "datasets" / "multiturn" / "scenarios"
AUDIO_DIR = ROOT / "results" / "multiturn" / "audio"
RESULTS_DIR = ROOT / "results" / "full_run" / "multiturn"

# Generative models with turn audio
MODELS = ["cascaded_groq_cartesia", "cascaded_groq_deepgram", "ultravox"]

_whisper_model = None


def transcribe(audio_path: str) -> str:
    global _whisper_model
    import whisper  # type: ignore
    if _whisper_model is None:
        print("Loading Whisper-base …")
        _whisper_model = whisper.load_model("base", device="cpu")
    result = _whisper_model.transcribe(audio_path, language="en", fp16=False)
    return result["text"].strip()


def build_transcript(scenario: Scenario, model: str, scenario_id: str) -> Optional[str]:
    """Build a full conversation transcript from scenario text + transcribed agent audio."""
    model_dir = AUDIO_DIR / scenario_id / model
    lines = []

    for ut in scenario.user_turns:
        # User turn text comes from scenario YAML
        lines.append(f"[USER]: {ut.text}")

        # Agent turn audio = turn_index + 1
        agent_turn_idx = ut.turn_index + 1
        agent_audio = model_dir / f"{scenario_id}_turn{agent_turn_idx:03d}.wav"

        if agent_audio.exists():
            agent_text = transcribe(str(agent_audio))
            lines.append(f"[AGENT]: {agent_text}")
        else:
            lines.append("[AGENT]: (no response)")

    return "\n".join(lines) if lines else None


def evaluate_task_completion(
    transcript: str,
    scenario: Scenario,
) -> Optional[float]:
    """Run LLM judge on transcript to evaluate task completion."""
    criteria = scenario.success_criteria
    if not criteria:
        return None

    required = criteria.get("required", []) or []
    optional = criteria.get("optional", []) or []

    if not required and not optional:
        return None

    all_criteria = []
    for i, c in enumerate(required):
        all_criteria.append({"id": f"R{i+1}", "type": "required", "criterion": c})
    for i, c in enumerate(optional):
        all_criteria.append({"id": f"O{i+1}", "type": "optional", "criterion": c})

    criteria_text = "\n".join(
        f"  {c['id']} ({c['type']}): {c['criterion']}" for c in all_criteria
    )

    prompt = (
        "You are evaluating a voice agent's task completion in a multi-turn dialogue.\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"SUCCESS CRITERIA:\n{criteria_text}\n\n"
        "For each criterion, respond with 1 if met or 0 if not met.\n"
        "Return ONLY valid JSON in this format:\n"
        '{"results": {"R1": 1, "R2": 0, "O1": 1, ...}}'
    )

    raw = _call_llm(prompt)
    if raw is None:
        print("      LLM returned None!")
        return None

    # Parse response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        print(f"      Could not parse LLM response: {raw[:100]}")
        return None
    parsed = json.loads(match.group(0))
    results = parsed.get("results", {})

    # Score required criteria
    required_scores: List[float] = []
    for i in range(len(required)):
        key = f"R{i+1}"
        val = results.get(key)
        if val is not None:
            score = float(min(1, max(0, int(val))))
            required_scores.append(score)
            status = "MET" if score > 0 else "NOT MET"
            print(f"      {key}: {status} — {required[i][:60]}")

    # Score optional criteria
    optional_scores: List[float] = []
    for i in range(len(optional)):
        key = f"O{i+1}"
        val = results.get(key)
        if val is not None:
            score = float(min(1, max(0, int(val))))
            optional_scores.append(score)
            status = "MET" if score > 0 else "NOT MET"
            print(f"      {key}: {status} — {optional[i][:60]}")

    # Weighted combination
    req_avg = sum(required_scores) / len(required_scores) if required_scores else 0.0
    opt_avg = sum(optional_scores) / len(optional_scores) if optional_scores else 0.0

    if required and optional:
        score = 0.8 * req_avg + 0.2 * opt_avg
    elif required:
        score = req_avg
    else:
        score = opt_avg

    return float(max(0.0, min(1.0, score)))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    scenarios = load_all_scenarios(SCENARIOS_DIR)
    scenario_map: Dict[str, Scenario] = {s.scenario_id: s for s in scenarios}
    print(f"Loaded {len(scenarios)} scenarios\n")

    for model in MODELS:
        json_path = RESULTS_DIR / f"{model}_multiturn.json"
        if not json_path.exists():
            print(f"Skipping {model} — no result file")
            continue

        data = json.loads(json_path.read_text())
        sessions = data["sessions"]

        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        for session in sessions:
            sid = session["scenario_id"]
            sc = scenario_map.get(sid)
            if not sc:
                print(f"  {sid}: scenario not found, skipping")
                continue

            print(f"\n  {sid}:")

            # Build transcript from audio
            transcript = build_transcript(sc, model, sid)
            if not transcript:
                print("    No transcript — skipping")
                continue

            # Show transcript preview
            lines = transcript.split("\n")
            print(f"    Transcript: {len(lines)} lines")

            # Run LLM judge
            new_tc = evaluate_task_completion(transcript, sc)

            old_tc = session.get("task_completion")
            session["task_completion"] = new_tc
            tc_change = ""
            if old_tc is not None and new_tc is not None:
                tc_change = f" (was {old_tc:.2f}, delta {new_tc - old_tc:+.2f})"
            print(f"    → task_completion: {new_tc}{tc_change}")

            # Run session verdict
            verdict = evaluate_session_verdict(transcript, sc)
            if verdict:
                for vk, vv in verdict.items():
                    session[vk] = vv
                v_pass = "PASS" if verdict.get("session_verdict", 0) > 0.5 else "FAIL"
                print(f"    → session_verdict: {v_pass}")
                if verdict.get("session_verdict_impossible", 0) > 0.5:
                    print(f"      ⚠ impossible_task flagged")
                if verdict.get("session_verdict_audio_issues", 0) > 0.5:
                    print(f"      ⚠ audio_issues flagged")
            else:
                print(f"    → session_verdict: (no result)")

        # Recompute aggregate
        data["aggregate"] = aggregate_multiturn_sessions(
            sessions, model_type="generative"
        )

        json_path.write_text(json.dumps(data, indent=2))
        agg = data["aggregate"]["raw_means"]
        print(f"\n  Saved {json_path.name}")
        print(f"  Aggregate task_completion: {agg.get('task_completion')}")
        print(f"  Aggregate session_verdict: {agg.get('session_verdict')}")

    print("\n\nDone! Run: python3 scripts/export_excel.py")


if __name__ == "__main__":
    main()
