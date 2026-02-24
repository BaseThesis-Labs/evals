#!/usr/bin/env python3
"""
Re-evaluate context retention probes AND context checks using Gemini LLM judge.

Reads existing agent response audio, transcribes with Whisper, then
re-evaluates using keyword fast-pass + Gemini LLM judge fallback.
Updates the multiturn JSON result files with new context_retention
and error_recovery scores and recomputes aggregates.

Usage:
    cd s2s_benchmark
    python3 scripts/reeval_context_probes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from datasets.multiturn.scenario_builder import (
    load_all_scenarios, Scenario, ContextProbe, ContextCheck,
)
from metrics.multiturn.context_retention import evaluate_probe_llm
from scoring.aggregate import aggregate_multiturn_sessions

SCENARIOS_DIR = ROOT / "datasets" / "multiturn" / "scenarios"
AUDIO_DIR = ROOT / "results" / "multiturn" / "audio"
RESULTS_DIR = ROOT / "results" / "full_run" / "multiturn"

# Generative models with probe/check audio
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


# ── Probe re-evaluation ─────────────────────────────────────────────────────

def eval_probe(probe: ContextProbe, scenario_id: str, model: str) -> Dict:
    """Re-evaluate a single context probe."""
    turn_idx = probe.inject_after_turn
    audio_path = AUDIO_DIR / scenario_id / model / f"{scenario_id}_probe{turn_idx:03d}.wav"

    if not audio_path.exists():
        return {
            "probe_text": probe.probe_text,
            "response_text": "",
            "expected": probe.expected_contains,
            "found": [],
            "passed": False,
            "eval_method": "no_audio",
        }

    response_text = transcribe(str(audio_path))

    # Fast-pass: keyword matching
    expected_lower = [e.lower() for e in probe.expected_contains]
    response_lower = response_text.lower()
    found = [e for e in expected_lower if e in response_lower]

    if found:
        return {
            "probe_text": probe.probe_text,
            "response_text": response_text,
            "expected": probe.expected_contains,
            "found": found,
            "passed": True,
            "eval_method": "keyword",
        }

    # LLM judge fallback
    passed = evaluate_probe_llm(
        probe_text=probe.probe_text,
        model_response=response_text,
        expected_contains=probe.expected_contains,
    )
    return {
        "probe_text": probe.probe_text,
        "response_text": response_text,
        "expected": probe.expected_contains,
        "found": [],
        "passed": passed,
        "eval_method": "llm_judge",
    }


# ── Context check re-evaluation ─────────────────────────────────────────────

def eval_context_check(
    cc: ContextCheck,
    agent_text: str,
    conversation_history: List[Dict],
) -> tuple[bool, str]:
    """Re-evaluate a single context check with LLM judge fallback."""
    text_lower = agent_text.lower()

    if cc.check_type == "response_contains_all":
        expected_lower = [e.lower() for e in cc.expected]
        missing = [e for e in expected_lower if e not in text_lower]
        if not missing:
            return True, "OK"
        # LLM fallback
        passed = evaluate_probe_llm(
            probe_text=cc.prompt or f"Check response contains: {cc.expected}",
            model_response=agent_text,
            expected_contains=cc.expected,
        )
        return passed, "LLM judge pass" if passed else f"Missing: {missing}"

    elif cc.check_type == "response_contains_any":
        expected_lower = [e.lower() for e in cc.expected]
        found = [e for e in expected_lower if e in text_lower]
        if found:
            return True, f"Found: {found}"
        # LLM fallback
        passed = evaluate_probe_llm(
            probe_text=cc.prompt or f"Check response mentions any of: {cc.expected}",
            model_response=agent_text,
            expected_contains=cc.expected,
        )
        return passed, "LLM judge pass" if passed else f"None of {cc.expected}"

    elif cc.check_type == "llm_judge":
        # Already uses LLM — re-run with Gemini via evaluate_probe_llm
        passed = evaluate_probe_llm(
            probe_text=cc.prompt,
            model_response=agent_text,
            expected_contains=cc.expected if cc.expected else [cc.description],
        )
        return passed, "LLM judge pass" if passed else "LLM judge fail"

    return True, "No check"


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

            model_dir = AUDIO_DIR / sid / model

            # ── Re-evaluate context checks (for error_recovery) ──────────
            # Build per-turn context check results
            check_results: List[Optional[bool]] = []  # indexed by user turn order
            conversation_history: List[Dict] = []

            print(f"\n  {sid}:")

            for ut in sc.user_turns:
                # Agent turn audio = turn_index + 1
                agent_turn_idx = ut.turn_index + 1
                agent_audio = model_dir / f"{sid}_turn{agent_turn_idx:03d}.wav"

                agent_text = ""
                if agent_audio.exists():
                    agent_text = transcribe(str(agent_audio))

                conversation_history.append({"role": "user", "text": ut.text})
                conversation_history.append({"role": "agent", "text": agent_text})

                if ut.context_check is not None:
                    passed, details = eval_context_check(
                        ut.context_check, agent_text, conversation_history
                    )
                    check_results.append(passed)
                    status = "PASS" if passed else "FAIL"
                    print(f"    Check @turn{agent_turn_idx}: {status} — {details}")
                    if not passed:
                        print(f"      Agent: \"{agent_text[:100]}\"")
                else:
                    check_results.append(None)

            # Compute error_recovery from check_results
            check_bools = [(i, p) for i, p in enumerate(check_results) if p is not None]
            if not check_bools:
                new_er = None
            else:
                failures = [i for i, (_, p) in enumerate(check_bools) if not p]
                if not failures:
                    new_er = 1.0
                else:
                    recoveries = 0
                    for fail_pos in failures:
                        for sub_pos in range(fail_pos + 1, len(check_bools)):
                            _, p = check_bools[sub_pos]
                            if p:
                                recoveries += 1
                                break
                    new_er = recoveries / len(failures)

            old_er = session.get("error_recovery")
            session["error_recovery"] = new_er
            er_change = ""
            if old_er is not None and new_er is not None:
                er_change = f" (was {old_er}, delta {new_er - old_er:+.2f})"
            print(f"    → error_recovery: {new_er}{er_change}")

            # ── Re-evaluate context probes (for context_retention) ────────
            if sc.context_probes:
                probe_results = []
                for probe in sc.context_probes:
                    result = eval_probe(probe, sid, model)
                    probe_results.append(result)
                    status = "PASS" if result["passed"] else "FAIL"
                    method = result["eval_method"]
                    print(f"    Probe @{probe.inject_after_turn}: {status} ({method})")
                    print(f"      Expected: {probe.expected_contains}")
                    print(f"      Agent:    \"{result['response_text'][:100]}\"")

                total = len(probe_results)
                passed_count = sum(1 for p in probe_results if p["passed"])
                new_cr = passed_count / total

                old_cr = session.get("context_retention")
                session["context_retention"] = new_cr
                cr_change = ""
                if old_cr is not None:
                    cr_change = f" (was {old_cr}, delta {new_cr - old_cr:+.2f})"
                print(f"    → context_retention: {new_cr:.2f}{cr_change}")

        # Recompute aggregate
        data["aggregate"] = aggregate_multiturn_sessions(
            sessions, model_type="generative"
        )

        json_path.write_text(json.dumps(data, indent=2))
        agg = data["aggregate"]["raw_means"]
        print(f"\n  Saved {json_path.name}")
        print(f"  Aggregate context_retention: {agg.get('context_retention')}")
        print(f"  Aggregate error_recovery:    {agg.get('error_recovery')}")

    print("\n\nDone! Run: python3 scripts/export_excel.py")


if __name__ == "__main__":
    main()
