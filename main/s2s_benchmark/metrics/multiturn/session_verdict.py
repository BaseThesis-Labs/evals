"""
Structured LLM session verdict for multi-turn S2S evaluation.

Uses Gemini 2.5 Flash to produce a structured pass/fail verdict with
detailed reasoning, failure explanation, and flags for impossible tasks
and audio issues — evaluated against success_criteria from scenario YAMLs.

Exposed functions:
    compute_session_verdict(session_result, scenario) -> Dict
    evaluate_session_verdict(transcript, scenario) -> Dict
"""
from __future__ import annotations

import json
import re
from typing import Dict, Optional

from metrics.multiturn.task_completion import _build_transcript, _call_llm

_VERDICT_SYSTEM_PROMPT = (
    "You are an expert evaluator of task-oriented dialogue systems."
)

_VERDICT_USER_TEMPLATE = (
    "You are evaluating a voice agent's performance in a multi-turn dialogue.\n\n"
    "TRANSCRIPT:\n{transcript}\n\n"
    "SUCCESS CRITERIA:\n{criteria_text}\n\n"
    "Evaluate the agent's overall session performance and return a structured JSON verdict.\n\n"
    "Your response MUST be valid JSON with exactly these fields:\n"
    '{{\n'
    '    "reasoning": "Detailed analysis: what the agent did well, what it missed, '
    'how it handled context, quality of responses, and overall assessment.",\n'
    '    "verdict": true or false,\n'
    '    "failure_reason": "If verdict is false: max 5 sentences explaining why the agent failed. '
    'If verdict is true: empty string.",\n'
    '    "impossible_task": true or false,\n'
    '    "audio_issues": true or false\n'
    '}}\n\n'
    "Field definitions:\n"
    "- reasoning: Thorough analysis of the agent's performance across all criteria.\n"
    "- verdict: true if the agent satisfactorily completed the task (met most required criteria), false otherwise.\n"
    "- failure_reason: If verdict=false, explain the primary reasons for failure (max 5 sentences). If verdict=true, use empty string.\n"
    "- impossible_task: true if the task was impossible for the agent to complete due to scenario design, "
    "missing information, or unreasonable expectations. false otherwise.\n"
    "- audio_issues: true if audio quality, transcription errors, or speech recognition problems "
    "significantly affected the session outcome. false otherwise.\n\n"
    "Return ONLY valid JSON, no other text."
)


def _build_criteria_text(scenario) -> Optional[str]:
    """Extract and format success criteria from a scenario."""
    criteria = getattr(scenario, "success_criteria", None)
    if criteria is None:
        return None

    if isinstance(criteria, dict):
        required = criteria.get("required", []) or []
        optional = criteria.get("optional", []) or []
    else:
        required = getattr(criteria, "required", []) or []
        optional = getattr(criteria, "optional", []) or []

    if not required and not optional:
        return None

    all_criteria = []
    for i, c in enumerate(required):
        all_criteria.append(f"  R{i+1} (required): {c}")
    for i, c in enumerate(optional):
        all_criteria.append(f"  O{i+1} (optional): {c}")

    return "\n".join(all_criteria)


def _parse_verdict(raw: str) -> Optional[Dict]:
    """Parse structured JSON verdict from LLM response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if "verdict" not in parsed:
        return None

    return {
        "session_verdict": 1.0 if parsed.get("verdict") else 0.0,
        "session_verdict_reasoning": str(parsed.get("reasoning", "")),
        "session_verdict_failure_reason": str(parsed.get("failure_reason", "")),
        "session_verdict_impossible": 1.0 if parsed.get("impossible_task") else 0.0,
        "session_verdict_audio_issues": 1.0 if parsed.get("audio_issues") else 0.0,
    }


def compute_session_verdict(session_result, scenario) -> Dict:
    """LLM-judged structured session verdict.

    Builds a transcript from session_result.turns, then asks the LLM judge
    to produce a structured pass/fail verdict with reasoning.

    Args:
        session_result: SessionResult with .turns list of TurnResult objects.
        scenario:       Scenario object with .success_criteria.

    Returns:
        Dict with session_verdict (float), reasoning (str),
        failure_reason (str), impossible (float), audio_issues (float).
        Returns empty dict on failure.
    """
    try:
        criteria_text = _build_criteria_text(scenario)
        if criteria_text is None:
            return {}

        transcript = _build_transcript(session_result.turns)
        if not transcript.strip():
            return {}

        return _evaluate_verdict(transcript, criteria_text)

    except Exception as exc:
        print(f"  [multiturn] session_verdict error: {exc}")
        return {}


def evaluate_session_verdict(transcript: str, scenario) -> Dict:
    """Run session verdict on a pre-built transcript (for re-evaluation scripts).

    Args:
        transcript: Full conversation transcript string.
        scenario:   Scenario object with .success_criteria.

    Returns:
        Dict with verdict fields, or empty dict on failure.
    """
    try:
        criteria_text = _build_criteria_text(scenario)
        if criteria_text is None:
            return {}

        if not transcript.strip():
            return {}

        return _evaluate_verdict(transcript, criteria_text)

    except Exception as exc:
        print(f"  [multiturn] session_verdict re-eval error: {exc}")
        return {}


def _evaluate_verdict(transcript: str, criteria_text: str) -> Dict:
    """Core verdict evaluation: build prompt, call LLM, parse response."""
    prompt = _VERDICT_USER_TEMPLATE.format(
        transcript=transcript,
        criteria_text=criteria_text,
    )

    raw = _call_llm(prompt)
    if raw is None:
        return {}

    result = _parse_verdict(raw)
    if result is None:
        print(f"  [multiturn] session_verdict: could not parse LLM response: {raw[:100]}")
        return {}

    return result
