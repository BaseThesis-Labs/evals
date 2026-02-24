"""
Context retention metric for multi-turn S2S evaluation.

Measures how well the voice agent retains information across turns by
evaluating the fraction of context probes that pass.

Exposed functions:
    compute_context_retention(session_result) -> float
    evaluate_probe_llm(probe_text, model_response, expected_contains) -> bool
"""
from __future__ import annotations

import os
from typing import List, Optional


def compute_context_retention(session_result) -> Optional[float]:
    """Fraction of context probes passed across the session.

    Args:
        session_result: SessionResult with .probe_results list of dicts,
                        each containing a "passed" bool key.

    Returns:
        Score in [0, 1] where 1.0 = all probes passed.
        None if there are no probes.
    """
    try:
        probes = getattr(session_result, "probe_results", None)
        if not probes:
            return None

        total = len(probes)
        passed = sum(1 for p in probes if p.get("passed", False))
        return float(passed / total)

    except Exception as exc:
        print(f"  [multiturn] context_retention error: {exc}")
        return None


def evaluate_probe_llm(
    probe_text: str,
    model_response: str,
    expected_contains: List[str],
) -> bool:
    """Use LLM judge to evaluate whether the agent recalled context correctly.

    Uses Gemini 2.5 Flash to determine semantic equivalence — catches cases
    where the agent uses different words/phrasing than the expected keywords.

    Args:
        probe_text:        The probe question that was asked.
        model_response:    The agent's transcribed response.
        expected_contains: List of expected keywords/phrases.

    Returns:
        True if the agent correctly recalled the relevant information.
    """
    system_prompt = (
        "You evaluate whether a voice agent correctly recalled "
        "information from earlier in the conversation."
    )
    user_prompt = (
        f'Probe question: "{probe_text}"\n\n'
        f'Agent\'s response: "{model_response}"\n\n'
        f"The agent should have recalled information related to: "
        f"{expected_contains}\n\n"
        f"Did the agent correctly recall the relevant information, "
        f"even if using different words?\n"
        f'Answer ONLY "yes" or "no".'
    )

    import time
    import random

    # Gemini (multi-key rotation with retry on rate limit)
    gemini_keys: List[str] = []
    primary = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if primary:
        gemini_keys.append(primary)
    for k in ("GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"):
        v = os.environ.get(k)
        if v and v not in gemini_keys:
            gemini_keys.append(v)

    if not gemini_keys:
        print("  [multiturn] evaluate_probe_llm: no Gemini keys found")
        return False

    import google.generativeai as genai

    max_retries = 3
    for attempt in range(max_retries):
        for api_key in gemini_keys:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction=system_prompt,
                )
                resp = model.generate_content(
                    user_prompt, request_options={"timeout": 30}
                )
                answer = resp.text.strip().lower()
                return answer == "yes"
            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "rate" in exc_str or "quota" in exc_str:
                    time.sleep(2 + random.random() * 2)
                    continue
                continue

        # All keys failed this attempt — back off before retrying
        if attempt < max_retries - 1:
            wait = (attempt + 1) * 5
            print(f"  [multiturn] evaluate_probe_llm: rate limited, waiting {wait}s …")
            time.sleep(wait)

    print("  [multiturn] evaluate_probe_llm: all Gemini keys exhausted after retries")
    return False
