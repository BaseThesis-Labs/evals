"""
Task completion metric for multi-turn S2S evaluation.

Uses GPT-4o to judge whether the voice agent successfully completed
the scenario's required and optional success criteria.

Exposed functions:
    compute_task_completion(session_result, scenario) -> float
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional


def compute_task_completion(session_result, scenario) -> Optional[float]:
    """LLM-judged task completion score.

    Builds a transcript from session_result.turns, then asks GPT-4o to
    evaluate each success criterion as met (1) or not met (0).

    Score = 0.8 * mean(required) + 0.2 * mean(optional)

    Args:
        session_result: SessionResult with .turns list of TurnResult objects.
        scenario:       Scenario object with .success_criteria containing
                        .required (list[str]) and .optional (list[str]).

    Returns:
        Score in [0, 1]. None if no criteria defined or API unavailable.
    """
    try:
        # Extract criteria
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

        # Build transcript
        transcript = _build_transcript(session_result.turns)
        if not transcript.strip():
            return None

        # Build criteria list for the prompt
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
            return None

        # Parse response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        parsed = json.loads(match.group(0))
        results = parsed.get("results", {})

        # Score required criteria
        required_scores: List[float] = []
        for i in range(len(required)):
            key = f"R{i+1}"
            val = results.get(key)
            if val is not None:
                required_scores.append(float(min(1, max(0, int(val)))))

        # Score optional criteria
        optional_scores: List[float] = []
        for i in range(len(optional)):
            key = f"O{i+1}"
            val = results.get(key)
            if val is not None:
                optional_scores.append(float(min(1, max(0, int(val)))))

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

    except Exception as exc:
        print(f"  [multiturn] task_completion error: {exc}")
        return None


def _build_transcript(turns) -> str:
    """Build a readable transcript from a list of TurnResult objects."""
    lines = []
    for turn in turns:
        role = turn.role.upper()
        text = turn.output_text or turn.input_text or "(no text)"
        lines.append(f"[{role}]: {text}")
    return "\n".join(lines)


def _call_llm(prompt: str) -> Optional[str]:
    """Call LLM judge for task completion: Gemini (with retry) → Groq → OpenAI."""
    import time
    import random

    system_prompt = "You are an expert evaluator of task-oriented dialogue systems."

    # 1. Gemini (primary, multi-key rotation with retry on rate limit)
    _gemini_keys = []
    primary = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if primary:
        _gemini_keys.append(primary)
    for k in ("GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"):
        v = os.getenv(k)
        if v and v not in _gemini_keys:
            _gemini_keys.append(v)
    if _gemini_keys:
        try:
            import google.generativeai as genai
            max_retries = 3
            for attempt in range(max_retries):
                for api_key in _gemini_keys:
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(
                            model_name="gemini-2.5-flash",
                            system_instruction=system_prompt,
                        )
                        resp = model.generate_content(
                            prompt, request_options={"timeout": 30}
                        )
                        return resp.text
                    except Exception as exc:
                        exc_str = str(exc).lower()
                        if "429" in exc_str or "rate" in exc_str or "quota" in exc_str:
                            time.sleep(2 + random.random() * 2)
                            continue
                        continue
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 5
                    print(f"  [multiturn] task_completion: Gemini rate limited, waiting {wait}s …")
                    time.sleep(wait)
        except ImportError:
            pass

    # 2. Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=512, timeout=15,
            )
            return resp.choices[0].message.content
        except Exception:
            pass

    # 3. OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key, timeout=30)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=512,
            )
            return resp.choices[0].message.content
        except Exception:
            pass

    print("  [multiturn] task_completion: all LLM providers failed")
    return None
