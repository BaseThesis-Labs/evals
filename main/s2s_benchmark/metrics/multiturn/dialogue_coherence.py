"""
Dialogue coherence metric for multi-turn S2S evaluation.

Uses GPT-4o to judge the overall coherence of a multi-turn dialogue
transcript on a 1-5 scale, normalized to [0, 1].

Exposed functions:
    compute_dialogue_coherence(session_result) -> float
"""
from __future__ import annotations

import os
import re
from typing import Optional


def compute_dialogue_coherence(session_result) -> Optional[float]:
    """LLM-judged dialogue coherence score.

    Sends the full transcript to GPT-4o and asks for a coherence rating
    on a 1-5 scale, then normalizes to [0, 1].

    Args:
        session_result: SessionResult with .turns list of TurnResult objects.

    Returns:
        Score in [0, 1] where 1.0 = maximally coherent dialogue.
        None if API is unavailable or transcript is empty.
    """
    try:
        transcript = _build_transcript(session_result.turns)
        if not transcript.strip():
            return None

        prompt = (
            "You are evaluating the coherence of a multi-turn voice agent dialogue.\n\n"
            f"TRANSCRIPT:\n{transcript}\n\n"
            "Rate the overall dialogue coherence on a scale of 1-5:\n"
            "  1 = Completely incoherent, contradictory, or nonsensical\n"
            "  2 = Mostly incoherent with major logical gaps\n"
            "  3 = Partially coherent but with noticeable issues\n"
            "  4 = Mostly coherent with minor issues\n"
            "  5 = Fully coherent, logically consistent, natural flow\n\n"
            "Consider: topic consistency, logical flow, reference resolution,\n"
            "appropriate responses to context, and natural turn-taking.\n\n"
            "Reply with ONLY a single integer from 1 to 5."
        )

        raw = _call_llm(prompt)
        if raw is None:
            return None

        # Parse the score
        match = re.search(r"([1-5])", raw.strip())
        if not match:
            return None

        score = int(match.group(1))
        # Normalize 1-5 to 0-1
        return float((score - 1) / 4.0)

    except Exception as exc:
        print(f"  [multiturn] dialogue_coherence error: {exc}")
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
    """Call LLM judge for dialogue coherence: Gemini → Groq → OpenAI."""
    system_prompt = "You are an expert evaluator of dialogue systems."

    # 1. Gemini (primary, multi-key rotation)
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
            for api_key in _gemini_keys:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(
                        model_name="gemini-2.5-flash",
                        system_instruction=system_prompt,
                    )
                    resp = model.generate_content(prompt, request_options={"timeout": 30})
                    return resp.text
                except Exception:
                    continue
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
                temperature=0.0, max_tokens=16, timeout=15,
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
                temperature=0.0, max_tokens=16,
            )
            return resp.choices[0].message.content
        except Exception:
            pass

    return None
