"""
Multi-turn session runner for S2S agent evaluation.

Orchestrates multi-turn dialogue sessions:
1. Loads scenario YAML → list of user turns + probes
2. Synthesises user speech via TTS API
3. Sends to adapter via process_turn() (session-aware)
4. Runs ASR on agent output
5. Evaluates context checks and probes
6. Returns SessionResult for metric computation

Usage:
    runner = MultiTurnRunner(adapter, config)
    result = runner.run_session(scenario)
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf

# Ensure s2s_benchmark root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.multiturn.scenario_builder import Scenario, UserTurn, ContextProbe
from datasets.multiturn.tts_synthesizer import TTSSynthesizer
from inference.adapters.base import BaseS2SAdapter


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn_index: int
    role: str                              # "user" or "agent"
    input_text: str
    input_audio_path: str
    output_audio_path: Optional[str] = None
    output_text: Optional[str] = None
    ttfb_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    duration_s: float = 0.0
    context_check_passed: Optional[bool] = None
    context_check_details: Optional[str] = None


@dataclass
class SessionResult:
    scenario_id: str
    model_name: str
    turns: List[TurnResult] = field(default_factory=list)
    total_duration_s: float = 0.0
    n_user_turns: int = 0
    n_agent_turns: int = 0
    probe_results: List[Dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Whisper ASR (cached singleton)
# ─────────────────────────────────────────────────────────────────────────────

_WHISPER_MODEL = None


def _transcribe(audio_path: str) -> Optional[str]:
    """Transcribe audio using Whisper-base (CPU)."""
    global _WHISPER_MODEL
    try:
        import whisper
        if _WHISPER_MODEL is None:
            print("  [multiturn] Loading Whisper-base for agent ASR …")
            _WHISPER_MODEL = whisper.load_model("base", device="cpu")
        result = _WHISPER_MODEL.transcribe(audio_path, language="en", fp16=False)
        return result.get("text", "").strip()
    except Exception as exc:
        print(f"  [multiturn] ASR error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Context check evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_context_check(
    agent_text: str,
    check_type: str,
    expected: List[str],
    prompt: str = "",
    conversation_history: Optional[List[Dict]] = None,
) -> tuple[bool, str]:
    """Evaluate a context check on agent response text.

    Keyword checks (response_contains_all/any) use keyword matching as a
    fast-pass. If keywords fail, falls back to Gemini LLM judge for
    semantic equivalence.
    """
    from metrics.multiturn.context_retention import evaluate_probe_llm

    text_lower = agent_text.lower()

    if check_type == "response_contains_all":
        expected_lower = [e.lower() for e in expected]
        missing = [e for e in expected_lower if e not in text_lower]
        if not missing:
            return True, "OK"
        # LLM judge fallback
        passed = evaluate_probe_llm(
            probe_text=prompt or f"Check that response contains: {expected}",
            model_response=agent_text,
            expected_contains=expected,
        )
        return passed, "LLM judge pass" if passed else f"Missing: {missing}"

    elif check_type == "response_contains_any":
        expected_lower = [e.lower() for e in expected]
        found = [e for e in expected_lower if e in text_lower]
        if found:
            return True, f"Found: {found}"
        # LLM judge fallback
        passed = evaluate_probe_llm(
            probe_text=prompt or f"Check that response mentions any of: {expected}",
            model_response=agent_text,
            expected_contains=expected,
        )
        return passed, "LLM judge pass" if passed else f"None of {expected}"

    elif check_type == "llm_judge":
        return _llm_judge_check(agent_text, prompt, conversation_history or [])

    return True, "No check"


def _llm_judge_check(
    agent_response: str,
    check_prompt: str,
    conversation_history: List[Dict],
) -> tuple[bool, str]:
    """Use LLM judge for context check: Gemini → Groq → OpenAI fallback."""
    system_prompt = 'Evaluate if a voice AI agent\'s response meets a criterion. Reply ONLY with JSON: {"passed": true/false, "reason": "..."}'
    history_str = "\n".join(
        f"{'User' if h.get('role') == 'user' else 'Agent'}: {h.get('text', '')}"
        for h in conversation_history
    )
    user_prompt = f"Conversation:\n{history_str}\n\nAgent response: {agent_response}\n\nCriterion: {check_prompt}"

    raw = None

    # 1. Gemini (primary, multi-key rotation)
    _gemini_keys = []
    primary = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if primary:
        _gemini_keys.append(primary)
    for k in ("GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"):
        v = os.environ.get(k)
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
                    resp = model.generate_content(user_prompt, request_options={"timeout": 30})
                    raw = resp.text
                    break
                except Exception:
                    continue
        except ImportError:
            pass

    # 2. Groq fallback
    if raw is None:
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    temperature=0.0, max_tokens=200, timeout=15,
                )
                raw = resp.choices[0].message.content
            except Exception:
                pass

    # 3. OpenAI fallback
    if raw is None:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key, timeout=30)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0, max_tokens=200,
                )
                raw = resp.choices[0].message.content
            except Exception as exc:
                print(f"  [multiturn] LLM judge check error: {exc}")

    if raw is None:
        return True, "LLM judge unavailable (all providers exhausted)"

    try:
        result = json.loads(raw)
        return result.get("passed", False), result.get("reason", "")
    except (json.JSONDecodeError, TypeError):
        lower = raw.lower()
        if "true" in lower or "passed" in lower or "yes" in lower:
            return True, raw
        return False, raw


# ─────────────────────────────────────────────────────────────────────────────
# Multi-turn runner
# ─────────────────────────────────────────────────────────────────────────────

class MultiTurnRunner:
    """Run multi-turn dialogue sessions against voice agents."""

    def __init__(self, adapter: BaseS2SAdapter, config: Dict):
        self.adapter = adapter
        self.config = config

        mt_cfg = config.get("multiturn", {})
        self.tts = TTSSynthesizer(
            provider=mt_cfg.get("tts_provider", "deepgram"),
            voice=mt_cfg.get("tts_voice", "aura-asteria-en"),
        )
        self.output_dir = Path(mt_cfg.get("output_dir", "results/multiturn"))

    def run_session(self, scenario: Scenario) -> SessionResult:
        """Run a full multi-turn session for one scenario."""
        result = SessionResult(
            scenario_id=scenario.scenario_id,
            model_name=self.adapter.model_name,
        )

        # Start session on adapter
        session_id = self.adapter.start_session(scenario.system_prompt)
        session_start = time.time()

        # Build probe lookup
        probes_by_turn = {
            p.inject_after_turn: p
            for p in scenario.context_probes
        }

        # Audio output directory for this session
        session_audio_dir = self.output_dir / "audio" / scenario.scenario_id / self.adapter.model_name
        session_audio_dir.mkdir(parents=True, exist_ok=True)

        for user_turn in scenario.user_turns:
            # 1. Synthesize user speech
            user_audio_path = self._synthesize_user_turn(
                text=user_turn.text,
                style=user_turn.speech_style,
                scenario_id=scenario.scenario_id,
                turn_index=user_turn.turn_index,
            )

            try:
                info = sf.info(str(user_audio_path))
                user_duration = info.duration
            except Exception:
                user_duration = 0.0

            result.turns.append(TurnResult(
                turn_index=user_turn.turn_index,
                role="user",
                input_text=user_turn.text,
                input_audio_path=str(user_audio_path),
                duration_s=user_duration,
            ))
            result.n_user_turns += 1

            # 2. Send to agent
            t_start = time.perf_counter()
            try:
                agent_response = self.adapter.process_turn(
                    audio_in_path=str(user_audio_path),
                    utterance_id=f"{scenario.scenario_id}_turn{user_turn.turn_index + 1:03d}",
                    output_dir=str(session_audio_dir),
                    session_id=session_id,
                )
            except Exception as exc:
                print(f"  [multiturn] Adapter error on turn {user_turn.turn_index}: {exc}")
                result.turns.append(TurnResult(
                    turn_index=user_turn.turn_index + 1,
                    role="agent",
                    input_text=user_turn.text,
                    input_audio_path=str(user_audio_path),
                ))
                result.n_agent_turns += 1
                continue

            e2e_ms = (time.perf_counter() - t_start) * 1000

            # 3. Transcribe agent response
            agent_text = agent_response.asr_transcript or ""
            if not agent_text and agent_response.audio_out_path and Path(agent_response.audio_out_path).exists():
                agent_text = _transcribe(agent_response.audio_out_path) or ""

            try:
                agent_duration = sf.info(agent_response.audio_out_path).duration if agent_response.audio_out_path and Path(agent_response.audio_out_path).exists() else 0.0
            except Exception:
                agent_duration = 0.0

            agent_turn = TurnResult(
                turn_index=user_turn.turn_index + 1,
                role="agent",
                input_text=user_turn.text,
                input_audio_path=str(user_audio_path),
                output_audio_path=agent_response.audio_out_path,
                output_text=agent_text,
                ttfb_ms=agent_response.ttfb_ms,
                e2e_latency_ms=e2e_ms,
                duration_s=agent_duration,
            )

            # 4. Context checks
            if user_turn.context_check is not None:
                cc = user_turn.context_check
                history = [
                    {"role": t.role, "text": t.output_text or t.input_text}
                    for t in result.turns
                ]
                passed, details = _evaluate_context_check(
                    agent_text=agent_text,
                    check_type=cc.check_type,
                    expected=cc.expected,
                    prompt=cc.prompt,
                    conversation_history=history,
                )
                agent_turn.context_check_passed = passed
                agent_turn.context_check_details = details

            result.turns.append(agent_turn)
            result.n_agent_turns += 1

            # 5. Context probes (injected after specific agent turns)
            agent_turn_index = user_turn.turn_index + 1
            if agent_turn_index in probes_by_turn:
                probe = probes_by_turn[agent_turn_index]
                probe_result = self._run_context_probe(
                    probe=probe,
                    session_id=session_id,
                    scenario_id=scenario.scenario_id,
                    session_audio_dir=session_audio_dir,
                )
                result.probe_results.append(probe_result)

        result.total_duration_s = time.time() - session_start

        # End session
        self.adapter.end_session(session_id)

        return result

    def _synthesize_user_turn(
        self,
        text: str,
        style: str,
        scenario_id: str,
        turn_index: int,
    ) -> Path:
        """Synthesize user speech via API TTS."""
        audio_dir = self.output_dir / "user_audio" / scenario_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        output_path = audio_dir / f"user_turn_{turn_index:03d}.wav"
        self.tts.synthesize(text=text, output_path=str(output_path), style=style)
        return output_path

    def _run_context_probe(
        self,
        probe: ContextProbe,
        session_id: str,
        scenario_id: str,
        session_audio_dir: Path,
    ) -> Dict:
        """Inject a context probe question and evaluate the response.

        Uses keyword matching as a fast-pass: if any expected keyword is found,
        the probe passes immediately. Otherwise, falls back to an LLM judge
        that can detect semantic equivalence (e.g. "sport utility vehicle"
        matching the expected keyword "SUV").
        """
        # Synthesize probe audio
        probe_audio_dir = self.output_dir / "user_audio" / f"{scenario_id}_probes"
        probe_audio_dir.mkdir(parents=True, exist_ok=True)
        probe_audio = probe_audio_dir / f"probe_{probe.inject_after_turn:03d}.wav"
        self.tts.synthesize(
            text=probe.probe_text,
            output_path=str(probe_audio),
            style="neutral",
        )

        # Send probe to agent
        t_start = time.perf_counter()
        try:
            response = self.adapter.process_turn(
                audio_in_path=str(probe_audio),
                utterance_id=f"{scenario_id}_probe{probe.inject_after_turn:03d}",
                output_dir=str(session_audio_dir),
                session_id=session_id,
            )
        except Exception as exc:
            return {
                "probe_text": probe.probe_text,
                "response_text": "",
                "expected": probe.expected_contains,
                "found": [],
                "passed": False,
                "eval_method": "error",
                "measures": probe.measures,
                "error": str(exc),
            }

        ttfb_ms = response.ttfb_ms

        # Transcribe response
        response_text = response.asr_transcript or ""
        if not response_text and response.audio_out_path and Path(response.audio_out_path).exists():
            response_text = _transcribe(response.audio_out_path) or ""

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
                "measures": probe.measures,
                "ttfb_ms": ttfb_ms,
            }

        # LLM judge fallback — keyword matching failed, check semantically
        from metrics.multiturn.context_retention import evaluate_probe_llm

        passed = evaluate_probe_llm(
            probe_text=probe.probe_text,
            model_response=response_text,
            expected_contains=probe.expected_contains,
        )

        return {
            "probe_text": probe.probe_text,
            "response_text": response_text,
            "expected": probe.expected_contains,
            "found": found,
            "passed": passed,
            "eval_method": "llm_judge",
            "measures": probe.measures,
            "ttfb_ms": ttfb_ms,
        }


def run_all_sessions(
    adapter: BaseS2SAdapter,
    scenarios: List[Scenario],
    config: Dict,
) -> List[SessionResult]:
    """Run all scenarios for a single model. Returns list of SessionResults."""
    runner = MultiTurnRunner(adapter, config)
    results = []
    for i, scenario in enumerate(scenarios):
        print(f"  [{i + 1}/{len(scenarios)}] {scenario.scenario_id} ({scenario.category})")
        try:
            result = runner.run_session(scenario)
            results.append(result)
            n_ok = sum(1 for t in result.turns if t.role == "agent" and t.output_audio_path)
            n_probes = len(result.probe_results)
            n_probe_pass = sum(1 for p in result.probe_results if p.get("passed"))
            print(f"    ✓ {result.n_agent_turns} agent turns, "
                  f"{n_probes} probes ({n_probe_pass} passed), "
                  f"{result.total_duration_s:.1f}s total")
        except Exception as exc:
            print(f"    ✗ Session failed: {exc}")
    return results
