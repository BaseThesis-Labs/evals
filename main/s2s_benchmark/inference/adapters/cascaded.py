"""
Cascaded S2S adapter: Whisper ASR → existing TTS client.

This adapter:
1. Transcribes the input audio using Whisper (on CPU).
2. Feeds the transcript into a tts_benchmark TTS client.
3. Returns an S2SResult with timing split into `asr_latency_ms` + `tts_latency_ms`.

Requirements:
  - tts_benchmark is in the same repo at ../../tts_benchmark/
  - Both directories are accessible via the SHARED venv.
"""
from __future__ import annotations

import os
import sys
import time
import random
import importlib
from typing import Optional

import numpy as np
import soundfile as sf

# ── Inject tts_benchmark onto sys.path so its clients are importable ─────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TTS_BENCH = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "tts_benchmark")
)
if _TTS_BENCH not in sys.path:
    sys.path.insert(0, _TTS_BENCH)

from inference.adapters.base import BaseS2SAdapter, S2SResult  # noqa: E402


# ── Whisper loader (shared singleton per process) ────────────────────────────
_WHISPER_MODELS: dict = {}


def _normalise_whisper_name(name: str) -> str:
    """Strip 'whisper-' prefix if present (config may use 'whisper-base' style)."""
    return name.removeprefix("whisper-") if name.startswith("whisper-") else name


def _load_whisper(model_name: str = "base", device: str = "cpu"):
    """Load (or return cached) Whisper model."""
    key = (model_name, device)
    if key not in _WHISPER_MODELS:
        import whisper  # type: ignore
        print(f"  [cascaded] Loading Whisper {model_name} on {device} …")
        _WHISPER_MODELS[key] = whisper.load_model(model_name, device=device)
    return _WHISPER_MODELS[key]


_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. "
    "Respond naturally in 1-3 sentences as if speaking aloud. "
    "Do not use markdown, lists, or special formatting."
)


class CascadedS2SAdapter(BaseS2SAdapter):
    """Cascaded Speech-to-Speech: Whisper ASR → LLM → TTS client.

    Config keys:
        tts_model     (str): tts_benchmark model name, e.g. "elevenlabs"
        voice         (str): voice/speaker identifier
        asr_model     (str): whisper model size, default "base"
        asr_device    (str): "cpu" or "cuda"
        llm_provider  (str): "anthropic" | "openai" | "none"  (default "none")
                             "none" = echo mode (no LLM — ASR transcript piped to TTS)
        llm_model     (str): model id, e.g. "claude-haiku-4-5-20251001"
        system_prompt (str): system prompt for the LLM (optional)
    """

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        self.tts_model_key = config.get("tts_model", "deepgram")
        self.voice = config.get("voice")
        self.asr_model_name = _normalise_whisper_name(config.get("asr_model", "base"))
        self.asr_device = config.get("asr_device", "cpu")

        # LLM response generation
        self.llm_provider = config.get("llm_provider", "none").lower()
        self.llm_model = config.get("llm_model", "claude-haiku-4-5-20251001")
        self.system_prompt = config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        # Custom base URL for OpenAI-compatible APIs (Ollama, Groq, Together, LM Studio, vLLM)
        self.llm_base_url = config.get("llm_base_url")     # None = use provider default

        # Build TTS client config (strip adapter-specific keys)
        _adapter_keys = {"tts_model", "asr_model", "asr_device", "llm_provider", "llm_model", "system_prompt"}
        self._tts_config = {k: v for k, v in config.items() if k not in _adapter_keys}

        self._tts_client = None  # lazy init

    # ── Lazy TTS client initialisation ───────────────────────────────────────

    def _get_tts_client(self):
        if self._tts_client is not None:
            return self._tts_client

        # Discover client module inside tts_benchmark/models/
        module_name = f"models.{self.tts_model_key}_client"
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Cannot import TTS client '{module_name}'. "
                f"Make sure {_TTS_BENCH} is on sys.path and the client exists."
            ) from exc

        # Guess class name: deepgram → DeepgramClient
        class_name = "".join(p.capitalize() for p in self.tts_model_key.split("_")) + "Client"
        cls = getattr(mod, class_name, None)
        if cls is None:
            raise AttributeError(
                f"Module '{module_name}' has no class '{class_name}'. "
                "Check models.yaml for the correct class_path."
            )

        self._tts_client = cls(**self._tts_config)
        return self._tts_client

    # ── Core process method ───────────────────────────────────────────────────

    def process(
        self,
        audio_in_path: str,
        reference_text: str,
        utterance_id: str,
        output_dir: str,
    ) -> S2SResult:
        t_start = time.perf_counter()

        # ── Step 1: ASR ───────────────────────────────────────────────────────
        asr_transcript, asr_latency_ms = self._run_asr(audio_in_path)

        # ── Step 2: LLM response (optional) ──────────────────────────────────
        if self.llm_provider != "none" and asr_transcript:
            llm_response, llm_latency_ms = self._run_llm(asr_transcript)
        else:
            llm_response = asr_transcript   # echo mode: repeat transcript
            llm_latency_ms = 0.0

        # ── Step 3: TTS ───────────────────────────────────────────────────────
        tts_result, tts_latency_ms, error = self._run_tts(
            text=llm_response or asr_transcript,
            utterance_id=utterance_id,
        )

        e2e_latency_ms = (time.perf_counter() - t_start) * 1_000

        if error is not None:
            return S2SResult(
                audio_out_path="",
                model_name=self.model_name,
                utterance_id=utterance_id,
                asr_transcript=asr_transcript,
                asr_latency_ms=asr_latency_ms,
                e2e_latency_ms=e2e_latency_ms,
                error=error,
            )

        # ── Save output audio ─────────────────────────────────────────────────
        out_path = self._make_output_path(output_dir, utterance_id)
        audio_np, sr = _to_numpy(tts_result)
        sf.write(out_path, audio_np, sr, subtype="PCM_16")

        # ── RTF (based on TTS stage only) ─────────────────────────────────────
        duration_s = len(audio_np) / sr if sr > 0 else 1.0
        rtf = (tts_latency_ms / 1_000) / max(duration_s, 1e-6)

        return S2SResult(
            audio_out_path=out_path,
            sample_rate=sr,
            asr_transcript=asr_transcript,
            ttfb_ms=getattr(tts_result, "ttfa_ms", 0.0),
            e2e_latency_ms=e2e_latency_ms,
            asr_latency_ms=asr_latency_ms,
            tts_latency_ms=tts_latency_ms,
            rtf=rtf,
            model_name=self.model_name,
            utterance_id=utterance_id,
            extra={
                "tts_inference_time_ms": getattr(tts_result, "inference_time_ms", 0.0),
                "llm_response": llm_response,
                "llm_latency_ms": llm_latency_ms,
            },
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_asr(self, audio_path: str) -> tuple[str, float]:
        """Run Whisper and return (transcript, latency_ms)."""
        whisper_model = _load_whisper(self.asr_model_name, self.asr_device)
        t0 = time.perf_counter()
        result = whisper_model.transcribe(audio_path, language="en", fp16=False)
        latency_ms = (time.perf_counter() - t0) * 1_000
        transcript = result.get("text", "").strip()
        return transcript, latency_ms

    def _run_llm(self, user_text: str) -> tuple[str, float]:
        """Generate LLM response from ASR transcript. Returns (response_text, latency_ms)."""
        t0 = time.perf_counter()
        try:
            if self.llm_provider == "anthropic":
                import anthropic  # type: ignore
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                resp = client.messages.create(
                    model=self.llm_model,
                    max_tokens=512,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": user_text}],
                )
                text = resp.content[0].text
            elif self.llm_provider in ("openai", "groq", "together", "ollama", "lmstudio", "vllm"):
                from openai import OpenAI  # type: ignore
                # Resolve API key and base URL per provider
                _key_map = {
                    "openai":   ("OPENAI_API_KEY",   None),
                    "groq":     ("GROQ_API_KEY",     "https://api.groq.com/openai/v1"),
                    "together": ("TOGETHER_API_KEY", "https://api.together.xyz/v1"),
                    "ollama":   ("OLLAMA_API_KEY",   "http://localhost:11434/v1"),
                    "lmstudio": ("LMSTUDIO_API_KEY", "http://localhost:1234/v1"),
                    "vllm":     ("VLLM_API_KEY",     "http://localhost:8000/v1"),
                }
                env_key, default_url = _key_map.get(self.llm_provider, ("OPENAI_API_KEY", None))
                base_url = self.llm_base_url or default_url

                # Collect all keys for this provider (support multi-key rotation)
                _all_keys = []
                primary = os.getenv(env_key)
                if primary:
                    _all_keys.append(primary)
                for suffix in ("_2", "_3"):
                    v = os.getenv(env_key + suffix)
                    if v and v not in _all_keys:
                        _all_keys.append(v)
                if not _all_keys:
                    _all_keys.append("ollama")  # ollama ignores key

                text = None
                for _api_key in _all_keys:
                    try:
                        client_kwargs = {"api_key": _api_key}
                        if base_url:
                            client_kwargs["base_url"] = base_url
                        client = OpenAI(**client_kwargs)
                        resp = client.chat.completions.create(
                            model=self.llm_model,
                            messages=[
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": user_text},
                            ],
                            max_tokens=512,
                        )
                        text = resp.choices[0].message.content
                        break
                    except Exception as key_exc:
                        if "429" in str(key_exc) or "rate" in str(key_exc).lower():
                            continue  # try next key
                        raise  # re-raise non-rate-limit errors
                if text is None:
                    raise RuntimeError("All API keys exhausted (rate limited)")
            else:
                text = user_text  # echo fallback
        except Exception as exc:
            print(f"  [cascaded] LLM error ({self.llm_provider}): {exc}")
            text = user_text  # fallback to echo on error
        latency_ms = (time.perf_counter() - t0) * 1_000
        return text.strip(), latency_ms

    def _run_tts(
        self,
        text: str,
        utterance_id: str,
        max_retries: int = 4,
    ):
        """Call TTS client with exponential backoff. Returns (result, latency_ms, error)."""
        client = self._get_tts_client()
        delay = 2.0

        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                result = client.generate(
                    text=text,
                    reference_audio_path=None,
                    speaker_id=self.voice,
                    utterance_id=utterance_id,
                )
                latency_ms = (time.perf_counter() - t0) * 1_000
                return result, latency_ms, None

            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1_000
                err_str = str(exc).lower()
                is_rate_limit = (
                    "429" in str(exc)
                    or "too many requests" in err_str
                    or "rate limit" in err_str
                    or "ratelimit" in err_str
                )
                if is_rate_limit and attempt < max_retries:
                    jitter = random.uniform(0, delay * 0.25)
                    wait = delay + jitter
                    print(
                        f"  [cascaded] Rate-limited on {utterance_id} "
                        f"(attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait:.1f}s…"
                    )
                    time.sleep(wait)
                    delay *= 2
                else:
                    return None, latency_ms, str(exc)

        return None, 0.0, "Max retries exceeded"

    # ── Multi-turn session support ────────────────────────────────────────

    def start_session(self, system_prompt: str) -> str:
        sid = super().start_session(system_prompt)
        # Override system prompt for this session
        self._sessions[sid]["system_prompt"] = system_prompt
        return sid

    def process_turn(
        self,
        audio_in_path: str,
        utterance_id: str,
        output_dir: str,
        session_id: str,
    ) -> S2SResult:
        """Process one turn with conversation history."""
        session = self._sessions.get(session_id, {})
        t_start = time.perf_counter()

        # ASR
        asr_transcript, asr_latency_ms = self._run_asr(audio_in_path)

        # LLM with history
        if self.llm_provider != "none" and asr_transcript:
            session.setdefault("history", []).append(
                {"role": "user", "content": asr_transcript}
            )
            # Build messages with full history
            n_turns = len(session.get("history", []))
            if n_turns >= 3:
                print(f"  [cascaded] Turn {n_turns}: history has {n_turns} messages")
            llm_response, llm_latency_ms = self._run_llm_with_history(session)
            session["history"].append(
                {"role": "assistant", "content": llm_response}
            )
        else:
            llm_response = asr_transcript
            llm_latency_ms = 0.0

        # TTS
        tts_result, tts_latency_ms, error = self._run_tts(
            text=llm_response or asr_transcript,
            utterance_id=utterance_id,
        )

        e2e_latency_ms = (time.perf_counter() - t_start) * 1_000

        if error is not None:
            return S2SResult(
                audio_out_path="",
                model_name=self.model_name,
                utterance_id=utterance_id,
                asr_transcript=asr_transcript,
                asr_latency_ms=asr_latency_ms,
                e2e_latency_ms=e2e_latency_ms,
                error=error,
            )

        out_path = self._make_output_path(output_dir, utterance_id)
        audio_np, sr = _to_numpy(tts_result)
        sf.write(out_path, audio_np, sr, subtype="PCM_16")

        duration_s = len(audio_np) / sr if sr > 0 else 1.0
        rtf = (tts_latency_ms / 1_000) / max(duration_s, 1e-6)

        session["turn_count"] = session.get("turn_count", 0) + 1

        return S2SResult(
            audio_out_path=out_path,
            sample_rate=sr,
            asr_transcript=asr_transcript,
            ttfb_ms=getattr(tts_result, "ttfa_ms", 0.0),
            e2e_latency_ms=e2e_latency_ms,
            asr_latency_ms=asr_latency_ms,
            tts_latency_ms=tts_latency_ms,
            rtf=rtf,
            model_name=self.model_name,
            utterance_id=utterance_id,
            extra={
                "llm_response": llm_response,
                "llm_latency_ms": llm_latency_ms,
                "turn_count": session.get("turn_count", 0),
            },
        )

    def _run_llm_with_history(self, session: dict) -> tuple[str, float]:
        """Run LLM with full conversation history."""
        t0 = time.perf_counter()
        system_prompt = session.get("system_prompt", self.system_prompt)
        history = session.get("history", [])

        try:
            if self.llm_provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                resp = client.messages.create(
                    model=self.llm_model,
                    max_tokens=512,
                    system=system_prompt,
                    messages=history,
                )
                text = resp.content[0].text
            elif self.llm_provider in ("openai", "groq", "together", "ollama", "lmstudio", "vllm"):
                from openai import OpenAI
                _key_map = {
                    "openai":   ("OPENAI_API_KEY",   None),
                    "groq":     ("GROQ_API_KEY",     "https://api.groq.com/openai/v1"),
                    "together": ("TOGETHER_API_KEY", "https://api.together.xyz/v1"),
                    "ollama":   ("OLLAMA_API_KEY",   "http://localhost:11434/v1"),
                    "lmstudio": ("LMSTUDIO_API_KEY", "http://localhost:1234/v1"),
                    "vllm":     ("VLLM_API_KEY",     "http://localhost:8000/v1"),
                }
                env_key, default_url = _key_map.get(self.llm_provider, ("OPENAI_API_KEY", None))
                api_key = os.getenv(env_key) or "ollama"
                base_url = self.llm_base_url or default_url
                client_kwargs = {"api_key": api_key}
                if base_url:
                    client_kwargs["base_url"] = base_url
                client = OpenAI(**client_kwargs)
                messages = [{"role": "system", "content": system_prompt}] + history
                resp = client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    max_tokens=512,
                )
                text = resp.choices[0].message.content
            else:
                text = history[-1]["content"] if history else ""
        except Exception as exc:
            print(f"  [cascaded] LLM history error ({self.llm_provider}): {exc}")
            text = history[-1]["content"] if history else ""

        latency_ms = (time.perf_counter() - t0) * 1_000
        return text.strip(), latency_ms

    def cleanup(self) -> None:
        super().cleanup()
        if self._tts_client is not None:
            try:
                self._tts_client.cleanup()
            except Exception:
                pass
            self._tts_client = None


# ── Utility ───────────────────────────────────────────────────────────────────

def _to_numpy(tts_result) -> tuple[np.ndarray, int]:
    """Extract (audio_array, sample_rate) from a TTSResult-like object."""
    audio = tts_result.audio
    sr = tts_result.sample_rate
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    audio = audio.astype(np.float32)
    return audio, sr
