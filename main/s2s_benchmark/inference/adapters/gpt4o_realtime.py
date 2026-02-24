"""
GPT-4o Realtime Speech-to-Speech adapter.

Uses the OpenAI Realtime API (WebSocket) to process audio end-to-end.
No intermediate ASR/TTS — the model hears and speaks natively.

Model options:
    gpt-4o-realtime-preview-2024-10-01
    gpt-4o-mini-realtime-preview-2024-12-17

Requirements:
    pip install openai>=1.37 websockets numpy soundfile

Config keys:
    model         (str): OpenAI realtime model ID
    voice         (str): "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer"
    system_prompt (str): optional system instructions
    timeout_s     (int): max seconds to wait for response (default 30)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

from inference.adapters.base import BaseS2SAdapter, S2SResult

_DEFAULT_SYSTEM = (
    "You are a helpful, concise voice assistant. "
    "Respond naturally in 1-3 sentences as if speaking aloud."
)


def _wav_to_pcm16_b64(audio_path: str, target_sr: int = 24_000) -> str:
    """Load a WAV file, resample to target_sr, return base64-encoded PCM16."""
    import soundfile as sf   # type: ignore
    import librosa           # type: ignore

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Float32 → Int16 PCM
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    raw_bytes = audio_int16.tobytes()
    return base64.b64encode(raw_bytes).decode()


def _pcm16_bytes_to_float32(raw: bytes, sr: int = 24_000) -> np.ndarray:
    """PCM16 raw bytes → float32 numpy array."""
    n_samples = len(raw) // 2
    samples = struct.unpack(f"<{n_samples}h", raw)
    return np.array(samples, dtype=np.float32) / 32768.0


async def _run_realtime_session(
    api_key: str,
    model: str,
    voice: str,
    system_prompt: str,
    audio_b64: str,
    timeout_s: int,
) -> tuple[Optional[bytes], Optional[str], float]:
    """
    Run one turn through the GPT-4o Realtime WebSocket API.

    Returns (audio_pcm16_bytes, transcript_text, ttfb_s).
    """
    import websockets  # type: ignore

    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    audio_chunks: list[bytes] = []
    transcript_parts: list[str] = []
    ttfb_s: float = 0.0
    t_send = time.perf_counter()

    async with websockets.connect(url, additional_headers=headers) as ws:
        # Configure session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "voice": voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": None,  # manual turn management for eval
                "instructions": system_prompt,
            },
        }))

        # Send input audio
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))

        # Collect response
        try:
            async with asyncio.timeout(timeout_s):
                async for raw_msg in ws:
                    event = json.loads(raw_msg)
                    etype = event.get("type", "")

                    if etype == "response.audio.delta":
                        chunk = base64.b64decode(event.get("delta", ""))
                        if chunk:
                            if not audio_chunks:
                                ttfb_s = time.perf_counter() - t_send
                            audio_chunks.append(chunk)

                    elif etype == "response.audio_transcript.delta":
                        transcript_parts.append(event.get("delta", ""))

                    elif etype == "response.done":
                        break

                    elif etype == "error":
                        raise RuntimeError(f"API error: {event.get('error', {})}")

        except asyncio.TimeoutError:
            pass  # return whatever was collected

    audio_bytes = b"".join(audio_chunks) if audio_chunks else None
    transcript = "".join(transcript_parts).strip() or None
    return audio_bytes, transcript, ttfb_s


class GPT4ORealtimeAdapter(BaseS2SAdapter):
    """GPT-4o Realtime Speech-to-Speech adapter."""

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        self.oai_model = config.get("model", "gpt-4o-realtime-preview-2024-10-01")
        self.voice = config.get("voice", "alloy")
        self.system_prompt = config.get("system_prompt", _DEFAULT_SYSTEM)
        self.timeout_s = int(config.get("timeout_s", 30))
        self._api_key = os.getenv("OPENAI_API_KEY", "")

    def process(
        self,
        audio_in_path: str,
        reference_text: str,
        utterance_id: str,
        output_dir: str,
    ) -> S2SResult:
        import soundfile as sf  # type: ignore

        t_start = time.perf_counter()

        if not self._api_key:
            return S2SResult(
                audio_out_path="",
                model_name=self.model_name,
                utterance_id=utterance_id,
                e2e_latency_ms=0.0,
                error="OPENAI_API_KEY not set",
            )

        try:
            # Encode input to PCM16 base64 (24kHz, as required by realtime API)
            audio_b64 = _wav_to_pcm16_b64(audio_in_path, target_sr=24_000)

            # Run async session
            audio_bytes, transcript, ttfb_s = asyncio.run(
                _run_realtime_session(
                    api_key=self._api_key,
                    model=self.oai_model,
                    voice=self.voice,
                    system_prompt=self.system_prompt,
                    audio_b64=audio_b64,
                    timeout_s=self.timeout_s,
                )
            )

            e2e_latency_ms = (time.perf_counter() - t_start) * 1_000
            ttfb_ms = ttfb_s * 1_000

            if not audio_bytes:
                return S2SResult(
                    audio_out_path="",
                    model_name=self.model_name,
                    utterance_id=utterance_id,
                    asr_transcript=transcript,
                    ttfb_ms=ttfb_ms,
                    e2e_latency_ms=e2e_latency_ms,
                    error="no_audio_in_response",
                )

            # Decode PCM16 → float32
            audio_np = _pcm16_bytes_to_float32(audio_bytes, sr=24_000)
            out_path = self._make_output_path(output_dir, utterance_id)
            sf.write(out_path, audio_np, 24_000, subtype="PCM_16")

            duration_s = len(audio_np) / 24_000
            rtf = (e2e_latency_ms / 1_000) / max(duration_s, 1e-6)

            return S2SResult(
                audio_out_path=out_path,
                sample_rate=24_000,
                asr_transcript=transcript,
                ttfb_ms=ttfb_ms,
                e2e_latency_ms=e2e_latency_ms,
                rtf=rtf,
                model_name=self.model_name,
                utterance_id=utterance_id,
            )

        except Exception as exc:
            return S2SResult(
                audio_out_path="",
                model_name=self.model_name,
                utterance_id=utterance_id,
                e2e_latency_ms=(time.perf_counter() - t_start) * 1_000,
                error=str(exc),
            )

    # ── Multi-turn session support ────────────────────────────────────────

    def start_session(self, system_prompt: str) -> str:
        """Start a persistent WebSocket session for multi-turn dialogue."""
        sid = super().start_session(system_prompt)
        # Override system prompt
        self._sessions[sid]["system_prompt"] = system_prompt
        return sid

    def process_turn(
        self,
        audio_in_path: str,
        utterance_id: str,
        output_dir: str,
        session_id: str,
    ) -> S2SResult:
        """Process a single turn — uses the session's system prompt."""
        session = self._sessions.get(session_id, {})
        # GPT-4o Realtime creates a new WS per call (no persistent WS state
        # in the eval harness), but we use the session's system prompt.
        old_prompt = self.system_prompt
        self.system_prompt = session.get("system_prompt", old_prompt)
        try:
            result = self.process(
                audio_in_path=audio_in_path,
                reference_text="",
                utterance_id=utterance_id,
                output_dir=output_dir,
            )
        finally:
            self.system_prompt = old_prompt
        session["turn_count"] = session.get("turn_count", 0) + 1
        return result

    def cleanup(self) -> None:
        super().cleanup()
