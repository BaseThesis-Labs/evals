"""
Ultravox Speech-to-Speech adapter — hosted API version.

Uses the Ultravox cloud API (api.ultravox.ai) — no GPU required.
Flow: POST /api/calls → WebSocket joinUrl → send audio → receive audio.

Requirements:
    pip install websockets requests soundfile numpy
    ULTRAVOX_API_KEY env var  (get from app.ultravox.ai)

Config keys:
    model         (str): e.g. "fixie-ai/ultravox-v0_5-llama-3_1-8b"
    voice         (str): e.g. "Mark" | "Tanya" | "Spencer" (default "Mark")
    system_prompt (str): system instruction for the model
    timeout_s     (int): max seconds to wait for full response (default 30)
    input_sr      (int): input sample rate sent to API (default 16000)
"""
from __future__ import annotations

import asyncio
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
    "Respond naturally in 1-3 sentences as if speaking aloud. "
    "Do not use markdown or special formatting."
)

_API_BASE = "https://api.ultravox.ai/api"


def _to_pcm16_bytes(audio_path: str, target_sr: int = 16_000) -> tuple[bytes, int]:
    """Load audio file and convert to PCM16 bytes at target_sr."""
    import soundfile as sf  # type: ignore

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        try:
            import librosa  # type: ignore
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            pass  # use as-is if librosa not installed
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return pcm16, target_sr


async def _run_ultravox_session(
    api_key: str,
    model: str,
    voice: str,
    system_prompt: str,
    audio_bytes: bytes,
    input_sr: int,
    timeout_s: int,
) -> tuple[Optional[bytes], Optional[str], float]:
    """
    One-turn Ultravox session via hosted API.
    Returns (audio_pcm16_bytes_at_8kHz, transcript, ttfb_s).
    """
    import requests    # type: ignore
    import websockets  # type: ignore

    # ── Step 1: Create call ───────────────────────────────────────────────
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    payload = {
        "systemPrompt": system_prompt,
        "model": model,
        "voice": voice,
        "firstSpeaker": "FIRST_SPEAKER_USER",
        "medium": {"serverWebSocket": {
            "inputSampleRate": input_sr,
            "outputSampleRate": 8000,
        }},
    }

    resp = requests.post(f"{_API_BASE}/calls", headers=headers, json=payload, timeout=15)
    if not resp.ok:
        raise RuntimeError(f"Ultravox /calls failed {resp.status_code}: {resp.text}")
    join_url = resp.json()["joinUrl"]

    # ── Step 2: WebSocket session ─────────────────────────────────────────
    audio_chunks: list[bytes] = []
    transcript_parts: list[str] = []
    ttfb_s: float = 0.0
    t_send = time.perf_counter()

    ws_headers = {"X-API-Key": api_key}
    frame_size = (input_sr * 30) // 1000 * 2   # 30ms PCM16 chunk

    async with websockets.connect(join_url, additional_headers=ws_headers) as ws:

        async def _send_audio():
            """Send user audio then silence to trigger VAD."""
            # Wait briefly for session init
            await asyncio.sleep(0.3)
            offset = 0
            while offset < len(audio_bytes):
                await ws.send(audio_bytes[offset: offset + frame_size])
                offset += frame_size
                await asyncio.sleep(0.01)
            # 1.5s silence → triggers Ultravox VAD end-of-speech
            silence = b"\x00" * frame_size
            for _ in range(int(input_sr * 1.5) * 2 // frame_size):
                await ws.send(silence)
                await asyncio.sleep(0.01)

        # Start sending audio concurrently while we listen
        send_task = asyncio.create_task(_send_audio())

        try:
            async with asyncio.timeout(timeout_s):
                async for raw in ws:
                    if isinstance(raw, bytes) and raw:
                        # Binary PCM16 audio from the agent
                        if not audio_chunks:
                            ttfb_s = time.perf_counter() - t_send
                        audio_chunks.append(raw)
                    elif isinstance(raw, str):
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        msg_type = msg.get("type", "")

                        if msg_type == "transcript":
                            role = msg.get("role", "")
                            final = msg.get("final", False)
                            text = msg.get("text", "")
                            if role == "agent" and final and text:
                                transcript_parts.append(text)

                        elif msg_type in ("call_ended", "hang", "disconnect"):
                            break
        except asyncio.TimeoutError:
            pass
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass

    audio_out = b"".join(audio_chunks) if audio_chunks else None
    transcript = " ".join(transcript_parts).strip() or None
    return audio_out, transcript, ttfb_s


class UltravoxAdapter(BaseS2SAdapter):
    """Ultravox hosted API Speech-to-Speech adapter."""

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        self.model = config.get("model", "fixie-ai/ultravox-v0_5-llama-3_1-8b")
        self.voice = config.get("voice", "Mark")
        self.system_prompt = config.get("system_prompt", _DEFAULT_SYSTEM)
        self.timeout_s = int(config.get("timeout_s", 30))
        self.input_sr = int(config.get("input_sr", 16_000))
        self.output_sr = 8_000   # Ultravox API returns 8kHz PCM16
        self._api_key = os.getenv("ULTRAVOX_API_KEY", "")

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
                error="ULTRAVOX_API_KEY not set",
            )

        try:
            audio_bytes, _ = _to_pcm16_bytes(audio_in_path, self.input_sr)

            audio_out_bytes, transcript, ttfb_s = asyncio.run(
                _run_ultravox_session(
                    api_key=self._api_key,
                    model=self.model,
                    voice=self.voice,
                    system_prompt=self.system_prompt,
                    audio_bytes=audio_bytes,
                    input_sr=self.input_sr,
                    timeout_s=self.timeout_s,
                )
            )

            e2e_latency_ms = (time.perf_counter() - t_start) * 1_000
            ttfb_ms = ttfb_s * 1_000

            if not audio_out_bytes:
                return S2SResult(
                    audio_out_path="",
                    model_name=self.model_name,
                    utterance_id=utterance_id,
                    asr_transcript=transcript,
                    ttfb_ms=ttfb_ms,
                    e2e_latency_ms=e2e_latency_ms,
                    error="no_audio_in_response",
                )

            # PCM16 bytes → float32
            n = len(audio_out_bytes) // 2
            samples = struct.unpack(f"<{n}h", audio_out_bytes)
            audio_np = np.array(samples, dtype=np.float32) / 32768.0

            out_path = self._make_output_path(output_dir, utterance_id)
            sf.write(out_path, audio_np, self.output_sr, subtype="PCM_16")

            duration_s = len(audio_np) / max(self.output_sr, 1)
            rtf = (e2e_latency_ms / 1_000) / max(duration_s, 1e-6)

            return S2SResult(
                audio_out_path=out_path,
                sample_rate=self.output_sr,
                asr_transcript=transcript,
                ttfb_ms=ttfb_ms,
                e2e_latency_ms=e2e_latency_ms,
                rtf=rtf,
                model_name=self.model_name,
                utterance_id=utterance_id,
                extra={"model": self.model, "voice": self.voice},
            )

        except Exception as exc:
            return S2SResult(
                audio_out_path="",
                model_name=self.model_name,
                utterance_id=utterance_id,
                e2e_latency_ms=(time.perf_counter() - t_start) * 1_000,
                error=str(exc),
            )

    def cleanup(self) -> None:
        pass
