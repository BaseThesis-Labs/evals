"""
Gemini 2.5 Flash Live Speech-to-Speech adapter.

Uses the Google Gemini Live API (bidirectional WebSocket) for end-to-end S2S.
The model receives audio and responds with audio — no explicit ASR/TTS step.

Model: gemini-2.5-flash-preview-native-audio-dialog
       gemini-2.0-flash-live  (older stable option)

Requirements:
    pip install google-genai>=0.7 soundfile numpy

Config keys:
    model         (str): Gemini model ID
    voice         (str): "Puck" | "Charon" | "Kore" | "Fenrir" | "Aoede" | "Orbit" | "Zephyr"
    system_prompt (str): optional system instruction
    timeout_s     (int): max seconds to wait for full response (default 30)
    language      (str): BCP-47 language code (default "en-US")
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from inference.adapters.base import BaseS2SAdapter, S2SResult

_DEFAULT_SYSTEM = (
    "You are a helpful, concise voice assistant. "
    "Respond naturally in 1-3 sentences as if speaking aloud."
)


def _load_audio_bytes(audio_path: str, target_sr: int = 16_000) -> tuple[bytes, int]:
    """Load and resample audio, return raw PCM16 bytes + actual sample rate."""
    import soundfile as sf  # type: ignore
    import librosa           # type: ignore

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return pcm16, target_sr


async def _run_gemini_live_session(
    api_key: str,
    model: str,
    voice: str,
    system_prompt: str,
    audio_bytes: bytes,
    input_sr: int,
    timeout_s: int,
    language: str,
) -> tuple[Optional[bytes], Optional[str], float]:
    """
    Run one turn via Gemini Live API.
    Returns (audio_pcm16_bytes, transcript, ttfb_s).
    """
    from google import genai          # type: ignore
    from google.genai import types    # type: ignore

    client = genai.Client(api_key=api_key)

    audio_chunks: list[bytes] = []
    transcript_parts: list[str] = []
    ttfb_s: float = 0.0
    t_send = time.perf_counter()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
        system_instruction=system_prompt,
    )

    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            # Send audio as inline blob
            await session.send(
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(data=audio_bytes, mime_type=f"audio/pcm;rate={input_sr}")
                    ]
                )
            )

            # Signal end of turn
            await session.send(input=types.LiveClientContent(turn_complete=True))

            # Collect response
            try:
                async with asyncio.timeout(timeout_s):
                    async for response in session.receive():
                        # Audio data
                        if response.data:
                            if not audio_chunks:
                                ttfb_s = time.perf_counter() - t_send
                            audio_chunks.append(response.data)

                        # Text transcript from output_audio_transcription
                        if response.server_content:
                            sc = response.server_content
                            if sc.output_transcription and sc.output_transcription.text:
                                transcript_parts.append(sc.output_transcription.text)
                            if sc.turn_complete:
                                break

            except asyncio.TimeoutError:
                pass

    except Exception as exc:
        raise RuntimeError(f"Gemini Live session error: {exc}") from exc

    audio_out = b"".join(audio_chunks) if audio_chunks else None
    transcript = "".join(transcript_parts).strip() or None
    return audio_out, transcript, ttfb_s


class GeminiLiveAdapter(BaseS2SAdapter):
    """Gemini 2.5 Flash Live Speech-to-Speech adapter."""

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        self.gemini_model = config.get(
            "model", "gemini-2.5-flash-preview-native-audio-dialog"
        )
        self.voice = config.get("voice", "Puck")
        self.system_prompt = config.get("system_prompt", _DEFAULT_SYSTEM)
        self.timeout_s = int(config.get("timeout_s", 30))
        self.language = config.get("language", "en-US")
        self._api_key = (
            os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        )
        # Gemini Live outputs PCM16 at 24kHz
        self.output_sr = 24_000

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
                error="GOOGLE_API_KEY / GEMINI_API_KEY not set",
            )

        try:
            # Gemini Live accepts PCM at 16kHz input
            audio_bytes, input_sr = _load_audio_bytes(audio_in_path, target_sr=16_000)

            audio_out_bytes, transcript, ttfb_s = asyncio.run(
                _run_gemini_live_session(
                    api_key=self._api_key,
                    model=self.gemini_model,
                    voice=self.voice,
                    system_prompt=self.system_prompt,
                    audio_bytes=audio_bytes,
                    input_sr=input_sr,
                    timeout_s=self.timeout_s,
                    language=self.language,
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

            # Convert PCM16 bytes → float32 array
            import struct
            n = len(audio_out_bytes) // 2
            samples = struct.unpack(f"<{n}h", audio_out_bytes)
            audio_np = np.array(samples, dtype=np.float32) / 32768.0

            out_path = self._make_output_path(output_dir, utterance_id)
            sf.write(out_path, audio_np, self.output_sr, subtype="PCM_16")

            duration_s = len(audio_np) / self.output_sr
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
        sid = super().start_session(system_prompt)
        self._sessions[sid]["system_prompt"] = system_prompt
        return sid

    def process_turn(
        self,
        audio_in_path: str,
        utterance_id: str,
        output_dir: str,
        session_id: str,
    ) -> S2SResult:
        session = self._sessions.get(session_id, {})
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
