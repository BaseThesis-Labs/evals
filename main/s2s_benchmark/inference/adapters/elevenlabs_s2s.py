"""
ElevenLabs Speech-to-Speech adapter.

Uses the ElevenLabs /v1/speech-to-speech API (voice conversion).
The model receives audio and converts it to the target voice while
preserving the prosody, emotion, and speaking style of the input.

⚠️  This is a VOICE CONVERSION model — it does NOT have an LLM and
    does NOT generate a new response. The content of the speech is
    preserved; only the voice identity is changed.

    Use-cases in this benchmark:
      - Voice cloning / speaker similarity evaluation
      - Prosody/emotion preservation evaluation
      - Quality comparison vs cascaded TTS

API endpoint: POST /v1/speech-to-speech/{voice_id}
SDK:          elevenlabs>=1.0  (elevenlabs.client.ElevenLabs)

Models available:
    eleven_english_sts_v2        — English, highest quality
    eleven_multilingual_sts_v2   — Multilingual

Requirements:
    pip install elevenlabs>=1.0
    ELEVENLABS_API_KEY env var

Config keys:
    voice_id        (str): ElevenLabs voice ID (default "21m00Tcm4TlvDq8ikWAM" = Rachel)
    model_id        (str): STS model (default "eleven_english_sts_v2")
    stability       (float): voice stability 0-1 (default 0.5)
    similarity_boost(float): voice similarity 0-1 (default 0.75)
    style           (float): style exaggeration 0-1 (default 0.0)
    output_format   (str): "pcm_24000" | "mp3_44100_192" (default "pcm_24000")
"""
from __future__ import annotations

import io
import os
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

from inference.adapters.base import BaseS2SAdapter, S2SResult

# Rachel voice (free tier default)
_DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
_DEFAULT_MODEL = "eleven_english_sts_v2"


class ElevenLabsS2SAdapter(BaseS2SAdapter):
    """ElevenLabs Speech-to-Speech (voice conversion) adapter.

    NOTE: Because this is voice conversion (no LLM), the ASR transcript
    of the output will be nearly identical to the input. Content metrics
    (WER, BERT score) are therefore not meaningful. Focus evaluation on:
      - Speaker similarity (SECS, sim_wavlm)
      - Audio quality (UTMOS, DNSMOS)
      - Prosody/emotion preservation (E-SIM, emotion_sim)
    """

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        self.voice_id = config.get("voice_id", _DEFAULT_VOICE_ID)
        self.sts_model = config.get("model_id", _DEFAULT_MODEL)
        self.stability = float(config.get("stability", 0.5))
        self.similarity_boost = float(config.get("similarity_boost", 0.75))
        self.style = float(config.get("style", 0.0))
        self.output_format = config.get("output_format", "pcm_24000")
        self._api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self._client = None   # lazy init

    def _get_client(self):
        if self._client is not None:
            return self._client
        from elevenlabs.client import ElevenLabs  # type: ignore
        self._client = ElevenLabs(api_key=self._api_key)
        return self._client

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
                error="ELEVENLABS_API_KEY not set",
            )

        try:
            client = self._get_client()

            # Load input audio as bytes (API accepts wav/mp3/ogg/flac/m4a)
            with open(audio_in_path, "rb") as f:
                audio_bytes = f.read()

            t_api = time.perf_counter()

            # Call Speech-to-Speech API
            audio_gen = client.speech_to_speech.convert(
                voice_id=self.voice_id,
                audio=audio_bytes,
                model_id=self.sts_model,
                output_format=self.output_format,
            )

            # Collect streamed response
            raw_chunks: list[bytes] = []
            ttfb_ms: Optional[float] = None

            for chunk in audio_gen:
                if isinstance(chunk, bytes) and chunk:
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - t_api) * 1_000
                    raw_chunks.append(chunk)

            if not raw_chunks:
                return S2SResult(
                    audio_out_path="",
                    model_name=self.model_name,
                    utterance_id=utterance_id,
                    e2e_latency_ms=(time.perf_counter() - t_start) * 1_000,
                    error="empty_response",
                )

            raw = b"".join(raw_chunks)
            e2e_latency_ms = (time.perf_counter() - t_start) * 1_000

            # Decode audio
            out_path = self._make_output_path(output_dir, utterance_id)

            if "pcm" in self.output_format:
                # PCM16 raw bytes — determine sample rate from format name
                sr = int(self.output_format.split("_")[1])   # "pcm_24000" → 24000
                n = len(raw) // 2
                samples = struct.unpack(f"<{n}h", raw)
                audio_np = np.array(samples, dtype=np.float32) / 32768.0
                sf.write(out_path, audio_np, sr, subtype="PCM_16")
            else:
                # MP3 — decode via soundfile / pydub
                try:
                    audio_np, sr = sf.read(io.BytesIO(raw), dtype="float32")
                    if audio_np.ndim > 1:
                        audio_np = audio_np.mean(axis=-1)
                    sf.write(out_path, audio_np, sr, subtype="PCM_16")
                except Exception:
                    # Fallback: write raw MP3 and let pipeline skip wav metrics
                    mp3_path = out_path.replace(".wav", ".mp3")
                    with open(mp3_path, "wb") as f:
                        f.write(raw)
                    return S2SResult(
                        audio_out_path=mp3_path,
                        model_name=self.model_name,
                        utterance_id=utterance_id,
                        ttfb_ms=ttfb_ms,
                        e2e_latency_ms=e2e_latency_ms,
                        error="mp3_decode_failed_use_pcm_format",
                    )

            duration_s = len(audio_np) / max(sr, 1)
            rtf = (e2e_latency_ms / 1_000) / max(duration_s, 1e-6)

            # The output text = input text (voice conversion preserves content)
            return S2SResult(
                audio_out_path=out_path,
                sample_rate=sr,
                asr_transcript=reference_text,   # content unchanged by design
                ttfb_ms=ttfb_ms,
                e2e_latency_ms=e2e_latency_ms,
                rtf=rtf,
                model_name=self.model_name,
                utterance_id=utterance_id,
                extra={"voice_id": self.voice_id, "sts_model": self.sts_model},
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
        self._client = None
