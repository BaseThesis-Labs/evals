"""
TTS synthesizer for multi-turn user turns.
Uses Deepgram Aura by default (fast, cheap, CPU-friendly API call).
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional


class TTSSynthesizer:
    """Synthesize user speech for multi-turn scenarios via API TTS."""

    def __init__(self, provider: str = "deepgram", voice: str = "aura-asteria-en"):
        self.provider = provider
        self.voice = voice

    def synthesize(self, text: str, output_path: str, style: str = "neutral") -> str:
        """Synthesize text to WAV file. Returns output_path."""
        if Path(output_path).exists():
            return output_path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.provider == "deepgram":
            self._synthesize_deepgram(text, output_path)
        elif self.provider == "cartesia":
            self._synthesize_cartesia(text, output_path)
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")

        return output_path

    def _synthesize_deepgram(self, text: str, output_path: str) -> None:
        import httpx
        api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        resp = httpx.post(
            "https://api.deepgram.com/v1/speak",
            params={"model": self.voice, "encoding": "linear16", "sample_rate": "24000"},
            headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
            json={"text": text},
            timeout=30,
        )
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)

    def _synthesize_cartesia(self, text: str, output_path: str) -> None:
        import cartesia
        client = cartesia.Cartesia(api_key=os.environ.get("CARTESIA_API_KEY", ""))
        output = client.tts.bytes(
            model_id="sonic-english",
            transcript=text,
            voice_id=self.voice,
            output_format={"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000},
        )
        with open(output_path, "wb") as f:
            f.write(output)
