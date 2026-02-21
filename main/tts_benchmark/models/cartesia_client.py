"""Cartesia Sonic TTS client - FIXED VERSION."""
import time
import os
import numpy as np
from typing import Optional
import soundfile as sf
import tempfile
import requests

from .base_client import BaseTTSClient, TTSResult


class CartesiaClient(BaseTTSClient):
    """Cartesia Sonic TTS API - using REST API directly."""

    def __init__(self, voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"):
        super().__init__()
        self.name = "cartesia"
        self.supports_cloning = False  # Simplified for now
        self.supports_streaming = False
        self.voice_id = voice_id
        self.api_key = os.getenv("CARTESIA_API_KEY")

        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY environment variable not set")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with Cartesia using REST API."""
        start_time = time.time()

        # Cartesia TTS endpoint
        url = "https://api.cartesia.ai/tts/bytes"

        headers = {
            "X-API-Key": self.api_key,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json"
        }

        payload = {
            "model_id": "sonic-english",
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": self.voice_id
            },
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 24000
            }
        }

        # Make request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        ttfa_ms = (time.time() - start_time) * 1000

        # Save to temp file and read back
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(response.content)
            temp_path = tmp.name

        try:
            audio, sr = sf.read(temp_path)
        finally:
            os.remove(temp_path)

        inference_time_ms = (time.time() - start_time) * 1000

        # Ensure mono float32
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        duration = len(audio) / sr

        # Rate limiting
        time.sleep(0.2)

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_seconds=duration,
            inference_time_ms=inference_time_ms,
            ttfa_ms=ttfa_ms,
            model_name=self.name,
            utterance_id=utterance_id
        )
