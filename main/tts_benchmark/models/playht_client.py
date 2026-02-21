"""Play.ht TTS client."""
import time
import os
import numpy as np
from typing import Optional
import requests
import tempfile
import soundfile as sf

from .base_client import BaseTTSClient, TTSResult


class PlayHTClient(BaseTTSClient):
    """Play.ht TTS API."""

    def __init__(self, voice: str = "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json"):
        super().__init__()
        self.name = "playht"
        self.supports_cloning = True
        self.supports_streaming = False
        self.voice = voice

        self.api_key = os.getenv("PLAYHT_API_KEY")
        self.user_id = os.getenv("PLAYHT_USER_ID")

        if not self.api_key or not self.user_id:
            raise ValueError("PLAYHT_API_KEY and PLAYHT_USER_ID environment variables must be set")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with Play.ht."""
        start_time = time.time()

        # Play.ht API endpoint
        url = "https://api.play.ht/api/v2/tts"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-User-ID": self.user_id,
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "voice": self.voice,
            "output_format": "wav",
            "sample_rate": 24000
        }

        # Make request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        ttfa_ms = (time.time() - start_time) * 1000

        # Save audio to temp file
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

        # Rate limiting for free tier
        time.sleep(1.0)

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_seconds=duration,
            inference_time_ms=inference_time_ms,
            ttfa_ms=ttfa_ms,
            model_name=self.name,
            utterance_id=utterance_id
        )
