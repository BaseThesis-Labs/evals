"""LMNT TTS client."""
import time
import os
import numpy as np
from typing import Optional
import requests
import io
import soundfile as sf

from .base_client import BaseTTSClient, TTSResult


class LMNTClient(BaseTTSClient):
    """LMNT Speech Synthesis API."""

    def __init__(self, voice: str = "lily"):
        super().__init__()
        self.name = "lmnt"
        self.supports_cloning = True
        self.supports_streaming = True
        self.voice = voice

        # Load API key from environment
        from dotenv import load_dotenv
        load_dotenv()

        self.api_key = os.getenv("LMNT_API_KEY")
        if not self.api_key:
            raise ValueError("LMNT_API_KEY not set in .env file")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with LMNT."""
        start_time = time.time()

        # LMNT API endpoint
        url = "https://api.lmnt.com/v1/ai/speech"

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "voice": self.voice,
            "format": "wav",
            "sample_rate": 24000
        }

        # Make request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        ttfa_ms = (time.time() - start_time) * 1000

        # Parse response - LMNT returns audio in 'audio' field as base64
        result = response.json()

        # Decode base64 audio
        import base64
        audio_bytes = base64.b64decode(result['audio'])

        # Convert to numpy array
        audio_io = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_io)

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
        time.sleep(0.5)

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_seconds=duration,
            inference_time_ms=inference_time_ms,
            ttfa_ms=ttfa_ms,
            model_name=self.name,
            utterance_id=utterance_id
        )
