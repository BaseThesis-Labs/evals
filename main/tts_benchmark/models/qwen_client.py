"""Qwen Audio TTS client (Alibaba Cloud)."""
import time
import os
import numpy as np
from typing import Optional
import requests
import io
import soundfile as sf
import json
import hmac
import hashlib
import base64
from datetime import datetime

from .base_client import BaseTTSClient, TTSResult


class QwenClient(BaseTTSClient):
    """Qwen Audio TTS (Alibaba Cloud Model Studio)."""

    def __init__(self, voice: str = "zhifeng_emo"):
        super().__init__()
        self.name = "qwen"
        self.supports_cloning = False
        self.supports_streaming = False
        self.voice = voice

        # Load credentials from environment
        from dotenv import load_dotenv
        load_dotenv()

        self.access_key_id = os.getenv("ALIBABA_ACCESS_KEY_ID")
        self.access_key_secret = os.getenv("ALIBABA_ACCESS_KEY_SECRET")
        self.region = os.getenv("ALIBABA_REGION", "cn-shanghai")

        if not self.access_key_id or not self.access_key_secret:
            raise ValueError("ALIBABA_ACCESS_KEY_ID and ALIBABA_ACCESS_KEY_SECRET not set in .env")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with Qwen Audio."""
        start_time = time.time()

        # Alibaba Cloud API endpoint
        url = f"https://dashscope.aliyuncs.com/api/v1/services/audio/tts/synthesis"

        headers = {
            "Authorization": f"Bearer {self.access_key_id}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "false"
        }

        payload = {
            "model": "qwen-audio-tts",
            "input": {
                "text": text
            },
            "parameters": {
                "voice": self.voice,
                "format": "wav",
                "sample_rate": 24000
            }
        }

        # Make request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        ttfa_ms = (time.time() - start_time) * 1000

        result = response.json()

        # Get audio URL or base64 data
        if 'output' in result and 'audio_url' in result['output']:
            # Download audio from URL
            audio_response = requests.get(result['output']['audio_url'])
            audio_response.raise_for_status()
            audio_io = io.BytesIO(audio_response.content)
        elif 'output' in result and 'audio' in result['output']:
            # Base64 encoded audio
            audio_bytes = base64.b64decode(result['output']['audio'])
            audio_io = io.BytesIO(audio_bytes)
        else:
            raise RuntimeError(f"Unexpected response format: {result}")

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
