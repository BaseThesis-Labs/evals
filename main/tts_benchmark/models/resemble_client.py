"""Resemble AI TTS client."""
import time
import os
import numpy as np
from typing import Optional
import requests
import tempfile
import soundfile as sf

from .base_client import BaseTTSClient, TTSResult


class ResembleClient(BaseTTSClient):
    """Resemble AI TTS API."""

    def __init__(self, voice_uuid: str = None, project_uuid: str = None):
        super().__init__()
        self.name = "resemble"
        self.supports_cloning = True
        self.supports_streaming = False

        # Load from .env file
        from dotenv import load_dotenv
        load_dotenv()

        self.api_key = os.getenv("RESEMBLE_API_KEY")
        self.project_uuid = project_uuid or os.getenv("RESEMBLE_PROJECT_UUID")
        self.voice_uuid = voice_uuid or os.getenv("RESEMBLE_VOICE_UUID")

        if not self.api_key:
            raise ValueError("RESEMBLE_API_KEY not set in .env file")
        if not self.project_uuid or not self.voice_uuid:
            raise ValueError("RESEMBLE_PROJECT_UUID and RESEMBLE_VOICE_UUID must be set in .env file")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with Resemble AI."""
        start_time = time.time()

        # Resemble AI API endpoint
        url = f"https://app.resemble.ai/api/v2/projects/{self.project_uuid}/clips"

        headers = {
            "Authorization": f"Token token={self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "data": {
                "voice_uuid": self.voice_uuid,
                "body": text,
                "title": utterance_id,
                "sample_rate": 24000,
                "output_format": "wav"
            }
        }

        # Create clip
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        clip_data = response.json()

        # Poll for completion
        clip_uuid = clip_data['item']['uuid']
        status_url = f"https://app.resemble.ai/api/v2/projects/{self.project_uuid}/clips/{clip_uuid}"

        max_attempts = 60
        for _ in range(max_attempts):
            time.sleep(1)
            status_response = requests.get(status_url, headers=headers)
            status_response.raise_for_status()
            clip_status = status_response.json()

            if clip_status['item']['status'] == 'completed':
                audio_url = clip_status['item']['audio_src']
                break
        else:
            raise TimeoutError("TTS generation timeout")

        ttfa_ms = (time.time() - start_time) * 1000

        # Download audio
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_response.content)
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

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_seconds=duration,
            inference_time_ms=inference_time_ms,
            ttfa_ms=ttfa_ms,
            model_name=self.name,
            utterance_id=utterance_id
        )
