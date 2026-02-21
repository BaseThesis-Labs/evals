"""ElevenLabs TTS client."""
import time
import os
import numpy as np
from typing import Optional
import io
import soundfile as sf

from .base_client import BaseTTSClient, TTSResult


class ElevenLabsClient(BaseTTSClient):
    """ElevenLabs TTS API."""

    def __init__(self, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):  # Default: Rachel
        super().__init__()
        self.name = "elevenlabs"
        self.supports_cloning = True
        self.supports_streaming = True
        self.voice_id = voice_id
        self.api_key = os.getenv("ELEVENLABS_API_KEY")

        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")

        # Initialize client
        from elevenlabs.client import ElevenLabs
        self.client = ElevenLabs(api_key=self.api_key)

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with ElevenLabs."""
        start_time = time.time()

        # Generate audio using text_to_speech
        # Using eleven_turbo_v2_5 which is available on free tier
        audio_generator = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id="eleven_turbo_v2_5"  # Free tier model (newer)
        )

        # Collect audio chunks
        audio_bytes = b"".join(audio_generator)

        ttfa_ms = (time.time() - start_time) * 1000

        # Convert bytes to numpy array
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
