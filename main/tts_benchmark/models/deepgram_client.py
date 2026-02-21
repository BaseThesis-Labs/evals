"""Deepgram Aura TTS client."""
import time
import os
import numpy as np
from typing import Optional
import soundfile as sf
import tempfile

from .base_client import BaseTTSClient, TTSResult


class DeepgramClient(BaseTTSClient):
    """Deepgram Aura TTS API."""

    def __init__(self, voice: str = "aura-asteria-en"):
        super().__init__()
        self.name = "deepgram"
        self.supports_cloning = False
        self.supports_streaming = True
        self.voice = voice
        self.api_key = os.getenv("DEEPGRAM_API_KEY")

        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable not set")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with Deepgram Aura."""
        from deepgram import DeepgramClient as DG, SpeakOptions

        start_time = time.time()

        # Create client
        client = DG(self.api_key)

        # Configure options
        options = SpeakOptions(
            model=self.voice,
            encoding="linear16",
            sample_rate=24000
        )

        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Generate and save
            response = client.speak.v("1").save(
                temp_path,
                {"text": text},
                options
            )

            ttfa_ms = (time.time() - start_time) * 1000

            # Read the saved audio
            audio, sr = sf.read(temp_path)

            inference_time_ms = (time.time() - start_time) * 1000

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

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
