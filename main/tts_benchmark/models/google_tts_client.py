"""Google Cloud TTS client."""
import time
import os
import numpy as np
from typing import Optional
import io

from .base_client import BaseTTSClient, TTSResult


class GoogleTTSClient(BaseTTSClient):
    """Google Cloud Text-to-Speech API."""

    def __init__(self, voice_name: str = "en-US-Neural2-F", language_code: str = "en-US"):
        super().__init__()
        self.name = "google_tts"
        self.supports_cloning = False
        self.supports_streaming = False
        self.voice_name = voice_name
        self.language_code = language_code

        # Check for credentials
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with Google Cloud TTS."""
        from google.cloud import texttospeech
        import soundfile as sf

        start_time = time.time()

        # Initialize client
        client = texttospeech.TextToSpeechClient()

        # Set up request
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            name=self.voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000
        )

        # Perform TTS
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        ttfa_ms = (time.time() - start_time) * 1000

        # Convert audio bytes to numpy array
        audio_bytes = io.BytesIO(response.audio_content)
        audio, sr = sf.read(audio_bytes)

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
