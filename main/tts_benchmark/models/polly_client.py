"""Amazon Polly TTS client."""
import time
import os
import numpy as np
from typing import Optional
import io
import soundfile as sf

from .base_client import BaseTTSClient, TTSResult


class PollyClient(BaseTTSClient):
    """Amazon Polly TTS API."""

    def __init__(self, voice_id: str = "Joanna", engine: str = "neural"):
        super().__init__()
        self.name = "polly"
        self.supports_cloning = False
        self.supports_streaming = True
        self.voice_id = voice_id
        self.engine = engine

        # AWS credentials should be in environment or ~/.aws/credentials
        # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with Amazon Polly."""
        import boto3

        start_time = time.time()

        # Initialize Polly client
        polly = boto3.client('polly', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))

        # Synthesize speech
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat='pcm',
            VoiceId=self.voice_id,
            Engine=self.engine,
            SampleRate='24000'
        )

        ttfa_ms = (time.time() - start_time) * 1000

        # Read audio stream
        audio_stream = response['AudioStream'].read()

        # Convert PCM bytes to numpy array
        audio = np.frombuffer(audio_stream, dtype=np.int16).astype(np.float32) / 32768.0
        sr = 24000

        inference_time_ms = (time.time() - start_time) * 1000

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        duration = len(audio) / sr

        # Rate limiting
        time.sleep(0.3)

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_seconds=duration,
            inference_time_ms=inference_time_ms,
            ttfa_ms=ttfa_ms,
            model_name=self.name,
            utterance_id=utterance_id
        )
