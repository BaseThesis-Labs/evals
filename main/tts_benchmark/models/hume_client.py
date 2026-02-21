"""Hume AI TTS client."""
import time
import os
import numpy as np
from typing import Optional
import io
import soundfile as sf
import base64
import asyncio

from .base_client import BaseTTSClient, TTSResult


class HumeClient(BaseTTSClient):
    """Hume AI Empathic Voice Interface."""

    def __init__(self, voice: str = "ITO"):
        super().__init__()
        self.name = "hume"
        self.supports_cloning = False
        self.supports_streaming = True
        self.voice = voice

        # Load API key from environment
        from dotenv import load_dotenv
        load_dotenv()

        self.api_key = os.getenv("HUME_API_KEY")
        if not self.api_key:
            raise ValueError("HUME_API_KEY not set in .env file")

        # Import Hume SDK
        from hume import AsyncHumeClient
        from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName

        self.AsyncHumeClient = AsyncHumeClient
        self.PostedUtterance = PostedUtterance
        self.PostedUtteranceVoiceWithName = PostedUtteranceVoiceWithName

    async def _generate_async(self, text: str, max_retries: int = 5) -> bytes:
        """Async audio generation with retry logic for rate limits."""
        client = self.AsyncHumeClient(api_key=self.api_key)

        utterance = self.PostedUtterance(
            text=text,
            voice=self.PostedUtteranceVoiceWithName(
                name=self.voice,
                provider='HUME_AI'
            )
        )

        # Retry logic with exponential backoff for rate limits
        from hume.core.api_error import ApiError
        
        for attempt in range(max_retries):
            try:
                # Use non-streaming version for simplicity
                response = await client.tts.synthesize_json(
                    utterances=[utterance],
                    version="1"
                )

                # Extract audio from first generation
                # ReturnTts has 'generations' not 'utterances'
                if response.generations and len(response.generations) > 0:
                    audio_b64 = response.generations[0].audio
                    return base64.b64decode(audio_b64)
                else:
                    raise RuntimeError("No audio returned from Hume API")
                    
            except ApiError as e:
                # Check if it's a rate limit error (429)
                if e.status_code == 429:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2^attempt seconds, max 60 seconds
                        wait_time = min(2 ** attempt, 60)
                        print(f"    Rate limit hit (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"Rate limit exceeded after {max_retries} retries: {e}")
                else:
                    # Non-429 API error, re-raise immediately
                    raise
            except Exception as e:
                # Other errors, re-raise immediately
                raise

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with Hume AI."""
        start_time = time.time()

        # Run async code synchronously
        audio_bytes = asyncio.run(self._generate_async(text))

        ttfa_ms = (time.time() - start_time) * 1000

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

        # Rate limiting - increased delay to avoid hitting limits
        time.sleep(2.0)  # Increased from 0.5s to 2s between requests

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_seconds=duration,
            inference_time_ms=inference_time_ms,
            ttfa_ms=ttfa_ms,
            model_name=self.name,
            utterance_id=utterance_id
        )
