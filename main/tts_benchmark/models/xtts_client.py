"""Coqui XTTS v2 client."""
import time
import numpy as np
from typing import Optional
import torch

from .base_client import BaseTTSClient, TTSResult


class XTTSClient(BaseTTSClient):
    """Coqui XTTS v2 model (voice cloning capable)."""

    def __init__(self, language: str = "en"):
        super().__init__()
        self.name = "xtts_v2"
        self.supports_cloning = True
        self.supports_streaming = False
        self.language = language
        self.model = None

    def _load_model(self):
        """Lazy load model on first use."""
        if self.model is None:
            from TTS.api import TTS
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with XTTS v2."""
        self._load_model()

        start_time = time.time()

        # Use reference audio for cloning if provided
        if reference_audio_path:
            audio = self.model.tts(
                text=text,
                speaker_wav=reference_audio_path,
                language=self.language
            )
        else:
            # Use default speaker
            audio = self.model.tts(
                text=text,
                language=self.language
            )

        inference_time_ms = (time.time() - start_time) * 1000

        # Convert to numpy array
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        audio = np.array(audio, dtype=np.float32)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        # XTTS outputs at 24kHz
        sr = 24000
        duration = len(audio) / sr

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_seconds=duration,
            inference_time_ms=inference_time_ms,
            ttfa_ms=inference_time_ms,
            model_name=self.name,
            utterance_id=utterance_id
        )
