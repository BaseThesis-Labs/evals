"""Piper TTS client - DISABLED (Python API incompatible)."""
import time
import numpy as np
from typing import Optional

from .base_client import BaseTTSClient, TTSResult


class PiperClient(BaseTTSClient):
    """Piper TTS model - Currently disabled due to API compatibility issues."""

    def __init__(self, model: str = "en_US-lessac-medium"):
        super().__init__()
        self.name = "piper"
        self.supports_cloning = False
        self.supports_streaming = False
        self.model_name = model

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Piper generation not implemented - Python package API differs from CLI."""
        raise NotImplementedError(
            "Piper TTS Python package has different API than CLI. "
            "Please install Piper CLI separately or use Kokoro for local TTS."
        )
