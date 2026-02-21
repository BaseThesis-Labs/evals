"""Base TTS client interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TTSResult:
    """Result from TTS generation."""
    audio: np.ndarray           # float32, mono
    sample_rate: int
    duration_seconds: float
    inference_time_ms: float    # wall-clock total
    ttfa_ms: float             # time to first audio (for streaming)
    model_name: str
    utterance_id: str


class BaseTTSClient(ABC):
    """Abstract base class for TTS clients."""

    def __init__(self):
        self.name: str = ""
        self.supports_cloning: bool = False
        self.supports_streaming: bool = False

    @abstractmethod
    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            reference_audio_path: Optional reference audio for voice cloning
            speaker_id: Optional speaker identifier
            utterance_id: Identifier for this utterance

        Returns:
            TTSResult with audio and metadata
        """
        pass

    def cleanup(self):
        """Optional cleanup method."""
        pass
