"""Kokoro ONNX TTS client."""
import time
import numpy as np
from typing import Optional
import soundfile as sf
import librosa
from pathlib import Path

from .base_client import BaseTTSClient, TTSResult


class KokoroClient(BaseTTSClient):
    """Kokoro ONNX TTS model (local, fast)."""

    def __init__(self, voice: str = "af"):
        super().__init__()
        self.name = "kokoro"
        self.supports_cloning = False
        self.supports_streaming = False
        self.voice = voice

        # Load model immediately (not lazy loading)
        try:
            from kokoro_onnx import Kokoro

            model_dir = Path(__file__).parent / "kokoro_models"
            model_path = model_dir / "kokoro-v0_19.onnx"
            voices_path = model_dir / "voices.bin"

            self.model = Kokoro(str(model_path), str(voices_path))
            print(f"  ✓ Loaded Kokoro model with voice: {voice}")
        except ImportError as e:
            raise ImportError(f"Failed to import kokoro_onnx: {e}. Install with: pip install kokoro-onnx")
        except Exception as e:
            raise RuntimeError(f"Failed to load Kokoro model: {e}")

    def _load_model(self):
        """Model already loaded in __init__, this is a no-op."""
        pass

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with Kokoro."""
        start_time = time.time()

        # Generate audio
        audio, sr = self.model.create(text, voice=self.voice, speed=1.0)

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
            ttfa_ms=inference_time_ms,  # Non-streaming
            model_name=self.name,
            utterance_id=utterance_id
        )
