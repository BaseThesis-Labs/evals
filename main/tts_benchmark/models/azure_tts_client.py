"""Microsoft Azure TTS client."""
import time
import os
import numpy as np
from typing import Optional
import tempfile
import soundfile as sf

from .base_client import BaseTTSClient, TTSResult


class AzureTTSClient(BaseTTSClient):
    """Microsoft Azure Text-to-Speech API (Neural voices)."""

    def __init__(self, voice_name: str = "en-US-JennyNeural"):
        super().__init__()
        self.name = "azure_tts"
        self.supports_cloning = False
        self.supports_streaming = False
        self.voice_name = voice_name

        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.service_region = os.getenv("AZURE_SERVICE_REGION", "eastus")

        if not self.speech_key:
            raise ValueError("AZURE_SPEECH_KEY environment variable not set")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        utterance_id: str = "unknown"
    ) -> TTSResult:
        """Generate speech with Azure TTS."""
        import azure.cognitiveservices.speech as speechsdk

        start_time = time.time()

        # Configure speech synthesis
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.service_region
        )
        speech_config.speech_synthesis_voice_name = self.voice_name
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )

        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name

        try:
            # Configure audio output
            audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            # Synthesize
            result = synthesizer.speak_text_async(text).get()

            ttfa_ms = (time.time() - start_time) * 1000

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Read generated audio
                audio, sr = sf.read(temp_path)
            else:
                raise RuntimeError(f"Speech synthesis failed: {result.reason}")

        finally:
            if os.path.exists(temp_path):
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
