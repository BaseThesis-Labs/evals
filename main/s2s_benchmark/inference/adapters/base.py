"""
Base interface for all S2S adapters.

All adapters (cascaded, GPT-4o Realtime, Gemini Live, …) must subclass
`BaseS2SAdapter` and implement `process()`.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class S2SResult:
    """Unified result object returned by every S2S adapter."""

    # ── Audio output ────────────────────────────────────────────────────────
    audio_out_path: str                   # path to saved WAV file
    sample_rate: int = 24_000

    # ── Transcription (intermediate ASR for cascaded; for native, STT on output) ──
    asr_transcript: Optional[str] = None  # what the ASR decoded from audio_in

    # ── Timing ──────────────────────────────────────────────────────────────
    ttfb_ms: float = 0.0                  # time-to-first-byte / time-to-first-audio
    e2e_latency_ms: float = 0.0           # full round-trip latency
    asr_latency_ms: float = 0.0           # time spent in ASR stage (cascaded only)
    tts_latency_ms: float = 0.0           # time spent in TTS stage (cascaded only)
    rtf: float = 0.0                      # real-time factor = latency / audio_duration

    # ── Metadata ────────────────────────────────────────────────────────────
    model_name: str = ""
    utterance_id: str = ""
    error: Optional[str] = None           # non-None if inference failed

    # ── Extra fields (adapter-specific) ─────────────────────────────────────
    extra: dict = field(default_factory=dict)


class BaseS2SAdapter(ABC):
    """Abstract base class for S2S adapters.

    Subclasses implement `process()`.  The framework calls `cleanup()` after
    all utterances are processed.
    """

    def __init__(self, model_name: str, config: dict):
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def process(
        self,
        audio_in_path: str,
        reference_text: str,
        utterance_id: str,
        output_dir: str,
    ) -> S2SResult:
        """Convert input speech → output speech.

        Args:
            audio_in_path: Path to the input audio file (WAV / MP3 / OPUS).
            reference_text: Ground-truth transcript of the input (for metric
                            computation downstream; adapters may ignore it).
            utterance_id: Unique ID used to name the output file.
            output_dir: Directory where the output WAV should be saved.

        Returns:
            S2SResult with at least `audio_out_path` and timing fields set.
        """

    # ── Multi-turn session lifecycle ────────────────────────────────────────

    def start_session(self, system_prompt: str) -> str:
        """Start a multi-turn dialogue session.

        Args:
            system_prompt: System instruction for the session.

        Returns:
            session_id: Unique identifier for this session.
        """
        import uuid
        sid = str(uuid.uuid4())
        if not hasattr(self, "_sessions"):
            self._sessions: dict[str, dict] = {}
        self._sessions[sid] = {
            "system_prompt": system_prompt,
            "history": [],
            "turn_count": 0,
        }
        return sid

    def end_session(self, session_id: str) -> None:
        """End a multi-turn session and release its resources."""
        if hasattr(self, "_sessions"):
            self._sessions.pop(session_id, None)

    def process_turn(
        self,
        audio_in_path: str,
        utterance_id: str,
        output_dir: str,
        session_id: str,
    ) -> S2SResult:
        """Process one turn within a multi-turn session.

        Default implementation delegates to process() — subclasses should
        override if they support stateful sessions.
        """
        session = getattr(self, "_sessions", {}).get(session_id, {})
        system_prompt = session.get("system_prompt", "")
        return self.process(
            audio_in_path=audio_in_path,
            reference_text="",
            utterance_id=utterance_id,
            output_dir=output_dir,
        )

    def cleanup(self) -> None:
        """Release resources (models, connections).  Safe to call multiple times."""
        if hasattr(self, "_sessions"):
            self._sessions.clear()

    # ── Convenience helpers ─────────────────────────────────────────────────

    def _make_output_path(self, output_dir: str, utterance_id: str) -> str:
        import os
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{utterance_id}.wav")

    def _wall_clock_ms(self, start: float) -> float:
        return (time.perf_counter() - start) * 1_000
