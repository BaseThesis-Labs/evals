"""
Streaming TTFB (Time-to-First-Byte) measurement utilities.

Used by full-duplex adapters (GPT-4o Realtime, Gemini Live, Moshi) to
measure how quickly the system starts responding after receiving audio input.

Phase 3 — partially implemented.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List, Optional


@dataclass
class StreamingMetrics:
    """Timing metrics for a streaming S2S interaction."""

    input_start_ms: float = 0.0
    input_end_ms: float = 0.0          # when last audio chunk was sent
    first_audio_ms: float = 0.0        # when first output audio byte arrived
    last_audio_ms: float = 0.0
    chunk_latencies_ms: List[float] = field(default_factory=list)

    @property
    def ttfb_ms(self) -> Optional[float]:
        """Time from end of input to first audio byte."""
        if self.first_audio_ms > 0 and self.input_end_ms > 0:
            return self.first_audio_ms - self.input_end_ms
        return None

    @property
    def total_duration_ms(self) -> Optional[float]:
        if self.last_audio_ms > 0 and self.input_start_ms > 0:
            return self.last_audio_ms - self.input_start_ms
        return None

    @property
    def p50_chunk_latency_ms(self) -> Optional[float]:
        if not self.chunk_latencies_ms:
            return None
        sorted_lat = sorted(self.chunk_latencies_ms)
        mid = len(sorted_lat) // 2
        return float(sorted_lat[mid])


class TTFBTimer:
    """Context manager for measuring TTFB in streaming scenarios.

    Usage:
        timer = TTFBTimer()
        timer.start_input()
        # ... send audio chunks ...
        timer.end_input()
        # ... receive first audio ...
        timer.first_audio_received()
        metrics = timer.metrics
    """

    def __init__(self):
        self._metrics = StreamingMetrics()

    def start_input(self) -> None:
        self._metrics.input_start_ms = _now_ms()

    def end_input(self) -> None:
        self._metrics.input_end_ms = _now_ms()

    def first_audio_received(self) -> None:
        if self._metrics.first_audio_ms == 0.0:
            self._metrics.first_audio_ms = _now_ms()

    def last_audio_received(self) -> None:
        self._metrics.last_audio_ms = _now_ms()

    def record_chunk_latency(self, sent_at_ms: float) -> None:
        latency = _now_ms() - sent_at_ms
        self._metrics.chunk_latencies_ms.append(latency)

    @property
    def metrics(self) -> StreamingMetrics:
        return self._metrics


@contextmanager
def measure_ttfb() -> Generator[TTFBTimer, None, None]:
    """Context manager that yields a TTFBTimer.

    Example::

        with measure_ttfb() as timer:
            timer.start_input()
            # ... stream audio in ...
            timer.end_input()
            # ... receive output ...
            timer.first_audio_received()

        print(timer.metrics.ttfb_ms)
    """
    timer = TTFBTimer()
    yield timer


def _now_ms() -> float:
    return time.perf_counter() * 1_000
