"""
Voice Evaluation Framework with Speaker Diarization
Comprehensive voice AI evaluation with per-speaker analysis
"""

from .core.pipeline import VoiceEvaluationPipeline
from .core.diarization import SpeakerDiarizer
from .metrics.enhanced_metrics import EnhancedVoiceMetrics
from .reports.formatter import format_report

__version__ = "2.0.0"
__all__ = [
    "VoiceEvaluationPipeline",
    "SpeakerDiarizer",
    "EnhancedVoiceMetrics",
    "format_report",
]
