"""
Multi-turn dialogue metrics for S2S evaluation.

Evaluates session-level properties of multi-turn voice agent conversations:
  context_retention   - fraction of context probes passed across the session
  voice_consistency   - speaker identity stability across turns (WavLM similarity)
  task_completion     - LLM-judged completion of required/optional success criteria
  degradation         - quality metric slopes over successive agent turns
  error_recovery      - recovery rate after failed context checks
  dialogue_coherence  - LLM-judged coherence of the full dialogue transcript
  session_quality     - aggregate of all session-level metrics
"""

from metrics.multiturn.context_retention import compute_context_retention
from metrics.multiturn.consistency import compute_voice_consistency
from metrics.multiturn.task_completion import compute_task_completion
from metrics.multiturn.degradation import compute_degradation
from metrics.multiturn.error_recovery import compute_error_recovery
from metrics.multiturn.dialogue_coherence import compute_dialogue_coherence
from metrics.multiturn.session_quality import compute_session_quality

__all__ = [
    "compute_context_retention",
    "compute_voice_consistency",
    "compute_task_completion",
    "compute_degradation",
    "compute_error_recovery",
    "compute_dialogue_coherence",
    "compute_session_quality",
]
