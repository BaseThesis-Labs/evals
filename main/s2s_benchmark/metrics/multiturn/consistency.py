"""
Voice consistency metric for multi-turn S2S evaluation.

Measures speaker identity stability across turns using WavLM speaker
embeddings.  Combines first-vs-last similarity with mean consecutive
turn similarity.

Exposed functions:
    compute_voice_consistency(session_result) -> float
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def compute_voice_consistency(session_result) -> Optional[float]:
    """Speaker identity consistency across agent turns.

    Uses WavLM-Base+ cosine similarity (via metrics.speaker.compute_sim_wavlm)
    to compare:
      - first agent turn vs last agent turn  (weight 0.6)
      - consecutive agent turn pairs         (weight 0.4)

    Args:
        session_result: SessionResult with .turns list of TurnResult objects.
                        Each TurnResult has .role ("user"|"agent") and
                        .output_audio_path (str or None).

    Returns:
        Score in [0, 1] where 1.0 = perfectly consistent voice.
        None if fewer than 2 agent turns with audio.
    """
    try:
        from metrics.speaker import compute_sim_wavlm

        # Collect agent turn audio paths
        agent_paths: List[str] = []
        for turn in session_result.turns:
            if turn.role == "agent" and turn.output_audio_path:
                agent_paths.append(turn.output_audio_path)

        if len(agent_paths) < 2:
            return None

        # First vs last agent turn similarity
        first_last_sim = compute_sim_wavlm(agent_paths[0], agent_paths[-1])
        if first_last_sim is None:
            return None

        # Consecutive turn similarities
        consecutive_sims: List[float] = []
        for i in range(len(agent_paths) - 1):
            sim = compute_sim_wavlm(agent_paths[i], agent_paths[i + 1])
            if sim is not None:
                consecutive_sims.append(sim)

        if not consecutive_sims:
            # Fall back to first-last only
            return float(max(0.0, min(1.0, first_last_sim)))

        mean_consecutive = float(np.mean(consecutive_sims))

        # Weighted combination: 0.6 * first-last + 0.4 * mean consecutive
        score = 0.6 * first_last_sim + 0.4 * mean_consecutive
        return float(max(0.0, min(1.0, score)))

    except Exception as exc:
        print(f"  [multiturn] voice_consistency error: {exc}")
        return None
