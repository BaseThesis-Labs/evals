"""
Error recovery metric for multi-turn S2S evaluation.

Measures the agent's ability to recover from context check failures.
After a failed context check, if a subsequent check passes, that
counts as a successful recovery.

Exposed functions:
    compute_error_recovery(session_result) -> float
"""
from __future__ import annotations

from typing import Optional


def compute_error_recovery(session_result) -> Optional[float]:
    """Recovery rate after failed context checks.

    Scans turns for context_check_passed results.  After each failure,
    checks if any subsequent turn has a passing context check.  The
    recovery rate is the fraction of failures that are eventually
    recovered.

    Args:
        session_result: SessionResult with .turns list of TurnResult objects.
                        Each TurnResult has .context_check_passed (Optional[bool]).

    Returns:
        Recovery rate in [0, 1]. None if there were no failed checks.
    """
    try:
        turns = session_result.turns

        # Collect indices of context check results
        check_results = []
        for i, turn in enumerate(turns):
            passed = getattr(turn, "context_check_passed", None)
            if passed is not None:
                check_results.append((i, passed))

        if not check_results:
            return None  # No context checks at all — metric not applicable

        # Find failures
        failure_indices = [idx for idx, (_, passed) in enumerate(check_results) if not passed]
        if not failure_indices:
            return 1.0  # All checks passed — no recovery needed = perfect

        # For each failure, check if any subsequent check passes
        recoveries = 0
        for fail_pos in failure_indices:
            # Look at all checks after this failure
            for subsequent_pos in range(fail_pos + 1, len(check_results)):
                _, passed = check_results[subsequent_pos]
                if passed:
                    recoveries += 1
                    break  # Only count one recovery per failure

        return float(recoveries / len(failure_indices))

    except Exception as exc:
        print(f"  [multiturn] error_recovery error: {exc}")
        return None
