"""
Metric normalization: raw value → [0, 1].

Reads bounds from config/normalization.yaml.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Optional

import yaml

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "config", "normalization.yaml",
)


@lru_cache(maxsize=1)
def _load_bounds() -> Dict:
    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("METRIC_BOUNDS", {})


def normalize_metric(value: float, metric: str) -> Optional[float]:
    """Normalize a raw metric value to [0, 1].

    Returns None if the metric is unknown or the value is NaN/None.
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    import math
    if math.isnan(value) or math.isinf(value):
        return None

    bounds = _load_bounds()
    if metric not in bounds:
        return None

    cfg = bounds[metric]
    floor: float = cfg.get("floor", 0.0)
    ceiling: float = cfg.get("ceiling", 1.0)
    direction: str = cfg.get("direction", "higher")

    # Clamp to [floor, ceiling]
    value = max(floor, min(ceiling, value))

    if direction == "higher":
        return (value - floor) / (ceiling - floor)

    elif direction == "lower":
        return 1.0 - (value - floor) / (ceiling - floor)

    elif direction == "target":
        target: float = cfg.get("target", (floor + ceiling) / 2)
        tolerance: float = cfg.get("tolerance", 0.1)
        deviation = abs(value - target)
        # Full score within tolerance, linear decay to 0 at floor/ceiling
        half_range = max(ceiling - target, target - floor)
        if deviation <= tolerance:
            return 1.0
        excess = deviation - tolerance
        max_excess = half_range - tolerance
        if max_excess <= 0:
            return 0.0
        return max(0.0, 1.0 - excess / max_excess)

    return None


def normalize_dict(metrics: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Normalize all values in a metrics dict."""
    return {k: normalize_metric(v, k) for k, v in metrics.items()}
