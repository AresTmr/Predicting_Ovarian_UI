from __future__ import annotations

from math import isfinite, sqrt
from typing import Any, Mapping


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    return number if isfinite(number) else float(default)


def _clip(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def ohss_visual_display_probability(raw_probability: Any, *, high_threshold: Any = 0.12) -> float:
    """Return the strict moderate-to-severe OHSS probability for display."""
    return _clip(_to_float(raw_probability, 0.0), 0.0, 1.0)


def with_ohss_display_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a row and add UI display-risk fields while preserving raw probability."""
    out = dict(row)
    raw = _to_float(out.get("ohss_risk_probability", out.get("safety_probability", 0.0)), 0.0)
    high = out.get("ohss_risk_threshold_high", 0.12)
    out["ohss_raw_probability"] = raw
    out["ohss_display_probability"] = ohss_visual_display_probability(raw, high_threshold=high)
    out["ohss_display_label"] = "\u4e2d\u91cd\u5ea6 OHSS \u98ce\u9669"
    out["ohss_display_note"] = "The same strict moderate-to-severe OHSS probability is used for display and candidate ranking; every candidate remains eligible."
    return out
