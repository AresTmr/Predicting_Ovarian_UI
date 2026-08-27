from __future__ import annotations

from math import hypot, isfinite
from typing import Any, Mapping, Sequence


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    return number if isfinite(number) else float(default)


def _clip(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _risk_probability(row: Mapping[str, Any]) -> float:
    value = row.get(
        "ohss_risk_probability",
        row.get("strict_ohss_probability", row.get("safety_probability", 1.0)),
    )
    return _clip(_to_float(value, 1.0), 0.0, 1.0)


def _oocytes(row: Mapping[str, Any]) -> float:
    value = row.get("o_selection_value", row.get("o"))
    return max(_to_float(value, 0.0), 0.0)


def _pareto_frontier(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for candidate in rows:
        candidate_oocytes = _oocytes(candidate)
        candidate_risk = _risk_probability(candidate)
        dominated = any(
            other is not candidate
            and _oocytes(other) >= candidate_oocytes
            and _risk_probability(other) <= candidate_risk
            and (
                _oocytes(other) > candidate_oocytes
                or _risk_probability(other) < candidate_risk
            )
            for other in rows
        )
        candidate["pareto_candidate"] = not dominated
        if not dominated:
            frontier.append(candidate)
    return frontier


def _assign_normalized_ideal_distances(rows: Sequence[dict[str, Any]]) -> None:
    oocyte_values = [_oocytes(row) for row in rows]
    risk_values = [_risk_probability(row) for row in rows]
    oocyte_low, oocyte_high = min(oocyte_values), max(oocyte_values)
    risk_low, risk_high = min(risk_values), max(risk_values)
    oocyte_range = oocyte_high - oocyte_low
    risk_range = risk_high - risk_low

    for row in rows:
        oocyte_gap = max(0.0, oocyte_high - _oocytes(row))
        risk_gap = max(0.0, _risk_probability(row) - risk_low)
        oocyte_gap_normalized = oocyte_gap / oocyte_range if oocyte_range > 1e-12 else 0.0
        risk_gap_normalized = risk_gap / risk_range if risk_range > 1e-12 else 0.0
        row["oocyte_ideal_gap"] = round(oocyte_gap, 6)
        row["ohss_ideal_gap"] = round(risk_gap, 6)
        row["oocyte_ideal_gap_normalized"] = round(oocyte_gap_normalized, 6)
        row["ohss_ideal_gap_normalized"] = round(risk_gap_normalized, 6)
        row["balance_distance"] = round(
            hypot(oocyte_gap_normalized, risk_gap_normalized),
            6,
        )


def apply_oocyte_ohss_balance_recommendation(
    rows: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Select a gate-free Pareto balance of more oocytes and lower mod/severe OHSS risk.

    The two candidate-set ranges are independently normalized to [0, 1]. The
    selected point is the Pareto candidate nearest the clinical ideal of the
    highest predicted oocyte count and the lowest strict moderate-to-severe OHSS
    probability. This is conditional scenario ranking for clinician review, not
    a causal treatment-effect estimate or an automatic prescription.
    """
    kwargs.pop("gate_threshold", None)
    output: list[dict[str, Any]] = []
    stale_fields = {
        "balance_rank",
        "balance_selection_mode",
        "mii_ideal_gap",
        "mii_clinical_scale",
        "mii_balance_normalized",
        "ovarian_response_category",
        "ovarian_response_category_zh",
        "ovarian_response_reference_oocytes",
        "response_selection_strategy",
        "response_selection_strategy_zh",
        "response_selection_mode",
        "strategy_rank",
        "utility_score",
        "utility_rank",
        "ohss_gate_threshold",
        "ohss_gate_pass",
        "ohss_gate_status",
        "composite_safety_probability",
    }
    for source in rows or []:
        row = dict(source)
        role = str(row.get("candidate_role", "candidate"))
        row.setdefault("dose_model_candidate_role", role)
        for field in stale_fields:
            row.pop(field, None)
        if role != "current":
            row["candidate_role"] = "candidate"
            row["pareto_candidate"] = False
            row["recommendation_basis"] = "predicted oocytes/moderate-to-severe OHSS Pareto balance"
        output.append(row)

    candidates = [row for row in output if row.get("candidate_role") != "current"]
    if not candidates:
        return output

    frontier = _pareto_frontier(candidates)
    _assign_normalized_ideal_distances(candidates)
    selected = min(
        frontier,
        key=lambda row: (
            _to_float(row.get("balance_distance"), float("inf")),
            -_oocytes(row),
            _risk_probability(row),
            _to_float(row.get("candidate_total"), float("inf")),
            _to_float(row.get("fsh"), float("inf")),
            _to_float(row.get("lh"), float("inf")),
            _to_float(row.get("hmg"), float("inf")),
        ),
    )
    selected["candidate_role"] = "recommended"
    selected["recommendation_source"] = "candidate_oocyte_modsev_ohss_balance_no_gate_v1"
    selected["balance_selection_mode"] = "oocyte_modsev_ohss_pareto_equal_normalized_no_gate"

    ranked = sorted(
        candidates,
        key=lambda row: (
            0 if row is selected else 1,
            -int(bool(row.get("pareto_candidate"))),
            _to_float(row.get("balance_distance"), float("inf")),
            -_oocytes(row),
            _risk_probability(row),
            _to_float(row.get("candidate_total"), float("inf")),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["balance_rank"] = rank
    return output
