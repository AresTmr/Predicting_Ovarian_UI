from __future__ import annotations

from itertools import product
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS

MINORITY_CTFE_CLASSES = ("stop", "high")


def _probability_columns(probability_prefix: str = "prob_") -> list[str]:
    return [f"{probability_prefix}{label}" for label in CTFE_DOSE_LABELS]


def _validate_probability_columns(frame: pd.DataFrame, probability_prefix: str) -> None:
    missing = [column for column in _probability_columns(probability_prefix) if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing CTFE probability columns: {missing}")


def apply_class_decision_weights(
    frame: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    probability_prefix: str = "prob_",
    output_probability_prefix: str = "calibrated_prob_",
    prediction_col: str = "ctfe_minority_prediction",
) -> pd.DataFrame:
    """Apply post-hoc class decision weights to CTFE class probabilities.

    This does not retrain the neural CTFE model. It only changes the final
    decision boundary for small classes, so it is safe to audit and roll back.
    """

    _validate_probability_columns(frame, probability_prefix)
    result = frame.copy()
    probabilities = result[_probability_columns(probability_prefix)].astype(float).to_numpy()
    multipliers = np.asarray([float(weights.get(label, 1.0)) for label in CTFE_DOSE_LABELS], dtype=float)
    if np.any(multipliers <= 0):
        raise ValueError("CTFE decision weights must be positive.")
    weighted = probabilities * multipliers.reshape(1, -1)
    row_sums = weighted.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    calibrated = weighted / safe_sums
    for idx, label in enumerate(CTFE_DOSE_LABELS):
        result[f"{output_probability_prefix}{label}"] = calibrated[:, idx]
    result[prediction_col] = np.asarray(CTFE_DOSE_LABELS, dtype=object)[np.argmax(calibrated, axis=1)]
    return result


def evaluate_weighted_predictions(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    split: str | None = None,
    true_col: str = "ctfe_next_fsh_dose_class",
) -> dict[str, float | int | str]:
    subset = frame.copy()
    if split is not None:
        subset = subset[subset["split"].astype(str) == str(split)].copy()
    if subset.empty:
        raise ValueError(f"No rows available for split={split!r}")
    true = subset[true_col].astype(str).to_numpy()
    pred = subset[prediction_col].astype(str).to_numpy()
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=CTFE_DOSE_LABELS,
        zero_division=0,
    )
    payload: dict[str, float | int | str] = {
        "split": split or "all",
        "sample_count": int(len(subset)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=CTFE_DOSE_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, labels=CTFE_DOSE_LABELS, average="weighted", zero_division=0)),
    }
    minority_values = []
    for idx, label in enumerate(CTFE_DOSE_LABELS):
        payload[f"precision_{label}"] = float(precision[idx])
        payload[f"recall_{label}"] = float(recall[idx])
        payload[f"f1_{label}"] = float(per_class_f1[idx])
        payload[f"support_{label}"] = int(support[idx])
        payload[f"pred_count_{label}"] = int((pred == label).sum())
        if label in MINORITY_CTFE_CLASSES:
            minority_values.append(float(per_class_f1[idx]))
    payload["minority_mean_f1"] = float(np.mean(minority_values)) if minority_values else 0.0
    return payload


def _as_float_grid(values: Iterable[float]) -> list[float]:
    parsed = sorted({float(value) for value in values})
    if not parsed:
        raise ValueError("CTFE minority weight grid cannot be empty.")
    return parsed


def search_minority_class_weights(
    frame: pd.DataFrame,
    *,
    stop_weights: Iterable[float] = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0),
    high_weights: Iterable[float] = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0),
    decrease_weights: Iterable[float] = (1.0,),
    low_weights: Iterable[float] = (1.0,),
    medium_weights: Iterable[float] = (1.0,),
    valid_weighted_f1_floor: float | None = None,
    prediction_col: str = "ctfe_neural_prediction",
    true_col: str = "ctfe_next_fsh_dose_class",
) -> dict[str, object]:
    """Search post-hoc class decision weights on the validation split."""

    baseline = evaluate_weighted_predictions(frame, prediction_col=prediction_col, split="valid", true_col=true_col)
    floor = float(valid_weighted_f1_floor) if valid_weighted_f1_floor is not None else float(baseline["weighted_f1"]) - 0.005
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for stop, decrease, low, medium, high in product(
        _as_float_grid(stop_weights),
        _as_float_grid(decrease_weights),
        _as_float_grid(low_weights),
        _as_float_grid(medium_weights),
        _as_float_grid(high_weights),
    ):
        weights = {"stop": stop, "decrease": decrease, "low": low, "medium": medium, "high": high}
        weighted = apply_class_decision_weights(frame, weights)
        metrics = evaluate_weighted_predictions(weighted, prediction_col="ctfe_minority_prediction", split="valid", true_col=true_col)
        row = {
            "weight_stop": stop,
            "weight_decrease": decrease,
            "weight_low": low,
            "weight_medium": medium,
            "weight_high": high,
            **metrics,
        }
        row["passes_guardrail"] = bool(float(metrics["weighted_f1"]) >= floor)
        rows.append(row)
        if not row["passes_guardrail"]:
            continue
        if float(metrics["minority_mean_f1"]) < float(baseline["minority_mean_f1"]):
            continue
        if best is None:
            best = row
            continue
        candidate_key = (
            float(metrics["minority_mean_f1"]),
            float(metrics["macro_f1"]),
            float(metrics["weighted_f1"]),
        )
        best_key = (
            float(best["minority_mean_f1"]),
            float(best["macro_f1"]),
            float(best["weighted_f1"]),
        )
        if candidate_key > best_key:
            best = row
    search_results = pd.DataFrame(rows).sort_values(
        ["passes_guardrail", "minority_mean_f1", "macro_f1", "weighted_f1"],
        ascending=[False, False, False, False],
    )
    if best is None:
        return {
            "accepted": False,
            "reason": "no_candidate_met_guardrails",
            "weights": {label: 1.0 for label in CTFE_DOSE_LABELS},
            "valid_metrics": baseline,
            "baseline_valid_metrics": baseline,
            "search_results": search_results,
        }
    weights = {
        "stop": float(best["weight_stop"]),
        "decrease": float(best["weight_decrease"]),
        "low": float(best["weight_low"]),
        "medium": float(best["weight_medium"]),
        "high": float(best["weight_high"]),
    }
    return {
        "accepted": bool(any(weights[label] != 1.0 for label in CTFE_DOSE_LABELS)),
        "reason": "minority_decision_weights_selected",
        "weights": weights,
        "valid_metrics": {key: value for key, value in best.items() if not str(key).startswith("weight_")},
        "baseline_valid_metrics": baseline,
        "search_results": search_results,
    }
