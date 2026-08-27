from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS


PROBABILITY_COLUMNS = [f"prob_{label}" for label in CTFE_DOSE_LABELS]
STAGE_PROBABILITY_COLUMNS = [f"stage_prob_{label}" for label in CTFE_DOSE_LABELS]
CTFE_STAGE_TO_GN_DAY_GROUPS = {
    "early": ["d0_3", "d4_6"],
    "mid": ["d7_9"],
    "late": ["d10_12"],
    "lateplus": ["d13_plus"],
    "unknown": ["unknown"],
}


def _cut_as_object(values: pd.Series, *, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, right=True).astype("object").fillna("unknown")


def add_ctfe_stage_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add current-time CTFE stage columns for sliding-window/stage routing."""

    result = frame.copy()
    if "gn_day_group" in result.columns:
        result["gn_day_group"] = result["gn_day_group"].astype("object").fillna("unknown")
    elif "gn_day" in result.columns:
        result["gn_day_group"] = _cut_as_object(
            result["gn_day"],
            bins=[-np.inf, 3, 6, 9, 12, np.inf],
            labels=["d0_3", "d4_6", "d7_9", "d10_12", "d13_plus"],
        )
    else:
        result["gn_day_group"] = "unknown"

    stage_map = {}
    for stage, groups in CTFE_STAGE_TO_GN_DAY_GROUPS.items():
        for group in groups:
            stage_map[group] = stage
    if "ctfe_stage_group" in result.columns:
        result["ctfe_stage_group"] = result["ctfe_stage_group"].astype("object").fillna("unknown")
    else:
        result["ctfe_stage_group"] = result["gn_day_group"].astype(str).map(stage_map).fillna("unknown")

    if "current_fsh_band" in result.columns:
        result["current_fsh_band"] = result["current_fsh_band"].astype("object").fillna("unknown")
    elif "current_fsh_daily_dose" in result.columns:
        result["current_fsh_band"] = _cut_as_object(
            result["current_fsh_daily_dose"],
            bins=[-np.inf, 0, 80, 160, 240, np.inf],
            labels=["0", "0_80", "80_160", "160_240", "gt240"],
        )
    else:
        result["current_fsh_band"] = "unknown"
    if "gn_fsh_combo" not in result.columns:
        result["gn_fsh_combo"] = result["gn_day_group"].astype(str) + "|" + result["current_fsh_band"].astype(str)
    else:
        result["gn_fsh_combo"] = result["gn_fsh_combo"].astype("object").fillna("unknown|unknown")
    return result


def stage_values_for_stage(stage_name: str) -> list[str]:
    if stage_name not in CTFE_STAGE_TO_GN_DAY_GROUPS:
        raise ValueError(f"Unknown CTFE stage name: {stage_name}")
    return list(CTFE_STAGE_TO_GN_DAY_GROUPS[stage_name])


def _validate_prediction_frame(frame: pd.DataFrame, *, prediction_col: str, probability_prefix: str) -> None:
    required = {"split", "ctfe_next_fsh_dose_class", "ctfe_stage_group", prediction_col}
    required.update({f"{probability_prefix}{label}" for label in CTFE_DOSE_LABELS})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing CTFE stage prediction columns: {missing}")


def _assert_same_rows(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if len(left) != len(right):
        raise ValueError("CTFE stage prediction frames must have the same row count.")
    key_cols = [column for column in ["visit_uid", "cycle_uid", "monitoring_order"] if column in left.columns and column in right.columns]
    if key_cols and not left[key_cols].astype(str).reset_index(drop=True).equals(right[key_cols].astype(str).reset_index(drop=True)):
        raise ValueError(f"CTFE stage prediction frames are not aligned by {key_cols}.")


def evaluate_stage_prediction_frame(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    probability_prefix: str,
    split: str,
    stage_name: str | None = None,
) -> dict[str, float | int | str]:
    prepared = add_ctfe_stage_columns(frame)
    _validate_prediction_frame(prepared, prediction_col=prediction_col, probability_prefix=probability_prefix)
    subset = prepared[prepared["split"].astype(str) == str(split)].copy()
    if stage_name is not None:
        subset = subset[subset["ctfe_stage_group"].astype(str) == str(stage_name)].copy()
    if subset.empty:
        raise ValueError(f"No CTFE rows available for split={split!r}, stage={stage_name!r}")
    y_true = subset["ctfe_next_fsh_dose_class"].astype(str).to_numpy()
    y_pred = subset[prediction_col].astype(str).to_numpy()
    payload: dict[str, float | int | str] = {
        "split": split,
        "stage_name": stage_name or "all",
        "sample_count": int(len(subset)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average="weighted", zero_division=0)),
    }
    proba = subset[[f"{probability_prefix}{label}" for label in CTFE_DOSE_LABELS]].astype(float).to_numpy()
    proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)
    y_true_id = np.asarray([CTFE_DOSE_LABELS.index(value) for value in y_true], dtype=int)
    payload["log_loss"] = float(log_loss(y_true_id, proba, labels=list(range(len(CTFE_DOSE_LABELS)))))
    return payload


def select_stage_model_for_stage(
    fallback: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    stage_name: str,
    candidate_run_id: str,
    max_valid_macro_drop: float = 0.01,
    min_valid_weighted_gain: float = 0.0,
) -> dict[str, float | int | str | bool]:
    """Select a stage model using validation metrics against the fallback."""

    fallback_prepared = add_ctfe_stage_columns(fallback)
    candidate_prepared = add_ctfe_stage_columns(candidate)
    _assert_same_rows(fallback_prepared, candidate_prepared)
    fallback_metrics = evaluate_stage_prediction_frame(
        fallback_prepared,
        prediction_col="ctfe_stage_specialized_prediction" if "ctfe_stage_specialized_prediction" in fallback_prepared.columns else "ctfe_neural_prediction",
        probability_prefix="stage_prob_" if "stage_prob_stop" in fallback_prepared.columns else "prob_",
        split="valid",
        stage_name=stage_name,
    )
    candidate_metrics = evaluate_stage_prediction_frame(
        candidate_prepared,
        prediction_col="ctfe_neural_prediction",
        probability_prefix="prob_",
        split="valid",
        stage_name=stage_name,
    )
    accepted = bool(
        float(candidate_metrics["weighted_f1"]) >= float(fallback_metrics["weighted_f1"]) + float(min_valid_weighted_gain)
        and float(candidate_metrics["macro_f1"]) >= float(fallback_metrics["macro_f1"]) - float(max_valid_macro_drop)
    )
    return {
        "stage_name": stage_name,
        "run_id": candidate_run_id,
        "accepted": accepted,
        "reason": "valid_weighted_f1_with_macro_guardrail" if accepted else "no_valid_gain_or_macro_guardrail_failed",
        "valid_accuracy": float(candidate_metrics["accuracy"]),
        "valid_macro_f1": float(candidate_metrics["macro_f1"]),
        "valid_weighted_f1": float(candidate_metrics["weighted_f1"]),
        "valid_log_loss": float(candidate_metrics["log_loss"]),
        "valid_count": int(candidate_metrics["sample_count"]),
        "fallback_valid_accuracy": float(fallback_metrics["accuracy"]),
        "fallback_valid_macro_f1": float(fallback_metrics["macro_f1"]),
        "fallback_valid_weighted_f1": float(fallback_metrics["weighted_f1"]),
        "fallback_valid_log_loss": float(fallback_metrics["log_loss"]),
    }


def _copy_probabilities(target: pd.DataFrame, source: pd.DataFrame, row_mask: pd.Series, *, source_prefix: str = "prob_") -> None:
    for label in CTFE_DOSE_LABELS:
        target.loc[row_mask, f"stage_prob_{label}"] = source.loc[row_mask, f"{source_prefix}{label}"].astype(float).to_numpy()


def compose_stage_specialized_predictions(
    fallback: pd.DataFrame,
    stage_candidates: Mapping[str, pd.DataFrame],
    selections: Iterable[Mapping[str, object]],
    *,
    fallback_prediction_col: str | None = None,
    fallback_probability_prefix: str | None = None,
) -> pd.DataFrame:
    """Route accepted stage-specific CTFE models and fallback elsewhere."""

    result = add_ctfe_stage_columns(fallback).reset_index(drop=True)
    fallback_prediction_col = fallback_prediction_col or (
        "ctfe_stage_specialized_prediction"
        if "ctfe_stage_specialized_prediction" in result.columns
        else "ctfe_stratified_ensemble_prediction"
        if "ctfe_stratified_ensemble_prediction" in result.columns
        else "ctfe_ensemble_prediction"
        if "ctfe_ensemble_prediction" in result.columns
        else "ctfe_neural_prediction"
    )
    fallback_probability_prefix = fallback_probability_prefix or (
        "stage_prob_"
        if "stage_prob_stop" in result.columns
        else "ensemble_prob_"
        if "ensemble_prob_stop" in result.columns
        else "prob_"
    )
    _validate_prediction_frame(result, prediction_col=fallback_prediction_col, probability_prefix=fallback_probability_prefix)
    result["ctfe_stage_specialized_prediction"] = result[fallback_prediction_col].astype(str)
    result["ctfe_stage_source_run"] = "fallback"
    for label in CTFE_DOSE_LABELS:
        result[f"stage_prob_{label}"] = result[f"{fallback_probability_prefix}{label}"].astype(float)

    for selection in selections:
        if not bool(selection.get("accepted")):
            continue
        stage_name = str(selection["stage_name"])
        if stage_name not in stage_candidates:
            continue
        candidate = add_ctfe_stage_columns(stage_candidates[stage_name]).reset_index(drop=True)
        _assert_same_rows(result, candidate)
        row_mask = result["ctfe_stage_group"].astype(str).eq(stage_name)
        if not row_mask.any():
            continue
        result.loc[row_mask, "ctfe_stage_specialized_prediction"] = candidate.loc[row_mask, "ctfe_neural_prediction"].astype(str).to_numpy()
        result.loc[row_mask, "ctfe_stage_source_run"] = str(selection.get("run_id", stage_name))
        _copy_probabilities(result, candidate, row_mask, source_prefix="prob_")
    return result
