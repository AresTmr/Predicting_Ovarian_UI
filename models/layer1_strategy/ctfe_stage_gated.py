from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS
from models.layer1_strategy.ctfe_ensemble import evaluate_ctfe_prediction_frame

CONTROL_VARIANT = "control_fusion"
CANDIDATE_VARIANT = "stage_gated_candidate"
HIGH_ALLOWED_STAGES = frozenset({"d0_3", "d4_6"})
STOP_ALLOWED_STAGES = frozenset({"d10_12", "d13_plus"})
ENSEMBLE_PROBABILITY_COLUMNS = [f"ensemble_prob_{label}" for label in CTFE_DOSE_LABELS]
STAGE_GATED_PROBABILITY_COLUMNS = [f"stage_gated_prob_{label}" for label in CTFE_DOSE_LABELS]


def _ensure_input_columns(frame: pd.DataFrame) -> None:
    required = {"split", "gn_day_group", "ctfe_next_fsh_dose_class", *ENSEMBLE_PROBABILITY_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Stage-gated CTFE input missing columns: {missing}")


def apply_stage_gated_calibration(
    frame: pd.DataFrame,
    *,
    high_multiplier: float,
    stop_multiplier: float,
    minimum_probability: float,
    maximum_margin: float,
    prediction_col: str = "ctfe_stage_gated_prediction",
) -> pd.DataFrame:
    """Boost only stage-plausible extreme classes when the base model is uncertain."""

    _ensure_input_columns(frame)
    if high_multiplier < 1.0 or stop_multiplier < 1.0:
        raise ValueError("Stage-gated multipliers must not be below 1.0.")
    if not 0.0 <= minimum_probability <= 1.0 or not 0.0 <= maximum_margin <= 1.0:
        raise ValueError("Stage-gated probability constraints must be between 0 and 1.")
    result = frame.copy().reset_index(drop=True)
    probabilities = result[ENSEMBLE_PROBABILITY_COLUMNS].astype(float).to_numpy()
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    label_array = np.asarray(CTFE_DOSE_LABELS, dtype=object)
    base_predictions = label_array[probabilities.argmax(axis=1)]
    if "ctfe_stratified_ensemble_prediction" in result.columns:
        base_predictions = result["ctfe_stratified_ensemble_prediction"].astype(str).to_numpy()
    stages = result["gn_day_group"].astype(str).to_numpy()
    top_probabilities = probabilities.max(axis=1)
    adjustments = np.full(len(result), "none", dtype=object)
    for target, multiplier, allowed_stages in [
        ("high", float(high_multiplier), HIGH_ALLOWED_STAGES),
        ("stop", float(stop_multiplier), STOP_ALLOWED_STAGES),
    ]:
        target_idx = CTFE_DOSE_LABELS.index(target)
        target_probability = probabilities[:, target_idx]
        eligible = (
            np.isin(stages, list(allowed_stages))
            & (base_predictions != target)
            & (target_probability >= float(minimum_probability))
            & ((top_probabilities - target_probability) <= float(maximum_margin) + 1e-12)
            & (float(multiplier) > 1.0)
        )
        probabilities[eligible, target_idx] *= float(multiplier)
        adjustments[eligible] = target
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    for idx, label in enumerate(CTFE_DOSE_LABELS):
        result[f"stage_gated_prob_{label}"] = probabilities[:, idx]
    result[prediction_col] = label_array[probabilities.argmax(axis=1)]
    result["stage_gated_adjustment"] = adjustments
    result["stage_gated_changed_prediction"] = result[prediction_col].astype(str).to_numpy() != base_predictions
    return result


def evaluate_stage_gated_frame(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    split: str,
    probability_prefix: str = "stage_gated_prob_",
) -> dict[str, object]:
    metrics = evaluate_ctfe_prediction_frame(
        frame,
        prediction_col=prediction_col,
        probability_prefix=probability_prefix,
        split=split,
    )
    subset = frame[frame["split"].astype(str).eq(split)]
    y_true = subset["ctfe_next_fsh_dose_class"].astype(str).to_numpy()
    y_pred = subset[prediction_col].astype(str).to_numpy()
    class_scores = f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average=None, zero_division=0)
    for label, value in zip(CTFE_DOSE_LABELS, class_scores):
        metrics[f"f1_{label}"] = float(value)
    metrics["minority_mean_f1"] = (float(metrics["f1_stop"]) + float(metrics["f1_high"])) / 2.0
    if "stage_gated_changed_prediction" in subset.columns:
        metrics["changed_predictions"] = int(subset["stage_gated_changed_prediction"].sum())
    return metrics


def _baseline_metrics(frame: pd.DataFrame, *, split: str) -> dict[str, object]:
    return evaluate_stage_gated_frame(
        frame,
        prediction_col="ctfe_stratified_ensemble_prediction",
        probability_prefix="ensemble_prob_",
        split=split,
    )


def search_stage_gated_parameters(
    frame: pd.DataFrame,
    *,
    high_multipliers: Iterable[float] = (1.0, 1.05, 1.10, 1.15, 1.20),
    stop_multipliers: Iterable[float] = (1.0, 1.05, 1.10, 1.15),
    minimum_probabilities: Iterable[float] = (0.05, 0.10, 0.15),
    maximum_margins: Iterable[float] = (0.05, 0.10, 0.15, 0.20),
    selection_split: str = "train",
    maximum_log_loss_increase: float = 0.002,
) -> dict[str, object]:
    """Select conservative stage gates on train predictions only."""

    baseline = _baseline_metrics(frame, split=selection_split)
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for high_multiplier, stop_multiplier, min_probability, max_margin in product(
        high_multipliers, stop_multipliers, minimum_probabilities, maximum_margins
    ):
        candidate = apply_stage_gated_calibration(
            frame,
            high_multiplier=float(high_multiplier),
            stop_multiplier=float(stop_multiplier),
            minimum_probability=float(min_probability),
            maximum_margin=float(max_margin),
        )
        metrics = evaluate_stage_gated_frame(candidate, prediction_col="ctfe_stage_gated_prediction", split=selection_split)
        changed = int(metrics.get("changed_predictions", 0))
        nontrivial = changed > 0
        passes = bool(
            nontrivial
            and float(metrics["weighted_f1"]) >= float(baseline["weighted_f1"])
            and float(metrics["macro_f1"]) >= float(baseline["macro_f1"])
            and float(metrics["minority_mean_f1"]) >= float(baseline["minority_mean_f1"])
            and float(metrics["log_loss"]) <= float(baseline["log_loss"]) + float(maximum_log_loss_increase)
        )
        row = {
            "high_multiplier": float(high_multiplier),
            "stop_multiplier": float(stop_multiplier),
            "minimum_probability": float(min_probability),
            "maximum_margin": float(max_margin),
            "changed_predictions": changed,
            "passes_train_guardrail": passes,
            **{f"train_{key}": value for key, value in metrics.items() if key not in {"split"}},
        }
        rows.append(row)
        if not passes:
            continue
        if best is None or (
            float(metrics["minority_mean_f1"]), float(metrics["weighted_f1"]), float(metrics["macro_f1"]), -float(metrics["log_loss"]), -changed
        ) > (
            float(best["train_minority_mean_f1"]), float(best["train_weighted_f1"]), float(best["train_macro_f1"]), -float(best["train_log_loss"]), -int(best["changed_predictions"])
        ):
            best = row
    search_results = pd.DataFrame(rows).sort_values(
        ["passes_train_guardrail", "train_minority_mean_f1", "train_weighted_f1", "train_macro_f1", "train_log_loss"],
        ascending=[False, False, False, False, True],
    )
    if best is None:
        best = {"high_multiplier": 1.0, "stop_multiplier": 1.0, "minimum_probability": 1.0, "maximum_margin": 0.0, "changed_predictions": 0}
        reason = "no_nontrivial_train_guarded_stage_gate"
        accepted_for_validation = False
    else:
        reason = "train_guarded_stage_gate_selected"
        accepted_for_validation = True
    return {
        **best,
        "accepted_for_validation": accepted_for_validation,
        "reason": reason,
        "selection_split": selection_split,
        "test_used_for_selection": False,
        "baseline_metrics": baseline,
        "search_results": search_results,
    }


def summarize_stage_gated_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["accuracy", "macro_f1", "weighted_f1", "log_loss", "f1_stop", "f1_high", "minority_mean_f1", "changed_predictions"]
    rows: list[dict[str, object]] = []
    for (scope, variant, split), group in metrics.groupby(["scope", "variant", "split"], sort=True):
        row: dict[str, object] = {"scope": scope, "variant": variant, "split": split, "n_seeds": int(group["seed"].nunique())}
        for metric in metric_columns:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def assess_stage_gated_promotion(
    metrics: pd.DataFrame,
    *,
    scope: str,
    min_improved_seeds: int,
    minimum_minority_gain: float = 0.01,
    maximum_log_loss_increase: float = 0.002,
) -> dict[str, object]:
    """Apply strict validation-only gates for a stage-gated candidate."""

    valid = metrics[(metrics["scope"].astype(str).eq(scope)) & (metrics["split"].astype(str).eq("valid"))].copy()
    grouped = valid.groupby("variant")[["weighted_f1", "macro_f1", "minority_mean_f1", "log_loss"]].mean()
    if CONTROL_VARIANT not in grouped.index or CANDIDATE_VARIANT not in grouped.index:
        raise ValueError(f"Stage-gated comparison for {scope!r} requires both variants on validation rows.")
    control = grouped.loc[CONTROL_VARIANT]
    candidate = grouped.loc[CANDIDATE_VARIANT]
    paired = valid.pivot_table(index="seed", columns="variant", values="weighted_f1", aggfunc="first").dropna()
    improved_seeds = int((paired[CANDIDATE_VARIANT] >= paired[CONTROL_VARIANT]).sum())
    guards = {
        "weighted_f1_noninferior": bool(candidate["weighted_f1"] >= control["weighted_f1"]),
        "macro_f1_noninferior": bool(candidate["macro_f1"] >= control["macro_f1"]),
        "minority_gain_met": bool(candidate["minority_mean_f1"] >= control["minority_mean_f1"] + float(minimum_minority_gain) - 1e-12),
        "log_loss_tolerance_met": bool(candidate["log_loss"] <= control["log_loss"] + float(maximum_log_loss_increase) + 1e-12),
        "paired_weighted_f1_seed_count": bool(improved_seeds >= int(min_improved_seeds)),
    }
    return {
        "scope": scope,
        "accepted": bool(all(guards.values())),
        "reason": "all_validation_stage_gated_guardrails_passed" if all(guards.values()) else "validation_stage_gated_guardrail_failed",
        "parameter_selection_split": "train",
        "selection_split": "valid",
        "test_used_for_selection": False,
        "minimum_minority_gain": float(minimum_minority_gain),
        "maximum_log_loss_increase": float(maximum_log_loss_increase),
        "min_improved_seeds": int(min_improved_seeds),
        "improved_seed_count": improved_seeds,
        "guards": guards,
        "control_valid_metrics": {name: float(value) for name, value in control.to_dict().items()},
        "candidate_valid_metrics": {name: float(value) for name, value in candidate.to_dict().items()},
        "validation_delta_candidate_minus_control": {name: float(candidate[name] - control[name]) for name in grouped.columns},
    }