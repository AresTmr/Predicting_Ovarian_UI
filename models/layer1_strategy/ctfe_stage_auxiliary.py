from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS

CONTROL_VARIANT = "control_fusion"
CANDIDATE_VARIANT = "stage_auxiliary_candidate"
TARGET_ALLOWED_STAGES = {
    "high": frozenset({"d0_3", "d4_6"}),
    "stop": frozenset({"d10_12", "d13_plus"}),
}
AUXILIARY_FEATURES = [
    "afc",
    "amh",
    "gn_day",
    "current_e2",
    "current_fsh_daily_dose",
    "current_gn_dose",
    "total_follicle_count",
    "mature_follicle_count",
]
FORBIDDEN_AUXILIARY_FRAGMENTS = (
    "next_",
    "target_",
    "observed_",
    "oocytes",
    "mii",
    "ohss",
    "clinical_pregnancy",
    "live_birth",
    "embryo",
    "transfer",
)
ENSEMBLE_PROBABILITY_COLUMNS = [f"ensemble_prob_{label}" for label in CTFE_DOSE_LABELS]
AUXILIARY_PROBABILITY_COLUMNS = [f"stage_auxiliary_prob_{label}" for label in CTFE_DOSE_LABELS]


@dataclass
class ConstantProbabilityHead:
    probability: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), float(self.probability), dtype=float)
        return np.column_stack([1.0 - positive, positive])


def validate_auxiliary_feature_columns(feature_columns: Sequence[str]) -> None:
    bad = [name for name in feature_columns if any(fragment in name.lower() for fragment in FORBIDDEN_AUXILIARY_FRAGMENTS)]
    if bad:
        raise ValueError(f"Stage auxiliary feature list contains leakage-prone columns: {bad}")


def _ensure_prediction_columns(frame: pd.DataFrame) -> None:
    required = {"split", "gn_day_group", "ctfe_next_fsh_dose_class", "ctfe_stratified_ensemble_prediction", *ENSEMBLE_PROBABILITY_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Stage auxiliary input missing columns: {missing}")


def assign_inner_training_partition(
    frame: pd.DataFrame,
    *,
    seed: int,
    tune_ratio: float = 0.20,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Split only training patients/cycles into inner fit and tune partitions."""

    if not 0.0 < float(tune_ratio) < 1.0:
        raise ValueError("tune_ratio must be between 0 and 1.")
    result = frame.copy().reset_index(drop=True)
    if "split" not in result.columns:
        raise ValueError("Stage auxiliary partitioning requires split column.")
    if group_column is None:
        group_column = "art_id" if "art_id" in result.columns else "cycle_uid"
    if group_column not in result.columns:
        raise ValueError("Stage auxiliary partitioning requires art_id or cycle_uid.")
    result["auxiliary_partition"] = "external_holdout"
    train_mask = result["split"].astype(str).eq("train").to_numpy()
    train_indices = np.flatnonzero(train_mask)
    if len(train_indices) < 2:
        raise ValueError("Stage auxiliary inner split requires at least two training rows.")
    groups = result.loc[train_mask, group_column].fillna(result.loc[train_mask, "cycle_uid"] if "cycle_uid" in result.columns else "missing").astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("Stage auxiliary inner split requires at least two training groups.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(tune_ratio), random_state=int(seed))
    fit_position, tune_position = next(splitter.split(np.zeros(len(train_indices)), groups=groups))
    result.loc[train_indices[fit_position], "auxiliary_partition"] = "inner_fit"
    result.loc[train_indices[tune_position], "auxiliary_partition"] = "inner_tune"
    return result


def _build_head(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", C=0.50, solver="liblinear", max_iter=500, random_state=int(seed))),
        ]
    )


def fit_stage_auxiliary_heads(
    training_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = AUXILIARY_FEATURES,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit stage-specific rare-class heads from training rows only."""

    _ensure_prediction_columns(training_frame)
    validate_auxiliary_feature_columns(feature_columns)
    missing = sorted(set(feature_columns).difference(training_frame.columns))
    if missing:
        raise ValueError(f"Stage auxiliary feature columns missing: {missing}")
    heads: dict[str, object] = {}
    support_rows: list[dict[str, object]] = []
    for target, allowed_stages in TARGET_ALLOWED_STAGES.items():
        subset = training_frame[training_frame["gn_day_group"].astype(str).isin(allowed_stages)].copy()
        y = subset["ctfe_next_fsh_dose_class"].astype(str).eq(target).astype(int).to_numpy()
        positive = int(y.sum())
        negative = int(len(y) - positive)
        if not len(subset):
            heads[target] = ConstantProbabilityHead(0.0)
        elif positive == 0 or negative == 0:
            heads[target] = ConstantProbabilityHead(float(positive > 0))
        else:
            model = _build_head(int(seed) + (1 if target == "high" else 2))
            model.fit(subset[list(feature_columns)], y)
            heads[target] = model
        support_rows.append({"target": target, "allowed_stages": ",".join(sorted(allowed_stages)), "sample_count": int(len(subset)), "positive_count": positive, "negative_count": negative})
    return heads, pd.DataFrame(support_rows)


def add_auxiliary_probabilities(
    frame: pd.DataFrame,
    heads: Mapping[str, object],
    *,
    feature_columns: Sequence[str] = AUXILIARY_FEATURES,
) -> pd.DataFrame:
    _ensure_prediction_columns(frame)
    validate_auxiliary_feature_columns(feature_columns)
    result = frame.copy().reset_index(drop=True)
    for target, allowed_stages in TARGET_ALLOWED_STAGES.items():
        probabilities = np.zeros(len(result), dtype=float)
        eligible = result["gn_day_group"].astype(str).isin(allowed_stages).to_numpy()
        if eligible.any():
            model = heads[target]
            probabilities[eligible] = np.asarray(model.predict_proba(result.loc[eligible, list(feature_columns)]))[:, 1]
        result[f"auxiliary_prob_{target}"] = probabilities
    return result


def apply_stage_auxiliary_fusion(
    frame: pd.DataFrame,
    *,
    high_weight: float,
    stop_weight: float,
    minimum_aux_probability: float,
    maximum_margin: float,
    prediction_col: str = "ctfe_stage_auxiliary_prediction",
) -> pd.DataFrame:
    """Blend auxiliary evidence into stage-allowed rare classes only."""

    _ensure_prediction_columns(frame)
    for column in ["auxiliary_prob_high", "auxiliary_prob_stop"]:
        if column not in frame.columns:
            raise ValueError(f"Missing auxiliary probability column: {column}")
    for value in [high_weight, stop_weight, minimum_aux_probability, maximum_margin]:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("Stage auxiliary fusion parameters must be between 0 and 1.")
    result = frame.copy().reset_index(drop=True)
    probabilities = result[ENSEMBLE_PROBABILITY_COLUMNS].astype(float).to_numpy()
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    labels = np.asarray(CTFE_DOSE_LABELS, dtype=object)
    base_prediction = result["ctfe_stratified_ensemble_prediction"].astype(str).to_numpy()
    adjustment = np.full(len(result), "none", dtype=object)
    for target, weight in [("high", float(high_weight)), ("stop", float(stop_weight))]:
        if weight <= 0.0:
            continue
        target_idx = CTFE_DOSE_LABELS.index(target)
        target_base = probabilities[:, target_idx].copy()
        top_base = probabilities.max(axis=1)
        auxiliary = pd.to_numeric(result[f"auxiliary_prob_{target}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        allowed = result["gn_day_group"].astype(str).isin(TARGET_ALLOWED_STAGES[target]).to_numpy()
        proposed = (1.0 - weight) * target_base + weight * auxiliary
        eligible = allowed & (base_prediction != target) & (auxiliary >= float(minimum_aux_probability)) & ((top_base - target_base) <= float(maximum_margin) + 1e-12) & (proposed > target_base)
        for index in np.flatnonzero(eligible):
            old_target = float(probabilities[index, target_idx])
            new_target = float(min(max(proposed[index], 0.0), 1.0))
            old_other = max(1.0 - old_target, 1e-12)
            new_other = max(1.0 - new_target, 0.0)
            probabilities[index, :] *= new_other / old_other
            probabilities[index, target_idx] = new_target
            adjustment[index] = target
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    for index, label in enumerate(CTFE_DOSE_LABELS):
        result[f"stage_auxiliary_prob_{label}"] = probabilities[:, index]
    result[prediction_col] = labels[probabilities.argmax(axis=1)]
    result["stage_auxiliary_adjustment"] = adjustment
    result["stage_auxiliary_changed_prediction"] = result[prediction_col].astype(str).to_numpy() != base_prediction
    return result


def evaluate_stage_auxiliary_frame(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    probability_prefix: str = "stage_auxiliary_prob_",
) -> dict[str, object]:
    if frame.empty:
        raise ValueError("Cannot evaluate empty stage auxiliary frame.")
    y_true = frame["ctfe_next_fsh_dose_class"].astype(str).to_numpy()
    y_pred = frame[prediction_col].astype(str).to_numpy()
    probability_columns = [f"{probability_prefix}{label}" for label in CTFE_DOSE_LABELS]
    probabilities = frame[probability_columns].astype(float).to_numpy()
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    class_f1 = f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average=None, zero_division=0)
    metrics: dict[str, object] = {
        "sample_count": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(np.asarray([CTFE_DOSE_LABELS.index(value) for value in y_true], dtype=int), probabilities, labels=list(range(len(CTFE_DOSE_LABELS))))),
    }
    for label, value in zip(CTFE_DOSE_LABELS, class_f1):
        metrics[f"f1_{label}"] = float(value)
    metrics["minority_mean_f1"] = (float(metrics["f1_stop"]) + float(metrics["f1_high"])) / 2.0
    if "stage_auxiliary_changed_prediction" in frame.columns:
        metrics["changed_predictions"] = int(frame["stage_auxiliary_changed_prediction"].sum())
    return metrics


def _baseline_metrics(frame: pd.DataFrame) -> dict[str, object]:
    return evaluate_stage_auxiliary_frame(frame, prediction_col="ctfe_stratified_ensemble_prediction", probability_prefix="ensemble_prob_")


def search_stage_auxiliary_fusion_parameters(
    frame: pd.DataFrame,
    *,
    selection_mask: np.ndarray,
    high_weights: Iterable[float] = (0.05, 0.10, 0.15, 0.20, 0.25),
    stop_weights: Iterable[float] = (0.05, 0.10, 0.15, 0.20, 0.25),
    minimum_aux_probabilities: Iterable[float] = (0.35, 0.45, 0.55, 0.65),
    maximum_margins: Iterable[float] = (0.05, 0.10, 0.15, 0.20),
    maximum_log_loss_increase: float = 0.002,
) -> dict[str, object]:
    """Select fusion parameters using only the inner-tune subset of training rows."""

    mask = np.asarray(selection_mask, dtype=bool)
    if len(mask) != len(frame) or not mask.any():
        raise ValueError("Stage auxiliary selection requires a non-empty aligned inner-tune mask.")
    tune = frame.loc[mask].reset_index(drop=True)
    baseline = _baseline_metrics(tune)
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for high_weight, stop_weight, min_probability, max_margin in product(high_weights, stop_weights, minimum_aux_probabilities, maximum_margins):
        candidate = apply_stage_auxiliary_fusion(tune, high_weight=float(high_weight), stop_weight=float(stop_weight), minimum_aux_probability=float(min_probability), maximum_margin=float(max_margin))
        metrics = evaluate_stage_auxiliary_frame(candidate, prediction_col="ctfe_stage_auxiliary_prediction")
        changed = int(metrics.get("changed_predictions", 0))
        passes = bool(changed > 0 and float(metrics["weighted_f1"]) >= float(baseline["weighted_f1"]) and float(metrics["macro_f1"]) >= float(baseline["macro_f1"]) and float(metrics["minority_mean_f1"]) >= float(baseline["minority_mean_f1"]) and float(metrics["log_loss"]) <= float(baseline["log_loss"]) + float(maximum_log_loss_increase))
        result = {"high_weight": float(high_weight), "stop_weight": float(stop_weight), "minimum_aux_probability": float(min_probability), "maximum_margin": float(max_margin), "changed_predictions": changed, "passes_inner_tune_guardrail": passes, **{f"tune_{name}": value for name, value in metrics.items()}}
        rows.append(result)
        if passes and (best is None or (float(result["tune_minority_mean_f1"]), float(result["tune_weighted_f1"]), float(result["tune_macro_f1"]), -float(result["tune_log_loss"]), -int(changed)) > (float(best["tune_minority_mean_f1"]), float(best["tune_weighted_f1"]), float(best["tune_macro_f1"]), -float(best["tune_log_loss"]), -int(best["changed_predictions"]))):
            best = result
    search_results = pd.DataFrame(rows).sort_values(["passes_inner_tune_guardrail", "tune_minority_mean_f1", "tune_weighted_f1", "tune_macro_f1", "tune_log_loss"], ascending=[False, False, False, False, True])
    if best is None:
        best = {"high_weight": 0.0, "stop_weight": 0.0, "minimum_aux_probability": 1.0, "maximum_margin": 0.0, "changed_predictions": 0}
        reason = "no_nontrivial_inner_tune_guarded_fusion"
        accepted_for_validation = False
    else:
        reason = "inner_tune_guarded_fusion_selected"
        accepted_for_validation = True
    return {**best, "accepted_for_validation": accepted_for_validation, "reason": reason, "parameter_selection_split": "inner_tune", "test_used_for_selection": False, "baseline_metrics": baseline, "search_results": search_results}


def summarize_stage_auxiliary_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["accuracy", "macro_f1", "weighted_f1", "log_loss", "f1_stop", "f1_high", "minority_mean_f1", "changed_predictions"]
    rows: list[dict[str, object]] = []
    for (scope, variant, split), group in metrics.groupby(["scope", "variant", "split"], sort=True):
        entry: dict[str, object] = {"scope": scope, "variant": variant, "split": split, "n_seeds": int(group["seed"].nunique())}
        for name in metric_names:
            if name in group.columns:
                values = pd.to_numeric(group[name], errors="coerce")
                entry[f"{name}_mean"] = float(values.mean())
                entry[f"{name}_std"] = float(values.std(ddof=0))
        rows.append(entry)
    return pd.DataFrame(rows)


def assess_stage_auxiliary_promotion(
    metrics: pd.DataFrame,
    *,
    scope: str,
    min_improved_seeds: int,
    minimum_minority_gain: float = 0.01,
    maximum_log_loss_increase: float = 0.002,
) -> dict[str, object]:
    valid = metrics[(metrics["scope"].astype(str).eq(scope)) & (metrics["split"].astype(str).eq("valid"))].copy()
    grouped = valid.groupby("variant")[["weighted_f1", "macro_f1", "minority_mean_f1", "log_loss"]].mean()
    if CONTROL_VARIANT not in grouped.index or CANDIDATE_VARIANT not in grouped.index:
        raise ValueError(f"Stage auxiliary comparison for {scope!r} requires both validation variants.")
    control = grouped.loc[CONTROL_VARIANT]
    candidate = grouped.loc[CANDIDATE_VARIANT]
    paired = valid.pivot_table(index="seed", columns="variant", values="weighted_f1", aggfunc="first").dropna()
    improved_seed_count = int((paired[CANDIDATE_VARIANT] >= paired[CONTROL_VARIANT]).sum())
    guards = {
        "weighted_f1_noninferior": bool(candidate["weighted_f1"] >= control["weighted_f1"]),
        "macro_f1_noninferior": bool(candidate["macro_f1"] >= control["macro_f1"]),
        "minority_gain_met": bool(candidate["minority_mean_f1"] >= control["minority_mean_f1"] + float(minimum_minority_gain) - 1e-12),
        "log_loss_tolerance_met": bool(candidate["log_loss"] <= control["log_loss"] + float(maximum_log_loss_increase) + 1e-12),
        "paired_weighted_f1_seed_count": bool(improved_seed_count >= int(min_improved_seeds)),
    }
    accepted = bool(all(guards.values()))
    return {
        "scope": scope,
        "accepted": accepted,
        "reason": "all_validation_stage_auxiliary_guardrails_passed" if accepted else "validation_stage_auxiliary_guardrail_failed",
        "parameter_selection_split": "inner_tune",
        "selection_split": "valid",
        "test_used_for_selection": False,
        "minimum_minority_gain": float(minimum_minority_gain),
        "maximum_log_loss_increase": float(maximum_log_loss_increase),
        "min_improved_seeds": int(min_improved_seeds),
        "improved_seed_count": improved_seed_count,
        "guards": guards,
        "control_valid_metrics": {name: float(value) for name, value in control.to_dict().items()},
        "candidate_valid_metrics": {name: float(value) for name, value in candidate.to_dict().items()},
        "validation_delta_candidate_minus_control": {name: float(candidate[name] - control[name]) for name in grouped.columns},
    }