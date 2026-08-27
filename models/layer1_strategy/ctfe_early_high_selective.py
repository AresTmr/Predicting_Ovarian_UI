from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.layer1_strategy.ctfe_auxiliary import CTFE_DOSE_LABELS

CONTROL_VARIANT = "control_fusion"
CANDIDATE_VARIANT = "early_high_selective_candidate"
LOGISTIC_VARIANT = "early_high_logistic"
CATBOOST_VARIANT = "early_high_catboost"
MODEL_TYPES = [LOGISTIC_VARIANT, CATBOOST_VARIANT]
EARLY_STAGE_GROUPS = frozenset({"d0_3", "d4_6"})
BOUNDARY_LABELS = ["medium", "high"]
EARLY_HIGH_FEATURES = [
    "afc",
    "amh",
    "gn_day",
    "current_e2",
    "current_fsh_daily_dose",
    "current_gn_dose",
    "total_follicle_count",
    "mature_follicle_count",
]
FORBIDDEN_EARLY_HIGH_FRAGMENTS = (
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
EARLY_HIGH_PROBABILITY_COLUMNS = [f"early_high_prob_{label}" for label in CTFE_DOSE_LABELS]
LABEL_INDEX = {label: index for index, label in enumerate(CTFE_DOSE_LABELS)}


@dataclass
class ConstantBinaryHead:
    positive_probability: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), float(self.positive_probability), dtype=float)
        return np.column_stack([1.0 - positive, positive])


def validate_early_high_feature_columns(feature_columns: Sequence[str]) -> None:
    bad = [
        str(name)
        for name in feature_columns
        if any(fragment in str(name).lower() for fragment in FORBIDDEN_EARLY_HIGH_FRAGMENTS)
    ]
    if bad:
        raise ValueError(f"Early high feature list contains leakage-prone columns: {bad}")


def _validate_features(frame: pd.DataFrame, feature_columns: Sequence[str]) -> None:
    validate_early_high_feature_columns(feature_columns)
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Early high feature columns missing: {missing}")


def _require_train_only(frame: pd.DataFrame) -> None:
    if "split" in frame.columns and not frame["split"].astype(str).eq("train").all():
        raise ValueError("Early high fitting is restricted to train rows.")


def _require_prediction_columns(frame: pd.DataFrame) -> None:
    required = {
        "split",
        "gn_day_group",
        "ctfe_next_fsh_dose_class",
        "ctfe_stratified_ensemble_prediction",
        *ENSEMBLE_PROBABILITY_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Early high input missing columns: {missing}")


def _normalise(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    probabilities = frame[list(columns)].astype(float).to_numpy()
    return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)


def select_early_high_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    _require_prediction_columns(frame)
    target = frame["ctfe_next_fsh_dose_class"].astype(str)
    invalid = sorted(set(target).difference(CTFE_DOSE_LABELS))
    if invalid:
        raise ValueError(f"Unknown CTFE dose labels: {invalid}")
    mask = frame["gn_day_group"].astype(str).isin(EARLY_STAGE_GROUPS) & target.isin(BOUNDARY_LABELS)
    return frame.loc[mask].copy().reset_index(drop=True)


def _build_model(model_type: str, *, seed: int) -> object:
    if model_type == LOGISTIC_VARIANT:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        C=0.5,
                        solver="liblinear",
                        max_iter=500,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    if model_type == CATBOOST_VARIANT:
        return CatBoostClassifier(
            iterations=120,
            depth=3,
            learning_rate=0.05,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
            random_seed=int(seed),
            thread_count=4,
        )
    raise ValueError(f"Unknown early high model type: {model_type}")


def fit_early_high_head(
    training_frame: pd.DataFrame,
    *,
    model_type: str,
    feature_columns: Sequence[str] = EARLY_HIGH_FEATURES,
    seed: int,
) -> tuple[object, pd.DataFrame]:
    _require_train_only(training_frame)
    _validate_features(training_frame, feature_columns)
    subset = select_early_high_training_rows(training_frame)
    y = subset["ctfe_next_fsh_dose_class"].astype(str).eq("high").astype(int)
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if subset.empty or positive == 0 or negative == 0:
        model: object = ConstantBinaryHead(float(positive > 0))
    else:
        model = _build_model(model_type, seed=int(seed))
        model.fit(subset[list(feature_columns)], y.to_numpy(dtype=int))
    support_rows = []
    for label in BOUNDARY_LABELS:
        count = int(subset["ctfe_next_fsh_dose_class"].astype(str).eq(label).sum())
        support_rows.append(
            {
                "head": "early_medium_high_boundary",
                "model_type": model_type,
                "truth_label": label,
                "sample_count": count,
                "training_stage_groups": ",".join(sorted(EARLY_STAGE_GROUPS)),
            }
        )
    return model, pd.DataFrame(support_rows)


def score_early_high_head(
    frame: pd.DataFrame,
    model: object,
    *,
    feature_columns: Sequence[str] = EARLY_HIGH_FEATURES,
) -> pd.DataFrame:
    _require_prediction_columns(frame)
    _validate_features(frame, feature_columns)
    result = frame.copy().reset_index(drop=True)
    probability = np.asarray(model.predict_proba(result[list(feature_columns)]), dtype=float)[:, 1]
    result["early_high_aux_probability"] = np.clip(probability, 0.0, 1.0)
    return result


def _medium_high_pair_is_top_boundary(probabilities: np.ndarray) -> np.ndarray:
    medium_idx = LABEL_INDEX["medium"]
    high_idx = LABEL_INDEX["high"]
    non_pair = [idx for idx in range(probabilities.shape[1]) if idx not in {medium_idx, high_idx}]
    return (probabilities[:, medium_idx] >= probabilities[:, high_idx]) & (
        probabilities[:, high_idx] >= probabilities[:, non_pair].max(axis=1) - 1e-12
    )


def apply_early_high_selective_fusion(
    frame: pd.DataFrame,
    *,
    high_probability_threshold: float,
    maximum_control_margin: float,
    blend_weight: float,
    prediction_col: str = "ctfe_early_high_prediction",
) -> pd.DataFrame:
    _require_prediction_columns(frame)
    if "early_high_aux_probability" not in frame.columns:
        raise ValueError("Early high fusion requires early_high_aux_probability.")
    for value in [high_probability_threshold, maximum_control_margin, blend_weight]:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("Early high fusion parameters must be between zero and one.")
    result = frame.copy().reset_index(drop=True)
    probabilities = _normalise(result, ENSEMBLE_PROBABILITY_COLUMNS)
    labels = np.asarray(CTFE_DOSE_LABELS, dtype=object)
    medium_idx = LABEL_INDEX["medium"]
    high_idx = LABEL_INDEX["high"]
    base_prediction = result["ctfe_stratified_ensemble_prediction"].astype(str).to_numpy()
    aux = pd.to_numeric(result["early_high_aux_probability"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    early = result["gn_day_group"].astype(str).isin(EARLY_STAGE_GROUPS).to_numpy()
    pair_top = _medium_high_pair_is_top_boundary(probabilities)
    margin = probabilities[:, medium_idx] - probabilities[:, high_idx]
    eligible = (
        early
        & (base_prediction == "medium")
        & pair_top
        & (aux >= float(high_probability_threshold))
        & (margin >= -1e-12)
        & (margin <= float(maximum_control_margin) + 1e-12)
    )
    adjustment = np.full(len(result), "none", dtype=object)
    candidate_probabilities = probabilities.copy()
    for row_index in np.flatnonzero(eligible):
        pair_sum = float(probabilities[row_index, medium_idx] + probabilities[row_index, high_idx])
        proposed_high = (1.0 - float(blend_weight)) * float(probabilities[row_index, high_idx]) + float(blend_weight) * float(aux[row_index])
        proposed_high = min(max(proposed_high, 0.0), pair_sum)
        proposed_medium = max(pair_sum - proposed_high, 0.0)
        row = candidate_probabilities[row_index].copy()
        row[medium_idx] = proposed_medium
        row[high_idx] = proposed_high
        row = row / max(row.sum(), 1e-12)
        if int(row.argmax()) == high_idx:
            candidate_probabilities[row_index] = row
            adjustment[row_index] = "medium_to_high"
    candidate_probabilities = candidate_probabilities / np.maximum(candidate_probabilities.sum(axis=1, keepdims=True), 1e-12)
    for index, label in enumerate(CTFE_DOSE_LABELS):
        result[f"early_high_prob_{label}"] = candidate_probabilities[:, index]
    result[prediction_col] = labels[candidate_probabilities.argmax(axis=1)]
    result["early_high_adjustment"] = adjustment
    result["early_high_changed_prediction"] = result[prediction_col].astype(str).to_numpy() != base_prediction
    return result


def evaluate_early_high_frame(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    probability_prefix: str = "early_high_prob_",
) -> dict[str, float | int]:
    if frame.empty:
        raise ValueError("Cannot evaluate an empty early high frame.")
    true = frame["ctfe_next_fsh_dose_class"].astype(str).to_numpy()
    pred = frame[prediction_col].astype(str).to_numpy()
    probabilities = _normalise(frame, [f"{probability_prefix}{label}" for label in CTFE_DOSE_LABELS])
    class_f1 = f1_score(true, pred, labels=CTFE_DOSE_LABELS, average=None, zero_division=0)
    distances = np.asarray([abs(LABEL_INDEX[a] - LABEL_INDEX[b]) for a, b in zip(true, pred)])
    metrics: dict[str, float | int] = {
        "sample_count": int(len(frame)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=CTFE_DOSE_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, labels=CTFE_DOSE_LABELS, average="weighted", zero_division=0)),
        "log_loss": float(log_loss([LABEL_INDEX[value] for value in true], probabilities, labels=list(range(len(CTFE_DOSE_LABELS))))),
        "adjacent_errors": int((distances == 1).sum()),
        "far_errors": int((distances >= 2).sum()),
    }
    for label, score in zip(CTFE_DOSE_LABELS, class_f1):
        metrics[f"f1_{label}"] = float(score)
    if "early_high_changed_prediction" in frame.columns:
        changed = frame["early_high_changed_prediction"].astype(bool)
        adjustment = frame.get("early_high_adjustment", pd.Series(["none"] * len(frame))).astype(str)
        medium_to_high = changed & adjustment.eq("medium_to_high")
        correct = medium_to_high & frame["ctfe_next_fsh_dose_class"].astype(str).eq("high")
        wrong = medium_to_high & ~frame["ctfe_next_fsh_dose_class"].astype(str).eq("high")
        metrics["changed_predictions"] = int(changed.sum())
        metrics["medium_to_high_changes"] = int(medium_to_high.sum())
        metrics["medium_to_high_correct"] = int(correct.sum())
        metrics["medium_to_high_wrong"] = int(wrong.sum())
        metrics["medium_to_high_net_correct"] = int(correct.sum() - wrong.sum())
    return metrics


def _control_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return evaluate_early_high_frame(
        frame,
        prediction_col="ctfe_stratified_ensemble_prediction",
        probability_prefix="ensemble_prob_",
    )


def search_early_high_parameters(
    frame: pd.DataFrame,
    *,
    selection_mask: np.ndarray,
    blend_weights: Iterable[float] = (0.20, 0.40, 0.60, 0.80),
    high_probability_thresholds: Iterable[float] = (0.55, 0.60, 0.65, 0.70, 0.75),
    maximum_control_margins: Iterable[float] = (0.02, 0.05, 0.08, 0.10, 0.15),
    maximum_log_loss_increase: float = 0.001,
) -> dict[str, object]:
    mask = np.asarray(selection_mask, dtype=bool)
    if len(mask) != len(frame) or not mask.any():
        raise ValueError("Early high parameter search requires a non-empty aligned inner-tune mask.")
    tune = frame.loc[mask].reset_index(drop=True)
    baseline = _control_metrics(tune)
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for blend_weight, threshold, margin in product(blend_weights, high_probability_thresholds, maximum_control_margins):
        candidate = apply_early_high_selective_fusion(
            tune,
            high_probability_threshold=float(threshold),
            maximum_control_margin=float(margin),
            blend_weight=float(blend_weight),
        )
        metrics = evaluate_early_high_frame(candidate, prediction_col="ctfe_early_high_prediction")
        changed = int(metrics.get("changed_predictions", 0))
        passes = bool(
            changed > 0
            and float(metrics["weighted_f1"]) > float(baseline["weighted_f1"])
            and float(metrics["accuracy"]) >= float(baseline["accuracy"])
            and float(metrics["macro_f1"]) >= float(baseline["macro_f1"])
            and float(metrics["f1_high"]) > float(baseline["f1_high"])
            and float(metrics["log_loss"]) <= float(baseline["log_loss"]) + float(maximum_log_loss_increase) + 1e-12
            and int(metrics["adjacent_errors"]) <= int(baseline["adjacent_errors"])
        )
        result = {
            "blend_weight": float(blend_weight),
            "high_probability_threshold": float(threshold),
            "maximum_control_margin": float(margin),
            "changed_predictions": changed,
            "passes_inner_tune_guardrail": passes,
            **{f"tune_{name}": value for name, value in metrics.items()},
        }
        rows.append(result)
        key = (
            float(result["tune_weighted_f1"]),
            float(result["tune_accuracy"]),
            float(result["tune_macro_f1"]),
            float(result["tune_f1_high"]),
            -float(result["tune_log_loss"]),
            int(result.get("tune_medium_to_high_net_correct", 0)),
            -changed,
        )
        if passes and (best is None or key > (
            float(best["tune_weighted_f1"]),
            float(best["tune_accuracy"]),
            float(best["tune_macro_f1"]),
            float(best["tune_f1_high"]),
            -float(best["tune_log_loss"]),
            int(best.get("tune_medium_to_high_net_correct", 0)),
            -int(best["changed_predictions"]),
        )):
            best = result
    search_results = pd.DataFrame(rows).sort_values(
        ["passes_inner_tune_guardrail", "tune_weighted_f1", "tune_accuracy", "tune_macro_f1", "tune_f1_high", "tune_log_loss"],
        ascending=[False, False, False, False, False, True],
    )
    if best is None:
        best = {
            "blend_weight": 0.0,
            "high_probability_threshold": 1.0,
            "maximum_control_margin": 0.0,
            "changed_predictions": 0,
        }
        accepted_for_validation = False
        reason = "no_nontrivial_inner_tune_guarded_early_high_fusion"
    else:
        accepted_for_validation = True
        reason = "inner_tune_guarded_early_high_fusion_selected"
    return {
        **best,
        "accepted_for_validation": accepted_for_validation,
        "reason": reason,
        "parameter_selection_split": "inner_tune",
        "test_used_for_selection": False,
        "baseline_metrics": baseline,
        "search_results": search_results,
    }


def summarize_early_high_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "f1_stop",
        "f1_decrease",
        "f1_low",
        "f1_medium",
        "f1_high",
        "adjacent_errors",
        "far_errors",
        "changed_predictions",
        "medium_to_high_changes",
        "medium_to_high_net_correct",
    ]
    rows: list[dict[str, object]] = []
    for (scope, variant, split), group in metrics.groupby(["scope", "variant", "split"], sort=True):
        entry: dict[str, object] = {
            "scope": scope,
            "variant": variant,
            "split": split,
            "n_seeds": int(group["seed"].nunique()),
        }
        for name in metric_names:
            if name in group.columns:
                values = pd.to_numeric(group[name], errors="coerce")
                entry[f"{name}_mean"] = float(values.mean())
                entry[f"{name}_std"] = float(values.std(ddof=0))
        rows.append(entry)
    return pd.DataFrame(rows)


def assess_early_high_promotion(
    metrics: pd.DataFrame,
    *,
    scope: str,
    min_improved_seeds: int,
    maximum_log_loss_increase: float = 0.001,
) -> dict[str, object]:
    valid = metrics[(metrics["scope"].astype(str).eq(scope)) & (metrics["split"].astype(str).eq("valid"))].copy()
    grouped = valid.groupby("variant")[["accuracy", "weighted_f1", "macro_f1", "log_loss", "f1_high", "adjacent_errors", "changed_predictions"]].mean()
    if CONTROL_VARIANT not in grouped.index or CANDIDATE_VARIANT not in grouped.index:
        raise ValueError(f"Early high comparison for {scope!r} requires both validation variants.")
    control = grouped.loc[CONTROL_VARIANT]
    candidate = grouped.loc[CANDIDATE_VARIANT]
    paired = valid.pivot_table(index="seed", columns="variant", values="weighted_f1", aggfunc="first").dropna()
    improved_seed_count = int((paired[CANDIDATE_VARIANT] > paired[CONTROL_VARIANT]).sum())
    guards = {
        "weighted_f1_improved": bool(candidate["weighted_f1"] > control["weighted_f1"]),
        "accuracy_noninferior": bool(candidate["accuracy"] >= control["accuracy"]),
        "macro_f1_noninferior": bool(candidate["macro_f1"] >= control["macro_f1"]),
        "high_f1_improved": bool(candidate["f1_high"] > control["f1_high"]),
        "log_loss_tolerance_met": bool(candidate["log_loss"] <= control["log_loss"] + float(maximum_log_loss_increase) + 1e-12),
        "adjacent_errors_noninferior": bool(candidate["adjacent_errors"] <= control["adjacent_errors"]),
        "changed_rows_positive": bool(candidate["changed_predictions"] > 0),
        "paired_weighted_f1_seed_count": bool(improved_seed_count >= int(min_improved_seeds)),
    }
    accepted = bool(all(guards.values()))
    return {
        "scope": scope,
        "accepted": accepted,
        "reason": "all_validation_early_high_guardrails_passed" if accepted else "validation_early_high_guardrail_failed",
        "parameter_selection_split": "inner_tune",
        "selection_split": "valid",
        "test_used_for_selection": False,
        "maximum_log_loss_increase": float(maximum_log_loss_increase),
        "min_improved_seeds": int(min_improved_seeds),
        "improved_seed_count": improved_seed_count,
        "guards": guards,
        "control_valid_metrics": {name: float(value) for name, value in control.to_dict().items()},
        "candidate_valid_metrics": {name: float(value) for name, value in candidate.to_dict().items()},
        "validation_delta_candidate_minus_control": {name: float(candidate[name] - control[name]) for name in grouped.columns},
    }
