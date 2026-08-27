from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

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
CANDIDATE_VARIANT = "hierarchical_candidate"
LOGISTIC_VARIANT = "hierarchical_logistic"
CATBOOST_VARIANT = "hierarchical_catboost"
FOUR_CLASS_LABELS = ["stop", "decrease", "low", "upper_dose"]
UPPER_DOSE_INTERNAL_LABELS = frozenset({"medium", "high"})
EARLY_STAGE_GROUPS = frozenset({"d0_3", "d4_6"})
HIERARCHICAL_FEATURES = [
    "afc",
    "amh",
    "gn_day",
    "current_e2",
    "current_fsh_daily_dose",
    "current_gn_dose",
    "total_follicle_count",
    "mature_follicle_count",
]
FORBIDDEN_HIERARCHICAL_FRAGMENTS = (
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
HIERARCHICAL_PROBABILITY_COLUMNS = [f"hierarchical_prob_{label}" for label in CTFE_DOSE_LABELS]
FOUR_CLASS_PROBABILITY_COLUMNS = [f"four_class_prob_{label}" for label in FOUR_CLASS_LABELS]


@dataclass
class ConstantBinaryHead:
    probability: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), float(self.probability), dtype=float)
        return np.column_stack([1.0 - positive, positive])


@dataclass
class ConstantMulticlassModel:
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        self.classes_ = np.asarray(FOUR_CLASS_LABELS, dtype=object)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.probabilities.reshape(1, -1), len(frame), axis=0)


def add_hierarchical_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add target columns without changing the existing five-class internal labels."""

    if "ctfe_next_fsh_dose_class" not in frame.columns:
        raise ValueError("Hierarchical CTFE targets require ctfe_next_fsh_dose_class.")
    result = frame.copy()
    target = result["ctfe_next_fsh_dose_class"].astype(str)
    invalid = sorted(set(target).difference(CTFE_DOSE_LABELS))
    if invalid:
        raise ValueError(f"Unknown CTFE dose labels: {invalid}")
    result["upper_dose_target"] = target.isin(UPPER_DOSE_INTERNAL_LABELS).astype(int)
    result["high_dose_target"] = target.eq("high").astype(int)
    result["four_class_target"] = target.replace({"medium": "upper_dose", "high": "upper_dose"})
    return result


def validate_hierarchical_feature_columns(feature_columns: Sequence[str]) -> None:
    bad = [
        name
        for name in feature_columns
        if any(
            fragment in str(name).lower()
            for fragment in FORBIDDEN_HIERARCHICAL_FRAGMENTS
        )
    ]
    if bad:
        raise ValueError(f"Hierarchical CTFE feature list contains leakage-prone columns: {bad}")


def _validate_features(frame: pd.DataFrame, feature_columns: Sequence[str]) -> None:
    validate_hierarchical_feature_columns(feature_columns)
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Hierarchical CTFE feature columns missing: {missing}")


def _require_train_only(frame: pd.DataFrame) -> None:
    if "split" in frame.columns and not frame["split"].astype(str).eq("train").all():
        raise ValueError("Hierarchical CTFE fitting is restricted to train rows.")


def _build_model(model_type: str, *, seed: int, multiclass: bool) -> object:
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
                        solver="lbfgs" if multiclass else "liblinear",
                        max_iter=500,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    if model_type == CATBOOST_VARIANT:
        return CatBoostClassifier(
            iterations=120,
            depth=4,
            learning_rate=0.05,
            loss_function="MultiClass" if multiclass else "Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
            random_seed=int(seed),
            thread_count=4,
        )
    raise ValueError(f"Unknown hierarchical CTFE model type: {model_type}")


def _fit_binary_head(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    model_type: str,
    feature_columns: Sequence[str],
    seed: int,
) -> object:
    positive = int(target.sum())
    if frame.empty or positive == 0 or positive == len(frame):
        return ConstantBinaryHead(float(positive > 0))
    model = _build_model(model_type, seed=seed, multiclass=False)
    model.fit(frame[list(feature_columns)], target.to_numpy(dtype=int))
    return model


def fit_hierarchical_heads(
    training_frame: pd.DataFrame,
    *,
    model_type: str,
    feature_columns: Sequence[str] = HIERARCHICAL_FEATURES,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit upper-dose and early high-dose heads using train rows only."""

    _require_train_only(training_frame)
    _validate_features(training_frame, feature_columns)
    prepared = add_hierarchical_targets(training_frame)
    upper_target = prepared["upper_dose_target"].astype(int)
    upper_head = _fit_binary_head(
        prepared,
        upper_target,
        model_type=model_type,
        feature_columns=feature_columns,
        seed=int(seed) + 1,
    )
    splitter_rows = prepared[
        prepared["gn_day_group"].astype(str).isin(EARLY_STAGE_GROUPS)
        & prepared["ctfe_next_fsh_dose_class"].astype(str).isin(UPPER_DOSE_INTERNAL_LABELS)
    ].copy()
    splitter_target = splitter_rows["high_dose_target"].astype(int)
    splitter_head = _fit_binary_head(
        splitter_rows,
        splitter_target,
        model_type=model_type,
        feature_columns=feature_columns,
        seed=int(seed) + 2,
    )
    support = pd.DataFrame(
        [
            {
                "head": "upper_dose_gate",
                "model_type": model_type,
                "sample_count": int(len(prepared)),
                "positive_count": int(upper_target.sum()),
                "negative_count": int(len(prepared) - upper_target.sum()),
            },
            {
                "head": "early_high_splitter",
                "model_type": model_type,
                "sample_count": int(len(splitter_rows)),
                "positive_count": int(splitter_target.sum()),
                "negative_count": int(len(splitter_rows) - splitter_target.sum()),
            },
        ]
    )
    return {"upper_dose_gate": upper_head, "early_high_splitter": splitter_head}, support


def add_hierarchical_probabilities(
    frame: pd.DataFrame,
    heads: Mapping[str, object],
    *,
    feature_columns: Sequence[str] = HIERARCHICAL_FEATURES,
) -> pd.DataFrame:
    _validate_features(frame, feature_columns)
    if "gn_day_group" not in frame.columns:
        raise ValueError("Hierarchical CTFE scoring requires gn_day_group.")
    result = frame.copy().reset_index(drop=True)
    result["hierarchical_upper_probability"] = np.asarray(
        heads["upper_dose_gate"].predict_proba(result[list(feature_columns)])
    )[:, 1]
    high_probability = np.zeros(len(result), dtype=float)
    early = result["gn_day_group"].astype(str).isin(EARLY_STAGE_GROUPS).to_numpy()
    if early.any():
        high_probability[early] = np.asarray(
            heads["early_high_splitter"].predict_proba(result.loc[early, list(feature_columns)])
        )[:, 1]
    result["hierarchical_high_given_upper_probability"] = high_probability
    result["hierarchical_splitter_applied"] = early
    return result


def _normalised_control_probabilities(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(ENSEMBLE_PROBABILITY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Hierarchical CTFE input missing control probabilities: {missing}")
    probabilities = frame[ENSEMBLE_PROBABILITY_COLUMNS].astype(float).to_numpy()
    return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)


def compose_hierarchical_probabilities(
    frame: pd.DataFrame,
    *,
    upper_weight: float,
    split_weight: float,
    prediction_col: str = "ctfe_hierarchical_prediction",
) -> pd.DataFrame:
    """Blend upper-region evidence and reallocate medium/high only during early monitoring."""

    if not 0.0 <= float(upper_weight) <= 1.0 or not 0.0 <= float(split_weight) <= 1.0:
        raise ValueError("Hierarchical CTFE fusion weights must be between 0 and 1.")
    required = {
        "gn_day_group",
        "hierarchical_upper_probability",
        "hierarchical_high_given_upper_probability",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Hierarchical CTFE scoring input missing columns: {missing}")
    result = frame.copy().reset_index(drop=True)
    probabilities = _normalised_control_probabilities(result)
    medium_idx = CTFE_DOSE_LABELS.index("medium")
    high_idx = CTFE_DOSE_LABELS.index("high")
    base_upper = probabilities[:, medium_idx] + probabilities[:, high_idx]
    gate = pd.to_numeric(result["hierarchical_upper_probability"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    updated_upper = (1.0 - float(upper_weight)) * base_upper + float(upper_weight) * gate
    updated_upper = np.clip(updated_upper, 0.0, 1.0)
    high_share = np.divide(
        probabilities[:, high_idx],
        np.maximum(base_upper, 1e-12),
        out=np.full(len(result), 0.5, dtype=float),
        where=base_upper > 1e-12,
    )
    early = result["gn_day_group"].astype(str).isin(EARLY_STAGE_GROUPS).to_numpy()
    splitter = pd.to_numeric(
        result["hierarchical_high_given_upper_probability"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    high_share[early] = (
        (1.0 - float(split_weight)) * high_share[early]
        + float(split_weight) * splitter[early]
    )
    old_other = np.maximum(1.0 - base_upper, 1e-12)
    new_other = np.maximum(1.0 - updated_upper, 0.0)
    for index in range(len(CTFE_DOSE_LABELS)):
        if index not in {medium_idx, high_idx}:
            probabilities[:, index] *= new_other / old_other
    probabilities[:, medium_idx] = updated_upper * (1.0 - high_share)
    probabilities[:, high_idx] = updated_upper * high_share
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    for index, label in enumerate(CTFE_DOSE_LABELS):
        result[f"hierarchical_prob_{label}"] = probabilities[:, index]
    result[prediction_col] = np.asarray(CTFE_DOSE_LABELS, dtype=object)[probabilities.argmax(axis=1)]
    base_prediction = (
        result["ctfe_stratified_ensemble_prediction"].astype(str).to_numpy()
        if "ctfe_stratified_ensemble_prediction" in result.columns
        else np.asarray(CTFE_DOSE_LABELS, dtype=object)[_normalised_control_probabilities(result).argmax(axis=1)]
    )
    result["hierarchical_changed_prediction"] = result[prediction_col].astype(str).to_numpy() != base_prediction
    return result


def evaluate_hierarchical_frame(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    probability_prefix: str,
) -> dict[str, float | int]:
    if frame.empty:
        raise ValueError("Cannot evaluate an empty hierarchical CTFE frame.")
    y_true = frame["ctfe_next_fsh_dose_class"].astype(str).to_numpy()
    y_pred = frame[prediction_col].astype(str).to_numpy()
    probabilities = frame[[f"{probability_prefix}{label}" for label in CTFE_DOSE_LABELS]].astype(float).to_numpy()
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    scores = f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average=None, zero_division=0)
    metrics: dict[str, float | int] = {
        "sample_count": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CTFE_DOSE_LABELS, average="weighted", zero_division=0)),
        "log_loss": float(
            log_loss(
                np.asarray([CTFE_DOSE_LABELS.index(value) for value in y_true], dtype=int),
                probabilities,
                labels=list(range(len(CTFE_DOSE_LABELS))),
            )
        ),
    }
    for label, value in zip(CTFE_DOSE_LABELS, scores):
        metrics[f"f1_{label}"] = float(value)
    metrics["f1_high_dose"] = float(metrics["f1_high"])
    if "hierarchical_changed_prediction" in frame.columns:
        metrics["changed_predictions"] = int(frame["hierarchical_changed_prediction"].sum())
    return metrics


def search_hierarchical_fusion_parameters(
    frame: pd.DataFrame,
    *,
    selection_mask: np.ndarray,
    upper_weights: Iterable[float] = (0.0, 0.10, 0.20, 0.30),
    split_weights: Iterable[float] = (0.0, 0.10, 0.20, 0.30, 0.40, 0.60),
    maximum_log_loss_increase: float = 0.002,
) -> dict[str, object]:
    mask = np.asarray(selection_mask, dtype=bool)
    if len(mask) != len(frame) or not mask.any():
        raise ValueError("Hierarchical CTFE selection requires aligned non-empty inner-tune rows.")
    tune = frame.loc[mask].reset_index(drop=True)
    baseline = evaluate_hierarchical_frame(
        tune,
        prediction_col="ctfe_stratified_ensemble_prediction",
        probability_prefix="ensemble_prob_",
    )
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for upper_weight, split_weight in product(upper_weights, split_weights):
        candidate = compose_hierarchical_probabilities(
            tune, upper_weight=float(upper_weight), split_weight=float(split_weight)
        )
        metrics = evaluate_hierarchical_frame(
            candidate,
            prediction_col="ctfe_hierarchical_prediction",
            probability_prefix="hierarchical_prob_",
        )
        passes = bool(
            int(metrics.get("changed_predictions", 0)) > 0
            and float(metrics["weighted_f1"]) >= float(baseline["weighted_f1"])
            and float(metrics["macro_f1"]) >= float(baseline["macro_f1"])
            and float(metrics["f1_high"]) >= float(baseline["f1_high"])
            and float(metrics["log_loss"]) <= float(baseline["log_loss"]) + float(maximum_log_loss_increase)
        )
        row = {
            "upper_weight": float(upper_weight),
            "split_weight": float(split_weight),
            "passes_inner_tune_guardrail": passes,
            **{f"tune_{name}": value for name, value in metrics.items()},
        }
        rows.append(row)
        key = (
            float(metrics["f1_high"]),
            float(metrics["macro_f1"]),
            float(metrics["weighted_f1"]),
            -float(metrics["log_loss"]),
        )
        if passes and (
            best is None
            or key
            > (
                float(best["tune_f1_high"]),
                float(best["tune_macro_f1"]),
                float(best["tune_weighted_f1"]),
                -float(best["tune_log_loss"]),
            )
        ):
            best = row
    search_results = pd.DataFrame(rows).sort_values(
        ["passes_inner_tune_guardrail", "tune_f1_high", "tune_macro_f1", "tune_weighted_f1", "tune_log_loss"],
        ascending=[False, False, False, False, True],
    )
    if best is None:
        best = {"upper_weight": 0.0, "split_weight": 0.0}
        accepted_for_validation = False
        reason = "no_nontrivial_inner_tune_hierarchical_fusion"
    else:
        accepted_for_validation = True
        reason = "inner_tune_hierarchical_fusion_selected"
    return {
        **best,
        "accepted_for_validation": accepted_for_validation,
        "reason": reason,
        "parameter_selection_split": "inner_tune",
        "test_used_for_selection": False,
        "baseline_metrics": baseline,
        "search_results": search_results,
    }


def assess_hierarchical_promotion(
    metrics: pd.DataFrame,
    *,
    scope: str,
    min_improved_seeds: int,
    minimum_high_gain: float,
    maximum_stop_drop: float = 0.005,
    maximum_log_loss_increase: float = 0.002,
) -> dict[str, object]:
    valid = metrics[
        metrics["scope"].astype(str).eq(scope) & metrics["split"].astype(str).eq("valid")
    ].copy()
    grouped = valid.groupby("variant")[["weighted_f1", "macro_f1", "f1_high", "f1_stop", "log_loss"]].mean()
    if CONTROL_VARIANT not in grouped.index or CANDIDATE_VARIANT not in grouped.index:
        raise ValueError(f"Hierarchical comparison for {scope!r} requires both validation variants.")
    control = grouped.loc[CONTROL_VARIANT]
    candidate = grouped.loc[CANDIDATE_VARIANT]
    paired = valid.pivot_table(index="seed", columns="variant", values="weighted_f1", aggfunc="first").dropna()
    improved_seed_count = int((paired[CANDIDATE_VARIANT] >= paired[CONTROL_VARIANT]).sum())
    guards = {
        "weighted_f1_noninferior": bool(candidate["weighted_f1"] >= control["weighted_f1"]),
        "macro_f1_noninferior": bool(candidate["macro_f1"] >= control["macro_f1"]),
        "high_dose_gain_met": bool(candidate["f1_high"] >= control["f1_high"] + float(minimum_high_gain) - 1e-12),
        "stop_drop_within_limit": bool(candidate["f1_stop"] >= control["f1_stop"] - float(maximum_stop_drop) - 1e-12),
        "log_loss_tolerance_met": bool(candidate["log_loss"] <= control["log_loss"] + float(maximum_log_loss_increase) + 1e-12),
        "paired_weighted_f1_seed_count": bool(improved_seed_count >= int(min_improved_seeds)),
    }
    accepted = bool(all(guards.values()))
    return {
        "scope": scope,
        "accepted": accepted,
        "reason": "all_validation_hierarchical_guardrails_passed" if accepted else "validation_hierarchical_guardrail_failed",
        "parameter_selection_split": "inner_tune",
        "selection_split": "valid",
        "test_used_for_selection": False,
        "minimum_high_gain": float(minimum_high_gain),
        "maximum_stop_drop": float(maximum_stop_drop),
        "maximum_log_loss_increase": float(maximum_log_loss_increase),
        "min_improved_seeds": int(min_improved_seeds),
        "improved_seed_count": improved_seed_count,
        "guards": guards,
        "control_valid_metrics": {name: float(value) for name, value in control.to_dict().items()},
        "candidate_valid_metrics": {name: float(value) for name, value in candidate.to_dict().items()},
        "validation_delta_candidate_minus_control": {
            name: float(candidate[name] - control[name]) for name in grouped.columns
        },
    }


def fit_four_class_sensitivity_model(
    training_frame: pd.DataFrame,
    *,
    model_type: str,
    feature_columns: Sequence[str] = HIERARCHICAL_FEATURES,
    seed: int,
) -> tuple[object, pd.DataFrame]:
    _require_train_only(training_frame)
    _validate_features(training_frame, feature_columns)
    prepared = add_hierarchical_targets(training_frame)
    target = prepared["four_class_target"].astype(str)
    counts = target.value_counts().reindex(FOUR_CLASS_LABELS, fill_value=0)
    if (counts > 0).sum() < 2:
        probabilities = counts.to_numpy(dtype=float)
        probabilities = probabilities / max(float(probabilities.sum()), 1.0)
        model: object = ConstantMulticlassModel(probabilities)
    else:
        model = _build_model(model_type, seed=int(seed), multiclass=True)
        model.fit(prepared[list(feature_columns)], target.to_numpy())
    support = pd.DataFrame(
        {
            "four_class_target": FOUR_CLASS_LABELS,
            "sample_count": [int(counts[label]) for label in FOUR_CLASS_LABELS],
            "model_type": model_type,
        }
    )
    return model, support


def score_four_class_sensitivity(
    frame: pd.DataFrame,
    model: object,
    *,
    feature_columns: Sequence[str] = HIERARCHICAL_FEATURES,
) -> pd.DataFrame:
    _validate_features(frame, feature_columns)
    result = add_hierarchical_targets(frame) if "ctfe_next_fsh_dose_class" in frame.columns else frame.copy()
    raw = np.asarray(model.predict_proba(result[list(feature_columns)]), dtype=float)
    classes = [str(value) for value in getattr(model, "classes_", FOUR_CLASS_LABELS)]
    probabilities = np.zeros((len(result), len(FOUR_CLASS_LABELS)), dtype=float)
    for position, label in enumerate(classes):
        if label in FOUR_CLASS_LABELS:
            probabilities[:, FOUR_CLASS_LABELS.index(label)] = raw[:, position]
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    for position, label in enumerate(FOUR_CLASS_LABELS):
        result[f"four_class_prob_{label}"] = probabilities[:, position]
    result["ctfe_four_class_prediction"] = np.asarray(FOUR_CLASS_LABELS, dtype=object)[probabilities.argmax(axis=1)]
    return result


def evaluate_four_class_sensitivity(
    frame: pd.DataFrame,
    *,
    split: str | None = None,
    prediction_col: str = "ctfe_four_class_prediction",
) -> dict[str, float | int | str]:
    subset = frame if split is None else frame[frame["split"].astype(str).eq(str(split))]
    if subset.empty:
        raise ValueError("Cannot evaluate an empty four-class sensitivity frame.")
    y_true = subset["four_class_target"].astype(str).to_numpy()
    y_pred = subset[prediction_col].astype(str).to_numpy()
    probabilities = subset[FOUR_CLASS_PROBABILITY_COLUMNS].astype(float).to_numpy()
    metrics: dict[str, float | int | str] = {
        "split": split or "all",
        "sample_count": int(len(subset)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=FOUR_CLASS_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=FOUR_CLASS_LABELS, average="weighted", zero_division=0)),
        "log_loss": float(
            log_loss(
                np.asarray([FOUR_CLASS_LABELS.index(value) for value in y_true], dtype=int),
                probabilities,
                labels=list(range(len(FOUR_CLASS_LABELS))),
            )
        ),
    }
    scores = f1_score(y_true, y_pred, labels=FOUR_CLASS_LABELS, average=None, zero_division=0)
    for label, value in zip(FOUR_CLASS_LABELS, scores):
        metrics[f"f1_{label}"] = float(value)
    return metrics
